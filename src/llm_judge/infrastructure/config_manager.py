"""Thread-safe configuration management."""

import os
import threading
from typing import Any, Dict, Optional, cast, Mapping
from pathlib import Path
import json

import yaml

from ..services import IConfigurationManager


class ConfigurationManager(IConfigurationManager):
    """Thread-safe configuration manager with file and environment support."""

    def __init__(self, config_file: Optional[Path | str] = None, auto_reload: bool = False):
        """Initialize configuration manager.

        Args:
            config_file: Path to configuration file (YAML or JSON)
            auto_reload: Whether to automatically reload config on access
        """
        if isinstance(config_file, str):
            config_path: Optional[Path] = Path(config_file)
        else:
            config_path = config_file

        self._config_file: Optional[Path] = config_path
        self._auto_reload = auto_reload
        self._lock = threading.RLock()
        self._config: Dict[str, Any] = {}
        self._loaded = False

        if config_file:
            self.reload()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Supports dot notation for nested keys (e.g., 'api.timeout').
        Environment variables override file config if present.
        """
        with self._lock:
            if self._auto_reload and not self._loaded:
                self.reload()

            # Check environment variable first (uppercase with underscores)
            env_key = key.upper().replace(".", "_")
            env_value = os.getenv(env_key)
            if env_value is not None:
                return self._parse_env_value(env_value)

            # Navigate nested keys
            keys = key.split(".")
            current: Any = self._config

            for k in keys:
                if isinstance(current, Mapping):
                    current_map = cast(Mapping[str, Any], current)
                    if k not in current_map:
                        return default
                    next_value: Any = current_map.get(k)
                    if next_value is None:
                        return default
                    current = next_value
                else:
                    return default

            return current if current is not None else default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value.

        Note: This only sets in-memory config, does not persist to file.
        """
        with self._lock:
            keys = key.split(".")
            config = self._config

            # Navigate to parent
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]

            # Set the final key
            config[keys[-1]] = value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        with self._lock:
            if self._auto_reload and not self._loaded:
                self.reload()

            value = self.get(section, {})
            if isinstance(value, dict):
                return cast(Dict[str, Any], value)
            return {}

    def reload(self) -> None:
        """Reload configuration from file."""
        with self._lock:
            if not self._config_file:
                self._config = {}
                self._loaded = True
                return

            if not self._config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {self._config_file}")

            # Load based on file extension
            suffix = self._config_file.suffix.lower()

            data: Any
            with open(self._config_file, "r", encoding="utf-8") as f:
                if suffix in {".yaml", ".yml"}:
                    data = yaml.safe_load(f)
                elif suffix == ".json":
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {suffix}")

            if data is None or not isinstance(data, dict):
                raise TypeError("Configuration file must contain a mapping/object")

            data_dict: Dict[str, Any] = cast(Dict[str, Any], data)
            self._config = dict(data_dict)
            self._loaded = True

    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration as a dictionary."""
        with self._lock:
            if self._auto_reload and not self._loaded:
                self.reload()
            return self._config.copy()

    def merge(self, config: Mapping[str, Any]) -> None:
        """Merge configuration dictionary into current config."""
        with self._lock:
            merge_data: Dict[str, Any] = {str(key): value for key, value in config.items()}
            self._deep_merge(self._config, merge_data)

    @staticmethod
    def _deep_merge(target: Dict[str, Any], source: Mapping[str, Any]) -> None:
        """Deep merge source into target dictionary."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                ConfigurationManager._deep_merge(target[key], value)
            else:
                target[key] = value

    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try to parse as JSON for complex types
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            # Return as string if not valid JSON
            return value


class SingletonConfigurationManager:
    """Singleton wrapper for ConfigurationManager with thread-safe initialization."""

    _instance: Optional[ConfigurationManager] = None
    _lock = threading.RLock()

    @classmethod
    def get_instance(
        cls, config_file: Optional[Path] = None, auto_reload: bool = False, force_reload: bool = False
    ) -> ConfigurationManager:
        """Get or create singleton instance.

        Args:
            config_file: Path to configuration file (only used on first call)
            auto_reload: Enable auto-reload (only used on first call)
            force_reload: Force recreation of instance
        """
        with cls._lock:
            if cls._instance is None or force_reload:
                cls._instance = ConfigurationManager(config_file=config_file, auto_reload=auto_reload)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (mainly for testing)."""
        with cls._lock:
            cls._instance = None


__all__ = ["ConfigurationManager", "SingletonConfigurationManager"]
