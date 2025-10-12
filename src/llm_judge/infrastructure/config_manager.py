"""Thread-safe configuration management."""

import os
import threading
from typing import Any, Dict, Optional
from pathlib import Path
import json

import yaml

from ..services import IConfigurationManager


class ConfigurationManager(IConfigurationManager):
    """Thread-safe configuration manager with file and environment support."""

    def __init__(self, config_file: Optional[Path] = None, auto_reload: bool = False):
        """Initialize configuration manager.

        Args:
            config_file: Path to configuration file (YAML or JSON)
            auto_reload: Whether to automatically reload config on access
        """
        self._config_file = config_file
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
            value = self._config

            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                    if value is None:
                        return default
                else:
                    return default

            return value if value is not None else default

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
            return value if isinstance(value, dict) else {}

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

            with open(self._config_file, "r", encoding="utf-8") as f:
                if suffix in {".yaml", ".yml"}:
                    data = yaml.safe_load(f)
                elif suffix == ".json":
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {suffix}")

            if not isinstance(data, dict):
                raise TypeError("Configuration file must contain a mapping/object")

            self._config = data
            self._loaded = True

    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration as a dictionary."""
        with self._lock:
            if self._auto_reload and not self._loaded:
                self.reload()
            return self._config.copy()

    def merge(self, config: Dict[str, Any]) -> None:
        """Merge configuration dictionary into current config."""
        with self._lock:
            self._deep_merge(self._config, config)

    @staticmethod
    def _deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
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
