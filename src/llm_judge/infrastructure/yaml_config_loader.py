"""YAML configuration loader for llm-judge."""

from __future__ import annotations

import copy
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class YAMLConfigLoader:
    """Loads configuration from config.yaml or config.example.yaml.

    This class provides a thread-safe way to load and access YAML configuration.
    It looks for config.yaml in the project root first, and falls back to
    config.example.yaml if not found (useful for tests).
    """

    _instance: Optional[YAMLConfigLoader] = None
    _lock = threading.RLock()

    def __init__(self, config_path: Optional[Path] = None, use_example: bool = False):
        """Initialize the YAML config loader.

        Args:
            config_path: Optional explicit path to config file
            use_example: If True, use config.example.yaml instead of config.yaml
        """
        self._config: Dict[str, Any] = {}
        self._loaded = False
        self._config_path = config_path
        self._use_example = use_example

    @classmethod
    def get_instance(cls, config_path: Optional[Path] = None, use_example: bool = False) -> YAMLConfigLoader:
        """Get or create singleton instance.

        Args:
            config_path: Optional explicit path to config file
            use_example: If True, use config.example.yaml instead of config.yaml

        Returns:
            Singleton YAMLConfigLoader instance
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = YAMLConfigLoader(config_path=config_path, use_example=use_example)
                cls._instance.load()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (mainly for testing)."""
        with cls._lock:
            cls._instance = None

    def _find_config_file(self) -> Path:
        """Find the config file to load.

        Returns:
            Path to config file

        Raises:
            FileNotFoundError: If no config file is found
        """
        if self._config_path:
            if not self._config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self._config_path}")
            return self._config_path

        # Find project root (where config files should be)
        project_root = Path(__file__).resolve().parents[3]

        # Check for config.yaml or config.example.yaml
        if self._use_example or os.getenv("TESTING"):
            config_file = project_root / "config.example.yaml"
        else:
            config_file = project_root / "config.yaml"
            if not config_file.exists():
                config_file = project_root / "config.example.yaml"

        if not config_file.exists():
            raise FileNotFoundError(
                f"No configuration file found. Expected {project_root / 'config.yaml'} "
                f"or {project_root / 'config.example.yaml'}"
            )

        return config_file

    def load(self) -> None:
        """Load the configuration from YAML file."""
        with self._lock:
            # Early return if already loaded
            if self._loaded:
                return

            config_file = self._find_config_file()

            with open(config_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                raise TypeError(f"Config file must contain a YAML mapping, got {type(data)}")

            self._config = data
            self._loaded = True

    def get(self, key: str, default: Any = None) -> Any:  # pyright: ignore[reportUnknownVariableType]
        """Get a configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., 'inference.endpoint')
            default: Default value if key is not found

        Returns:
            Configuration value or default
        """
        with self._lock:
            if not self._loaded:
                self.load()

            keys = key.split(".")
            value: Any = self._config

            for k in keys:
                if not isinstance(value, dict):
                    return default
                next_value: Any = value.get(k)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                if next_value is None:
                    return default
                value = next_value  # pyright: ignore[reportUnknownVariableType]

            return value  # pyright: ignore[reportUnknownVariableType]

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary.

        Returns:
            Deep copy of entire configuration to prevent mutation of internal state
        """
        with self._lock:
            if not self._loaded:
                self.load()
            return copy.deepcopy(self._config)


def load_config(config_path: Optional[Path] = None, use_example: bool = False) -> YAMLConfigLoader:
    """Load configuration from YAML file.

    This is a convenience function that returns the singleton instance.

    Args:
        config_path: Optional explicit path to config file
        use_example: If True, use config.example.yaml instead of config.yaml

    Returns:
        YAMLConfigLoader instance
    """
    return YAMLConfigLoader.get_instance(config_path=config_path, use_example=use_example)


__all__ = ["YAMLConfigLoader", "load_config"]
