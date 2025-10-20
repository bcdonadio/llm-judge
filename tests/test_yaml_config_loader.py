"""Tests for YAML configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llm_judge.infrastructure.yaml_config_loader import YAMLConfigLoader, load_config


def test_yaml_config_loader_loads_from_explicit_path(tmp_path: Path) -> None:
    """Test loading config from explicit path."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
inference:
  endpoint: https://test.example.com/api
  key: test_key_123
  retries:
    max: 5
    timeout: 60
log_level: INFO
"""
    )

    loader = YAMLConfigLoader(config_path=config_file)
    loader.load()

    assert loader.get("inference.endpoint") == "https://test.example.com/api"
    assert loader.get("inference.key") == "test_key_123"
    assert loader.get("inference.retries.max") == 5
    assert loader.get("inference.retries.timeout") == 60
    assert loader.get("log_level") == "INFO"


def test_yaml_config_loader_get_with_default(tmp_path: Path) -> None:
    """Test get with default value."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("key: value\n")

    loader = YAMLConfigLoader(config_path=config_file)
    loader.load()

    assert loader.get("missing.key", "default") == "default"
    assert loader.get("key") == "value"


def test_yaml_config_loader_get_all(tmp_path: Path) -> None:
    """Test get_all returns entire config."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
a: 1
b:
  c: 2
  d: 3
"""
    )

    loader = YAMLConfigLoader(config_path=config_file)
    loader.load()

    all_config = loader.get_all()
    assert all_config == {"a": 1, "b": {"c": 2, "d": 3}}

    # Test that returned dict is a copy
    all_config["a"] = 999
    assert loader.get("a") == 1


def test_yaml_config_loader_file_not_found(tmp_path: Path) -> None:
    """Test error when config file doesn't exist."""
    loader = YAMLConfigLoader(config_path=tmp_path / "nonexistent.yaml")

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        loader.load()


def test_yaml_config_loader_invalid_yaml(tmp_path: Path) -> None:
    """Test error when YAML is not a mapping."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("- item1\n- item2\n")  # YAML list, not dict

    loader = YAMLConfigLoader(config_path=config_file)

    with pytest.raises(TypeError, match="must contain a YAML mapping"):
        loader.load()


def test_yaml_config_loader_nested_key_not_dict(tmp_path: Path) -> None:
    """Test accessing nested key when intermediate value is not a dict."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("string_value: plain text\n")

    loader = YAMLConfigLoader(config_path=config_file)
    loader.load()

    # Trying to access nested key on string should return default
    assert loader.get("string_value.nested", "default") == "default"


def test_yaml_config_loader_none_value(tmp_path: Path) -> None:
    """Test handling of None values in config."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("null_value: null\n")

    loader = YAMLConfigLoader(config_path=config_file)
    loader.load()

    # None value should return default
    assert loader.get("null_value", "default") == "default"


def test_yaml_config_loader_singleton() -> None:
    """Test singleton pattern."""
    YAMLConfigLoader.reset_instance()

    # First call creates instance
    instance1 = YAMLConfigLoader.get_instance(use_example=True)
    assert isinstance(instance1, YAMLConfigLoader)

    # Second call returns same instance
    instance2 = YAMLConfigLoader.get_instance()
    assert instance1 is instance2

    # Reset and verify new instance is created
    YAMLConfigLoader.reset_instance()
    instance3 = YAMLConfigLoader.get_instance(use_example=True)
    assert instance3 is not instance1


def test_yaml_config_loader_uses_example_config() -> None:
    """Test that loader can use config.example.yaml."""
    YAMLConfigLoader.reset_instance()

    loader = YAMLConfigLoader.get_instance(use_example=True)

    # Verify it loaded config.example.yaml
    assert loader.get("inference.endpoint") is not None
    assert loader.get("log_level") is not None
    assert loader.get("defaults.judge") is not None


def test_yaml_config_loader_testing_env_uses_example(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that TESTING env var causes example config to be used."""
    YAMLConfigLoader.reset_instance()
    monkeypatch.setenv("TESTING", "1")

    loader = YAMLConfigLoader.get_instance()

    # Should use example config when TESTING env var is set
    assert loader.get("inference.endpoint") is not None


def test_yaml_config_loader_no_config_file_raises_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error when no config file is found."""
    YAMLConfigLoader.reset_instance()
    monkeypatch.delenv("TESTING", raising=False)

    # Patch the project root to a directory without config files
    import llm_judge.infrastructure.yaml_config_loader as yaml_module

    original_file = yaml_module.__file__

    # Create a fake module path in tmp_path
    fake_module_path = tmp_path / "fake" / "module" / "path" / "file.py"
    fake_module_path.parent.mkdir(parents=True)
    fake_module_path.write_text("")

    monkeypatch.setattr(yaml_module, "__file__", str(fake_module_path))

    loader = YAMLConfigLoader()

    with pytest.raises(FileNotFoundError, match="No configuration file found"):
        loader.load()

    # Restore
    monkeypatch.setattr(yaml_module, "__file__", original_file)


def test_load_config_convenience_function() -> None:
    """Test load_config convenience function."""
    YAMLConfigLoader.reset_instance()

    config = load_config(use_example=True)
    assert isinstance(config, YAMLConfigLoader)
    assert config.get("inference.endpoint") is not None


def test_yaml_config_loader_get_triggers_load(tmp_path: Path) -> None:
    """Test that get() triggers load if not already loaded."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("key: value\n")

    loader = YAMLConfigLoader(config_path=config_file)

    # get() should trigger load automatically
    assert loader.get("key") == "value"


def test_yaml_config_loader_get_all_triggers_load(tmp_path: Path) -> None:
    """Test that get_all() triggers load if not already loaded."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("key: value\n")

    loader = YAMLConfigLoader(config_path=config_file)

    # get_all() should trigger load automatically
    all_config = loader.get_all()
    assert all_config == {"key": "value"}


def test_yaml_config_loader_thread_safety(tmp_path: Path) -> None:
    """Test thread-safe loading and access."""
    import threading

    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        """
counter: 0
data:
  value: test
"""
    )

    loader = YAMLConfigLoader(config_path=config_file)
    results: list[Any] = []
    errors: list[Exception] = []

    def worker() -> None:
        try:
            loader.load()
            value = loader.get("data.value")
            results.append(value)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(10)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert len(errors) == 0
    assert all(r == "test" for r in results)


def test_yaml_config_loader_fallback_to_example(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test fallback to config.example.yaml when config.yaml doesn't exist."""
    import llm_judge.infrastructure.yaml_config_loader as yaml_module

    # Create a fake project structure
    fake_root = tmp_path / "project"
    fake_root.mkdir()
    fake_module_path = fake_root / "src" / "llm_judge" / "infrastructure" / "yaml_config_loader.py"
    fake_module_path.parent.mkdir(parents=True)
    fake_module_path.write_text("")

    # Create only config.example.yaml (not config.yaml)
    example_config = fake_root / "config.example.yaml"
    example_config.write_text("fallback_test: true\n")

    # Patch __file__ to point to our fake module
    monkeypatch.setattr(yaml_module, "__file__", str(fake_module_path))
    monkeypatch.delenv("TESTING", raising=False)

    YAMLConfigLoader.reset_instance()
    loader = YAMLConfigLoader()
    loader.load()

    assert loader.get("fallback_test") is True


def test_yaml_config_loader_complex_structure(tmp_path: Path) -> None:
    """Test loading complex nested YAML structure."""
    config_file = tmp_path / "complex.yaml"
    config_file.write_text(
        """
listen:
  frontend:
    host: 0.0.0.0
    port: 5173
  proxy:
    host: 0.0.0.0
    port: 5000
inference:
  endpoint: https://openrouter.ai/api
  key: test_key
  retries:
    max: 3
    timeout: 30
log_level: DEBUG
defaults:
  max_rounds: 1
  judge: x-ai/grok-4-fast
  models:
    - qwen/qwen3-next-80b-a3b-instruct
    - anthropic/claude-3.5-sonnet
"""
    )

    loader = YAMLConfigLoader(config_path=config_file)
    loader.load()

    # Test nested access
    assert loader.get("listen.frontend.host") == "0.0.0.0"
    assert loader.get("listen.frontend.port") == 5173
    assert loader.get("inference.retries.max") == 3
    assert loader.get("defaults.max_rounds") == 1

    # Test list access
    models = loader.get("defaults.models")
    assert isinstance(models, list)
    models_list: list[Any] = models  # pyright: ignore[reportUnknownVariableType]
    assert len(models_list) == 2
    assert "qwen/qwen3-next-80b-a3b-instruct" in models_list
