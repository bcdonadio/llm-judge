# pyright: reportPrivateUsage=false
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from llm_judge.infrastructure.config_manager import ConfigurationManager, SingletonConfigurationManager


def test_configuration_manager_loads_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
api:
  timeout: 20
  retries: 2
models:
  primary: foo
"""
    )

    manager = ConfigurationManager(config_file=config_file)
    assert manager.get("api.timeout") == 20
    assert manager.get("models.primary") == "foo"
    assert manager.get("missing", default="fallback") == "fallback"

    manager.set("api.retries", 3)
    assert manager.get("api.retries") == 3

    section = manager.get_section("api")
    assert section["timeout"] == 20

    monkeypatch.setenv("API_TIMEOUT", "15")
    assert manager.get("api.timeout") == 15

    manager.merge({"api": {"timeout": 30, "new": "yes"}})
    monkeypatch.delenv("API_TIMEOUT", raising=False)
    assert manager.get("api.timeout") == 30
    assert manager.get("api.new") == "yes"
    assert manager.get_all()["api"]["timeout"] == 30


def test_configuration_manager_json_and_auto_reload(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"alpha": 1}))

    manager = ConfigurationManager(config_file=config_file, auto_reload=True)
    assert manager.get("alpha") == 1

    config_file.write_text(json.dumps({"alpha": 2, "beta": {"gamma": "x"}}))
    manager.reload()
    assert manager.get("beta.gamma") == "x"


def test_auto_reload_detects_file_updates(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("value: 1\n")

    manager = ConfigurationManager(config_file=config_file, auto_reload=True)
    assert manager.get("value") == 1

    original_mtime = config_file.stat().st_mtime
    config_file.write_text("value: 2\n")
    os.utime(config_file, (original_mtime + 1, original_mtime + 1))

    assert manager.get("value") == 2


def test_configuration_manager_without_file(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ConfigurationManager()
    assert manager.get("anything") is None

    manager.set("path.to.value", 7)
    assert manager.get("path.to.value") == 7

    os.environ.pop("PATH_TO_VALUE", None)
    assert manager.get_section("missing") == {}


def test_configuration_manager_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ConfigurationManager(config_file=tmp_path / "missing.yaml")

    bad_file = tmp_path / "config.txt"
    bad_file.write_text("noop")
    manager = ConfigurationManager()
    manager.merge({"outer": {"inner": {"value": 1}}})
    with pytest.raises(ValueError):
        ConfigurationManager(config_file=bad_file)


def test_singleton_configuration_manager(tmp_path: Path) -> None:
    SingletonConfigurationManager.reset()
    cfg = tmp_path / "singleton.yaml"
    cfg.write_text("foo: 1")
    instance = SingletonConfigurationManager.get_instance(config_file=cfg, force_reload=True)
    assert isinstance(instance, ConfigurationManager)
    same_instance = SingletonConfigurationManager.get_instance()
    assert same_instance is instance


def test_parse_env_value_json(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ConfigurationManager()
    monkeypatch.setenv("JSON_VALUE", '{"a": 1}')
    assert manager.get("json.value")["a"] == 1


def test_auto_reload_without_file() -> None:
    manager = ConfigurationManager(auto_reload=True)
    assert manager.get("missing", default="fallback") == "fallback"


def test_nested_lookup_handles_none_and_non_mapping() -> None:
    manager = ConfigurationManager()
    manager.set("path.to.value", None)
    assert manager.get("path.to.value", default="fallback") == "fallback"

    manager.set("path", "string")
    assert manager.get("path.deeper", default=5) == 5


def test_get_section_non_dict_returns_empty() -> None:
    manager = ConfigurationManager()
    manager.set("section", "value")
    assert manager.get_section("section") == {}


def test_reload_without_file_sets_loaded() -> None:
    manager = ConfigurationManager()
    manager.reload()
    assert manager.get_all() == {}


def test_get_section_auto_reload(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("section:\n  key: value\n")
    manager = ConfigurationManager(config_file=cfg, auto_reload=True)
    manager._loaded = False
    section = manager.get_section("section")
    assert section["key"] == "value"


def test_reload_rejects_non_mapping(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("- item\n")
    with pytest.raises(TypeError):
        ConfigurationManager(config_file=cfg)


def test_get_all_triggers_auto_reload(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"value": 1}))
    manager = ConfigurationManager(config_file=cfg, auto_reload=True)
    manager._loaded = False
    data = manager.get_all()
    assert data["value"] == 1


def test_parse_env_value_string(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ConfigurationManager()
    monkeypatch.setenv("STRING_VALUE", "plain")
    assert manager.get("string.value") == "plain"
