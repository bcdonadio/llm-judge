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
