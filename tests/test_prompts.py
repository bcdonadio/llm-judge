# pyright: reportPrivateUsage=false

import io

import pytest

import llm_judge.prompts as prompts_module
from llm_judge import CORE_PROMPTS, FOLLOW_UP, PROBES


def test_internal_list_validation_helpers() -> None:
    assert prompts_module._is_str_list(["a", "b"]) is True
    assert prompts_module._is_str_list("not-a-list") is False
    assert prompts_module._is_str_list(["ok", 2]) is False
    assert prompts_module._all_str_elements(["x", "y"]) is True
    assert prompts_module._all_str_elements(["x", 3]) is False


def test_expect_helpers_raise_for_invalid_types() -> None:
    with pytest.raises(TypeError):
        prompts_module._expect_str_list({"bad": 3}, "bad")

    with pytest.raises(TypeError):
        prompts_module._expect_str({"bad": 3}, "bad")


def test_load_prompts_requires_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts_module._load_prompts.cache_clear()
    original_files = prompts_module.resources.files

    class DummyResource:
        def open(self, mode: str, encoding: str) -> io.StringIO:
            return io.StringIO("- not a mapping")

    class DummyFiles:
        def __truediv__(self, name: str) -> "DummyResource":
            assert name == "prompts.yaml"
            return DummyResource()

    def fake_files(package: str) -> DummyFiles:
        return DummyFiles()

    monkeypatch.setattr(prompts_module.resources, "files", fake_files)

    with pytest.raises(TypeError):
        prompts_module._load_prompts()

    prompts_module._load_prompts.cache_clear()
    monkeypatch.setattr(prompts_module.resources, "files", original_files)
    prompts_module._load_prompts()


def test_prompts_loaded_from_yaml() -> None:
    assert CORE_PROMPTS, "Expected core prompts to be populated from YAML."
    assert PROBES, "Expected probes to be populated from YAML."
    assert isinstance(FOLLOW_UP, str) and FOLLOW_UP


def test_prompts_are_string_collections() -> None:
    assert all(isinstance(item, str) for item in CORE_PROMPTS)
    assert all(isinstance(item, str) for item in PROBES)
