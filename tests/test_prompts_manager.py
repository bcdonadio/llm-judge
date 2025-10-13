from __future__ import annotations

from pathlib import Path

import pytest

from llm_judge.infrastructure.prompts_manager import PromptsManager


def test_prompts_manager_loads_and_caches(tmp_path: Path) -> None:
    prompts_file = tmp_path / "prompts.yaml"
    prompts_file.write_text(
        """
core_prompts:
  - First prompt
  - Second prompt
  - 123
follow_up: Follow prompt
probes:
  - probe-a
  - 123
"""
    )

    manager = PromptsManager(prompts_file=prompts_file)
    prompts = manager.get_core_prompts()
    assert len(prompts) == 2
    assert prompts[0].text == "First prompt"
    assert manager.get_follow_up() == "Follow prompt"
    assert manager.get_probes() == ["probe-a"]

    # Ensure cache is used but reload clears it
    manager.reload()
    prompts_after_reload = manager.get_core_prompts()
    assert prompts_after_reload[1].text == "Second prompt"


def test_prompts_manager_defaults(tmp_path: Path) -> None:
    prompts_file = tmp_path / "prompts.yaml"
    prompts_file.write_text("core_prompts: 5\nfollow_up: 123\nprobes: 5\n")
    manager = PromptsManager(prompts_file=prompts_file)
    assert manager.get_core_prompts() == []
    assert manager.get_follow_up() == ""
    assert manager.get_probes() == []


def test_prompts_manager_resource_loads() -> None:
    manager = PromptsManager()
    prompts = manager.get_core_prompts()
    assert isinstance(prompts, list)


def test_prompts_manager_type_error(tmp_path: Path) -> None:
    bad = tmp_path / "prompts.yaml"
    bad.write_text("- item\n")
    manager = PromptsManager(prompts_file=bad)
    with pytest.raises(TypeError):
        manager.get_core_prompts()
