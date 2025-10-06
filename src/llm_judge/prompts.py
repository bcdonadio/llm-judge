"""Prompt sets used by the LLM judge runner."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any, Dict, List, Sequence, cast, TypeGuard

import yaml


def _is_str_list(value: Any) -> TypeGuard[List[str]]:
    if not isinstance(value, list):
        return False
    items: Sequence[Any] = cast(Sequence[Any], value)
    return _all_str_elements(items)


def _all_str_elements(items: Sequence[Any]) -> bool:
    for element in items:
        if not isinstance(element, str):
            return False
    return True


@lru_cache(maxsize=1)
def _load_prompts() -> Dict[str, Any]:
    """Load prompt definitions from the YAML resource."""
    resource = resources.files("llm_judge") / "prompts.yaml"
    with resource.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError("Prompts YAML must define a mapping at the top level.")
    return cast(Dict[str, Any], data)


def _expect_str_list(data: Dict[str, Any], key: str) -> List[str]:
    """Fetch and validate a list of strings from the loaded YAML data."""
    value = data.get(key)
    if not _is_str_list(value):
        raise TypeError(f"Expected a list of strings for '{key}' in prompts.yaml.")
    return list(value)


def _expect_str(data: Dict[str, Any], key: str) -> str:
    """Fetch and validate a single string value from the loaded YAML data."""
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"Expected a string for '{key}' in prompts.yaml.")
    return value


_PROMPTS = _load_prompts()
CORE_PROMPTS = _expect_str_list(_PROMPTS, "core_prompts")
PROBES = _expect_str_list(_PROMPTS, "probes")
FOLLOW_UP = _expect_str(_PROMPTS, "follow_up")

__all__ = ["CORE_PROMPTS", "PROBES", "FOLLOW_UP"]
