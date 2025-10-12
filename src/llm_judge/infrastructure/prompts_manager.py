"""Thread-safe prompts management implementation."""

import threading
from typing import List, Dict, Any, Optional, cast
from pathlib import Path
from importlib import resources

import yaml

from ..domain import Prompt
from ..services import IPromptsManager


class PromptsManager(IPromptsManager):
    """Thread-safe prompts management without global state."""

    def __init__(self, prompts_file: Optional[Path] = None):
        self._lock = threading.RLock()
        self._prompts_cache: Optional[Dict[str, Any]] = None
        self._prompts_file = prompts_file

    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML file."""
        with self._lock:
            if self._prompts_cache is not None:
                return self._prompts_cache

            if self._prompts_file:
                with open(self._prompts_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            else:
                # Default to package resource
                resource = resources.files("llm_judge") / "prompts.yaml"
                with resource.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

            if not isinstance(data, dict):
                raise TypeError("Prompts YAML must be a mapping")

            prompts_data = cast(Dict[str, Any], data)
            self._prompts_cache = prompts_data
            return prompts_data

    def get_core_prompts(self) -> List[Prompt]:
        """Get core prompts as domain objects."""
        data = self._load_prompts()
        prompts_list_raw = data.get("core_prompts", [])
        if not isinstance(prompts_list_raw, list):
            return []

        prompts: List[Prompt] = []
        for i, text in enumerate(prompts_list_raw):
            if isinstance(text, str):
                prompts.append(Prompt(text=text, category="core", index=i))
        return prompts

    def get_follow_up(self) -> str:
        """Get follow-up prompt text."""
        data = self._load_prompts()
        follow_up = data.get("follow_up", "")
        return follow_up if isinstance(follow_up, str) else ""

    def get_probes(self) -> List[str]:
        """Get probe prompts."""
        data = self._load_prompts()
        probes_raw = data.get("probes", [])
        if not isinstance(probes_raw, list):
            return []
        return [probe for probe in probes_raw if isinstance(probe, str)]

    def reload(self) -> None:
        """Force reload of prompts from disk."""
        with self._lock:
            self._prompts_cache = None


__all__ = ["PromptsManager"]
