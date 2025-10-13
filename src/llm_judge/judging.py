"""Judge configuration and evaluation utilities."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from importlib import resources
from typing import Any, Dict, cast

import yaml

from .api import openrouter_chat
from .infrastructure.utility_services import ResponseParser

LOGGER = logging.getLogger(__name__)
_RESPONSE_PARSER = ResponseParser()


@lru_cache(maxsize=1)
def _load_judge_config() -> Dict[str, Any]:
    """Load judge configuration from the YAML resource."""
    resource = resources.files("llm_judge") / "judge_config.yaml"
    with resource.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError("Judge configuration must be a mapping.")
    return cast(Dict[str, Any], data)


def _expect_str(data: Dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"Expected a string for '{key}' in judge_config.yaml.")
    return value


def _expect_mapping(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping for '{key}' in judge_config.yaml.")
    return cast(Dict[str, Any], value)


def _ensure_dict(value: Any) -> Dict[str, Any]:
    return cast(Dict[str, Any], value) if isinstance(value, dict) else {}


_CONFIG = _load_judge_config()
JUDGE_SYSTEM = _expect_str(_CONFIG, "system")
JUDGE_SCHEMA = _expect_mapping(_CONFIG, "schema")
INSTRUCTIONS_TEMPLATE = _expect_str(_CONFIG, "instructions")
JUDGE_INSTRUCTIONS = INSTRUCTIONS_TEMPLATE + f"\n\n{json.dumps(JUDGE_SCHEMA, ensure_ascii=False)}"


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def _find_json_snippet(candidate: str, start: int) -> str | None:
    depth = 0
    for idx in range(start, len(candidate)):
        char = candidate[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                snippet = candidate[start : idx + 1]
                if _is_json(snippet):
                    return snippet
    return None


def extract_json_object(text: str) -> str:
    """Normalize judge output into a JSON string."""
    candidate = text.strip()
    if not candidate:
        raise ValueError("Empty judge response content.")

    if _is_json(candidate):
        return candidate

    start = candidate.find("{")
    if start == -1:
        raise ValueError("No JSON object found in judge response.")

    snippet = _find_json_snippet(candidate, start)
    if snippet is None:
        raise ValueError("Unable to isolate JSON object in judge response.")
    return snippet


def judge_decide(
    judge_model: str,
    prompt: str,
    initial_resp: str,
    follow_resp: str,
    max_tokens: int,
    temperature: float,
    meta: Dict[str, str],
    prompt_index: int | None = None,
    use_color: bool = False,
) -> Dict[str, Any]:
    """Request a judgment from the specified judge model."""
    LOGGER.debug(
        "Evaluating judge model=%s prompt_length=%d initial_length=%d follow_length=%d",
        judge_model,
        len(prompt),
        len(initial_resp or ""),
        len(follow_resp or ""),
    )
    user_block = (
        "USER_PROMPT:\n"
        + prompt.strip()
        + "\n\nMODEL_RESPONSE_INITIAL:\n"
        + (initial_resp or "").strip()
        + "\n\nMODEL_RESPONSE_FOLLOWUP:\n"
        + (follow_resp or "").strip()
        + "\n\nReturn ONLY the JSON."
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_INSTRUCTIONS},
        {"role": "user", "content": user_block},
    ]

    attempts = 0
    current_max_tokens = max_tokens
    last_error = "Judge response missing JSON content."
    data: Dict[str, Any] = {}

    while attempts < 3:
        attempts += 1
        step_label = "Judge" if attempts == 1 else f"Judge retry {attempts - 1}"
        data = openrouter_chat(
            judge_model,
            messages,
            max_tokens=current_max_tokens,
            temperature=temperature,
            metadata=meta,
            response_format={"type": "json_object"},
            step=step_label,
            prompt_index=prompt_index if prompt_index is not None else attempts - 1,
            use_color=use_color,
        )
        raw_content = _RESPONSE_PARSER.extract_text(data)
        choices_obj = data.get("choices")
        choice_dict = _ensure_dict(choices_obj[0]) if isinstance(choices_obj, list) and choices_obj else {}
        finish_value = choice_dict.get("finish_reason") or choice_dict.get("native_finish_reason")
        finish_reason = finish_value.lower() if isinstance(finish_value, str) else ""

        if not raw_content.strip():
            if finish_reason in {"length", "max_output_tokens"} and current_max_tokens < 40960:
                current_max_tokens = min(current_max_tokens * 2, 40960)
                LOGGER.debug(
                    "Judge response empty (finish_reason=%s); retrying with max_tokens=%d",
                    finish_reason,
                    current_max_tokens,
                )
                continue
            last_error = "Judge response missing JSON content."
            break

        try:
            json_text = extract_json_object(raw_content)
            decision = cast(Dict[str, Any], json.loads(json_text))
            return {"ok": True, "raw": data, "decision": decision}
        except ValueError as exc:
            last_error = str(exc)
            if finish_reason in {"length", "max_output_tokens"} and current_max_tokens < 40960:
                current_max_tokens = min(current_max_tokens * 2, 40960)
                LOGGER.debug(
                    "Judge JSON truncated (finish_reason=%s); retrying with max_tokens=%d",
                    finish_reason,
                    current_max_tokens,
                )
                continue
            LOGGER.debug("Unable to parse judge JSON: %s", exc)
            break

    return {"ok": False, "error": last_error, "raw": data or {}}


__all__ = [
    "JUDGE_SYSTEM",
    "JUDGE_SCHEMA",
    "JUDGE_INSTRUCTIONS",
    "judge_decide",
    "extract_json_object",
]
