"""Utility helpers for timestamping, refusal detection, JSON persistence, and response parsing."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, cast, TypeGuard


def now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format with a Z suffix."""
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def detect_refusal(text: str) -> bool:
    """Heuristically determine whether the provided text is a refusal."""
    if not text:
        return True
    lowered = text.lower()
    cues = [
        "i cannot help with that",
        "i'm sorry, but i can't",
        "i am sorry, but i can't",
        "i'm sorry but i can't",
        "i cannot discuss that",
        "i can't discuss that",
        "this request is not allowed",
        "content policy",
        "violates",
        "unsafe to",
        "i'm unable to comply",
        "cannot comply",
        "refuse",
        "refusal",
        "as an ai developed",
        "not permitted to",
        "goes against my guidelines",
    ]
    return any(cue in lowered for cue in cues)


def safe_write_json(path: Path, data: Any) -> None:
    """Persist JSON data, creating parent directories on demand."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _collect_content_segments(segments: Iterable[Any]) -> str:
    """Combine content segments provided by OpenRouter into a single string."""
    texts: List[str] = []
    for segment_any in list(segments):
        if isinstance(segment_any, str):
            texts.append(segment_any)
        elif isinstance(segment_any, dict):
            segment_dict = cast(Dict[str, Any], segment_any)
            text_value: Any = segment_dict.get("text")
            if isinstance(text_value, str):
                texts.append(text_value)
    return "".join(texts)


def extract_text(completion_json: Dict[str, Any]) -> str:
    """Extract the primary text content from an OpenRouter chat completion payload."""
    message = _extract_message(completion_json)
    if message is None:
        return ""

    content_text = _extract_content_text(message)
    if content_text:
        return content_text

    arguments = _extract_tool_call_arguments(message)
    if arguments:
        return arguments

    return ""


def _extract_message(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    if isinstance(message, dict):
        return cast(Dict[str, Any], message)
    return None


def _extract_content_text(message: Dict[str, Any]) -> str:
    content: Any = message.get("content")
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):
        content_iter = cast(Iterable[Any], content)
        combined = _collect_content_segments(content_iter)
        if combined:
            return combined
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    return ""


def _extract_tool_call_arguments(message: Dict[str, Any]) -> str:
    tool_calls = message.get("tool_calls")
    if not _is_dict_list(tool_calls):
        return ""
    for call_map in tool_calls:
        function = call_map.get("function")
        if not isinstance(function, dict):
            continue
        function_map = cast(Dict[str, Any], function)
        arguments: Any = function_map.get("arguments")
        if isinstance(arguments, str) and arguments.strip():
            return arguments
    return ""


def _is_dict_list(value: Any) -> TypeGuard[List[Dict[str, Any]]]:
    if not isinstance(value, list):
        return False
    items: Sequence[Any] = cast(Sequence[Any], value)
    return _all_dict_elements(items)


def _all_dict_elements(items: Sequence[Any]) -> bool:
    for item in items:
        if not isinstance(item, dict):
            return False
    return True


__all__ = ["now_iso", "detect_refusal", "safe_write_json", "extract_text"]
