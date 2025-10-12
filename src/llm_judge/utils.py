"""Utility helpers for timestamping, refusal detection, JSON persistence, and response parsing.

DEPRECATED: This module is maintained for backward compatibility only.
New code should use the service classes from llm_judge.infrastructure.utility_services instead.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional


from .infrastructure.utility_services import (
    TimeService,
    RefusalDetector,
    ResponseParser,
    FileSystemService,
)

_time_service = TimeService()
_refusal_detector = RefusalDetector()
_response_parser = ResponseParser()
_fs_service = FileSystemService()


def now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format with a Z suffix.

    DEPRECATED: Use TimeService.now_iso() instead.
    """
    return _time_service.now_iso()


def detect_refusal(text: str) -> bool:
    """Heuristically determine whether the provided text is a refusal.

    DEPRECATED: Use RefusalDetector.is_refusal() instead.
    """
    return _refusal_detector.is_refusal(text)


def safe_write_json(path: Path, data: Any) -> None:
    """Persist JSON data, creating parent directories on demand.

    DEPRECATED: Use FileSystemService.write_json() instead.
    """
    _fs_service.write_json(path, data)


def create_temp_outdir(*, prefix: str = "llm-judge-") -> Path:
    """Create and return a temporary directory for storing run artifacts.

    DEPRECATED: Use FileSystemService.create_temp_dir() instead.
    """
    return _fs_service.create_temp_dir(prefix)


def extract_text(completion_json: Dict[str, Any]) -> str:
    """Extract the primary text content from an OpenRouter chat completion payload.

    DEPRECATED: Use ResponseParser.extract_text() instead.
    """
    return _response_parser.extract_text(completion_json)


def collect_content_segments(segments: Iterable[Any]) -> str:
    """Compat wrapper for legacy helpers used in tests."""
    return ResponseParser._collect_content_segments(segments)


def is_dict_list(value: Any) -> bool:
    return ResponseParser._is_dict_list(value)


def all_dict_elements(items: Iterable[Any]) -> bool:
    return ResponseParser._all_dict_elements(list(items))


def extract_message(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return ResponseParser._extract_message(payload)


def extract_content_text(message: Dict[str, Any]) -> str:
    return _response_parser._extract_content_text(message)


def extract_tool_call_arguments(message: Dict[str, Any]) -> str:
    return ResponseParser._extract_tool_call_arguments(message)


# Backward compatibility aliases (Pyright sees these as used via the public names above)
_collect_content_segments = collect_content_segments  # pyright: ignore[reportPrivateUsage]
_is_dict_list = is_dict_list  # pyright: ignore[reportPrivateUsage]
_all_dict_elements = all_dict_elements  # pyright: ignore[reportPrivateUsage]
_extract_message = extract_message  # pyright: ignore[reportPrivateUsage]
_extract_content_text = extract_content_text  # pyright: ignore[reportPrivateUsage]
_extract_tool_call_arguments = extract_tool_call_arguments  # pyright: ignore[reportPrivateUsage]


__all__ = [
    "now_iso",
    "detect_refusal",
    "safe_write_json",
    "extract_text",
    "create_temp_outdir",
    "collect_content_segments",
    "is_dict_list",
    "all_dict_elements",
    "extract_message",
    "extract_content_text",
    "extract_tool_call_arguments",
]
