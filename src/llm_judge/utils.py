"""Utility helpers for timestamping, refusal detection, JSON persistence, and response parsing.

DEPRECATED: This module is maintained for backward compatibility only.
New code should use the service classes from llm_judge.infrastructure.utility_services instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


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


__all__ = ["now_iso", "detect_refusal", "safe_write_json", "extract_text", "create_temp_outdir"]
