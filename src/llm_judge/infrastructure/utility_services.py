"""Utility service implementations."""

import datetime as dt
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, TypeGuard, cast

from ..services import ITimeService, IRefusalDetector, IResponseParser, IFileSystemService


class TimeService(ITimeService):
    """Service for time-related operations."""

    def now_iso(self) -> str:
        """Return the current UTC timestamp in ISO-8601 format with a Z suffix."""
        return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class RefusalDetector(IRefusalDetector):
    """Service for detecting refusals in text."""

    def __init__(self, custom_cues: Optional[List[str]] = None):
        """Initialize with default or custom refusal cues."""
        self._cues = custom_cues or self._default_cues()

    @staticmethod
    def _default_cues() -> List[str]:
        """Get default refusal detection cues."""
        return [
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

    def is_refusal(self, text: str) -> bool:
        """Heuristically determine whether the provided text is a refusal."""
        if not text:
            return True
        lowered = text.lower()
        return any(cue in lowered for cue in self._cues)


class ResponseParser(IResponseParser):
    """Service for parsing API responses."""

    def extract_text(self, payload: Dict[str, Any]) -> str:
        """Extract the primary text content from an OpenRouter chat completion payload."""
        message = self._extract_message(payload)
        if message is None:
            return ""

        content_text = self._extract_content_text(message)
        if content_text:
            return content_text

        arguments = self._extract_tool_call_arguments(message)
        if arguments:
            return arguments

        return ""

    @staticmethod
    def _extract_message(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract message from payload."""
        try:
            message = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return None
        if isinstance(message, dict):
            return cast(Dict[str, Any], message)
        return None

    def _extract_content_text(self, message: Dict[str, Any]) -> str:
        """Extract text content from message."""
        content: Any = message.get("content")
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            content_iter = cast(Iterable[Any], content)
            combined = self._collect_content_segments(content_iter)
            if combined:
                return combined
        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
        return ""

    @staticmethod
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

    @staticmethod
    def _extract_tool_call_arguments(message: Dict[str, Any]) -> str:
        """Extract tool call arguments from message."""
        tool_calls = message.get("tool_calls")
        if not ResponseParser._is_dict_list(tool_calls):
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

    @staticmethod
    def _is_dict_list(value: Any) -> TypeGuard[List[Dict[str, Any]]]:
        """Check if value is a list of dictionaries."""
        if not isinstance(value, list):
            return False
        items: Sequence[Any] = cast(Sequence[Any], value)
        return ResponseParser._all_dict_elements(items)

    @staticmethod
    def _all_dict_elements(items: Sequence[Any]) -> bool:
        """Check if all items are dictionaries."""
        for item in items:
            if not isinstance(item, dict):
                return False
        return True


class FileSystemService(IFileSystemService):
    """Service for file system operations."""

    def write_json(self, path: Path, data: Any) -> None:
        """Persist JSON data, creating parent directories on demand."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    def create_temp_dir(self, prefix: str = "llm-judge-") -> Path:
        """Create and return a temporary directory for storing run artifacts."""
        temp_path = tempfile.mkdtemp(prefix=prefix)
        return Path(temp_path)


__all__ = [
    "TimeService",
    "RefusalDetector",
    "ResponseParser",
    "FileSystemService",
]
