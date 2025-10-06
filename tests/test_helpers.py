import json
import datetime as dt
from pathlib import Path
from typing import Any, Dict

from llm_judge import detect_refusal, extract_text, safe_write_json, now_iso


def test_detect_refusal_true_for_common_policy_language() -> None:
    refusal_text = "I'm sorry, but I can't help with that request because it violates policy."
    assert detect_refusal(refusal_text) is True


def test_detect_refusal_false_for_compliant_response() -> None:
    helpful_text = "Here is the information you requested presented clearly and accurately."
    assert detect_refusal(helpful_text) is False


def test_extract_text_returns_primary_message_contents() -> None:
    payload: Dict[str, Any] = {"choices": [{"message": {"content": "Hello world"}}]}
    assert extract_text(payload) == "Hello world"


def test_extract_text_handles_missing_content_gracefully() -> None:
    payload: Dict[str, Any] = {"choices": [{"message": {}}]}
    assert extract_text(payload) == ""


def test_extract_text_combines_segmented_content() -> None:
    payload: Dict[str, Any] = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"text": "Hello "},
                        {"text": "world"},
                    ]
                }
            }
        ]
    }
    assert extract_text(payload) == "Hello world"


def test_extract_text_uses_tool_call_arguments_when_content_empty() -> None:
    payload: Dict[str, Any] = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "structured",
                                "arguments": '{"answer": 1}',
                            }
                        }
                    ],
                }
            }
        ]
    }
    assert extract_text(payload) == '{"answer": 1}'


def test_safe_write_json_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "artifact.json"
    sample = {"status": "ok", "count": 3}

    safe_write_json(target, sample)

    assert target.exists()
    with target.open("r", encoding="utf-8") as fh:
        persisted = json.load(fh)
    assert persisted == sample


def test_now_iso_returns_utc_timestamp_with_z_suffix() -> None:
    timestamp = now_iso()
    assert timestamp.endswith("Z")

    parsed = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.tzinfo.utcoffset(parsed) == dt.timedelta(0)
