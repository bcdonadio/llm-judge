import json
import datetime as dt
from pathlib import Path
from typing import Any, Dict

import llm_judge.utils as utils_module
from llm_judge import detect_refusal, extract_text, safe_write_json, now_iso


def test_detect_refusal_true_for_common_policy_language() -> None:
    refusal_text = "I'm sorry, but I can't help with that request because it violates policy."
    assert detect_refusal(refusal_text) is True


def test_detect_refusal_false_for_compliant_response() -> None:
    helpful_text = "Here is the information you requested presented clearly and accurately."
    assert detect_refusal(helpful_text) is False


def test_detect_refusal_empty_text_is_refusal() -> None:
    assert detect_refusal("") is True


def test_extract_text_returns_primary_message_contents() -> None:
    payload: Dict[str, Any] = {"choices": [{"message": {"content": "Hello world"}}]}
    assert extract_text(payload) == "Hello world"


def test_extract_text_handles_missing_content_gracefully() -> None:
    payload: Dict[str, Any] = {"choices": [{"message": {}}]}
    assert extract_text(payload) == ""


def test_extract_text_handles_missing_message() -> None:
    payload: Dict[str, Any] = {}
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


def test_collect_content_segments_handles_strings() -> None:
    segments = ["Hello ", {"text": "world"}, 3]
    assert utils_module._collect_content_segments(segments) == "Hello world"


def test_collect_content_segments_ignores_non_string_dicts() -> None:
    assert utils_module._collect_content_segments([{ "text": 5 }]) == ""


def test_extract_text_uses_reasoning_when_available() -> None:
    payload: Dict[str, Any] = {
        "choices": [
            {
                "message": {
                    "content": [],
                    "reasoning": "Consider multiple factors",
                }
            }
        ]
    }
    assert extract_text(payload) == "Consider multiple factors"


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


def test_extract_text_ignores_invalid_tool_calls() -> None:
    message = {
        "content": "",
        "tool_calls": [
            {"function": "invalid"},
            {"function": {"arguments": "   "}},
            {"function": {"arguments": "{\"valid\": true}"}},
        ],
    }
    payload: Dict[str, Any] = {"choices": [{"message": message}]}
    assert extract_text(payload) == '{"valid": true}'


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


def test_internal_dict_utilities() -> None:
    assert utils_module._is_dict_list([{"a": 1}]) is True
    assert utils_module._is_dict_list("nope") is False
    assert utils_module._all_dict_elements([{"a": 1}, "bad"]) is False


def test_extract_message_and_content_failures() -> None:
    assert utils_module._extract_message({}) is None
    assert utils_module._extract_message({"choices": [{"message": "string"}]}) is None
    assert utils_module._extract_content_text({}) == ""
    assert utils_module._extract_tool_call_arguments({"tool_calls": "bad"}) == ""
    assert utils_module._extract_tool_call_arguments({"tool_calls": [{"function": {}}]}) == ""
