# pyright: reportPrivateUsage=false

import io
import json
import logging
from typing import Any, Dict, List

import pytest

import llm_judge.judging as judging_module
from llm_judge.judging import extract_json_object, judge_decide


def test_extract_json_object_with_wrapped_text() -> None:
    sample = '```json\n{\n  "value": 42\n}\n```'
    parsed = json.loads(extract_json_object(sample))
    assert parsed["value"] == 42


def test_extract_json_object_error_paths() -> None:
    with pytest.raises(ValueError):
        extract_json_object("   ")

    with pytest.raises(ValueError):
        extract_json_object("no braces present")

    with pytest.raises(ValueError):
        extract_json_object('{"value": 1')

    with pytest.raises(ValueError):
        extract_json_object('prefix {"value": } suffix')


def test_judge_decide_parses_tool_call_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    decision_json: Dict[str, Any] = {
        "initial": {
            "refusal": False,
            "completeness": 2,
            "sourcing_quality": "good",
            "safety_flags": [],
            "rationale": "ok",
        },
        "followup": {
            "refusal": False,
            "completeness": 2,
            "sourcing_quality": "good",
            "safety_flags": [],
            "rationale": "ok",
        },
        "asymmetry_leading": "none",
        "final_notes": "All good",
    }

    payload = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "structured_output",
                                "arguments": json.dumps(decision_json),
                            },
                        }
                    ],
                }
            }
        ]
    }

    def fake_chat(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return payload

    monkeypatch.setattr("llm_judge.judging.openrouter_chat", fake_chat)

    result = judge_decide(
        judge_model="openai/gpt-test",
        prompt="Prompt",
        initial_resp="Initial",
        follow_resp="Follow",
        max_tokens=100,
        temperature=0.0,
        meta={},
    )

    assert result["ok"] is True
    assert result["decision"] == decision_json


def test_judge_decide_retries_when_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    truncated_payload = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": '{"initial": {"refusal": false, "completeness": 2}',
                    "tool_calls": None,
                },
            }
        ]
    }

    full_decision: Dict[str, Any] = {
        "initial": {
            "refusal": False,
            "completeness": 2,
            "sourcing_quality": "weak",
            "safety_flags": [],
            "rationale": "ok",
        },
        "followup": {
            "refusal": False,
            "completeness": 2,
            "sourcing_quality": "good",
            "safety_flags": [],
            "rationale": "ok",
        },
        "asymmetry_leading": "none",
        "final_notes": "complete",
    }

    full_payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(full_decision), "tool_calls": None},
            }
        ]
    }

    calls: List[int] = []

    def fake_chat(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs["max_tokens"])
        return truncated_payload if len(calls) == 1 else full_payload

    monkeypatch.setattr("llm_judge.judging.openrouter_chat", fake_chat)

    result = judge_decide(
        judge_model="openai/gpt-test",
        prompt="Prompt",
        initial_resp="Initial",
        follow_resp="Follow",
        max_tokens=100,
        temperature=0.0,
        meta={},
    )

    assert result["ok"] is True
    assert result["decision"] == full_decision
    assert calls == [100, 200]


def test_judge_decide_retries_on_empty_length(monkeypatch: pytest.MonkeyPatch) -> None:
    decision_json: Dict[str, Any] = {
        "initial": {"refusal": False, "completeness": 1, "sourcing_quality": "ok"},
        "followup": {"refusal": False, "completeness": 1, "sourcing_quality": "ok"},
        "asymmetry_leading": "none",
        "final_notes": "",
    }

    payloads = [
        {
            "choices": [
                {
                    "message": {"content": ""},
                    "finish_reason": "length",
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {"content": json.dumps(decision_json)},
                    "finish_reason": "stop",
                }
            ]
        },
    ]

    calls: List[int] = []

    def fake_chat(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs["max_tokens"])
        return payloads.pop(0)

    monkeypatch.setattr(judging_module, "openrouter_chat", fake_chat)

    result = judge_decide(
        judge_model="openai/gpt-test",
        prompt="Prompt",
        initial_resp="Initial",
        follow_resp="Follow",
        max_tokens=50,
        temperature=0.0,
        meta={},
    )

    assert result["ok"] is True
    assert result["decision"] == decision_json
    assert calls == [50, 100]


def test_judge_decide_handles_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "choices": [
            {
                "message": {"content": ""},
                "finish_reason": "stop",
            }
        ]
    }

    def fake_chat(*args: object, **kwargs: object) -> Dict[str, Any]:
        return payload

    monkeypatch.setattr(judging_module, "openrouter_chat", fake_chat)

    result = judge_decide(
        judge_model="judge",
        prompt="Prompt",
        initial_resp="",
        follow_resp="",
        max_tokens=10,
        temperature=0.0,
        meta={},
    )

    assert result["ok"] is False
    assert result["error"] == "Judge response missing JSON content."


def test_judge_decide_max_attempts_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "choices": [
            {
                "message": {"content": ""},
                "finish_reason": "length",
            }
        ]
    }

    calls: List[int] = []

    def fake_chat(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs["max_tokens"])
        return payload

    monkeypatch.setattr(judging_module, "openrouter_chat", fake_chat)

    result = judge_decide(
        judge_model="judge",
        prompt="Prompt",
        initial_resp="",
        follow_resp="",
        max_tokens=10,
        temperature=0.0,
        meta={},
    )

    assert result["ok"] is False
    assert result["error"] == "Judge response missing JSON content."
    assert calls == [10, 20, 40]


def test_judge_decide_logs_parse_failure(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    payload = {
        "choices": [
            {
                "message": {"content": "not json"},
                "finish_reason": "stop",
            }
        ]
    }

    def fake_chat(*args: object, **kwargs: object) -> Dict[str, Any]:
        return payload

    monkeypatch.setattr(judging_module, "openrouter_chat", fake_chat)
    caplog.set_level(logging.DEBUG, logger="llm_judge.judging")

    result = judge_decide(
        judge_model="judge",
        prompt="Prompt",
        initial_resp="",
        follow_resp="",
        max_tokens=10,
        temperature=0.0,
        meta={},
        prompt_index=5,
    )

    assert result["ok"] is False
    assert "No JSON object found" in result["error"]
    assert any("Unable to parse judge JSON" in message for message in caplog.messages)


def test_judge_config_helpers_validate_types(monkeypatch: pytest.MonkeyPatch) -> None:
    judging_module._load_judge_config.cache_clear()
    original_files = judging_module.resources.files

    class DummyResource:
        def open(self, mode: str, encoding: str) -> io.StringIO:
            return io.StringIO("- not a mapping")

    class DummyFiles:
        def __truediv__(self, name: str) -> "DummyResource":
            assert name == "judge_config.yaml"
            return DummyResource()

    def fake_files(package: str) -> DummyFiles:
        return DummyFiles()

    monkeypatch.setattr(judging_module.resources, "files", fake_files)

    with pytest.raises(TypeError):
        judging_module._load_judge_config()

    monkeypatch.setattr(judging_module.resources, "files", original_files)
    judging_module._load_judge_config.cache_clear()
    judging_module._load_judge_config()

    with pytest.raises(TypeError):
        judging_module._expect_str({"system": 1}, "system")

    with pytest.raises(TypeError):
        judging_module._expect_mapping({"schema": 1}, "schema")
