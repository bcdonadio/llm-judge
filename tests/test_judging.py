import json
from typing import Any, Dict, List

import pytest

from llm_judge.judging import extract_json_object, judge_decide


def test_extract_json_object_with_wrapped_text() -> None:
    sample = '```json\n{\n  "value": 42\n}\n```'
    parsed = json.loads(extract_json_object(sample))
    assert parsed["value"] == 42


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
