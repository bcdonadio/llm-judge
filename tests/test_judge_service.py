# pyright: reportPrivateUsage=false
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import pytest

from llm_judge.domain import JudgeDecision, ModelResponse, RunConfiguration
from llm_judge.infrastructure.judge_service import JudgeService


class DummyAPIClient:
    def __init__(self, responses: List[ModelResponse]):
        self._responses = responses
        self.calls: List[Dict[str, Any]] = []

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        metadata: Dict[str, str],
        response_format: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "metadata": metadata,
                "response_format": response_format,
                **kwargs,
            }
        )
        if not self._responses:
            raise RuntimeError("no response queued")
        return self._responses.pop(0)

    def close(self) -> None:
        pass

    def list_models(self) -> List[Dict[str, Any]]:
        return []


@pytest.fixture()
def run_config(tmp_path: Path) -> RunConfiguration:
    return RunConfiguration(
        models=["model-x"],
        judge_model="judge-model",
        outdir=tmp_path,
        max_tokens=128,
        judge_max_tokens=256,
        temperature=0.1,
        judge_temperature=0.3,
        sleep_s=0.0,
        limit=None,
        verbose=False,
        use_color=False,
    )


def test_judge_service_evaluate_success(tmp_path: Path, run_config: RunConfiguration) -> None:
    config_file = tmp_path / "judge.yaml"
    config_file.write_text(
        textwrap.dedent(
            """
            system: "You are a judge"
            instructions: "Score responses"
            schema:
              type: object
            """
        )
    )

    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "initial": {
                                "refusal": False,
                                "completeness": 0.9,
                                "sourcing_quality": "ok",
                                "safety_flags": ["policy"],
                            },
                            "followup": {
                                "refusal": True,
                                "completeness": 0.1,
                                "sourcing_quality": "bad",
                                "safety_flags": "manual",
                            },
                            "asymmetry_leading": "initial",
                            "final_notes": "notes",
                        }
                    )
                }
            }
        ]
    }
    text_content = payload["choices"][0]["message"]["content"]
    response = ModelResponse(text=text_content, raw_payload=payload, finish_reason="stop")

    service = JudgeService(api_client=DummyAPIClient([response]), config_file=config_file)
    decision = service.evaluate("prompt", ModelResponse("a", {}), ModelResponse("b", {}), run_config)

    assert isinstance(decision, JudgeDecision)
    assert decision.success is True
    assert decision.follow_refusal is True
    assert decision.safety_flags_follow == ["manual"]


def test_judge_service_handles_invalid_json(tmp_path: Path, run_config: RunConfiguration) -> None:
    config_file = tmp_path / "judge.yaml"
    config_file.write_text("system: judge")

    response = ModelResponse(text="not json", raw_payload={"choices": []}, finish_reason="stop")

    service = JudgeService(api_client=DummyAPIClient([response, response, response]), config_file=config_file)
    decision = service.evaluate("prompt", ModelResponse("a", {}), ModelResponse("b", {}), run_config)
    assert decision.success is False
    assert "Error:" in decision.notes


def test_judge_service_uses_default_resource(run_config: RunConfiguration) -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "initial": {
                                "refusal": False,
                                "completeness": 1,
                                "sourcing_quality": "high",
                                "safety_flags": [],
                            },
                            "followup": {
                                "refusal": False,
                                "completeness": 1,
                                "sourcing_quality": "high",
                                "safety_flags": [],
                            },
                        }
                    )
                }
            }
        ]
    }
    response = ModelResponse(text=payload["choices"][0]["message"]["content"], raw_payload=payload)
    service = JudgeService(api_client=DummyAPIClient([response]))
    decision = service.evaluate("prompt", ModelResponse("i", {}), ModelResponse("f", {}), run_config)
    assert decision.success is True


def test_judge_service_parses_embedded_json(run_config: RunConfiguration) -> None:
    embedded = json.dumps(
        {
            "initial": {
                "refusal": False,
                "completeness": "2",
                "sourcing_quality": "ok",
                "safety_flags": [],
            },
            "followup": {
                "refusal": False,
                "completeness": 0,
                "sourcing_quality": "ok",
                "safety_flags": [],
            },
        }
    )
    noisy_payload = {"choices": [{"message": {"content": f"prefix {embedded} suffix"}}]}
    response = ModelResponse(text=noisy_payload["choices"][0]["message"]["content"], raw_payload=noisy_payload)
    service = JudgeService(api_client=DummyAPIClient([response]))
    decision = service.evaluate("prompt", ModelResponse("first", {}), ModelResponse("second", {}), run_config)
    assert decision.initial_completeness == 2.0


def test_judge_service_retry_on_length(tmp_path: Path, run_config: RunConfiguration) -> None:
    config_file = tmp_path / "judge.yaml"
    config_file.write_text("system: judge")

    first = ModelResponse(text="not json", raw_payload={"choices": []}, finish_reason="length")
    valid_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "initial": {"refusal": False, "completeness": 1, "sourcing_quality": "ok"},
                            "followup": {"refusal": False, "completeness": 1, "sourcing_quality": "ok"},
                        }
                    )
                }
            }
        ]
    }
    second = ModelResponse(text=valid_payload["choices"][0]["message"]["content"], raw_payload=valid_payload)

    client = DummyAPIClient([first, second])
    service = JudgeService(api_client=client, config_file=config_file)
    decision = service.evaluate("prompt", ModelResponse("a", {}), ModelResponse("b", {}), run_config)

    assert decision.success is True
    assert len(client.calls) == 2
    assert client.calls[1]["max_tokens"] == run_config.judge_max_tokens * 2


def test_judge_service_load_config_type_error(tmp_path: Path, run_config: RunConfiguration) -> None:
    cfg = tmp_path / "judge.yaml"
    cfg.write_text("- item\n")
    service = JudgeService(api_client=DummyAPIClient([]), config_file=cfg)
    with pytest.raises(TypeError):
        service.evaluate("prompt", ModelResponse("a", {}), ModelResponse("b", {}), run_config)


def test_parse_judge_response_errors(run_config: RunConfiguration) -> None:
    empty_response = ModelResponse(text="   ", raw_payload={})
    with pytest.raises(ValueError):
        JudgeService._parse_judge_response(empty_response)

    invalid_json = ModelResponse(text="{", raw_payload={})
    with pytest.raises(ValueError):
        JudgeService._parse_judge_response(invalid_json)

    array_json = ModelResponse(text="[]", raw_payload={})
    with pytest.raises(ValueError):
        JudgeService._parse_judge_response(array_json)


def test_create_decision_filters_flags(run_config: RunConfiguration) -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "initial": {
                                "refusal": False,
                                "completeness": "invalid",
                                "sourcing_quality": "ok",
                                "safety_flags": ["policy", 1],
                            },
                            "followup": {
                                "refusal": False,
                                "completeness": 1,
                                "sourcing_quality": "ok",
                                "safety_flags": 7,
                            },
                        }
                    )
                }
            }
        ]
    }
    response = ModelResponse(text=payload["choices"][0]["message"]["content"], raw_payload=payload)
    service = JudgeService(api_client=DummyAPIClient([response]))
    decision = service.evaluate("prompt", ModelResponse("x", {}), ModelResponse("y", {}), run_config)
    assert decision.safety_flags_initial == ["policy"]
    assert decision.safety_flags_follow == []
    assert decision.initial_completeness == 0.0


def test_judge_service_max_attempts_exceeded(tmp_path: Path, run_config: RunConfiguration) -> None:
    config_file = tmp_path / "judge.yaml"
    config_file.write_text("system: judge")

    failing = ModelResponse(text="not json", raw_payload={"choices": []}, finish_reason="length")
    service = JudgeService(api_client=DummyAPIClient([failing, failing, failing]), config_file=config_file)
    decision = service.evaluate("prompt", ModelResponse("a", {}), ModelResponse("b", {}), run_config)
    assert decision.success is False
    assert decision.error == "Max attempts exceeded"


def test_coerce_flags_non_list() -> None:
    assert JudgeService._coerce_flags(123) == []


def test_normalize_section_non_dict() -> None:
    assert JudgeService._normalize_section(None) == {}
