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

    response = ModelResponse(text="not json", raw_payload={"choices": []})

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
