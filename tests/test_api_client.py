from __future__ import annotations

import sys
from typing import Any, Dict, Optional

import pytest

from llm_judge.domain import ModelResponse
from llm_judge.infrastructure.api_client import OpenRouterClient, OpenAIError


class DummyElapsed:
    def __init__(self, seconds: float) -> None:
        self._seconds = seconds

    def total_seconds(self) -> float:
        return self._seconds


_dummy_elapsed_seconds: Optional[float] = 0.01


class DummyHTTPResponse:
    def __init__(self) -> None:
        self.status_code = 200
        self.elapsed = DummyElapsed(_dummy_elapsed_seconds) if _dummy_elapsed_seconds is not None else None


class DummyCompletion:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> Dict[str, Any]:
        return self._payload


class DummyRawResponse:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload
        self.http_response = DummyHTTPResponse()

    def parse(self) -> DummyCompletion:
        return DummyCompletion(self._payload)


class DummyChatCompletions:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    @property
    def with_raw_response(self) -> DummyChatCompletions:
        return self

    def create(self, **kwargs: Any) -> DummyRawResponse:
        self.last_kwargs = kwargs
        return DummyRawResponse(self._payload)


class DummyChat:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.completions = DummyChatCompletions(payload)


class DummyOpenAI:
    def __init__(self, *, payload: Dict[str, Any], **_: Any) -> None:
        self.chat = DummyChat(payload)


class DummyHTTPClient:
    def __init__(self, **_: Any) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _run_completion(
    monkeypatch: pytest.MonkeyPatch,
    payload: Dict[str, Any],
    *,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> ModelResponse:
    def factory(**kwargs: Any) -> DummyOpenAI:
        return DummyOpenAI(payload=payload, **kwargs)

    monkeypatch.setattr("llm_judge.infrastructure.api_client.OpenAI", factory)
    monkeypatch.setattr("llm_judge.infrastructure.api_client.httpx.Client", DummyHTTPClient)

    client = OpenRouterClient(api_key="test-key", base_url="http://example.com")
    result = client.chat_completion(
        model="model",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=32,
        temperature=0.5,
        metadata={"title": "Test"},
        **(extra_kwargs or {}),
    )
    client.close()
    return result


def test_chat_completion_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
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

    response = _run_completion(monkeypatch, payload)
    assert isinstance(response, ModelResponse)
    assert response.text == "Hello world"


def test_extract_text_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    payload_reason = {"choices": [{"message": {"content": [], "reasoning": "Use reasoning"}}]}
    payload_tool = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"arguments": '{"answer": 1}'}}],
                }
            }
        ]
    }

    assert _run_completion(monkeypatch, payload_reason).text == "Use reasoning"
    assert _run_completion(monkeypatch, payload_tool).text == '{"answer": 1}'


def test_extract_finish_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
    payload_native = {"choices": [{"message": {"content": ""}, "native_finish_reason": "length"}]}

    assert _run_completion(monkeypatch, payload).finish_reason == "stop"
    assert _run_completion(monkeypatch, payload_native).finish_reason == "length"


def test_logging_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "choices": [
            {
                "message": {"content": ""},
                "finish_reason": "done",
            }
        ]
    }

    # Simulate missing colorama and no elapsed
    monkeypatch.setitem(sys.modules, "colorama", None)
    global _dummy_elapsed_seconds
    _dummy_elapsed_seconds = None

    response = _run_completion(
        monkeypatch,
        payload,
        extra_kwargs={"step": "Initial", "prompt_index": 1, "use_color": True},
    )
    assert response.finish_reason == "done"

    _dummy_elapsed_seconds = 0.01


def test_chat_completion_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class RaisingOpenAI:
        def __init__(self, **_: Any) -> None:
            raise OpenAIError("boom")

    monkeypatch.setattr("llm_judge.infrastructure.api_client.OpenAI", RaisingOpenAI)
    monkeypatch.setattr("llm_judge.infrastructure.api_client.httpx.Client", DummyHTTPClient)

    client = OpenRouterClient(api_key="key")
    with pytest.raises(OpenAIError):
        client.chat_completion(
            model="m",
            messages=[],
            max_tokens=1,
            temperature=0.0,
            metadata={"title": "Test"},
        )
