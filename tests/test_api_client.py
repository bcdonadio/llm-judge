# pyright: reportPrivateUsage=false
from __future__ import annotations

import sys
import logging
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
        self.raise_error = False

    @property
    def with_raw_response(self) -> DummyChatCompletions:
        return self

    def create(self, **kwargs: Any) -> DummyRawResponse:
        self.last_kwargs = kwargs
        if self.raise_error:
            raise OpenAIError("failure")
        return DummyRawResponse(self._payload)


class DummyChat:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.completions = DummyChatCompletions(payload)


class DummyOpenAI:
    def __init__(self, *, payload: Dict[str, Any], **_: Any) -> None:
        self.chat = DummyChat(payload)


class CountingOpenAI(DummyOpenAI):
    calls = 0

    def __init__(self, *, payload: Dict[str, Any], **kwargs: Any) -> None:
        CountingOpenAI.calls += 1
        super().__init__(payload=payload, **kwargs)


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
    payload_reason: Dict[str, Any] = {"choices": [{"message": {"content": [], "reasoning": "Use reasoning"}}]}
    payload_tool: Dict[str, Any] = {
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


def test_logging_without_color(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    payload = {"choices": [{"message": {"content": ""}}]}
    caplog.set_level(logging.INFO)
    _run_completion(
        monkeypatch,
        payload,
        extra_kwargs={"step": "Initial", "prompt_index": 2, "use_color": False},
    )
    assert any("[Request 02]" in message for message in caplog.messages)


def test_client_reuses_cached_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"choices": [{"message": {"content": "cached"}}]}

    def factory(**kwargs: Any) -> CountingOpenAI:
        return CountingOpenAI(payload=payload, **kwargs)

    CountingOpenAI.calls = 0
    monkeypatch.setattr("llm_judge.infrastructure.api_client.OpenAI", factory)
    monkeypatch.setattr("llm_judge.infrastructure.api_client.httpx.Client", DummyHTTPClient)

    client = OpenRouterClient(api_key="key")
    client.chat_completion(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=2,
        temperature=0.1,
        metadata={"title": "t"},
    )
    client.chat_completion(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=2,
        temperature=0.1,
        metadata={"title": "t"},
    )
    client.close()

    assert CountingOpenAI.calls == 1


def test_logging_with_colorama(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    payload = {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}

    caplog.set_level(logging.INFO)

    response = _run_completion(
        monkeypatch,
        payload,
        extra_kwargs={"step": "Initial", "prompt_index": 3, "use_color": True},
    )
    assert response.finish_reason == "stop"
    assert any("Initial" in message for message in caplog.messages)


def test_chat_completion_logs_error(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def factory(**kwargs: Any) -> DummyOpenAI:
        dummy = DummyOpenAI(payload={"choices": [{"message": {"content": ""}}]}, **kwargs)
        dummy.chat.completions.raise_error = True
        return dummy

    monkeypatch.setattr("llm_judge.infrastructure.api_client.OpenAI", factory)
    monkeypatch.setattr("llm_judge.infrastructure.api_client.httpx.Client", DummyHTTPClient)

    client = OpenRouterClient(api_key="key")
    caplog.set_level(logging.ERROR)
    with pytest.raises(OpenAIError):
        client.chat_completion(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0.0,
            metadata={"title": "T"},
        )
    assert any("API request failed" in message for message in caplog.messages)


def test_extract_text_variants() -> None:
    payload_string = {"choices": [{"message": {"content": "plain"}}]}
    payload_list = {"choices": [{"message": {"content": ["a", {"text": "b"}]}}]}
    payload_invalid = {"invalid": True}

    assert OpenRouterClient._extract_text(payload_string) == "plain"
    assert OpenRouterClient._extract_text(payload_list) == "ab"
    assert OpenRouterClient._extract_text(payload_invalid) == ""


def test_extract_text_tool_call_missing_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"arguments": "  "}},
                        {"invalid": True},
                    ],
                }
            }
        ]
    }

    result = _run_completion(monkeypatch, payload)
    assert result.text == ""


def test_extract_text_segment_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"text": "alpha"},
                        {"text": b"bytes"},
                        "beta",
                    ]
                }
            }
        ]
    }

    result = _run_completion(monkeypatch, payload)
    assert result.text == "alphabytesbeta"


def test_extract_text_skips_non_string_text_value() -> None:
    payload = {"choices": [{"message": {"content": [{"text": 123}]}}]}
    assert OpenRouterClient._extract_text(payload) == ""


def test_context_manager_closes_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    def factory(**kwargs: Any) -> DummyOpenAI:
        return DummyOpenAI(payload={"choices": [{"message": {"content": "ctx"}}]}, **kwargs)

    monkeypatch.setattr("llm_judge.infrastructure.api_client.OpenAI", factory)
    monkeypatch.setattr("llm_judge.infrastructure.api_client.httpx.Client", DummyHTTPClient)

    with OpenRouterClient(api_key="key") as client:
        result = client.chat_completion(
            model="model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0.0,
            metadata={"title": "T"},
        )
        assert result.text == "ctx"
