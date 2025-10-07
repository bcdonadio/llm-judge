"""Tests for the OpenRouter API client helpers."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

import llm_judge.api as api
from openai import OpenAIError


class DummyElapsed:
    def total_seconds(self) -> float:
        return 0.321


class DummyRawResponse:
    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
        self.http_response = SimpleNamespace(
            status_code=200,
            elapsed=DummyElapsed(),
            text=json.dumps(payload),
        )

    def parse(self) -> SimpleNamespace:
        return SimpleNamespace(model_dump=lambda: self.payload)


class DummyChatWithRaw:
    def __init__(self, responses: List[Dict[str, Any]]):
        self.responses = responses
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> DummyRawResponse:
        self.calls.append(kwargs)
        payload = self.responses.pop(0)
        return DummyRawResponse(payload)


class DummyOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=SimpleNamespace(with_raw_response=None))


class DummyHttpClient:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class DummyTimeout:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def reset_api_state() -> None:
    api._client = None
    api._http_client = None


def test_get_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        api._get_client()


def test_openrouter_chat_uses_cached_client(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "testing-key")

    dummy_chat = DummyChatWithRaw([
        {"choices": [{"message": {"content": "Hello"}}]},
    ])

    def fake_openai(**kwargs: Any) -> DummyOpenAI:
        client = DummyOpenAI(**kwargs)
        client.chat.completions.with_raw_response = dummy_chat
        return client

    monkeypatch.setattr(api, "OpenAI", fake_openai)
    monkeypatch.setattr(api, "httpx", SimpleNamespace(Client=DummyHttpClient, Timeout=DummyTimeout))

    caplog.set_level(logging.DEBUG, logger="llm_judge.api")

    result = api.openrouter_chat(
        model="openrouter/model",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=64,
        temperature=0.1,
        metadata={"referer": "https://example.com", "title": "Suite"},
        response_format={"type": "json_object"},
        step="Initial",
        prompt_index=0,
        use_color=True,
    )

    assert result == {"choices": [{"message": {"content": "Hello"}}]}
    assert dummy_chat.calls[0]["max_tokens"] == 64
    assert api._client is not None and api._get_client() is api._client
    assert "Response preview" in caplog.text


def test_openrouter_chat_propagates_openai_error(monkeypatch: pytest.MonkeyPatch) -> None:
    error = OpenAIError("boom")

    class RaisingChat:
        def create(self, **kwargs: Any) -> Dict[str, Any]:
            raise error

    monkeypatch.setattr(api, "_get_client", lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(with_raw_response=RaisingChat()))))

    with pytest.raises(OpenAIError) as exc:
        api.openrouter_chat(
            model="test",
            messages=[],
            max_tokens=1,
            temperature=0.0,
            metadata={},
        )

    assert exc.value is error


def test_openrouter_chat_debug_logging(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "debug-key")

    dummy_chat = DummyChatWithRaw([
        {"choices": [{"message": {"content": "Plain"}}]},
        {"choices": [{"message": {"content": "Second"}}]},
    ])

    def fake_openai(**kwargs: Any) -> DummyOpenAI:
        client = DummyOpenAI(**kwargs)
        client.chat.completions.with_raw_response = dummy_chat
        return client

    monkeypatch.setattr(api, "OpenAI", fake_openai)
    monkeypatch.setattr(api, "httpx", SimpleNamespace(Client=DummyHttpClient, Timeout=DummyTimeout))

    caplog.set_level(logging.DEBUG, logger="llm_judge.api")

    response = api.openrouter_chat(
        model="openrouter/model",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=32,
        temperature=0.0,
        metadata={},
    )

    assert response == {"choices": [{"message": {"content": "Plain"}}]}
    assert any("POST" in message and "/chat/completions" in message for message in caplog.messages)
    caplog.clear()
    caplog.set_level(logging.INFO, logger="llm_judge.api")

    response_no_color = api.openrouter_chat(
        model="openrouter/model",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=16,
        temperature=0.0,
        metadata={},
        step="Judge",
        prompt_index=1,
    )

    assert response_no_color == {"choices": [{"message": {"content": "Second"}}]}
    assert not any("Response preview" in message for message in caplog.messages)
