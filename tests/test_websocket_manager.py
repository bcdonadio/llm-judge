"""Unit tests for the WebSocketManager utilities."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, cast

import pytest

from llm_judge.webapp.websocket import WebSocketManager
from fastapi import WebSocket


class StubWebSocket:
    """Lightweight stand-in for FastAPI's WebSocket."""

    def __init__(self) -> None:
        self.accepted = False
        self.sent: List[Dict[str, Any]] = []

    async def accept(self) -> None:  # pragma: no cover - called in tests
        self.accepted = True

    async def send_json(self, message: Dict[str, Any]) -> None:
        self.sent.append(message)


def test_websocket_manager_connect_send_and_keepalive() -> None:
    async def run() -> None:
        manager = WebSocketManager(keepalive_s=0.01)
        websocket = StubWebSocket()

        await manager.connect(cast(WebSocket, websocket))
        assert websocket.accepted

        background = asyncio.create_task(asyncio.sleep(1))
        background_tasks = getattr(manager, "_background_tasks")
        background_tasks[cast(WebSocket, websocket)] = background

        sender = asyncio.create_task(
            manager.send_events(
                cast(WebSocket, websocket), initial_events=[{"type": "status", "payload": {"state": "idle"}}]
            )
        )

        await asyncio.sleep(0.02)
        assert {"type": "status", "payload": {"state": "idle"}} in websocket.sent

        manager.publish({"type": "message", "payload": {"content": "hi"}})
        await asyncio.sleep(0.02)
        assert any(event["type"] == "message" for event in websocket.sent)

        await asyncio.sleep(0.02)
        assert any(event["type"] == "ping" for event in websocket.sent)

        sender.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sender

        await asyncio.sleep(0.01)
        event_queues = getattr(manager, "_event_queues")
        assert cast(WebSocket, websocket) not in event_queues
        assert background.cancelled()

    asyncio.run(run())


def test_websocket_manager_send_events_without_queue(caplog: pytest.LogCaptureFixture) -> None:
    async def run() -> None:
        manager = WebSocketManager()
        websocket = StubWebSocket()

        with caplog.at_level("WARNING"):
            await manager.send_events(cast(WebSocket, websocket))

        assert "No event queue" in caplog.text

    asyncio.run(run())


def test_websocket_manager_disconnect_handles_completed_task() -> None:
    async def run() -> None:
        manager = WebSocketManager()
        websocket = StubWebSocket()

        await manager.connect(cast(WebSocket, websocket))
        background = asyncio.create_task(asyncio.sleep(0))
        await background
        background_tasks = getattr(manager, "_background_tasks")
        background_tasks[cast(WebSocket, websocket)] = background

        await manager.disconnect(cast(WebSocket, websocket))

    asyncio.run(run())


def test_websocket_manager_publish_requires_type() -> None:
    manager = WebSocketManager()
    with pytest.raises(ValueError):
        manager.publish({"payload": {}})


def test_websocket_manager_publish_without_running_loop(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    manager = WebSocketManager()

    class DummyLoop:
        def is_running(self) -> bool:
            return False

    monkeypatch.setattr(asyncio, "get_event_loop", lambda: DummyLoop())

    with caplog.at_level("WARNING"):
        manager.publish({"type": "status", "payload": {}})

    assert "No event loop running" in caplog.text


def test_websocket_manager_publish_runtime_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    manager = WebSocketManager()

    def raise_runtime_error() -> Any:
        raise RuntimeError("no loop")

    monkeypatch.setattr(asyncio, "get_event_loop", raise_runtime_error)

    with caplog.at_level("WARNING"):
        manager.publish({"type": "status", "payload": {}})

    assert "Could not get event loop" in caplog.text
