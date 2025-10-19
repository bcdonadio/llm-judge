"""Unit tests for the WebSocketManager utilities."""

from __future__ import annotations

import asyncio
import threading
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

    asyncio.run(run())


def test_websocket_manager_connect_without_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        manager = WebSocketManager()
        websocket = StubWebSocket()

        monkeypatch.setattr(WebSocketManager, "_current_running_loop", staticmethod(lambda: None))

        await manager.connect(cast(WebSocket, websocket))
        assert websocket.accepted

    asyncio.run(run())


def test_websocket_manager_send_events_without_queue(caplog: pytest.LogCaptureFixture) -> None:
    async def run() -> None:
        manager = WebSocketManager()
        websocket = StubWebSocket()

        with caplog.at_level("WARNING"):
            await manager.send_events(cast(WebSocket, websocket))

        assert "No event queue" in caplog.text

    asyncio.run(run())


def test_websocket_manager_disconnect_removes_queue() -> None:
    async def run() -> None:
        manager = WebSocketManager()
        websocket = StubWebSocket()

        await manager.connect(cast(WebSocket, websocket))
        await manager.disconnect(cast(WebSocket, websocket))
        event_queues = getattr(manager, "_event_queues")
        assert cast(WebSocket, websocket) not in event_queues

    asyncio.run(run())


def test_websocket_manager_publish_requires_type() -> None:
    manager = WebSocketManager()
    with pytest.raises(ValueError):
        manager.publish({"payload": {}})


def test_websocket_manager_publish_without_running_loop(caplog: pytest.LogCaptureFixture) -> None:
    manager = WebSocketManager()

    with caplog.at_level("WARNING"):
        manager.publish({"type": "status", "payload": {}})

    assert "No running event loop" in caplog.text


def test_websocket_manager_publish_from_thread() -> None:
    async def run() -> None:
        manager = WebSocketManager()
        websocket = StubWebSocket()

        await manager.connect(cast(WebSocket, websocket))
        sender = asyncio.create_task(manager.send_events(cast(WebSocket, websocket)))

        published = threading.Event()

        def publish_from_thread() -> None:
            manager.publish({"type": "message", "payload": {"source": "thread"}})
            published.set()

        thread = threading.Thread(target=publish_from_thread)
        thread.start()
        thread.join()
        assert published.wait(timeout=1)

        await asyncio.sleep(0.05)
        assert any(event["type"] == "message" for event in websocket.sent)

        sender.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sender

        await asyncio.sleep(0.01)
        event_queues = getattr(manager, "_event_queues")
        assert cast(WebSocket, websocket) not in event_queues

    asyncio.run(run())
