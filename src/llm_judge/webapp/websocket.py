"""WebSocket manager for broadcasting events to connected clients."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, List
from weakref import WeakSet

from fastapi import WebSocket

LOGGER = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts events to all connected clients."""

    def __init__(self, *, keepalive_s: float = 15.0) -> None:
        """Initialize the WebSocket manager.

        Args:
            keepalive_s: Interval in seconds for sending ping messages to keep connections alive
        """
        self._keepalive = keepalive_s
        self._connections: WeakSet[WebSocket] = WeakSet()
        self._event_queues: Dict[WebSocket, asyncio.Queue[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_guard = threading.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Register a new WebSocket connection.

        Args:
            websocket: The WebSocket connection to register
        """
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
            self._event_queues[websocket] = asyncio.Queue()
            loop = self._current_running_loop()
            if loop is not None:
                self._remember_loop(loop)
        LOGGER.debug("WebSocket client connected. Total connections: %d", len(self._event_queues))

    async def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a WebSocket connection and clean up resources.

        Args:
            websocket: The WebSocket connection to unregister
        """
        async with self._lock:
            self._event_queues.pop(websocket, None)
        LOGGER.debug("WebSocket client disconnected. Total connections: %d", len(self._event_queues))

    def publish(self, event: Dict[str, Any]) -> None:
        """Publish an event to all connected WebSocket clients.

        This is a synchronous method that can be called from non-async code.
        It schedules the async broadcast as a background task.

        Args:
            event: Event dictionary with 'type' and 'payload' keys

        Raises:
            ValueError: If event does not contain a 'type' field
        """
        if "type" not in event:
            raise ValueError("WebSocket events must include a 'type' field.")

        running_loop = self._current_running_loop()
        if running_loop is not None:
            running_loop.create_task(self._async_publish(event))
            self._remember_loop(running_loop)
            return

        loop = self._get_stored_loop()
        if loop is None or not loop.is_running():
            LOGGER.warning("No running event loop, cannot publish WebSocket event")
            return

        try:
            asyncio.run_coroutine_threadsafe(self._async_publish(event), loop)
        except RuntimeError as exc:  # pragma: no cover - loop closed during shutdown
            LOGGER.warning("Failed to schedule WebSocket publish on stored loop: %s", exc)

    async def _async_publish(self, event: Dict[str, Any]) -> None:
        """Internal async method to publish events to all clients.

        Args:
            event: Event dictionary to broadcast
        """
        async with self._lock:
            queues = list(self._event_queues.values())

        for queue in queues:
            try:
                await queue.put(event)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to queue event for WebSocket client: %s", exc)

    async def send_events(
        self,
        websocket: WebSocket,
        initial_events: List[Dict[str, Any]] | None = None,
    ) -> None:
        """Send events to a specific WebSocket connection.

        This method handles sending initial state and then streaming events from the queue.

        Args:
            websocket: The WebSocket connection to send events to
            initial_events: Optional list of initial events to send immediately
        """
        try:
            # Send initial events if provided
            if initial_events:
                for event in initial_events:
                    await self._send_event(websocket, event)

            queue = self._event_queues.get(websocket)
            if queue is None:
                LOGGER.warning("No event queue found for WebSocket connection")
                return

            await self._stream_events(websocket, queue)

        except Exception as exc:  # pragma: no cover
            LOGGER.debug("WebSocket send loop ended: %s", exc)
        finally:
            await self.disconnect(websocket)

    async def _send_event(self, websocket: WebSocket, event: Dict[str, Any]) -> None:
        """Send a single event to a WebSocket client.

        Args:
            websocket: The WebSocket connection
            event: Event dictionary to send
        """
        message = {
            "type": event.get("type", "message"),
            "payload": event.get("payload", {}),
        }
        await websocket.send_json(message)

    async def _stream_events(self, websocket: WebSocket, queue: asyncio.Queue[Dict[str, Any]]) -> None:
        """Continuously forward queued events or keepalive pings."""

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=self._keepalive)
            except asyncio.TimeoutError:
                await self._send_event(
                    websocket,
                    {
                        "type": "ping",
                        "payload": {"ts": self._time_now_ms()},
                    },
                )
            else:
                await self._send_event(websocket, event)

    @staticmethod
    def _time_now_ms() -> int:
        """Return current timestamp in milliseconds.

        Returns:
            Integer millisecond timestamp
        """
        import time

        return int(time.time() * 1000)

    def _remember_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._loop_guard:
            if self._loop is None:
                self._loop = loop

    def _get_stored_loop(self) -> asyncio.AbstractEventLoop | None:
        with self._loop_guard:
            return self._loop

    @staticmethod
    def _current_running_loop() -> asyncio.AbstractEventLoop | None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        return loop if loop.is_running() else None
