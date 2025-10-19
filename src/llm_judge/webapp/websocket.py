"""WebSocket manager for broadcasting events to connected clients."""

from __future__ import annotations

import asyncio
import logging
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
        self._background_tasks: Dict[WebSocket, asyncio.Task[None]] = {}

    async def connect(self, websocket: WebSocket) -> None:
        """Register a new WebSocket connection.

        Args:
            websocket: The WebSocket connection to register
        """
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
            self._event_queues[websocket] = asyncio.Queue()
        LOGGER.debug("WebSocket client connected. Total connections: %d", len(self._event_queues))

    async def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a WebSocket connection and clean up resources.

        Args:
            websocket: The WebSocket connection to unregister
        """
        async with self._lock:
            if websocket in self._event_queues:
                del self._event_queues[websocket]
            if websocket in self._background_tasks:
                task = self._background_tasks.pop(websocket)
                if not task.done():
                    task.cancel()
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

        # Schedule the async broadcast without awaiting
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._async_publish(event))
            else:
                # If no loop is running, we can't publish
                LOGGER.warning("No event loop running, cannot publish WebSocket event")
        except RuntimeError:
            LOGGER.warning("Could not get event loop for WebSocket publish")

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

            # Get the event queue for this connection
            queue = self._event_queues.get(websocket)
            if queue is None:
                LOGGER.warning("No event queue found for WebSocket connection")
                return

            # Send events from queue with keepalive
            while True:
                try:
                    # Wait for event with timeout for keepalive
                    event = await asyncio.wait_for(queue.get(), timeout=self._keepalive)
                    await self._send_event(websocket, event)
                except asyncio.TimeoutError:
                    # Send ping/keepalive message
                    await self._send_event(
                        websocket,
                        {
                            "type": "ping",
                            "payload": {"ts": self._time_now_ms()},
                        },
                    )

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

    @staticmethod
    def _time_now_ms() -> int:
        """Return current timestamp in milliseconds.

        Returns:
            Integer millisecond timestamp
        """
        import time

        return int(time.time() * 1000)
