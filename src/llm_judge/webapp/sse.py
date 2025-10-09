"""Server-Sent Events broker utilities for the llm-judge web UI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional

if TYPE_CHECKING:  # pragma: no cover - used for static analysis only
    from queue import Empty as QueueEmptyType
    from queue import Queue as QueueType
    from threading import RLock as LockType
else:  # pragma: no cover - runtime selection
    try:
        import gevent.queue  # type: ignore[import-untyped]
        import gevent.lock  # type: ignore[import-untyped]

        QueueEmptyType = gevent.queue.Empty  # type: ignore[misc]
        QueueType = gevent.queue.Queue  # type: ignore[misc]
        LockType = gevent.lock.RLock  # type: ignore[misc]
    except ImportError:  # pragma: no cover - gevent not installed
        import queue
        import threading

        QueueEmptyType = queue.Empty
        QueueType = queue.Queue
        LockType = threading.RLock


class SSEBroker:
    """Fan-out broker that multiplexes events to connected SSE clients."""

    def __init__(self, *, keepalive_s: float = 15.0) -> None:
        self._keepalive = keepalive_s
        self._lock = LockType()
        self._subscribers: List[Any] = []

    def publish(self, event: Dict[str, Any]) -> None:
        """Publish a new event to all subscribers."""
        if "type" not in event:
            raise ValueError("SSE events must include a 'type' field.")
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.put(event)

    def stream(self, initial: Optional[Iterable[Dict[str, Any]]] = None) -> Iterator[str]:
        """Yield formatted SSE data for a single subscriber."""
        mailbox: QueueType[Dict[str, Any]] = QueueType()
        with self._lock:
            self._subscribers.append(mailbox)

        try:
            if initial:
                for entry in initial:
                    yield self._format(entry)
            while True:
                try:
                    event = mailbox.get(timeout=self._keepalive)
                    yield self._format(event)
                except QueueEmptyType:
                    yield self._format({"type": "ping", "payload": {"ts": time_now_ms()}})
        except GeneratorExit:
            # Client disconnected - clean up gracefully
            pass
        finally:
            with self._lock:
                if mailbox in self._subscribers:
                    self._subscribers.remove(mailbox)

    @staticmethod
    def _format(event: Dict[str, Any]) -> str:
        payload = {"type": event.get("type", "message"), "payload": event.get("payload", {})}
        body = json.dumps(payload, ensure_ascii=False)
        return f"event: {payload['type']}\ndata: {body}\n\n"


def time_now_ms() -> int:
    """Return an integer millisecond timestamp."""
    import time

    return int(time.time() * 1000)
