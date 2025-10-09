"""Server-Sent Events broker utilities for the llm-judge web UI."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Iterator, List, Optional

# Use gevent-compatible primitives when running under gevent workers
try:
    from gevent.queue import Queue
    from gevent.lock import RLock as Lock
except ImportError:
    from queue import Queue  # type: ignore
    from threading import Lock  # type: ignore


class SSEBroker:
    """Fan-out broker that multiplexes events to connected SSE clients."""

    def __init__(self, *, keepalive_s: float = 15.0) -> None:
        self._keepalive = keepalive_s
        self._lock = Lock()
        self._subscribers: List[Queue[Dict[str, Any]]] = []

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
        from queue import Empty  # Import Empty from the appropriate module

        mailbox: Queue[Dict[str, Any]] = Queue()
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
                except Empty:
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
