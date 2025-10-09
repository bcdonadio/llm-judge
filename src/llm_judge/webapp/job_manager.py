"""Background orchestration utilities for the llm-judge web interface."""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List

# Use gevent-compatible primitives when running under gevent workers
try:
    from gevent.event import Event
    from gevent.lock import RLock as Lock
except ImportError:
    from threading import Event, Lock  # type: ignore

from llm_judge.runner import LLMJudgeRunner, RunnerConfig, RunnerControl, RunnerEvent

from .sse import SSEBroker, time_now_ms

LOGGER = logging.getLogger(__name__)


class ThreadedRunnerControl(RunnerControl):
    """Thread-safe pause/cancel primitives for the runner."""

    def __init__(self, *, poll_interval: float = 0.1) -> None:
        super().__init__()
        self._pause_event = Event()
        self._pause_event.set()
        self._cancel_event = Event()
        self._poll_interval = poll_interval

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    def cancel(self) -> None:
        self._cancel_event.set()
        self._pause_event.set()

    def wait_if_paused(self) -> None:
        while not self._pause_event.is_set():
            if self._cancel_event.is_set():
                break
            time.sleep(self._poll_interval)

    def should_stop(self) -> bool:
        return self._cancel_event.is_set()


RunnerFactory = Callable[[RunnerConfig, Callable[[RunnerEvent], None], RunnerControl], LLMJudgeRunner]


class JobManager:
    """Coordinates background runs and surfaces updates via SSE."""

    def __init__(
        self,
        *,
        outdir: Path | None = None,
        runner_factory: RunnerFactory | None = None,
        history_limit: int = 500,
    ) -> None:
        self._lock = Lock()
        self._outdir = outdir or Path("results")
        self._runner_factory = runner_factory or self._default_runner_factory
        self._history_limit = history_limit

        self._broker = SSEBroker()
        self._thread: Any | None = None
        self._control = ThreadedRunnerControl()

        self._state: str = "idle"
        self._last_error: str | None = None
        self._summary: Dict[str, Any] | None = None
        self._artifacts: Dict[str, Any] | None = None
        self._history: List[Dict[str, Any]] = []
        self._active_config: Dict[str, Any] | None = None
        self._started_at: float | None = None
        self._finished_at: float | None = None

    # ------------------------------------------------------------------ #
    # Public API

    def start_run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        config = self._build_config(payload)
        config_dict = self._config_to_dict(config)

        with self._lock:
            if self._state in {"running", "paused", "cancelling"}:
                raise RuntimeError("A run is already in progress.")
            self._state = "running"
            self._last_error = None
            self._summary = None
            self._artifacts = None
            self._history = []
            self._active_config = config_dict
            self._started_at = time.time()
            self._finished_at = None
            self._control = ThreadedRunnerControl()

            # Use gevent-compatible thread if available
            try:
                import gevent

                worker = gevent.spawn(self._run_worker, config, self._control)
            except ImportError:
                import threading

                worker = threading.Thread(
                    target=self._run_worker,
                    args=(config, self._control),
                    daemon=True,
                )
                worker.start()
            self._thread = worker

        self._broker.publish(self._status_event())
        return config_dict

    def pause(self) -> bool:
        with self._lock:
            if self._state != "running":
                return False
            self._state = "paused"
        self._control.pause()
        self._broker.publish(self._status_event())
        return True

    def resume(self) -> bool:
        with self._lock:
            if self._state != "paused":
                return False
            self._state = "running"
        self._control.resume()
        self._broker.publish(self._status_event())
        return True

    def cancel(self) -> bool:
        with self._lock:
            if self._state not in {"running", "paused"}:
                return False
            self._state = "cancelling"
        self._control.cancel()
        self._broker.publish(self._status_event())
        return True

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            history_copy = copy.deepcopy(self._history)
            summary_copy = copy.deepcopy(self._summary)
            artifacts_copy = copy.deepcopy(self._artifacts)
            config_copy = copy.deepcopy(self._active_config)
            status = {
                "state": self._state,
                "error": self._last_error,
                "config": config_copy,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
            }
            if summary_copy is not None:
                status["summary"] = summary_copy
            if artifacts_copy is not None:
                status["artifacts"] = artifacts_copy

        return {
            "status": status,
            "history": history_copy,
        }

    def event_stream(self) -> Iterator[str]:
        with self._lock:
            status_payload = copy.deepcopy(self._status_payload())
            history = copy.deepcopy(self._history)
            summary_copy = copy.deepcopy(self._summary)
            artifacts_copy = copy.deepcopy(self._artifacts)

        initial_events: List[Dict[str, Any]] = [{"type": "status", "payload": status_payload}]
        initial_events.extend(history)
        if summary_copy is not None:
            initial_events.append({"type": "summary", "payload": {"summary": summary_copy}})
        if artifacts_copy is not None:
            initial_events.append({"type": "artifacts", "payload": artifacts_copy})
        return self._broker.stream(initial_events)

    def defaults(self) -> Dict[str, Any]:
        return {
            "models": ["qwen/qwen3-next-80b-a3b-instruct"],
            "judge_model": "x-ai/grok-4-fast",
            "limit": 1,
            "max_tokens": 8000,
            "judge_max_tokens": 6000,
            "temperature": 0.2,
            "judge_temperature": 0.0,
            "sleep_s": 0.2,
            "outdir": str(self._outdir),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers

    def _run_worker(self, config: RunnerConfig, control: ThreadedRunnerControl) -> None:
        def handler(event: RunnerEvent) -> None:
            self._handle_runner_event(event)

        try:
            runner = self._runner_factory(config, handler, control)
            artifacts = runner.run()
            with self._lock:
                if self._artifacts is None:
                    self._artifacts = {
                        "csv_path": str(artifacts.csv_path),
                        "runs_dir": str(artifacts.runs_dir),
                        "summary": copy.deepcopy(artifacts.summaries),
                    }
                if self._state not in {"cancelled", "error"}:
                    self._state = "completed"
                self._finished_at = time.time()
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception("Runner execution failed")
            with self._lock:
                self._last_error = str(exc)
                self._state = "error"
                self._finished_at = time.time()
        finally:
            with self._lock:
                self._thread = None
            self._broker.publish(self._status_event())

    def _handle_runner_event(self, event: RunnerEvent) -> None:
        event_dict = {"type": event.type, "payload": event.payload}
        store_in_history = event.type in {"run_started", "message", "judge", "run_completed", "run_cancelled"}
        publish_status = False

        if event.type == "summary":
            with self._lock:
                summary = event.payload.get("summary")
                self._summary = copy.deepcopy(summary)
        elif event.type in {"run_completed", "run_cancelled"}:
            with self._lock:
                self._artifacts = {
                    "csv_path": event.payload.get("csv_path"),
                    "runs_dir": event.payload.get("runs_dir"),
                    "summary": copy.deepcopy(event.payload.get("summary")),
                }
                if event.type == "run_cancelled":
                    self._state = "cancelled"
                elif self._state not in {"cancelled", "error"}:
                    self._state = "completed"
                self._finished_at = time.time()
                publish_status = True
        elif event.type == "run_started":
            with self._lock:
                self._started_at = time.time()

        if store_in_history:
            self._append_history(event_dict)

        self._broker.publish(event_dict)
        if publish_status:
            self._broker.publish(self._status_event())

    def _append_history(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self._history.append(copy.deepcopy(event))
            if len(self._history) > self._history_limit:
                self._history = self._history[-self._history_limit :]

    def _status_event(self) -> Dict[str, Any]:
        return {"type": "status", "payload": self._status_payload()}

    def _status_payload(self) -> Dict[str, Any]:
        with self._lock:
            payload = {
                "state": self._state,
                "error": self._last_error,
                "config": copy.deepcopy(self._active_config),
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "ts": time_now_ms(),
            }
            if self._summary is not None:
                payload["summary"] = copy.deepcopy(self._summary)
            if self._artifacts is not None:
                payload["artifacts"] = copy.deepcopy(self._artifacts)
        return payload

    def _build_config(self, payload: Dict[str, Any]) -> RunnerConfig:
        defaults = self.defaults()
        merged = {**defaults, **{k: v for k, v in payload.items() if v is not None}}

        models = merged.get("models")
        if isinstance(models, str):
            models = [m.strip() for m in models.split() if m.strip()]
        if not models:
            raise ValueError("Provide at least one model slug to evaluate.")

        outdir = Path(merged.get("outdir", self._outdir))
        outdir.mkdir(parents=True, exist_ok=True)

        return RunnerConfig(
            models=list(models),
            judge_model=str(merged.get("judge_model")),
            outdir=outdir,
            max_tokens=int(merged.get("max_tokens")),
            judge_max_tokens=int(merged.get("judge_max_tokens")),
            temperature=float(merged.get("temperature")),
            judge_temperature=float(merged.get("judge_temperature")),
            sleep_s=float(merged.get("sleep_s")),
            limit=int(merged["limit"]) if merged.get("limit") is not None else None,
            verbose=bool(merged.get("verbose", False)),
            use_color=False,
        )

    def _config_to_dict(self, config: RunnerConfig) -> Dict[str, Any]:
        return {
            "models": list(config.models),
            "judge_model": config.judge_model,
            "outdir": str(config.outdir),
            "max_tokens": config.max_tokens,
            "judge_max_tokens": config.judge_max_tokens,
            "temperature": config.temperature,
            "judge_temperature": config.judge_temperature,
            "sleep_s": config.sleep_s,
            "limit": config.limit,
            "verbose": config.verbose,
        }

    @staticmethod
    def _default_runner_factory(
        config: RunnerConfig,
        progress_callback: Callable[[RunnerEvent], None],
        control: RunnerControl,
    ) -> LLMJudgeRunner:
        return LLMJudgeRunner(config, progress_callback=progress_callback, control=control, logger=LOGGER)
