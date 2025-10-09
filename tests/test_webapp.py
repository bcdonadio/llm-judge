"""Tests covering the Flask web UI helpers."""

from __future__ import annotations
import time
from importlib import import_module
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Dict, Iterator, List, cast

import pytest

from llm_judge.runner import RunArtifacts, RunnerConfig, RunnerControl, RunnerEvent
from llm_judge.webapp import create_app
from llm_judge.webapp.job_manager import JobManager, ThreadedRunnerControl
from llm_judge.webapp.sse import SSEBroker


class StubRunner:
    """Deterministic runner used to simulate backend behaviour."""

    def __init__(
        self,
        config: RunnerConfig,
        callback: Callable[[RunnerEvent], None],
        control: RunnerControl,
        *,
        delay: float = 0.02,
        loops: int = 1,
    ) -> None:
        self.config = config
        self.callback = callback
        self.control = control
        self.delay = delay
        self.loops = loops

    def run(self) -> RunArtifacts:
        summary = _sample_summary()
        self.callback(RunnerEvent("run_started", {"models": list(self.config.models)}))
        self.callback(
            RunnerEvent(
                "message",
                {
                    "model": self.config.models[0],
                    "prompt_index": 0,
                    "role": "user",
                    "content": "Prompt text",
                },
            )
        )
        for _ in range(self.loops):
            time.sleep(self.delay)
            self.control.wait_if_paused()
            if self.control.should_stop():
                self.callback(
                    RunnerEvent(
                        "run_cancelled",
                        {"csv_path": str(Path("cancelled.csv")), "runs_dir": str(Path("runs")), "summary": summary},
                    )
                )
                return RunArtifacts(csv_path=Path("cancelled.csv"), runs_dir=Path("runs"), summaries=summary)
        self.callback(RunnerEvent("summary", {"summary": summary}))
        self.callback(
            RunnerEvent(
                "run_completed",
                {"csv_path": str(Path("result.csv")), "runs_dir": str(Path("runs")), "summary": summary},
            )
        )
        return RunArtifacts(csv_path=Path("result.csv"), runs_dir=Path("runs"), summaries=summary)


def _sample_summary() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "demo-model": [
            {
                "total": 1,
                "ok": 1,
                "issues": 0,
                "avg_initial_completeness": 1.0,
                "avg_followup_completeness": 1.0,
                "initial_refusal_rate": 0.0,
                "followup_refusal_rate": 0.0,
                "initial_sourcing_counts": {"primary": 1},
                "followup_sourcing_counts": {"primary": 1},
                "asymmetry_counts": {"none": 1},
                "error_counts": {},
            }
        ]
    }


def _wait_for(predicate: Callable[[], bool], timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
        try:
            gevent_module = import_module("gevent")
        except ModuleNotFoundError:
            gevent_module = None
        if gevent_module is not None:
            gevent_module.sleep(0)
    return False


def _runner_factory(
    delay: float = 0.02, loops: int = 1
) -> Callable[[RunnerConfig, Callable[[RunnerEvent], None], RunnerControl], StubRunner]:
    def factory(config: RunnerConfig, callback: Callable[[RunnerEvent], None], control: RunnerControl) -> StubRunner:
        return StubRunner(config, callback, control, delay=delay, loops=loops)

    return factory


def test_job_manager_completes_run(tmp_path: Path) -> None:
    manager = JobManager(outdir=tmp_path, runner_factory=_runner_factory())
    config_dict = manager.start_run({"models": ["demo-model"], "judge_model": "tester"})
    assert config_dict["models"] == ["demo-model"]

    assert _wait_for(lambda: manager.snapshot()["status"]["state"] == "completed")
    snapshot = manager.snapshot()
    status = snapshot["status"]
    summary_entries = status["summary"]["demo-model"]
    assert summary_entries[0]["total"] == 1
    assert any(event["type"] == "message" for event in snapshot["history"])


def test_job_manager_pause_resume_flow(tmp_path: Path) -> None:
    manager = JobManager(outdir=tmp_path, runner_factory=_runner_factory(delay=0.01, loops=100))
    manager.start_run({"models": ["demo-model"], "judge_model": "tester"})
    assert manager.pause() is True
    assert manager.snapshot()["status"]["state"] == "paused"
    assert manager.resume() is True
    assert _wait_for(lambda: manager.snapshot()["status"]["state"] == "completed")


def test_job_manager_cancel(tmp_path: Path) -> None:
    manager = JobManager(outdir=tmp_path, runner_factory=_runner_factory(delay=0.01, loops=200))
    manager.start_run({"models": ["demo-model"], "judge_model": "tester"})
    assert manager.cancel() is True
    assert _wait_for(lambda: manager.snapshot()["status"]["state"] == "cancelled")


def test_threaded_runner_control_waits_until_resumed() -> None:
    control = ThreadedRunnerControl(poll_interval=0.001)
    control.pause()
    resumed: List[bool] = []

    def worker() -> None:
        control.wait_if_paused()
        resumed.append(True)

    thread = Thread(target=worker)
    thread.start()
    time.sleep(0.01)
    control.resume()
    thread.join(timeout=1)
    assert resumed


def test_job_manager_gevent_branch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from llm_judge.webapp import job_manager as job_module

    calls: List[FakeGreenlet] = []

    class FakeGreenlet:
        def __init__(self, func: Callable[..., Any], args: tuple[Any, ...]) -> None:
            self._func = func
            self._args = args
            self.dead = False

        def run(self) -> None:
            self._func(*self._args)
            self.dead = True

    class FakeGevent:
        def spawn(self, func: Callable[..., Any], *args: Any) -> FakeGreenlet:
            greenlet = FakeGreenlet(func, args)
            calls.append(greenlet)
            return greenlet

    fake_gevent = FakeGevent()
    original_import = job_module.import_module

    def fake_import(name: str) -> Any:
        if name == "gevent":
            return fake_gevent
        return original_import(name)

    monkeypatch.setattr(job_module, "import_module", fake_import)

    manager = JobManager(outdir=tmp_path, runner_factory=_runner_factory())
    manager.start_run({"models": ["demo-model"], "judge_model": "tester"})
    assert calls
    for greenlet in calls:
        greenlet.run()


def test_sse_broker_publish_roundtrip() -> None:
    broker = SSEBroker(keepalive_s=0.001)
    stream = broker.stream(initial=[{"type": "ready"}])
    next(stream)  # consume initial event to register subscriber
    broker.publish({"type": "message", "payload": {"value": 42}})
    for _ in range(5):
        chunk = next(stream)
        if "value" in chunk:
            break
    else:
        pytest.fail("did not receive expected message event")


class DummyManager:
    def __init__(self) -> None:
        self.started_with: Dict[str, Any] | None = None
        self.state = "idle"

    def defaults(self) -> Dict[str, Any]:
        return {
            "models": ["qwen/qwen3-next-80b-a3b-instruct"],
            "judge_model": "x-ai/grok-4-fast",
            "limit": 1,
            "max_tokens": 100,
            "judge_max_tokens": 80,
            "temperature": 0.1,
            "judge_temperature": 0.0,
            "sleep_s": 0.1,
            "outdir": "results",
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "status": {
                "state": self.state,
                "summary": None,
                "config": None,
                "started_at": None,
                "finished_at": None,
            },
            "history": [],
        }

    def start_run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        models = payload.get("models")
        if isinstance(models, list) and not models:
            raise ValueError("Provide at least one model slug to evaluate.")
        self.started_with = payload
        self.state = "running"
        return payload

    def pause(self) -> bool:
        if self.state != "running":
            return False
        self.state = "paused"
        return True

    def resume(self) -> bool:
        if self.state != "paused":
            return False
        self.state = "running"
        return True

    def cancel(self) -> bool:
        if self.state not in {"running", "paused"}:
            return False
        self.state = "cancelled"
        return True

    def event_stream(self) -> Iterator[str]:
        yield 'event: status\ndata: {"type":"status","payload":{"state":"idle"}}\n\n'


def test_flask_api_routes(tmp_path: Path) -> None:
    app = create_app({"TESTING": True})
    dummy = DummyManager()
    app.config["JOB_MANAGER"] = dummy

    client = app.test_client()

    defaults_resp = client.get("/api/defaults")
    assert defaults_resp.status_code == 200
    defaults_payload = defaults_resp.get_json()
    assert defaults_payload["judge_model"] == "x-ai/grok-4-fast"
    assert defaults_payload["models"] == ["qwen/qwen3-next-80b-a3b-instruct"]

    run_error = client.post("/api/run", json={"models": [], "judge_model": "x-ai/grok-4-fast"})
    assert run_error.status_code == 400
    assert "Provide at least one model" in run_error.get_json()["error"]

    run_resp = client.post(
        "/api/run",
        json={"models": ["abc"], "judge_model": "x-ai/grok-4-fast"},
    )
    assert run_resp.status_code == 200
    assert dummy.started_with is not None
    assert dummy.started_with["models"] == ["abc"]

    assert client.post("/api/pause").status_code == 200
    assert client.post("/api/resume").status_code == 200
    assert client.post("/api/cancel").status_code == 200

    assert client.post("/api/pause").status_code == 400
    assert client.post("/api/resume").status_code == 400
    assert client.post("/api/cancel").status_code == 400

    events_resp = client.get("/api/events")
    first_chunk = next(iter(events_resp.response))
    chunk_text = first_chunk.decode("utf-8") if isinstance(first_chunk, bytes) else str(first_chunk)
    assert "event: status" in chunk_text


def test_frontend_missing_assets(tmp_path: Path) -> None:
    app = create_app({"TESTING": True, "FRONTEND_DIST": str(tmp_path / "missing")})
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 503


def test_frontend_serves_existing_asset(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    app = create_app({"TESTING": True, "FRONTEND_DIST": str(dist_dir)})
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"ok" in response.data


def test_job_manager_event_stream_includes_history(tmp_path: Path) -> None:
    manager = JobManager(outdir=tmp_path, runner_factory=_runner_factory())
    cast(Any, manager)._history = [{"type": "message", "payload": {"body": "hello"}}]
    cast(Any, manager)._summary = {"demo": [{"total": 0}]}
    stream = manager.event_stream()
    status_chunk = next(stream)
    assert "event: status" in status_chunk
    history_chunk = next(stream)
    assert "message" in history_chunk


def test_threaded_runner_control_wait_breaks_when_cancelled() -> None:
    control = ThreadedRunnerControl(poll_interval=0)
    control.pause()
    cast(Any, control)._cancel_event.set()
    cast(Any, control)._pause_event.clear()
    control.wait_if_paused()
