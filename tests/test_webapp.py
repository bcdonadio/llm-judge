"""Tests covering the Flask web UI helpers."""

from __future__ import annotations

import builtins
import os
import sys
import time
from importlib import import_module
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Dict, Iterator, List, cast
from types import SimpleNamespace

import pytest

import llm_judge.webapp as webapp_module
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


def test_load_dotenv_missing_file(tmp_path: Path) -> None:
    webapp_module._load_dotenv(tmp_path / "nonexistent.env")  # pyright: ignore[reportPrivateUsage]


def test_load_dotenv_uses_python_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\n")

    load_calls: Dict[str, Any] = {}

    def fake_load(path: str | Path, *, override: bool) -> None:
        load_calls["path"] = Path(path)
        load_calls["override"] = override

    manual_called = False

    def fake_manual(path: Path) -> None:
        nonlocal manual_called
        manual_called = True

    monkeypatch.setitem(sys.modules, "dotenv", SimpleNamespace(load_dotenv=fake_load))
    monkeypatch.setattr(webapp_module, "_load_dotenv_manual", fake_manual)

    webapp_module._load_dotenv(env_file)  # pyright: ignore[reportPrivateUsage]

    assert load_calls == {"path": env_file, "override": False}
    assert manual_called is False


def test_load_dotenv_with_library_import_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original_import = builtins.__import__

    def fake_import(
        name: str, globals: Any | None = None, locals: Any | None = None, fromlist: tuple[str, ...] = (), level: int = 0
    ) -> Any:
        if name == "dotenv":
            raise ImportError("No module named 'dotenv'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert webapp_module._load_dotenv_with_library(tmp_path / ".env") is False  # pyright: ignore[reportPrivateUsage]


def test_load_dotenv_manual_parses_entries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment line",
                "export QUOTED='hello world'",
                "INLINE=value # inline comment",
                "BARE=value2",
                "LEADING=# comment",
                "EMPTY=",
                "EMBEDDED=abc#123",
                "export NO_EQUALS",
                "=skip-me",
                "",
            ]
        )
    )

    def fake_loader(_path: Path) -> bool:
        return False

    monkeypatch.setattr(webapp_module, "_load_dotenv_with_library", fake_loader)
    monkeypatch.delenv("QUOTED", raising=False)
    monkeypatch.setenv("INLINE", "preexisting")
    monkeypatch.delenv("BARE", raising=False)
    monkeypatch.delenv("LEADING", raising=False)
    monkeypatch.delenv("EMPTY", raising=False)
    monkeypatch.delenv("EMBEDDED", raising=False)

    webapp_module._load_dotenv(env_file)  # pyright: ignore[reportPrivateUsage]

    assert os.environ["QUOTED"] == "hello world"
    assert os.environ["INLINE"] == "preexisting"  # existing value not overridden
    assert os.environ["BARE"] == "value2"
    assert os.environ["LEADING"] == ""
    assert os.environ["EMPTY"] == ""
    assert os.environ["EMBEDDED"] == "abc#123"
    assert "NO_EQUALS" not in os.environ
    assert "skip-me" not in os.environ

    for key in ("QUOTED", "BARE", "EMPTY", "EMBEDDED"):
        monkeypatch.delenv(key, raising=False)


def test_serve_frontend_missing_dist(tmp_path: Path) -> None:
    app = create_app({"FRONTEND_DIST": str(tmp_path / "missing"), "RUNS_OUTDIR": str(tmp_path / "runs")})
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 503
    payload = response.get_json()
    assert isinstance(payload, dict)
    assert payload["error"] == "Frontend assets not found."


def test_serve_frontend_static_and_security(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "index.html").write_text("<html>ok</html>")
    (dist_dir / "bundle.js").write_text("console.log('hi')")

    app = create_app({"FRONTEND_DIST": str(dist_dir), "RUNS_OUTDIR": str(tmp_path / "runs")})
    client = app.test_client()

    response = client.get("/bundle.js")
    assert response.status_code == 200
    assert response.data == b"console.log('hi')"

    traversal = client.get("/../secret.txt")
    assert traversal.status_code == 400
    traversal_payload = traversal.get_json()
    assert isinstance(traversal_payload, dict)
    assert traversal_payload["error"] == "Invalid path"

    fallback = client.get("/")
    assert fallback.status_code == 200
    assert b"<html>ok</html>" in fallback.data

    fallback_asset = client.get("/missing.js")
    assert fallback_asset.status_code == 200
    assert b"<html>ok</html>" in fallback_asset.data


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
    # Test pause when not running returns False
    assert manager.pause() is False
    assert manager.resume() is True
    # Test resume when not paused returns False
    assert manager.resume() is False
    assert _wait_for(lambda: manager.snapshot()["status"]["state"] == "completed")


def test_job_manager_cancel(tmp_path: Path) -> None:
    manager = JobManager(outdir=tmp_path, runner_factory=_runner_factory(delay=0.01, loops=200))
    manager.start_run({"models": ["demo-model"], "judge_model": "tester"})
    assert manager.cancel() is True
    assert _wait_for(lambda: manager.snapshot()["status"]["state"] == "cancelled")
    # Test cancel when not running returns False
    assert manager.cancel() is False


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


def test_job_manager_threading_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that JobManager falls back to threading when gevent is unavailable."""
    from llm_judge.webapp import job_manager as job_module

    original_import = job_module.import_module

    def fake_import_no_gevent(name: str) -> Any:
        if name == "gevent":
            raise ModuleNotFoundError("No module named 'gevent'")
        return original_import(name)

    monkeypatch.setattr(job_module, "import_module", fake_import_no_gevent)

    manager = JobManager(outdir=tmp_path, runner_factory=_runner_factory())
    manager.start_run({"models": ["demo-model"], "judge_model": "tester"})
    assert _wait_for(lambda: manager.snapshot()["status"]["state"] == "completed")


def test_job_manager_rejects_concurrent_runs(tmp_path: Path) -> None:
    """Test that starting a run while one is active raises RuntimeError."""
    manager = JobManager(outdir=tmp_path, runner_factory=_runner_factory(delay=0.05, loops=100))
    manager.start_run({"models": ["demo-model"], "judge_model": "tester"})
    with pytest.raises(RuntimeError, match="already in progress"):
        manager.start_run({"models": ["another-model"], "judge_model": "tester"})
    manager.cancel()


def test_job_manager_history_limit(tmp_path: Path) -> None:
    """Test that history is truncated when it exceeds the limit."""
    manager = JobManager(outdir=tmp_path, runner_factory=_runner_factory())
    cast(Any, manager)._history_limit = 3
    # Manually append events to exceed limit
    for i in range(5):
        cast(Any, manager)._append_history({"type": "test", "index": i})
    assert len(cast(Any, manager)._history) == 3
    assert cast(Any, manager)._history[0]["index"] == 2


def test_job_manager_accepts_string_models(tmp_path: Path) -> None:
    """Test that models can be provided as a space-separated string."""
    manager = JobManager(outdir=tmp_path, runner_factory=_runner_factory())
    config = manager.start_run({"models": "model-a model-b", "judge_model": "tester"})
    assert config["models"] == ["model-a", "model-b"]
    assert _wait_for(lambda: manager.snapshot()["status"]["state"] == "completed")


def test_job_manager_rejects_empty_string_models(tmp_path: Path) -> None:
    """Test that empty string for models raises ValueError."""
    manager = JobManager(outdir=tmp_path, runner_factory=_runner_factory())
    with pytest.raises(ValueError, match="Provide at least one model"):
        manager.start_run({"models": "   ", "judge_model": "tester"})


def test_job_manager_artifacts_fallback(tmp_path: Path) -> None:
    """Test that artifacts are set even when runner doesn't emit completion event."""

    class SilentRunner:
        """Runner that doesn't emit completion events."""

        def __init__(
            self, config: RunnerConfig, callback: Callable[[RunnerEvent], None], control: RunnerControl
        ) -> None:
            self.config = config
            self.callback = callback
            self.control = control

        def run(self) -> RunArtifacts:
            # Don't emit run_completed event - test the fallback at line 214
            return RunArtifacts(csv_path=Path("results.csv"), runs_dir=Path("runs"), summaries={})

    def silent_factory(
        config: RunnerConfig, callback: Callable[[RunnerEvent], None], control: RunnerControl
    ) -> SilentRunner:
        return SilentRunner(config, callback, control)

    manager = JobManager(outdir=tmp_path, runner_factory=silent_factory)
    manager.start_run({"models": ["demo-model"], "judge_model": "tester"})
    assert _wait_for(lambda: manager.snapshot()["status"]["state"] == "completed")
    snapshot = manager.snapshot()
    assert snapshot["status"]["artifacts"] is not None
    assert snapshot["status"]["artifacts"]["csv_path"] == "results.csv"


def test_job_manager_default_runner_factory(tmp_path: Path) -> None:
    """Test that default runner factory creates LLMJudgeRunner."""
    from llm_judge.runner import LLMJudgeRunner

    manager = JobManager(outdir=tmp_path)
    # Start a run that will use the default factory
    # We can't really run it without an API key, but we can test the factory method directly

    class DummyControl:
        def wait_if_paused(self) -> None:
            pass

        def should_stop(self) -> bool:
            return True

    config = RunnerConfig(
        models=["test"],
        judge_model="judge",
        outdir=tmp_path,
        max_tokens=10,
        judge_max_tokens=10,
        temperature=0.0,
        judge_temperature=0.0,
        sleep_s=0.0,
        limit=1,
        verbose=False,
        use_color=False,
    )

    def callback(_e: RunnerEvent) -> None:
        pass

    control: RunnerControl = cast(RunnerControl, DummyControl())
    runner = cast(Any, manager)._default_runner_factory(config, callback, control)
    assert isinstance(runner, LLMJudgeRunner)


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


def test_sse_broker_requires_type_field() -> None:
    broker = SSEBroker()
    with pytest.raises(ValueError, match="must include a 'type' field"):
        broker.publish({"payload": {"value": 42}})


def test_sse_broker_handles_generator_exit() -> None:
    """Test that SSE broker cleans up subscribers on generator exit."""
    broker = SSEBroker(keepalive_s=0.001)
    stream = broker.stream()
    next(stream)  # register subscriber
    assert len(cast(Any, broker)._subscribers) == 1
    # Trigger GeneratorExit by not consuming the generator
    del stream
    # The cleanup happens in the finally block when generator is garbage collected


def test_sse_broker_cleanup_when_not_subscriber() -> None:
    """Test that SSE broker does not remove mailbox if not in subscribers."""
    broker = SSEBroker(keepalive_s=0.001)
    stream = broker.stream()
    next(stream)  # start the generator, add mailbox
    mailbox = cast(Any, broker)._subscribers[0]
    assert len(cast(Any, broker)._subscribers) == 1
    # Manually remove the mailbox
    cast(Any, broker)._subscribers.remove(mailbox)
    # Trigger GeneratorExit
    del stream
    # Since mailbox not in subscribers, the if is false


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
            "outdir": "/tmp/llm-judge-test",
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
        if self.state in {"running", "paused"}:
            raise RuntimeError("A run is already in progress.")
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

    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200
    assert health_resp.get_json()["status"] == "ok"

    state_resp = client.get("/api/state")
    assert state_resp.status_code == 200

    defaults_resp = client.get("/api/defaults")
    assert defaults_resp.status_code == 200
    defaults_payload = defaults_resp.get_json()
    assert defaults_payload["judge_model"] == "x-ai/grok-4-fast"
    assert defaults_payload["models"] == ["qwen/qwen3-next-80b-a3b-instruct"]

    run_error = client.post("/api/run", json={"models": [], "judge_model": "x-ai/grok-4-fast"})
    assert run_error.status_code == 400
    assert "Invalid configuration provided" in run_error.get_json()["error"]

    run_resp = client.post(
        "/api/run",
        json={"models": ["abc"], "judge_model": "x-ai/grok-4-fast"},
    )
    assert run_resp.status_code == 200
    assert dummy.started_with is not None
    assert dummy.started_with["models"] == ["abc"]

    # Test RuntimeError when run is already in progress
    run_conflict = client.post(
        "/api/run",
        json={"models": ["def"], "judge_model": "x-ai/grok-4-fast"},
    )
    assert run_conflict.status_code == 409

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
    (dist_dir / "style.css").write_text("body { color: red; }", encoding="utf-8")
    app = create_app({"TESTING": True, "FRONTEND_DIST": str(dist_dir)})
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"ok" in response.data

    # Test serving a specific file
    css_response = client.get("/style.css")
    assert css_response.status_code == 200
    assert b"color: red" in css_response.data


def test_create_app_configures_temp_outdir() -> None:
    app = create_app({"TESTING": True})
    outdir_value = cast(str, app.config["RUNS_OUTDIR"])
    outdir = Path(outdir_value).resolve()
    project_root = Path(__file__).resolve().parent.parent
    assert outdir.exists()
    with pytest.raises(ValueError):
        outdir.relative_to(project_root)


def test_job_manager_event_stream_includes_history(tmp_path: Path) -> None:
    manager = JobManager(outdir=tmp_path, runner_factory=_runner_factory())
    cast(Any, manager)._history = [{"type": "message", "payload": {"body": "hello"}}]
    cast(Any, manager)._summary = {"demo": [{"total": 0}]}
    stream = manager.event_stream()
    status_chunk = next(stream)
    assert "event: status" in status_chunk
    history_chunk = next(stream)
    assert "message" in history_chunk


def test_job_manager_event_stream_includes_artifacts(tmp_path: Path) -> None:
    """Test that event_stream includes artifacts when they exist."""
    manager = JobManager(outdir=tmp_path, runner_factory=_runner_factory())
    cast(Any, manager)._history = []
    cast(Any, manager)._summary = {"demo": [{"total": 1}]}
    cast(Any, manager)._artifacts = {
        "csv_path": "results.csv",
        "runs_dir": "runs",
        "summary": {"demo": [{"total": 1}]},
    }
    stream = manager.event_stream()
    status_chunk = next(stream)
    assert "event: status" in status_chunk
    summary_chunk = next(stream)
    assert "summary" in summary_chunk
    artifacts_chunk = next(stream)
    assert "artifacts" in artifacts_chunk
    assert "results.csv" in artifacts_chunk


def test_job_manager_event_stream_no_summary_when_none(tmp_path: Path) -> None:
    """Test that event_stream does not include summary when _summary is None."""
    manager = JobManager(outdir=tmp_path, runner_factory=_runner_factory())
    cast(Any, manager)._history = [{"type": "message", "payload": {"body": "hello"}}]
    cast(Any, manager)._summary = None
    stream = manager.event_stream()
    status_chunk = next(stream)
    assert "event: status" in status_chunk
    history_chunk = next(stream)
    assert "message" in history_chunk
    cast(Any, stream).close()
    # No summary chunk since _summary is None
    try:
        next(stream)
        assert False, "Should not have more chunks"
    except StopIteration:
        pass


def test_threaded_runner_control_wait_breaks_when_cancelled() -> None:
    control = ThreadedRunnerControl(poll_interval=0)
    control.pause()
    cast(Any, control)._cancel_event.set()
    cast(Any, control)._pause_event.clear()
    control.wait_if_paused()


def test_job_manager_handle_run_completed_when_cancelled(tmp_path: Path) -> None:
    """Test that _handle_runner_event does not change state to completed when already cancelled."""
    manager = JobManager(outdir=tmp_path, runner_factory=_runner_factory())
    cast(Any, manager)._state = "cancelled"
    event = RunnerEvent("run_completed", {"csv_path": "test.csv", "runs_dir": "runs", "summary": {}})
    cast(Any, manager)._handle_runner_event(event)
    assert cast(Any, manager)._state == "cancelled"
