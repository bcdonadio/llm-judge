"""Tests covering the FastAPI web UI helpers."""

from __future__ import annotations

import builtins
import importlib
import os
import sys
import time
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Dict, List, cast
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import llm_judge.webapp as webapp_module
from llm_judge.runner import LLMJudgeRunner, RunArtifacts, RunnerConfig, RunnerControl, RunnerEvent
from llm_judge.webapp import create_app
from llm_judge.webapp.job_manager import JobManager, ThreadedRunnerControl


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
    return False


def _runner_factory(
    delay: float = 0.02, loops: int = 1
) -> Callable[[RunnerConfig, Callable[[RunnerEvent], None], RunnerControl], StubRunner]:
    def factory(config: RunnerConfig, callback: Callable[[RunnerEvent], None], control: RunnerControl) -> StubRunner:
        return StubRunner(config, callback, control, delay=delay, loops=loops)

    return factory


class StubContainer:
    def __init__(self, resolver: Dict[Any, Any]) -> None:
        self._resolver = resolver

    def resolve(self, key: Any) -> Any:
        if key not in self._resolver:
            raise KeyError(f"No registration for {key}")
        return self._resolver[key]


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
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 503
    payload = response.json()
    assert isinstance(payload, dict)
    assert payload["error"] == "Frontend assets not found."


def test_serve_frontend_static_and_security(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "assets").mkdir()  # FastAPI requires assets subdirectory
    (dist_dir / "index.html").write_text("<html>ok</html>")
    (dist_dir / "bundle.js").write_text("console.log('hi')")

    app = create_app({"FRONTEND_DIST": str(dist_dir), "RUNS_OUTDIR": str(tmp_path / "runs")})
    client = TestClient(app)

    response = client.get("/bundle.js")
    assert response.status_code == 200
    assert response.content == b"console.log('hi')"

    # Test path traversal - FastAPI normalizes the direct "../" path but our guard still
    # ensures the response falls back to the SPA shell rather than leaking files.
    traversal = client.get("/../../secret.txt")
    assert traversal.status_code == 200
    assert b"<html>ok</html>" in traversal.content

    # Encoded traversal should be rejected with an explicit 400 error.
    encoded = client.get("/..%2F..%2Fsecret.txt")
    assert encoded.status_code == 400
    assert encoded.json()["detail"] == "Invalid path"

    absolute = client.get("/%2Fetc/passwd")
    assert absolute.status_code == 400
    assert absolute.json()["detail"] == "Invalid path"

    fallback = client.get("/")
    assert fallback.status_code == 200
    assert b"<html>ok</html>" in fallback.content

    fallback_asset = client.get("/missing.js")
    assert fallback_asset.status_code == 200
    assert b"<html>ok</html>" in fallback_asset.content


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


def test_job_manager_supported_models(tmp_path: Path) -> None:
    manager = JobManager(outdir=tmp_path)
    catalog = [{"id": "model-a"}, {"id": "model-b"}]
    manager.set_supported_models(catalog)

    catalog.append({"id": "model-c"})
    retrieved = manager.get_supported_models()
    assert retrieved == [{"id": "model-a"}, {"id": "model-b"}]

    retrieved.append({"id": "model-d"})
    assert manager.get_supported_models() == [{"id": "model-a"}, {"id": "model-b"}]


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


def test_job_manager_handle_run_completed_when_cancelled(tmp_path: Path) -> None:
    """Test that _handle_runner_event does not change state to completed when already cancelled."""
    manager = JobManager(outdir=tmp_path, runner_factory=_runner_factory())
    cast(Any, manager)._state = "cancelled"
    event = RunnerEvent("run_completed", {"csv_path": "test.csv", "runs_dir": "runs", "summary": {}})
    cast(Any, manager)._handle_runner_event(event)
    assert cast(Any, manager)._state == "cancelled"


def test_job_manager_publish_event_invokes_websocket_manager(tmp_path: Path) -> None:
    manager = JobManager(outdir=tmp_path)
    recorded: List[Dict[str, Any]] = []

    class StubWebSocketManager:
        def publish(self, event: Dict[str, Any]) -> None:
            recorded.append(event)

    cast(Any, manager)._websocket_manager = StubWebSocketManager()
    event = {"type": "status", "payload": {"state": "idle"}}
    publish_event = getattr(manager, "_publish_event")
    publish_event(event)
    assert recorded == [event]


def test_threaded_runner_control_wait_breaks_when_cancelled() -> None:
    control = ThreadedRunnerControl(poll_interval=0)
    control.pause()
    cast(Any, control)._cancel_event.set()
    cast(Any, control)._pause_event.clear()
    control.wait_if_paused()


def test_websocket_endpoint_exists() -> None:
    """Test that WebSocket endpoint is registered."""
    app = create_app({"TESTING": True})

    # Verify WebSocket endpoint exists by checking app.routes
    has_ws_route = any(getattr(route, "path", None) == "/api/ws" for route in app.routes)
    assert has_ws_route, "WebSocket endpoint /api/ws not found in routes"


def test_websocket_connection_basic(tmp_path: Path) -> None:
    """Test basic WebSocket connection and initial state."""
    app = create_app({"TESTING": True, "RUNS_OUTDIR": str(tmp_path)})
    client = TestClient(app)

    with client.websocket_connect("/api/ws") as websocket:
        # Should receive initial status message
        data = websocket.receive_json()
        assert data["type"] == "status"
        assert "payload" in data
        assert data["payload"]["state"] == "idle"


def test_websocket_initial_events_include_summary_and_artifacts(tmp_path: Path) -> None:
    app = create_app({"TESTING": True, "RUNS_OUTDIR": str(tmp_path)})
    manager = cast(JobManager, app.state.job_manager)
    summary = {"model": {"total": 1, "ok": 1, "issues": 0}}
    artifacts = {"csv_path": "status.csv", "runs_dir": "runs"}
    cast(Any, manager)._summary = summary
    cast(Any, manager)._artifacts = artifacts
    cast(Any, manager)._history = [{"type": "message", "payload": {"content": "hello"}}]
    cast(Any, manager)._state = "running"

    client = TestClient(app)
    with client.websocket_connect("/api/ws") as websocket:
        events = [websocket.receive_json() for _ in range(4)]

    event_types = [event["type"] for event in events]
    assert event_types == ["status", "message", "summary", "artifacts"]
    assert events[2]["payload"]["summary"] == summary
    assert events[3]["payload"] == artifacts


def test_websocket_endpoint_handles_disconnect(tmp_path: Path) -> None:
    app = create_app({"TESTING": True, "RUNS_OUTDIR": str(tmp_path)})

    class SnapshotOnlyManager:
        def snapshot(self) -> Dict[str, Any]:
            return {"status": {"state": "idle"}, "history": []}

    app.state.job_manager = SnapshotOnlyManager()

    websocket_manager = app.state.websocket_manager
    original_send = websocket_manager.send_events
    original_disconnect = websocket_manager.disconnect
    disconnect_called = False
    raised = False

    async def failing_send_events(websocket: Any, initial_events: Any) -> None:
        nonlocal raised
        raised = True
        raise WebSocketDisconnect()

    async def spy_disconnect(websocket: Any) -> None:
        nonlocal disconnect_called
        disconnect_called = True
        await original_disconnect(websocket)

    setattr(websocket_manager, "send_events", failing_send_events)
    setattr(websocket_manager, "disconnect", spy_disconnect)

    websocket_route = next(route for route in app.routes if getattr(route, "path", None) == "/api/ws")
    websocket_endpoint = cast(Any, websocket_route).endpoint

    class StubWebSocket:
        async def accept(self) -> None:
            return None

        async def send_json(self, message: Dict[str, Any]) -> None:
            raise AssertionError("send_json should not be called")

    import asyncio

    async def invoke() -> None:
        await websocket_endpoint(StubWebSocket())

    asyncio.run(invoke())

    assert disconnect_called
    assert raised

    setattr(websocket_manager, "send_events", original_send)
    setattr(websocket_manager, "disconnect", original_disconnect)


class DummyManager:
    def __init__(self) -> None:
        self.started_with: Dict[str, Any] | None = None
        self.state = "idle"
        self._models = [{"id": "qwen/qwen3-next-80b-a3b-instruct"}]

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

    def get_supported_models(self) -> List[Dict[str, Any]]:
        return list(self._models)

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


def test_fastapi_api_routes(tmp_path: Path) -> None:
    app = create_app({"TESTING": True})
    dummy = DummyManager()
    app.state.job_manager = dummy

    client = TestClient(app)

    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "ok"

    state_resp = client.get("/api/state")
    assert state_resp.status_code == 200

    defaults_resp = client.get("/api/defaults")
    assert defaults_resp.status_code == 200
    defaults_payload = defaults_resp.json()
    assert defaults_payload["judge_model"] == "x-ai/grok-4-fast"
    assert defaults_payload["models"] == ["qwen/qwen3-next-80b-a3b-instruct"]

    models_resp = client.get("/api/models")
    assert models_resp.status_code == 200
    assert models_resp.json() == {"models": dummy.get_supported_models()}

    run_error = client.post("/api/run", json={"models": [], "judge_model": "x-ai/grok-4-fast"})
    assert run_error.status_code == 400
    assert "Invalid configuration provided" in run_error.json()["detail"]

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


def test_frontend_missing_assets(tmp_path: Path) -> None:
    app = create_app({"TESTING": True, "FRONTEND_DIST": str(tmp_path / "missing")})
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 503


def test_frontend_serves_existing_asset(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir(parents=True)
    (dist_dir / "assets").mkdir()  # FastAPI requires assets subdirectory
    (dist_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")
    (dist_dir / "style.css").write_text("body { color: red; }", encoding="utf-8")
    app = create_app({"TESTING": True, "FRONTEND_DIST": str(dist_dir)})
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert b"ok" in response.content

    # Test serving a specific file
    css_response = client.get("/style.css")
    assert css_response.status_code == 200
    assert b"color: red" in css_response.content


def test_create_app_configures_temp_outdir() -> None:
    app = create_app({"TESTING": True})
    outdir_value = cast(str, app.state.config["RUNS_OUTDIR"])
    outdir = Path(outdir_value).resolve()
    project_root = Path(__file__).resolve().parent.parent
    assert outdir.exists()
    with pytest.raises(ValueError):
        outdir.relative_to(project_root)
    models_cache = cast(List[Dict[str, Any]], app.state.config["OPENROUTER_MODELS"])
    assert isinstance(models_cache, list)


def test_webapp_without_di_support(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_judge.webapp as webapp

    original_import = builtins.__import__

    def fake_import(
        name: str, globals: Any | None = None, locals: Any | None = None, fromlist: tuple[str, ...] = (), level: int = 0
    ) -> Any:
        if name.endswith("container") or name.endswith("factories"):
            raise ImportError("forced")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    module = importlib.reload(webapp)
    assert module.has_di_support is False

    monkeypatch.setattr(builtins, "__import__", original_import)
    importlib.reload(webapp)


def test_create_app_with_container(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from llm_judge.services import IAPIClient, IJudgeService, IPromptsManager
    from llm_judge.domain import ModelResponse, JudgeDecision, Prompt

    class APIClientStub:
        def __init__(self) -> None:
            self.calls: Dict[str, Any] = {}

        def chat_completion(self, **_: Any) -> ModelResponse:
            return ModelResponse(text="", raw_payload={})

        def list_models(self) -> List[Dict[str, Any]]:
            self.calls["list_models"] = True
            return [{"id": "stub/model-a", "name": "Model A"}]

    class JudgeServiceStub:
        def evaluate(self, **_: Any) -> JudgeDecision:
            return JudgeDecision(
                success=True,
                initial_refusal=False,
                initial_completeness=1.0,
                initial_sourcing="",
                follow_refusal=False,
                follow_completeness=1.0,
                follow_sourcing="",
                asymmetry="none",
                safety_flags_initial=[],
                safety_flags_follow=[],
                notes="",
                raw_data={},
            )

    class PromptsManagerStub:
        def get_core_prompts(self) -> List[Prompt]:
            return [Prompt(text="p", category="c", index=0)]

        def get_follow_up(self) -> str:
            return "follow"

    api_client_stub = APIClientStub()
    container = StubContainer(
        {
            IAPIClient: api_client_stub,
            IJudgeService: JudgeServiceStub(),
            IPromptsManager: PromptsManagerStub(),
        }
    )

    app = create_app({"RUNS_OUTDIR": str(tmp_path)}, container=container)  # type: ignore[arg-type]
    manager = cast(JobManager, app.state.job_manager)
    assert isinstance(manager, JobManager)
    config = RunnerConfig(
        models=["m"],
        judge_model="judge",
        outdir=tmp_path,
        max_tokens=1,
        judge_max_tokens=1,
        temperature=0.0,
        judge_temperature=0.0,
        sleep_s=0.0,
    )
    factory = cast(
        Callable[[RunnerConfig, Callable[[RunnerEvent], None], RunnerControl], LLMJudgeRunner],
        getattr(manager, "_runner_factory"),
    )

    def noop(event: RunnerEvent) -> None:
        return None

    runner = factory(config, noop, RunnerControl())
    assert isinstance(runner, LLMJudgeRunner)
    assert app.state.config["OPENROUTER_MODELS"] == [{"id": "stub/model-a", "name": "Model A"}]
    assert api_client_stub.calls.get("list_models") is True
    assert manager.get_supported_models() == [{"id": "stub/model-a", "name": "Model A"}]


def test_create_app_respects_base_url(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "assets").mkdir()  # FastAPI requires assets subdirectory
    app = create_app(
        {
            "FRONTEND_DIST": str(dist_dir),
            "RUNS_OUTDIR": str(tmp_path / "runs"),
            "OPENROUTER_BASE_URL": "https://custom.example/api",
        }
    )
    assert app.state.config["OPENROUTER_BASE_URL"] == "https://custom.example/api"


def test_create_app_raises_when_di_factory_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    monkeypatch.setattr(webapp_module, "has_di_support", True)
    monkeypatch.setattr(webapp_module, "_runner_factory", None)

    dummy_container = cast(webapp_module.ServiceContainerType, StubContainer({}))

    with pytest.raises(RuntimeError, match="Dependency injection support is not available"):
        create_app({"FRONTEND_DIST": str(dist_dir), "RUNS_OUTDIR": str(tmp_path / "runs")}, container=dummy_container)


def test_load_supported_models_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    load_supported_models = getattr(webapp_module, "_load_supported_models")
    models = load_supported_models(None, base_url="https://example.com")
    assert models == []


def test_frontend_route_rejects_path_traversal(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    (dist_dir / "assets").mkdir(parents=True)
    (dist_dir / "index.html").write_text("<!doctype html>")
    app = create_app({"FRONTEND_DIST": str(dist_dir), "RUNS_OUTDIR": str(tmp_path / "runs")})
    client = TestClient(app)

    response = client.get("/%2e%2e/secret")
    assert response.status_code == 400


def test_frontend_route_missing_index(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "assets").mkdir()
    app = create_app({"FRONTEND_DIST": str(dist_dir), "RUNS_OUTDIR": str(tmp_path / "runs")})
    client = TestClient(app)

    response = client.get("/missing")
    assert response.status_code == 503
    assert "Frontend assets not found" in response.json()["detail"]
