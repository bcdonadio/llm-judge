"""Tests covering the FastAPI web UI helpers."""

from __future__ import annotations

import builtins
import importlib
import logging
import textwrap
import time
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Dict, List, cast

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


def test_load_yaml_config_success(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        textwrap.dedent(
            """
            listen:
              frontend:
                host: 127.0.0.1
                port: 5173
              proxy:
                host: 127.0.0.1
                port: 5000
            inference:
              endpoint: https://example.test/v1
              key: secret-key
            retries:
              max: 4
              timeout: 45
            log_level: DEBUG
            defaults:
              max_rounds: 7
              judge: judge/slim
              models:
                - model/a
                - model/b
            """
        ).strip()
    )

    settings = webapp_module._load_yaml_config(config_file)  # pyright: ignore[reportPrivateUsage]
    assert settings.inference.endpoint == "https://example.test/v1"
    assert settings.defaults.max_rounds == 7
    assert settings.defaults.models == ["model/a", "model/b"]


def test_settings_to_job_defaults_roundtrip() -> None:
    settings = webapp_module.ApplicationSettings(**_base_config_dict())
    defaults = webapp_module._settings_to_job_defaults(settings)  # pyright: ignore[reportPrivateUsage]
    assert defaults == {
        "models": ["model/a", "model/b"],
        "judge_model": "judge/slim",
        "limit": 7,
    }


def test_setup_app_config_without_overrides(tmp_path: Path) -> None:
    frontend = tmp_path / "dist"
    config, base_url, outdir = webapp_module._setup_app_config(frontend, None)  # pyright: ignore[reportPrivateUsage]
    assert config["FRONTEND_DIST"].endswith("dist")
    assert base_url == "https://openrouter.ai/api/v1"
    assert outdir.exists()


def test_setup_app_config_with_partial_overrides(tmp_path: Path) -> None:
    frontend = tmp_path / "dist"
    overrides = {"SOME_FLAG": True}
    config, base_url, _ = webapp_module._setup_app_config(frontend, overrides)  # pyright: ignore[reportPrivateUsage]
    assert config["SOME_FLAG"] is True
    assert base_url == "https://openrouter.ai/api/v1"


def test_load_yaml_config_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(RuntimeError, match="Configuration file not found"):
        webapp_module._load_yaml_config(missing)  # pyright: ignore[reportPrivateUsage]


def test_load_yaml_config_invalid_structure(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("[]")
    with pytest.raises(RuntimeError, match="Configuration must be a mapping"):
        webapp_module._load_yaml_config(config_file)  # pyright: ignore[reportPrivateUsage]


def test_load_yaml_config_validation_error(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("inference_endpoint: 123\n")
    with pytest.raises(RuntimeError, match="Invalid configuration file"):
        webapp_module._load_yaml_config(config_file)  # pyright: ignore[reportPrivateUsage]


def test_configure_logging_level_sets_root(monkeypatch: pytest.MonkeyPatch) -> None:
    root_logger = logging.getLogger()
    original_level = root_logger.level
    try:
        webapp_module._configure_logging_level("DEBUG")  # pyright: ignore[reportPrivateUsage]
        assert root_logger.level == logging.DEBUG
    finally:
        root_logger.setLevel(original_level)


def test_configure_logging_level_invalid() -> None:
    with pytest.raises(RuntimeError, match="Invalid log level"):
        webapp_module._configure_logging_level("NOT_A_LEVEL")  # pyright: ignore[reportPrivateUsage]


def test_coerce_int_helper() -> None:
    coerce = getattr(webapp_module, "_coerce_int")
    assert coerce(5) == 5
    assert coerce("10") == 10
    assert coerce("abc") is None
    assert coerce(None) is None


def test_serve_frontend_missing_dist(tmp_path: Path) -> None:
    app = create_app({"FRONTEND_DIST": str(tmp_path / "missing"), "RUNS_OUTDIR": str(tmp_path / "runs")})
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 503
    payload = response.json()
    assert isinstance(payload, dict)
    assert payload["detail"] == "Frontend assets not found."
    assert payload["hint"] == "Run `npm install && npm run build` inside webui/."


def test_serve_frontend_static_and_security(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "assets").mkdir()  # FastAPI requires assets subdirectory
    (dist_dir / "index.html").write_text("<html>ok</html>")
    (dist_dir / "bundle.js").write_text("console.log('hi')")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    symlink_target = dist_dir / "link"
    symlink_target.symlink_to(outside, target_is_directory=True)

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

    symlink = client.get("/link/secret.txt")
    assert symlink.status_code == 400
    assert symlink.json()["detail"] == "Invalid path"


def test_create_app_uses_yaml_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    app = create_app({"TESTING": True, "RUNS_OUTDIR": str(tmp_path)})
    manager = cast(JobManager, app.state.job_manager)
    defaults = manager.defaults()
    assert defaults["models"] == ["qwen/qwen3-next-80b-a3b-instruct"]
    assert defaults["judge_model"] == "x-ai/grok-fast-4"
    assert defaults["limit"] == 1
    import os as _os

    assert _os.getenv("OPENROUTER_API_KEY") == "your_api_key_here"
    assert app.state.settings.defaults.max_rounds == 1
    assert app.state.config["LISTEN_FRONTEND_HOST"] == "0.0.0.0"
    assert app.state.config["LISTEN_PROXY_PORT"] == 5000


def test_create_app_keeps_existing_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "preexisting")
    app = create_app({"TESTING": True, "RUNS_OUTDIR": str(tmp_path)})
    import os as _os

    assert _os.getenv("OPENROUTER_API_KEY") == "preexisting"
    manager = cast(JobManager, app.state.job_manager)
    assert manager.defaults()["limit"] == 1


def test_create_app_allows_limit_override(tmp_path: Path) -> None:
    app = create_app({"TESTING": True, "RUNS_OUTDIR": str(tmp_path), "DEFAULT_LIMIT": "9"})
    manager = cast(JobManager, app.state.job_manager)
    assert manager.defaults()["limit"] == 9


def test_create_app_nonlist_models_override_uses_defaults(tmp_path: Path) -> None:
    app = create_app(
        {
            "TESTING": True,
            "RUNS_OUTDIR": str(tmp_path),
            "DEFAULT_MODELS": "not-a-list",
        }
    )
    manager = cast(JobManager, app.state.job_manager)
    assert manager.defaults()["models"] == ["qwen/qwen3-next-80b-a3b-instruct"]


def test_job_manager_default_builder_skips_none(tmp_path: Path) -> None:
    manager = JobManager(outdir=tmp_path, defaults={"limit": 2, "temperature": None})
    defaults = manager.defaults()
    assert defaults["limit"] == 2
    assert defaults["temperature"] == 0.2


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
    original_disconnect = websocket_manager.disconnect
    original_send_event = websocket_manager._send_event
    disconnect_called = False
    raised = False

    async def failing_send_event(websocket: Any, event: Any) -> None:
        nonlocal raised
        raised = True
        raise WebSocketDisconnect()

    async def spy_disconnect(websocket: Any) -> None:
        nonlocal disconnect_called
        disconnect_called = True
        await original_disconnect(websocket)

    setattr(websocket_manager, "_send_event", failing_send_event)
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

    setattr(websocket_manager, "_send_event", original_send_event)
    setattr(websocket_manager, "disconnect", original_disconnect)


def test_serve_frontend_request_handles_missing_raw_path(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "assets").mkdir()
    (dist_dir / "index.html").write_text("<html>ok</html>")

    serve_frontend_request = getattr(webapp_module, "_serve_frontend_request")
    response = serve_frontend_request(dist_dir, "index.html", None)
    assert getattr(response, "status_code", 200) == 200


def test_normalise_raw_path_non_bytes() -> None:
    normalise_raw_path = getattr(webapp_module, "_normalise_raw_path")
    assert normalise_raw_path("not-bytes") is None
    assert normalise_raw_path(bytearray(b"/path")) == b"/path"


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
    assert app.state.config["DEFAULT_LIMIT"] == 1


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


def test_load_supported_models_applies_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "token")
    captured: Dict[str, Any] = {}

    class DummyClient:
        def __init__(
            self,
            *,
            api_key: str,
            base_url: str,
            timeout: int = 0,
            max_retries: int = 0,
            logger: Any | None = None,
        ) -> None:
            captured.update(
                {
                    "api_key": api_key,
                    "base_url": base_url,
                    "timeout": timeout,
                    "max_retries": max_retries,
                }
            )

        def list_models(self) -> List[Dict[str, Any]]:
            return [{"id": "demo"}]

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr("llm_judge.infrastructure.api_client.OpenRouterClient", DummyClient)

    models = webapp_module._load_supported_models(  # pyright: ignore[reportPrivateUsage]
        None,
        base_url="https://example.com",
        timeout=123,
        max_retries=4,
    )

    assert models == [{"id": "demo"}]
    assert captured == {
        "api_key": "token",
        "base_url": "https://example.com",
        "timeout": 123,
        "max_retries": 4,
        "closed": True,
    }


def test_resolve_api_client_applies_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "token")
    captured: Dict[str, Any] = {}

    class DummyClient:
        def __init__(
            self,
            *,
            api_key: str,
            base_url: str,
            timeout: int = 0,
            max_retries: int = 0,
            logger: Any | None = None,
        ) -> None:
            captured.update(
                {
                    "api_key": api_key,
                    "base_url": base_url,
                    "timeout": timeout,
                    "max_retries": max_retries,
                }
            )

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr("llm_judge.infrastructure.api_client.OpenRouterClient", DummyClient)

    client, should_close = webapp_module._resolve_api_client(  # pyright: ignore[reportPrivateUsage]
        None,
        base_url="https://example.com",
        timeout=10,
        max_retries=5,
    )

    assert captured == {
        "api_key": "token",
        "base_url": "https://example.com",
        "timeout": 10,
        "max_retries": 5,
    }
    assert should_close is True
    assert client is not None

    # Ensure cleanup path executes
    client.close()
    assert captured["closed"] is True


def test_resolve_api_client_without_optional_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "token")
    captured: Dict[str, Any] = {}

    class DummyClient:
        def __init__(
            self,
            *,
            api_key: str,
            base_url: str,
            timeout: int = 0,
            max_retries: int = 0,
            logger: Any | None = None,
        ) -> None:
            captured.update(
                {
                    "api_key": api_key,
                    "base_url": base_url,
                    "timeout": timeout,
                    "max_retries": max_retries,
                }
            )

    monkeypatch.setattr("llm_judge.infrastructure.api_client.OpenRouterClient", DummyClient)

    client, should_close = webapp_module._resolve_api_client(  # pyright: ignore[reportPrivateUsage]
        None,
        base_url="https://example.com",
        timeout=None,
        max_retries=None,
    )

    assert captured == {
        "api_key": "token",
        "base_url": "https://example.com",
        "timeout": 0,
        "max_retries": 0,
    }
    assert should_close is True
    assert client is not None


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


def _base_config_dict() -> Dict[str, Any]:
    return {
        "listen": {
            "frontend": {"host": "127.0.0.1", "port": 5173},
            "proxy": {"host": "127.0.0.1", "port": 5000},
        },
        "inference": {"endpoint": "https://example.test/v1", "key": "secret-key"},
        "retries": {"max": 4, "timeout": 45},
        "log_level": "DEBUG",
        "defaults": {
            "max_rounds": 7,
            "judge": "judge/slim",
            "models": ["model/a", "model/b"],
        },
    }
