"""Tests covering the Flask web UI helpers."""

from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator

from llm_judge.runner import RunArtifacts, RunnerConfig, RunnerControl, RunnerEvent
from llm_judge.webapp import create_app
from llm_judge.webapp.job_manager import JobManager


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


def _sample_summary() -> Dict[str, Dict[str, Any]]:
    return {
        "demo-model": {
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
    }


def _wait_for(predicate: Callable[[], bool], timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
        try:
            import gevent

            gevent.sleep(0)
        except Exception:
            pass
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
    assert status["summary"]["demo-model"]["total"] == 1
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


class DummyManager:
    def __init__(self) -> None:
        self.started_with: Dict[str, Any] | None = None

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
                "state": "idle",
                "summary": None,
                "config": None,
                "started_at": None,
                "finished_at": None,
            },
            "history": [],
        }

    def start_run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.started_with = payload
        return payload

    def pause(self) -> bool:
        return True

    def resume(self) -> bool:
        return True

    def cancel(self) -> bool:
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

    run_resp = client.post(
        "/api/run",
        json={"models": ["abc"], "judge_model": "x-ai/grok-4-fast"},
    )
    assert run_resp.status_code == 200
    assert dummy.started_with["models"] == ["abc"]

    assert client.post("/api/pause").status_code == 200
    assert client.post("/api/resume").status_code == 200
    assert client.post("/api/cancel").status_code == 200

    events_resp = client.get("/api/events")
    first_chunk = next(events_resp.response)
    assert b"event: status" in first_chunk
