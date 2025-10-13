# pyright: reportPrivateUsage=false

"""Tests covering the LLMJudgeRunner orchestration class."""

from __future__ import annotations

import csv
import io
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, cast

import pytest

from llm_judge.domain import JudgeDecision, ModelResponse, Prompt, RunConfiguration
from llm_judge.runner import (
    CSV_FIELDNAMES,
    LLMJudgeRunner,
    RunArtifacts,
    RunnerConfig,
    RunnerControl,
    RunnerEvent,
    run_suite,
)


def _fake_fetch_completion(*_: Any, **__: Any) -> tuple[str, Dict[str, Any]]:
    return "text", {"choices": [{"message": {"content": "text"}}]}


def _fake_judge_decision(**_: Any) -> Dict[str, Any]:
    return {
        "ok": True,
        "decision": {
            "initial": {"refusal": False, "completeness": 1, "sourcing_quality": ""},
            "followup": {"refusal": False, "completeness": 1, "sourcing_quality": ""},
            "asymmetry_leading": "none",
            "final_notes": "",
        },
    }


def _fake_now() -> str:
    return "now"


def _noop_emit(event_type: str, **payload: Any) -> None:
    return None


def _noop(*args: Any, **kwargs: Any) -> None:
    return None


def build_config(
    tmp_path: Path,
    *,
    models: Iterable[str] = (),
    limit: int | None = None,
    verbose: bool = False,
    use_color: bool = False,
) -> RunnerConfig:
    return RunnerConfig(
        models=list(models),
        judge_model="judge-model",
        outdir=tmp_path,
        max_tokens=16,
        judge_max_tokens=32,
        temperature=0.2,
        judge_temperature=0.0,
        sleep_s=0.0,
        limit=limit,
        verbose=verbose,
        use_color=use_color,
    )


def test_emit_normalises_payload(tmp_path: Path) -> None:
    events: List[RunnerEvent] = []
    runner = LLMJudgeRunner(build_config(tmp_path, models=["model"]), progress_callback=events.append)
    sample_path = tmp_path / "file.json"
    runner._emit("sample", path=sample_path, counts=Counter({"x": 2}), note="ok")
    assert events and events[0].payload["path"] == str(sample_path)
    assert events[0].payload["counts"] == {"x": 2}


def test_limited_prompts_respects_limit(tmp_path: Path) -> None:
    runner = LLMJudgeRunner(build_config(tmp_path, models=["model"], limit=1))
    prompts = runner._limited_prompts(["p1", "p2"])
    assert prompts == ["p1"]


def test_limited_prompts_logs_when_truncated(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    runner = LLMJudgeRunner(build_config(tmp_path, models=["model"], limit=1))
    caplog.set_level(logging.DEBUG, logger="llm_judge.runner")
    runner._limited_prompts(["p1", "p2"])
    assert any("Prompt limit applied" in message for message in caplog.messages)


def test_limited_prompts_no_log_when_not_truncated(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    runner = LLMJudgeRunner(build_config(tmp_path, models=["model"], limit=2))
    caplog.set_level(logging.DEBUG, logger="llm_judge.runner")
    runner._limited_prompts(["p1", "p2"])
    assert not any("Prompt limit applied" in message for message in caplog.messages)


class ImmediateStopControl:
    def wait_if_paused(self) -> None:
        return None

    def should_stop(self) -> bool:
        return True


def test_process_models_halts_when_control_requests(tmp_path: Path) -> None:
    runner = LLMJudgeRunner(build_config(tmp_path, models=["model"]))
    runner._control = cast(RunnerControl, ImmediateStopControl())
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    cancelled = runner._process_models(["prompt"], tmp_path, writer, {})
    assert cancelled is True


def test_process_models_stops_when_prompt_cancels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = LLMJudgeRunner(build_config(tmp_path, models=["model"]))

    def fake_process_single(*args: Any, **kwargs: Any) -> bool:
        return True

    monkeypatch.setattr(runner, "_process_single_model", fake_process_single)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    cancelled = runner._process_models(["prompt"], tmp_path, writer, {})
    assert cancelled is True


def test_process_prompt_stops_immediately(tmp_path: Path) -> None:
    runner = LLMJudgeRunner(build_config(tmp_path, models=["model"]))
    runner._control = cast(RunnerControl, ImmediateStopControl())
    writer = csv.DictWriter(io.StringIO(), fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    assert runner._process_prompt("model", tmp_path, 0, "prompt", writer, {}) is True


class StopAfterSleepControl:
    def __init__(self) -> None:
        self._stop = False

    def wait_if_paused(self) -> None:
        return None

    def should_stop(self) -> bool:
        return self._stop

    def trigger(self) -> None:
        self._stop = True


def test_sleep_with_stop_triggers_cancel(tmp_path: Path) -> None:
    control = StopAfterSleepControl()
    runner = LLMJudgeRunner(build_config(tmp_path, models=["model"]))
    runner._control = cast(RunnerControl, control)

    def fake_sleep(_: float) -> None:
        control.trigger()

    runner._sleep = fake_sleep
    assert runner._sleep_with_stop() is True


class ToggleControl:
    def __init__(self) -> None:
        self._stop = False
        self._calls = 0

    def wait_if_paused(self) -> None:
        return None

    def should_stop(self) -> bool:
        return self._stop

    def advance(self) -> None:
        if self._calls >= 1:
            self._stop = True
        self._calls += 1


def test_process_prompt_cancels_after_summary(tmp_path: Path) -> None:
    """Test cancellation after summary (line 557)."""
    runner = LLMJudgeRunner(build_config(tmp_path, models=["model"]))

    class StopAfterSummary:
        def __init__(self) -> None:
            self._sleep_count = 0

        def wait_if_paused(self) -> None:
            pass

        def should_stop(self) -> bool:
            # Stop after third sleep (after summary, line 557)
            return self._sleep_count >= 3

        def on_sleep(self) -> None:
            self._sleep_count += 1

    control = StopAfterSummary()
    runner._control = cast(RunnerControl, control)
    cast(Any, runner)._fetch_completion = _fake_fetch_completion
    cast(Any, runner)._judge_client = _fake_judge_decision
    cast(Any, runner)._now = _fake_now
    cast(Any, runner)._emit = _noop_emit
    cast(Any, runner)._verbose_log_prompt = _noop
    cast(Any, runner)._verbose_log_response = _noop
    cast(Any, runner)._verbose_log_judge = _noop

    def fake_sleep(_: float) -> None:
        control.on_sleep()

    runner._sleep = fake_sleep
    writer = csv.DictWriter(io.StringIO(), fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    summary: Dict[str, List[Dict[str, Any]]] = {}
    assert runner._process_prompt("model", tmp_path, 0, "prompt", writer, summary) is True


def test_process_prompt_cancels_after_followup(tmp_path: Path) -> None:
    """Test cancellation after followup prompt before followup response (line 450)."""
    runner = LLMJudgeRunner(build_config(tmp_path, models=["model"]))
    control = ToggleControl()
    control._calls = 1  # Will stop on next check (second sleep call)
    runner._control = cast(RunnerControl, control)
    cast(Any, runner)._fetch_completion = _fake_fetch_completion
    cast(Any, runner)._judge_client = _fake_judge_decision
    cast(Any, runner)._now = _fake_now
    cast(Any, runner)._emit = _noop_emit
    cast(Any, runner)._verbose_log_prompt = _noop
    cast(Any, runner)._verbose_log_response = _noop
    cast(Any, runner)._verbose_log_judge = _noop

    def fake_sleep(_: float) -> None:
        control.advance()

    runner._sleep = fake_sleep
    writer = csv.DictWriter(io.StringIO(), fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    summary: Dict[str, List[Dict[str, Any]]] = {}
    assert runner._process_prompt("model", tmp_path, 0, "prompt", writer, summary) is True


def test_process_prompt_cancels_before_judge(tmp_path: Path) -> None:
    """Test cancellation before judge call (line 480)."""
    runner = LLMJudgeRunner(build_config(tmp_path, models=["model"]))

    class StopBeforeJudge:
        def __init__(self) -> None:
            self._sleep_count = 0

        def wait_if_paused(self) -> None:
            pass

        def should_stop(self) -> bool:
            # Stop after second sleep (before judge, line 480)
            return self._sleep_count >= 2

        def on_sleep(self) -> None:
            self._sleep_count += 1

    control = StopBeforeJudge()
    runner._control = cast(RunnerControl, control)
    cast(Any, runner)._fetch_completion = _fake_fetch_completion
    cast(Any, runner)._judge_client = _fake_judge_decision
    cast(Any, runner)._now = _fake_now
    cast(Any, runner)._emit = _noop_emit
    cast(Any, runner)._verbose_log_prompt = _noop
    cast(Any, runner)._verbose_log_response = _noop
    cast(Any, runner)._verbose_log_judge = _noop

    def fake_sleep(_: float) -> None:
        control.on_sleep()

    runner._sleep = fake_sleep
    writer = csv.DictWriter(io.StringIO(), fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    summary: Dict[str, List[Dict[str, Any]]] = {}
    assert runner._process_prompt("model", tmp_path, 0, "prompt", writer, summary) is True


def test_process_single_model_breaks_on_prompt_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = LLMJudgeRunner(build_config(tmp_path, models=["model"]))

    def passthrough(prompts: Sequence[str]) -> Sequence[str]:
        return prompts

    def stop_prompt(*_: Any, **__: Any) -> bool:
        return True

    monkeypatch.setattr(runner, "_limited_prompts", passthrough)
    monkeypatch.setattr(runner, "_process_prompt", stop_prompt)
    writer = csv.DictWriter(io.StringIO(), fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    cancelled = runner._process_single_model("model", ["prompt"], tmp_path, writer, {})
    assert cancelled is True


def test_snippet_behaviour() -> None:
    assert LLMJudgeRunner._snippet("", limit=10) == "(empty)"
    text = "word " * 100
    truncated = LLMJudgeRunner._snippet(text, limit=20)
    assert truncated.endswith("…")
    assert len(truncated) == 20


def test_color_toggle(tmp_path: Path) -> None:
    runner = LLMJudgeRunner(build_config(tmp_path, models=[], use_color=True))
    plain_runner = LLMJudgeRunner(build_config(tmp_path, models=[], use_color=False))
    colored = runner._color("hello", "X")
    plain = plain_runner._color("hello", "X")
    assert colored.startswith("X") and colored.endswith("hello\x1b[0m")
    assert plain == "hello"


def test_verbose_logging_helpers(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="llm_judge.runner")
    cfg = build_config(tmp_path, models=[], verbose=True, use_color=False)
    runner = LLMJudgeRunner(cfg)

    runner._verbose_log_prompt(idx=1, prompt="Example prompt")
    runner._verbose_log_response(label="Initial", text="Answer")
    runner._verbose_log_response(label="Judge", text="Decision")
    runner._verbose_log_judge(
        judge_result={"ok": True},
        initial_decision={"refusal": False, "completeness": 3, "sourcing_quality": "high"},
        follow_decision={"refusal": True, "completeness": 1, "sourcing_quality": "low"},
        asymmetry="initial",
        final_notes="Detailed notes",
    )
    runner._verbose_log_judge(
        judge_result={"ok": False, "error": "timeout"},
        initial_decision={},
        follow_decision={},
        asymmetry="",
        final_notes="",
    )
    runner._verbose_log_prompt(idx=2, prompt="Quiet")
    runner._verbose_log_response(label="Judge", text="Ignored")
    runner._verbose_log_judge(
        judge_result={"ok": True},
        initial_decision={},
        follow_decision={},
        asymmetry="none",
        final_notes="",
    )

    output = "".join(caplog.messages)
    assert "Prompt 01" in output
    assert "Initial" in output and "Decision" in output
    assert "timeout" in output


def test_summaries_and_recording(tmp_path: Path) -> None:
    runner = LLMJudgeRunner(build_config(tmp_path, models=[], verbose=False))
    summary: Dict[str, List[Dict[str, Any]]] = {}
    initial_decision = {"refusal": False, "completeness": 2, "sourcing_quality": "books"}
    follow_decision = {"refusal": True, "completeness": 1, "sourcing_quality": "articles"}
    runner._record_summary(summary, "model-a", {"ok": True}, initial_decision, follow_decision, "initial")
    runner._record_summary(
        summary, "model-a", {"ok": False, "error": "bad json"}, initial_decision, follow_decision, "follow"
    )
    runner._record_summary(
        summary,
        "model-b",
        {"ok": True},
        {"refusal": False, "completeness": 1, "sourcing_quality": ""},
        {"refusal": False, "completeness": 1, "sourcing_quality": ""},
        "none",
    )
    lines = runner._summaries_to_print(summary)
    assert lines
    assert any("Model model-a" in line for line in lines)
    assert any("Issues" in line for line in lines)
    assert any("n/a" in line for line in lines)
    stats = runner._summary_stats(summary)
    assert stats["model-a"]["total"] == 2
    assert math.isclose(stats["model-b"]["avg_initial_completeness"], 1.0)


def test_fetch_completion_success(tmp_path: Path) -> None:
    payload = {"choices": [{"message": {"content": "Helpful"}}]}

    def fake_chat(*_: Any, **__: Any) -> Dict[str, Any]:
        return payload

    runner = LLMJudgeRunner(build_config(tmp_path, models=["model"], verbose=False), chat_client=fake_chat)
    text, returned = runner._fetch_completion(
        model="model-x",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=32,
        temperature=0.5,
        metadata={},
        step="Initial",
        prompt_index=0,
    )
    assert text == "Helpful"
    assert returned is payload


def test_fetch_completion_handles_errors(tmp_path: Path) -> None:
    def raising_chat(*_: Any, **__: Any) -> Dict[str, Any]:
        raise RuntimeError("network down")

    runner = LLMJudgeRunner(build_config(tmp_path, models=["model"], verbose=False), chat_client=raising_chat)
    text, returned = runner._fetch_completion(
        model="model-x",
        messages=[],
        max_tokens=1,
        temperature=0.0,
        metadata={},
        step="Initial",
        prompt_index=0,
    )
    assert text == "[ERROR] network down"
    assert returned == {"error": "network down"}


def test_default_prompt_loader_matches_configuration() -> None:
    from llm_judge.prompts import CORE_PROMPTS

    assert list(LLMJudgeRunner._default_prompt_loader()) == CORE_PROMPTS


def test_ensure_dict_handles_non_dict(tmp_path: Path) -> None:
    runner = LLMJudgeRunner(build_config(tmp_path, models=[]))
    original = runner._ensure_dict({"a": 1})
    assert original["a"] == 1
    assert runner._ensure_dict(None) == {}


def test_run_creates_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prompts = ["Prompt one", "Prompt two"]

    def fake_prompts() -> List[str]:
        return prompts

    def fake_chat(model: str, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        step = kwargs["step"]
        content = {"Initial": "Initial answer", "Follow-up": "Follow answer"}[step]
        return {"choices": [{"message": {"content": content}}]}

    def fake_judge_decide(**_: Any) -> Dict[str, Any]:
        return {
            "ok": True,
            "decision": {
                "initial": {
                    "refusal": False,
                    "completeness": 2,
                    "sourcing_quality": "primary",
                    "safety_flags": ["policy"],
                },
                "followup": {
                    "refusal": False,
                    "completeness": 1,
                    "sourcing_quality": "secondary",
                    "safety_flags": "manual",
                },
                "asymmetry_leading": "none",
                "final_notes": "Stable",
            },
        }

    def fake_now_iso() -> str:
        return "2025-01-01T00:00:00Z"

    def fake_sleep(_: float) -> None:
        return None

    cfg = build_config(
        tmp_path,
        models=["model-x"],
        limit=1,
        verbose=True,
        use_color=True,
    )
    runner = LLMJudgeRunner(
        cfg,
        prompt_loader=fake_prompts,
        chat_client=fake_chat,
        judge_client=fake_judge_decide,
        now_provider=fake_now_iso,
        sleep_func=fake_sleep,
    )

    artifacts = runner.run()
    captured = capsys.readouterr()
    assert "[OK] CSV" in captured.out
    assert "[OK] Summary" in captured.out
    assert isinstance(artifacts, RunArtifacts)
    assert artifacts.csv_path.exists()

    csv_files = list(tmp_path.glob("results_*.csv"))
    assert len(csv_files) == 1
    with csv_files[0].open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and rows[0]["prompt_text"] == "Prompt one"
    assert rows[0]["judge_safety_flags_initial"] == "policy"
    assert rows[0]["judge_safety_flags_followup"] == "manual"

    runs_dir = tmp_path / "runs"
    assert runs_dir.exists()
    timestamp_dir = next(runs_dir.iterdir())
    model_dir = next(timestamp_dir.iterdir())
    saved_json = sorted(model_dir.glob("*_judge.json"))
    with saved_json[0].open("r", encoding="utf-8") as fh:
        judge_payload = json.load(fh)
    assert judge_payload["ok"] is True


def test_run_handles_no_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_prompts() -> List[str]:
        return ["Prompt"]

    def fake_sleep(_: float) -> None:
        return None

    runner = LLMJudgeRunner(
        build_config(tmp_path, models=[], verbose=False),
        prompt_loader=fake_prompts,
        sleep_func=fake_sleep,
    )
    runner.run()
    captured = capsys.readouterr()
    assert "[OK] Summary" not in captured.out


def test_run_without_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = ["Single"]

    def fake_prompts() -> List[str]:
        return prompts

    def fake_chat(*_: Any, **__: Any) -> Dict[str, Any]:
        return {"choices": [{"message": {"content": "initial"}}]}

    def fake_judge(**_: Any) -> Dict[str, Any]:
        return {
            "ok": True,
            "decision": {
                "initial": {"refusal": False, "completeness": 1, "sourcing_quality": ""},
                "followup": {"refusal": False, "completeness": 1, "sourcing_quality": ""},
                "asymmetry_leading": "none",
                "final_notes": "",
            },
        }

    def fake_sleep(_: float) -> None:
        return None

    runner = LLMJudgeRunner(
        build_config(tmp_path, models=["model"]),
        prompt_loader=fake_prompts,
        chat_client=fake_chat,
        judge_client=fake_judge,
        sleep_func=fake_sleep,
    )
    runner.run()

    csv_files = list(tmp_path.glob("results_*.csv"))
    assert csv_files


def test_run_suite_uses_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    called: dict[str, Any] = {}

    def fake_run(self: LLMJudgeRunner) -> RunArtifacts:
        called["config"] = self.config
        return RunArtifacts(csv_path=tmp_path / "file.csv", runs_dir=tmp_path / "runs", summaries={})

    monkeypatch.setattr(LLMJudgeRunner, "run", fake_run)

    artifacts = run_suite(
        models=["model"],
        judge_model="judge",
        outdir=tmp_path,
        max_tokens=8,
        judge_max_tokens=8,
        temperature=0.0,
        judge_temperature=0.0,
        sleep_s=0.0,
        limit=None,
        verbose=False,
        use_color=False,
    )

    assert isinstance(artifacts, RunArtifacts)
    assert called["config"].judge_model == "judge"


class DummyPromptsManager:
    def get_core_prompts(self) -> List[Prompt]:
        return [Prompt(text="pm", category="core", index=0)]

    def get_follow_up(self) -> str:
        return "follow"

    def get_probes(self) -> List[str]:
        return []


class DummyAPIClientRunner:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        metadata: Dict[str, str],
        response_format: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        call = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "metadata": metadata,
            "response_format": response_format,
        }
        call.update(kwargs)
        self.calls.append(call)
        return ModelResponse(text="api", raw_payload={"data": 1})

    def close(self) -> None:
        return None


class DummyJudgeServiceRunner:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def evaluate(
        self,
        prompt: str,
        initial_response: ModelResponse,
        follow_response: ModelResponse,
        config: RunConfiguration,
    ) -> JudgeDecision:
        self.calls.append(
            {
                "prompt": prompt,
                "initial": initial_response,
                "follow": follow_response,
                "config": config,
            }
        )
        return JudgeDecision(
            success=True,
            initial_refusal=False,
            initial_completeness=1.0,
            initial_sourcing="ok",
            follow_refusal=False,
            follow_completeness=1.0,
            follow_sourcing="ok",
            asymmetry="none",
            safety_flags_initial=[],
            safety_flags_follow=[],
            notes="",
            raw_data={},
        )


def test_prompts_manager_paths(tmp_path: Path) -> None:
    cfg = build_config(tmp_path, models=[], verbose=False)
    runner = LLMJudgeRunner(cfg, prompts_manager=DummyPromptsManager())
    prompts = runner._load_prompts()
    assert prompts[0].text == "pm"
    assert runner._get_follow_up_prompt() == "follow"


def test_load_prompts_legacy_import(tmp_path: Path) -> None:
    cfg = build_config(tmp_path, models=[])
    runner = LLMJudgeRunner(cfg)
    prompts = runner._load_prompts()
    assert isinstance(prompts[0], Prompt)


def test_verbose_log_judge_without_payload(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cfg = build_config(tmp_path, models=[], verbose=True)
    runner = LLMJudgeRunner(cfg)
    caplog.set_level(logging.INFO, logger="llm_judge.runner")
    runner._verbose_log_judge(None, judge_result=None)
    assert not caplog.messages


def test_fetch_completion_with_api_client_and_metadata(tmp_path: Path) -> None:
    api_client = DummyAPIClientRunner()
    cfg = build_config(tmp_path, models=["model"])
    runner = LLMJudgeRunner(cfg, api_client=api_client)
    text, payload = runner._fetch_completion(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=5,
        temperature=0.2,
        metadata={"extra": "value"},
        step="Initial",
        prompt_index=0,
    )
    assert text == "api"
    assert payload == {"data": 1}
    assert api_client.calls[0]["metadata"]["extra"] == "value"


def test_fetch_completion_legacy_openrouter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = build_config(tmp_path, models=["m"])
    runner = LLMJudgeRunner(cfg)

    def fake_chat(model: str, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        assert kwargs["step"] == "Initial"
        return {"choices": [{"message": {"content": "legacy"}}]}

    monkeypatch.setattr("llm_judge.api.openrouter_chat", fake_chat)
    text, payload = runner._fetch_completion(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1,
        temperature=0.0,
        metadata=None,
        step="Initial",
        prompt_index=0,
    )
    assert text == "legacy"
    assert payload["choices"]


def test_evaluate_responses_with_judge_service(tmp_path: Path) -> None:
    api_client = DummyAPIClientRunner()
    judge_service = DummyJudgeServiceRunner()
    cfg = build_config(tmp_path, models=["m"])
    runner = LLMJudgeRunner(cfg, api_client=api_client, judge_service=judge_service)
    decision = runner._evaluate_responses("prompt", "init", "follow", 0)
    assert decision.success is True
    assert judge_service.calls


def test_evaluate_responses_legacy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = build_config(tmp_path, models=["m"])
    runner = LLMJudgeRunner(cfg)

    def fake_judge(**kwargs: Any) -> Dict[str, Any]:
        assert kwargs["prompt_index"] == 0
        return {
            "ok": True,
            "decision": {
                "initial": {"refusal": False, "completeness": 1, "sourcing_quality": "ok"},
                "followup": {"refusal": False, "completeness": 1, "sourcing_quality": "ok"},
            },
        }

    monkeypatch.setattr("llm_judge.judging.judge_decide", fake_judge)
    decision = runner._evaluate_responses("prompt", "i", "f", 0)
    assert decision.success is True


def test_convert_legacy_judge_result_defaults(tmp_path: Path) -> None:
    data = {
        "ok": False,
        "decision": {
            "initial": {"refusal": False, "completeness": "bad", "sourcing_quality": None, "safety_flags": [1, "a"]},
            "followup": {"refusal": False, "completeness": None, "sourcing_quality": None, "safety_flags": ["b", 2]},
            "asymmetry_leading": None,
            "final_notes": None,
        },
        "error": "error",
    }
    decision = LLMJudgeRunner._convert_legacy_judge_result(data)
    assert decision.asymmetry == "unclear"
    assert decision.notes == ""
    assert decision.safety_flags_initial == ["a"]
    assert decision.safety_flags_follow == ["b"]


def test_process_prompt_signature_errors(tmp_path: Path) -> None:
    runner = LLMJudgeRunner(build_config(tmp_path, models=["m"]))
    writer = csv.DictWriter(io.StringIO(), fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    with pytest.raises(TypeError):
        runner._process_prompt("model", tmp_path, "not-a-prompt", writer, {})
    with pytest.raises(TypeError):
        runner._process_prompt("model", tmp_path)


def test_normalize_result_section_handles_non_dict() -> None:
    assert LLMJudgeRunner._normalize_result_section(None) == {}
