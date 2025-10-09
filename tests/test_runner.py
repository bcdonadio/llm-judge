# pyright: reportPrivateUsage=false

"""Tests covering the LLMJudgeRunner orchestration class."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pytest

from llm_judge.runner import LLMJudgeRunner, RunArtifacts, RunnerConfig, run_suite


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
    assert pytest.approx(stats["model-b"]["avg_initial_completeness"]) == 1.0


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
