"""Tests covering the orchestration logic in ``runner.py``."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pytest

import llm_judge.runner as runner


def test_snippet_behaviour() -> None:
    assert runner._snippet("", limit=10) == "(empty)"
    text = "word " * 100
    truncated = runner._snippet(text, limit=20)
    assert truncated.endswith("…")
    assert len(truncated) == 20


def test_color_toggle() -> None:
    plain = runner._color("hello", "X", use_color=False)
    colored = runner._color("hello", "X", use_color=True)
    assert plain == "hello"
    assert colored.startswith("X") and colored.endswith("hello\x1b[0m")


def test_verbose_logging_helpers(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="llm_judge.runner")
    runner._verbose_log_prompt(verbose=True, idx=1, prompt="Example prompt", use_color=False)
    runner._verbose_log_response(verbose=True, label="Initial", text="Answer", use_color=False)
    runner._verbose_log_response(verbose=True, label="Judge", text="Decision", use_color=True)
    runner._verbose_log_judge(
        verbose=True,
        judge_result={"ok": True},
        initial_decision={"refusal": False, "completeness": 3, "sourcing_quality": "high"},
        follow_decision={"refusal": True, "completeness": 1, "sourcing_quality": "low"},
        asymmetry="initial",
        final_notes="Detailed notes",
        use_color=False,
    )
    runner._verbose_log_judge(
        verbose=True,
        judge_result={"ok": False, "error": "timeout"},
        initial_decision={},
        follow_decision={},
        asymmetry="",
        final_notes="",
        use_color=True,
    )
    runner._verbose_log_prompt(verbose=False, idx=2, prompt="Quiet", use_color=False)
    runner._verbose_log_response(verbose=False, label="Initial", text="Ignored", use_color=False)
    runner._verbose_log_judge(
        verbose=False,
        judge_result={},
        initial_decision={},
        follow_decision={},
        asymmetry="",
        final_notes="",
        use_color=False,
    )
    runner._verbose_log_judge(
        verbose=True,
        judge_result={"ok": True},
        initial_decision={},
        follow_decision={},
        asymmetry="none",
        final_notes="",
        use_color=False,
    )
    output = "".join(caplog.messages)
    assert "Prompt 01" in output
    assert "Initial" in output and "Decision" in output
    assert "timeout" in output


def test_summaries_and_recording() -> None:
    summary: Dict[str, List[Dict[str, Any]]] = {}
    initial_decision = {"refusal": False, "completeness": 2, "sourcing_quality": "books"}
    follow_decision = {"refusal": True, "completeness": 1, "sourcing_quality": "articles"}
    runner._record_summary(summary, "model-a", {"ok": True}, initial_decision, follow_decision, "initial")
    runner._record_summary(summary, "model-a", {"ok": False, "error": "bad json"}, initial_decision, follow_decision, "follow")
    runner._record_summary(
        summary,
        "model-b",
        {"ok": True},
        {"refusal": False, "completeness": 1, "sourcing_quality": ""},
        {"refusal": False, "completeness": 1, "sourcing_quality": ""},
        "none",
    )
    lines = runner._summaries_to_print(summary, use_color=False)
    assert lines
    assert any("Model model-a" in line for line in lines)
    assert any("Issues" in line for line in lines)
    assert any("n/a" in line for line in lines)


def test_fetch_completion_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"choices": [{"message": {"content": "Helpful"}}]}

    def fake_chat(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return payload

    monkeypatch.setattr(runner, "openrouter_chat", fake_chat)
    text, returned = runner._fetch_completion(
        model="model-x",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=32,
        temperature=0.5,
        metadata={},
        step="Initial",
        prompt_index=0,
        use_color=False,
    )
    assert text == "Helpful"
    assert returned is payload


def test_fetch_completion_handles_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def raising_chat(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("network down")

    monkeypatch.setattr(runner, "openrouter_chat", raising_chat)
    text, returned = runner._fetch_completion(
        model="model-x",
        messages=[],
        max_tokens=1,
        temperature=0.0,
        metadata={},
        step="Initial",
        prompt_index=0,
        use_color=False,
    )
    assert text == "[ERROR] network down"
    assert returned == {"error": "network down"}


def test_iter_prompts_matches_configuration() -> None:
    from llm_judge.prompts import CORE_PROMPTS, PROBES

    assert runner._iter_prompts(include_probes=False) == CORE_PROMPTS
    assert runner._iter_prompts(include_probes=True) == CORE_PROMPTS + PROBES


def test_run_suite_creates_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    prompts = ["Prompt one", "Prompt two"]
    monkeypatch.setattr(runner, "_iter_prompts", lambda include_probes: prompts)

    def fake_chat(model: str, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        step = kwargs.get("step")
        content = {
            "Initial": "Initial answer",
            "Follow-up": "Follow answer",
        }[step]
        return {"choices": [{"message": {"content": content}}]}

    def fake_judge_decide(**kwargs: Any) -> Dict[str, Any]:
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

    monkeypatch.setattr(runner, "openrouter_chat", fake_chat)
    monkeypatch.setattr(runner, "judge_decide", fake_judge_decide)
    monkeypatch.setattr(runner, "now_iso", lambda: "2025-01-01T00:00:00Z")
    monkeypatch.setattr(runner.time, "sleep", lambda *args: None)

    runner.run_suite(
        models=["model-x"],
        judge_model="judge-y",
        include_probes=True,
        outdir=tmp_path,
        max_tokens=16,
        judge_max_tokens=32,
        temperature=0.2,
        judge_temperature=0.4,
        sleep_s=0.0,
        limit=1,
        verbose=True,
        use_color=True,
    )

    captured = capsys.readouterr()
    assert "[OK] CSV" in captured.out
    assert "[OK] Summary" in captured.out

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


def test_run_suite_with_no_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(runner, "_iter_prompts", lambda include_probes: ["Prompt"])
    monkeypatch.setattr(runner.time, "sleep", lambda *args: None)

    runner.run_suite(
        models=[],
        judge_model="judge",
        include_probes=False,
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

    captured = capsys.readouterr()
    assert "[OK] Summary" not in captured.out


def test_run_suite_without_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = ["Single"]
    monkeypatch.setattr(runner, "_iter_prompts", lambda include_probes: prompts)

    def fake_chat(model: str, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        step = kwargs.get("step")
        content = "initial" if step == "Initial" else "follow"
        return {"choices": [{"message": {"content": content}}]}

    def fake_judge(**kwargs: Any) -> Dict[str, Any]:
        return {
            "ok": True,
            "decision": {
                "initial": {"refusal": False, "completeness": 1, "sourcing_quality": ""},
                "followup": {"refusal": False, "completeness": 1, "sourcing_quality": ""},
                "asymmetry_leading": "none",
                "final_notes": "",
            },
        }

    monkeypatch.setattr(runner, "openrouter_chat", fake_chat)
    monkeypatch.setattr(runner, "judge_decide", fake_judge)
    monkeypatch.setattr(runner, "now_iso", lambda: "2025-01-01T00:00:00Z")
    monkeypatch.setattr(runner.time, "sleep", lambda *args: None)

    runner.run_suite(
        models=["model"],
        judge_model="judge",
        include_probes=False,
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

    csv_files = list(tmp_path.glob("results_*.csv"))
    assert csv_files
