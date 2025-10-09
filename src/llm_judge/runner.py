"""Primary execution logic for the LLM judge test suite."""

from __future__ import annotations

import csv
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, cast

from colorama import Fore, Style

from .api import openrouter_chat
from .judging import judge_decide
from .prompts import CORE_PROMPTS, FOLLOW_UP
from .utils import detect_refusal, extract_text, now_iso, safe_write_json

LOGGER = logging.getLogger(__name__)

CSV_FIELDNAMES = [
    "timestamp_utc",
    "model",
    "prompt_index",
    "prompt_text",
    "response_initial",
    "response_followup",
    "heuristic_refused_initial",
    "heuristic_refused_followup",
    "judge_model",
    "judge_initial_refusal",
    "judge_initial_completeness",
    "judge_initial_sourcing",
    "judge_followup_refusal",
    "judge_followup_completeness",
    "judge_followup_sourcing",
    "judge_asymmetry_leading",
    "judge_safety_flags_initial",
    "judge_safety_flags_followup",
    "judge_final_notes",
    "raw_initial_path",
    "raw_followup_path",
    "raw_judge_path",
]


@dataclass(frozen=True)
class RunnerConfig:
    """Configuration describing a single judge run."""

    models: Sequence[str]
    judge_model: str
    outdir: Path
    max_tokens: int
    judge_max_tokens: int
    temperature: float
    judge_temperature: float
    sleep_s: float
    limit: int | None = None
    verbose: bool = False
    use_color: bool = False


@dataclass(frozen=True)
class RunArtifacts:
    """Artifacts produced by executing a judge run."""

    csv_path: Path
    runs_dir: Path
    summaries: Dict[str, List[Dict[str, Any]]]


@dataclass(frozen=True)
class RunnerEvent:
    """Lightweight payload emitted during runner progress updates."""

    type: str
    payload: Dict[str, Any]


class RunnerControl:
    """Control hooks allowing external supervisors to pause or cancel runs."""

    def wait_if_paused(self) -> None:
        """Block while the runner is paused; default no-op."""
        return None

    def should_stop(self) -> bool:
        """Return True when the run should halt early."""
        return False


class LLMJudgeRunner:
    """Object-oriented orchestration of the LLM judge workflow."""

    def __init__(
        self,
        config: RunnerConfig,
        *,
        prompt_loader: Callable[[], Iterable[str]] | None = None,
        chat_client: Callable[..., Dict[str, Any]] | None = None,
        judge_client: Callable[..., Dict[str, Any]] | None = None,
        now_provider: Callable[[], str] | None = None,
        sleep_func: Callable[[float], None] | None = None,
        logger: logging.Logger | None = None,
        progress_callback: Callable[[RunnerEvent], None] | None = None,
        control: RunnerControl | None = None,
    ) -> None:
        self.config = config
        self._prompt_loader = prompt_loader or self._default_prompt_loader
        self._chat_client = chat_client or openrouter_chat
        self._judge_client = judge_client or judge_decide
        self._now = now_provider or now_iso
        self._sleep = sleep_func or time.sleep
        self._logger = logger or LOGGER
        self._progress_callback = progress_callback
        self._control = control or RunnerControl()

    # --------------------------------------------------------------------- #
    # Configuration helpers

    @staticmethod
    def _default_prompt_loader() -> Iterable[str]:
        """Yield the configured prompt collection."""
        return CORE_PROMPTS

    # --------------------------------------------------------------------- #
    # Logging helpers

    @staticmethod
    def _snippet(text: str, limit: int = 220) -> str:
        """Create a compact one-line snippet of text for verbose output."""
        if not text:
            return "(empty)"
        collapsed = " ".join(text.strip().split())
        if len(collapsed) <= limit:
            return collapsed
        return collapsed[: limit - 1] + "…"

    def _color(self, text: str, code: str) -> str:
        if not self.config.use_color:
            return text
        return f"{code}{text}{Style.RESET_ALL}"

    def _verbose_log_prompt(self, idx: int, prompt: str) -> None:
        if not self.config.verbose:
            return
        self._logger.info(
            "%s",
            self._color(
                f"[Prompt {idx:02d}] {self._snippet(prompt, limit=180)}",
                Fore.MAGENTA + Style.BRIGHT,
            ),
        )

    def _verbose_log_response(self, label: str, text: str) -> None:
        if not self.config.verbose:
            return
        color = {
            "Initial": Fore.GREEN,
            "Follow-up": Fore.BLUE,
            "Judge": Fore.YELLOW,
        }.get(label, Fore.WHITE)
        self._logger.info("%s", self._color(f"  {label} ▶ {self._snippet(text)}", color))

    def _verbose_log_judge(
        self,
        judge_result: Dict[str, Any],
        initial_decision: Dict[str, Any],
        follow_decision: Dict[str, Any],
        asymmetry: str,
        final_notes: str,
    ) -> None:
        if not self.config.verbose:
            return
        if judge_result.get("ok"):
            details = (
                "  Judge ▶ init(refusal={r0}, comp={c0}, source={s0}) "
                "follow(refusal={r1}, comp={c1}, source={s1}) asym={asym}"
            ).format(
                r0=initial_decision.get("refusal"),
                c0=initial_decision.get("completeness"),
                s0=initial_decision.get("sourcing_quality"),
                r1=follow_decision.get("refusal"),
                c1=follow_decision.get("completeness"),
                s1=follow_decision.get("sourcing_quality"),
                asym=asymmetry,
            )
            self._logger.info("%s", self._color(details, Fore.YELLOW))
            if final_notes:
                self._logger.info("%s", self._color(f"  Notes ▶ {self._snippet(final_notes)}", Fore.LIGHTYELLOW_EX))
        else:
            self._logger.info(
                "%s",
                self._color(f"  Judge ▶ error={judge_result.get('error')}", Fore.RED + Style.BRIGHT),
            )

    # --------------------------------------------------------------------- #
    # Progress hooks

    def _emit(self, event_type: str, **payload: Any) -> None:
        if self._progress_callback is None:
            return
        safe_payload: Dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, Path):
                safe_payload[key] = str(value)
            elif isinstance(value, Counter):
                safe_payload[key] = dict(value)
            else:
                safe_payload[key] = value
        self._progress_callback(RunnerEvent(type=event_type, payload=safe_payload))

    # --------------------------------------------------------------------- #
    # Core execution helpers

    @staticmethod
    def _ensure_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return cast(Dict[str, Any], value)
        empty: Dict[str, Any] = {}
        return empty

    def _fetch_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        metadata: Dict[str, str],
        step: str,
        prompt_index: int,
    ) -> tuple[str, Dict[str, Any]]:
        try:
            payload = self._chat_client(
                model,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                metadata=metadata,
                step=step,
                prompt_index=prompt_index,
                use_color=self.config.use_color,
            )
            text = extract_text(payload)
        except Exception as exc:  # pragma: no cover - network failures
            text = f"[ERROR] {exc}"
            payload = {"error": str(exc)}
        return text, payload

    @staticmethod
    def _record_summary(
        summary_data: Dict[str, List[Dict[str, Any]]],
        model: str,
        judge_result: Dict[str, Any],
        initial_decision: Dict[str, Any],
        follow_decision: Dict[str, Any],
        asymmetry: str,
    ) -> None:
        summary_data.setdefault(model, [])
        summary_data[model].append(
            {
                "ok": bool(judge_result.get("ok")),
                "initial_comp": initial_decision.get("completeness") if judge_result.get("ok") else None,
                "follow_comp": follow_decision.get("completeness") if judge_result.get("ok") else None,
                "initial_refusal": initial_decision.get("refusal"),
                "follow_refusal": follow_decision.get("refusal"),
                "initial_source": initial_decision.get("sourcing_quality"),
                "follow_source": follow_decision.get("sourcing_quality"),
                "asymmetry": asymmetry,
                "error": None if judge_result.get("ok") else judge_result.get("error", "unknown error"),
            }
        )

    def _summary_stats(self, summary_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}
        for model, rows in summary_data.items():
            total = len(rows)
            ok_count = sum(1 for r in rows if r["ok"])
            fail_count = total - ok_count

            init_comps = [r["initial_comp"] for r in rows if r["initial_comp"] is not None]
            follow_comps = [r["follow_comp"] for r in rows if r["follow_comp"] is not None]
            avg_init = sum(init_comps) / len(init_comps) if init_comps else 0.0
            avg_follow = sum(follow_comps) / len(follow_comps) if follow_comps else 0.0

            init_refusal_rate = sum(1 for r in rows if r["initial_refusal"]) / total if total else 0.0
            follow_refusal_rate = sum(1 for r in rows if r["follow_refusal"]) / total if total else 0.0

            init_sources = _counter([r["initial_source"] for r in rows])
            follow_sources = _counter([r["follow_source"] for r in rows])
            asymmetry_counts = _counter([r["asymmetry"] for r in rows])
            errors = _counter([r["error"] for r in rows])

            stats[model] = {
                "total": total,
                "ok": ok_count,
                "issues": fail_count,
                "avg_initial_completeness": avg_init,
                "avg_followup_completeness": avg_follow,
                "initial_refusal_rate": init_refusal_rate,
                "followup_refusal_rate": follow_refusal_rate,
                "initial_sourcing_counts": dict(init_sources),
                "followup_sourcing_counts": dict(follow_sources),
                "asymmetry_counts": dict(asymmetry_counts),
                "error_counts": dict(errors),
            }

        return stats

    def _summaries_to_print(self, summary_data: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        lines: List[str] = []
        stats = self._summary_stats(summary_data)
        for model, model_stats in stats.items():
            total = model_stats["total"]
            ok_count = model_stats["ok"]
            fail_count = model_stats["issues"]
            avg_init = model_stats["avg_initial_completeness"]
            avg_follow = model_stats["avg_followup_completeness"]
            init_refusal_rate = model_stats["initial_refusal_rate"]
            follow_refusal_rate = model_stats["followup_refusal_rate"]

            init_sources: Counter[str] = Counter(model_stats["initial_sourcing_counts"])
            follow_sources: Counter[str] = Counter(model_stats["followup_sourcing_counts"])
            asymmetry_counts: Counter[str] = Counter(model_stats["asymmetry_counts"])
            errors: Counter[str] = Counter(model_stats["error_counts"])

            header = self._color(f"Model {model}", Fore.CYAN + Style.BRIGHT)
            lines.append(f"{header} • prompts {total} (ok {ok_count}, issues {fail_count})")
            lines.append(
                "  Initial: comp {:.2f}, refusal {:d}%, sourcing {}".format(
                    avg_init,
                    int(round(init_refusal_rate * 100)),
                    _format_counter(init_sources),
                )
            )
            lines.append(
                "  Follow : comp {:.2f}, refusal {:d}%, sourcing {}".format(
                    avg_follow,
                    int(round(follow_refusal_rate * 100)),
                    _format_counter(follow_sources),
                )
            )
            lines.append(f"  Asymmetry: {_format_counter(asymmetry_counts)}")
            if errors:
                lines.append(f"  Issues: {_format_counter(errors)}")
        return lines

    def _limited_prompts(self, prompts: Sequence[str]) -> Sequence[str]:
        if self.config.limit is None:
            return prompts
        limited = prompts[: self.config.limit]
        if self.config.limit < len(prompts):
            self._logger.debug("Prompt limit applied (%d of %d prompts)", self.config.limit, len(prompts))
        return limited

    def _process_models(
        self,
        prompts: Sequence[str],
        runs_dir: Path,
        writer: csv.DictWriter[Any],
        summary_data: Dict[str, List[Dict[str, Any]]],
    ) -> bool:
        for model in self.config.models:
            if self._control.should_stop():
                return True
            if self._process_single_model(model, prompts, runs_dir, writer, summary_data):
                return True
        return False

    def _process_single_model(
        self,
        model: str,
        prompts: Sequence[str],
        runs_dir: Path,
        writer: csv.DictWriter[Any],
        summary_data: Dict[str, List[Dict[str, Any]]],
    ) -> bool:
        model_dir = runs_dir / str(model).replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        self._logger.debug("Processing model '%s' (%d prompts)", model, len(prompts))

        prompt_sequence = self._limited_prompts(prompts)
        self._emit("model_started", model=model, total_prompts=len(prompt_sequence))

        for idx, prompt in enumerate(prompt_sequence):
            if self._process_prompt(model, model_dir, idx, prompt, writer, summary_data):
                return True
        return False

    def _process_prompt(
        self,
        model: str,
        model_dir: Path,
        idx: int,
        prompt: str,
        writer: csv.DictWriter[Any],
        summary_data: Dict[str, List[Dict[str, Any]]],
    ) -> bool:
        if self._should_stop():
            return True

        metadata: Dict[str, str] = {"referer": "https://openrouter.ai", "title": "AsymmetrySuite"}
        self._verbose_log_prompt(idx, prompt)
        self._emit(
            "message",
            model=model,
            prompt_index=idx,
            step="prompt",
            role="user",
            content=prompt,
        )

        messages_initial = [{"role": "user", "content": prompt}]
        text_initial, completion_initial = self._fetch_completion(
            model,
            messages_initial,
            self.config.max_tokens,
            self.config.temperature,
            metadata,
            "Initial",
            idx,
        )
        self._verbose_log_response("Initial", text_initial)
        self._emit(
            "message",
            model=model,
            prompt_index=idx,
            step="initial",
            role="assistant",
            content=text_initial,
        )

        messages_followup = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text_initial},
            {"role": "user", "content": FOLLOW_UP},
        ]
        self._emit(
            "message",
            model=model,
            prompt_index=idx,
            step="followup_prompt",
            role="user",
            content=FOLLOW_UP,
        )

        if self._sleep_with_stop():
            return True

        text_follow, completion_follow = self._fetch_completion(
            model,
            messages_followup,
            self.config.max_tokens,
            self.config.temperature,
            metadata,
            "Follow-up",
            idx,
        )
        self._verbose_log_response("Follow-up", text_follow)
        self._emit(
            "message",
            model=model,
            prompt_index=idx,
            step="followup",
            role="assistant",
            content=text_follow,
        )

        initial_path = model_dir / f"{idx:02d}_initial.json"
        follow_path = model_dir / f"{idx:02d}_followup.json"
        safe_write_json(initial_path, completion_initial)
        safe_write_json(follow_path, completion_follow)

        heur_refusal_initial = detect_refusal(text_initial)
        heur_refusal_follow = detect_refusal(text_follow)

        if self._sleep_with_stop():
            return True

        judge_result = self._judge_client(
            judge_model=self.config.judge_model,
            prompt=prompt,
            initial_resp=text_initial,
            follow_resp=text_follow,
            max_tokens=self.config.judge_max_tokens,
            temperature=self.config.judge_temperature,
            meta=metadata,
            prompt_index=idx,
            use_color=self.config.use_color,
        )

        judge_path = model_dir / f"{idx:02d}_judge.json"
        safe_write_json(judge_path, judge_result)

        decision_dict = self._ensure_dict(judge_result.get("decision"))
        initial_decision = self._ensure_dict(decision_dict.get("initial"))
        follow_decision = self._ensure_dict(decision_dict.get("followup"))
        asymmetry_value = decision_dict.get("asymmetry_leading")
        asymmetry = asymmetry_value if isinstance(asymmetry_value, str) else "unclear"
        notes_value = decision_dict.get("final_notes")
        final_notes = notes_value if isinstance(notes_value, str) else ""

        self._verbose_log_judge(judge_result, initial_decision, follow_decision, asymmetry, final_notes)
        self._emit(
            "judge",
            model=model,
            prompt_index=idx,
            asymmetry=asymmetry,
            notes=final_notes,
            decision=decision_dict,
            ok=bool(judge_result.get("ok")),
            error=judge_result.get("error"),
        )

        writer.writerow(
            {
                "timestamp_utc": self._now(),
                "model": model,
                "prompt_index": idx,
                "prompt_text": prompt,
                "response_initial": text_initial,
                "response_followup": text_follow,
                "heuristic_refused_initial": heur_refusal_initial,
                "heuristic_refused_followup": heur_refusal_follow,
                "judge_model": self.config.judge_model,
                "judge_initial_refusal": initial_decision.get("refusal"),
                "judge_initial_completeness": initial_decision.get("completeness"),
                "judge_initial_sourcing": initial_decision.get("sourcing_quality"),
                "judge_followup_refusal": follow_decision.get("refusal"),
                "judge_followup_completeness": follow_decision.get("completeness"),
                "judge_followup_sourcing": follow_decision.get("sourcing_quality"),
                "judge_asymmetry_leading": asymmetry,
                "judge_safety_flags_initial": (
                    ",".join(initial_decision.get("safety_flags", []))
                    if isinstance(initial_decision.get("safety_flags"), list)
                    else initial_decision.get("safety_flags")
                ),
                "judge_safety_flags_followup": (
                    ",".join(follow_decision.get("safety_flags", []))
                    if isinstance(follow_decision.get("safety_flags"), list)
                    else follow_decision.get("safety_flags")
                ),
                "judge_final_notes": final_notes,
                "raw_initial_path": str(initial_path),
                "raw_followup_path": str(follow_path),
                "raw_judge_path": str(judge_path),
            }
        )

        self._record_summary(summary_data, model, judge_result, initial_decision, follow_decision, asymmetry)
        snapshot = self._summary_stats(summary_data)
        self._emit("summary", summary=snapshot, model=model, prompt_index=idx)

        if self._sleep_with_stop():
            return True
        return False

    def _should_stop(self) -> bool:
        self._control.wait_if_paused()
        return self._control.should_stop()

    def _sleep_with_stop(self) -> bool:
        self._control.wait_if_paused()
        self._sleep(self.config.sleep_s)
        return self._control.should_stop()

    def _finalize_run(
        self,
        cancelled: bool,
        csv_path: Path,
        runs_dir: Path,
        summary_data: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        summary_snapshot = self._summary_stats(summary_data)
        event_type = "run_cancelled" if cancelled else "run_completed"
        self._emit(event_type, csv_path=csv_path, runs_dir=runs_dir, summary=summary_snapshot)

        print(self._color(f"[OK] CSV: {csv_path}", Fore.GREEN + Style.BRIGHT))
        print(self._color(f"[OK] Raw JSON per-call under: {runs_dir}", Fore.GREEN + Style.BRIGHT))
        summary_lines = self._summaries_to_print(summary_data)
        if summary_lines:
            print(self._color("[OK] Summary", Fore.GREEN + Style.BRIGHT))
            for line in summary_lines:
                print(line)

    # --------------------------------------------------------------------- #
    # Public API

    def run(self) -> RunArtifacts:
        """Execute the configured suite across all models."""
        prompts = list(self._prompt_loader())
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

        runs_dir = self.config.outdir / "runs" / timestamp
        runs_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.config.outdir / f"results_{timestamp}.csv"

        self._emit(
            "run_started",
            models=list(self.config.models),
            total_prompts=len(prompts),
            limit=self.config.limit,
            timestamp=timestamp,
        )

        self._logger.debug(
            "Prepared run directory at %s; models=%s total_prompts=%d",
            runs_dir,
            list(self.config.models),
            len(prompts),
        )

        summary_data: Dict[str, List[Dict[str, Any]]] = {}
        cancelled = False

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer: csv.DictWriter[Any] = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            cancelled = self._process_models(prompts, runs_dir, writer, summary_data)

        self._finalize_run(cancelled, csv_path, runs_dir, summary_data)
        return RunArtifacts(csv_path=csv_path, runs_dir=runs_dir, summaries=summary_data)


def _counter(items: Iterable[Any]) -> Counter[Any]:
    """Count occurrences of truthy string labels."""
    return Counter(item for item in items if item)


def _format_counter(counter: Counter[Any]) -> str:
    if not counter:
        return "n/a"
    return ", ".join(f"{key}×{value}" for key, value in sorted(counter.items(), key=lambda kv: kv[1], reverse=True))


def run_suite(
    *,
    models: Sequence[str],
    judge_model: str,
    outdir: Path,
    max_tokens: int,
    judge_max_tokens: int,
    temperature: float,
    judge_temperature: float,
    sleep_s: float,
    limit: int | None = None,
    verbose: bool = False,
    use_color: bool = False,
) -> RunArtifacts:
    """Backward-compatible functional entry point."""
    config = RunnerConfig(
        models=models,
        judge_model=judge_model,
        outdir=outdir,
        max_tokens=max_tokens,
        judge_max_tokens=judge_max_tokens,
        temperature=temperature,
        judge_temperature=judge_temperature,
        sleep_s=sleep_s,
        limit=limit,
        verbose=verbose,
        use_color=use_color,
    )
    runner = LLMJudgeRunner(config)
    return runner.run()


__all__ = ["LLMJudgeRunner", "RunnerConfig", "RunArtifacts", "RunnerEvent", "RunnerControl", "run_suite"]
