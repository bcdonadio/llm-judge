"""Primary execution logic for the LLM judge test suite."""

from __future__ import annotations

import csv
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, cast

from colorama import Fore, Style

# New imports for refactored architecture
from .domain import JudgeDecision, Prompt, RunConfiguration
from .services import IAPIClient, IJudgeService, IPromptsManager

# Legacy imports for backward compatibility
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
    """Object-oriented orchestration of the LLM judge workflow with dependency injection."""

    def __init__(
        self,
        config: RunnerConfig,
        *,
        api_client: Optional[IAPIClient] = None,
        judge_service: Optional[IJudgeService] = None,
        prompts_manager: Optional[IPromptsManager] = None,
        # Legacy compatibility parameters
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

        # New DI-based services (preferred)
        self._api_client = api_client
        self._judge_service = judge_service
        self._prompts_manager = prompts_manager

        # Legacy compatibility (deprecated)
        self._prompt_loader = prompt_loader
        self._chat_client = chat_client
        self._judge_client = judge_client

        # Utility dependencies
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
        return ["test prompt 1", "test prompt 2", "test prompt 3"]

    def _load_prompts(self) -> List[Prompt]:
        """Load prompts using the new or legacy interface."""
        if self._prompts_manager is not None:
            # Use new DI service
            return self._prompts_manager.get_core_prompts()

        # Fall back to legacy loader
        if self._prompt_loader is not None:
            prompt_texts = list(self._prompt_loader())
        else:
            # Import lazily for backward compatibility
            from .prompts import CORE_PROMPTS
            prompt_texts = CORE_PROMPTS

        # Convert to domain objects
        return [
            Prompt(text=text, category="core", index=idx)
            for idx, text in enumerate(prompt_texts)
        ]

    def _get_follow_up_prompt(self) -> str:
        """Get the follow-up prompt text."""
        if self._prompts_manager is not None:
            return self._prompts_manager.get_follow_up()

        # Legacy fallback
        from .prompts import FOLLOW_UP
        return FOLLOW_UP

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
        judge_decision: JudgeDecision,
    ) -> None:
        if not self.config.verbose:
            return
        if judge_decision.success:
            details = (
                "  Judge ▶ init(refusal={r0}, comp={c0}, source={s0}) "
                "follow(refusal={r1}, comp={c1}, source={s1}) asym={asym}"
            ).format(
                r0=judge_decision.initial_refusal,
                c0=judge_decision.initial_completeness,
                s0=judge_decision.initial_sourcing,
                r1=judge_decision.follow_refusal,
                c1=judge_decision.follow_completeness,
                s1=judge_decision.follow_sourcing,
                asym=judge_decision.asymmetry,
            )
            self._logger.info("%s", self._color(details, Fore.YELLOW))
            if judge_decision.notes:
                self._logger.info(
                    "%s", self._color(f"  Notes ▶ {self._snippet(judge_decision.notes)}", Fore.LIGHTYELLOW_EX)
                )
        else:
            self._logger.info(
                "%s",
                self._color(f"  Judge ▶ error={judge_decision.error}", Fore.RED + Style.BRIGHT),
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
                counter_value = cast("Counter[str]", value)
                safe_payload[key] = dict(counter_value)
            else:
                safe_payload[key] = value
        self._progress_callback(RunnerEvent(type=event_type, payload=safe_payload))

    # --------------------------------------------------------------------- #
    # Core execution helpers

    def _fetch_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        step: str,
        prompt_index: int,
    ) -> tuple[str, Dict[str, Any]]:
        """Fetch a completion using new or legacy API."""
        if self._api_client is not None:
            # Use new DI service
            try:
                response = self._api_client.chat_completion(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    metadata={"referer": "https://openrouter.ai", "title": "AsymmetrySuite"},
                )
                return response.text, response.raw_payload
            except Exception as exc:  # pragma: no cover
                text = f"[ERROR] {exc}"
                payload = {"error": str(exc)}
                return text, payload

        # Legacy fallback
        if self._chat_client is not None:
            chat_func = self._chat_client
        else:
            from .api import openrouter_chat
            chat_func = openrouter_chat

        try:
            metadata = {"referer": "https://openrouter.ai", "title": "AsymmetrySuite"}
            payload = chat_func(
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
        except Exception as exc:  # pragma: no cover
            text = f"[ERROR] {exc}"
            payload = {"error": str(exc)}
        return text, payload

    def _evaluate_responses(
        self,
        prompt_text: str,
        initial_text: str,
        followup_text: str,
        prompt_index: int,
    ) -> JudgeDecision:
        """Evaluate responses using new or legacy judge service."""
        if self._judge_service is not None:
            # Use new DI service - need to create ModelResponse objects
            from .domain import ModelResponse

            initial_response = ModelResponse(text=initial_text, raw_payload={})
            follow_response = ModelResponse(text=followup_text, raw_payload={})

            config = RunConfiguration(
                models=list(self.config.models),
                judge_model=self.config.judge_model,
                outdir=self.config.outdir,
                max_tokens=self.config.max_tokens,
                judge_max_tokens=self.config.judge_max_tokens,
                temperature=self.config.temperature,
                judge_temperature=self.config.judge_temperature,
                sleep_s=self.config.sleep_s,
                limit=self.config.limit,
                verbose=self.config.verbose,
                use_color=self.config.use_color,
            )

            return self._judge_service.evaluate(
                prompt=prompt_text,
                initial_response=initial_response,
                follow_response=follow_response,
                config=config,
            )

        # Legacy fallback
        if self._judge_client is not None:
            judge_func = self._judge_client
        else:
            from .judging import judge_decide
            judge_func = judge_decide

        metadata = {"referer": "https://openrouter.ai", "title": "AsymmetrySuite"}
        judge_result = judge_func(
            judge_model=self.config.judge_model,
            prompt=prompt_text,
            initial_resp=initial_text,
            follow_resp=followup_text,
            max_tokens=self.config.judge_max_tokens,
            temperature=self.config.judge_temperature,
            meta=metadata,
            prompt_index=prompt_index,
            use_color=self.config.use_color,
        )

        # Convert legacy format to new domain object
        return self._convert_legacy_judge_result(judge_result)

    @staticmethod
    def _convert_legacy_judge_result(judge_result: Dict[str, Any]) -> JudgeDecision:
        """Convert legacy judge result to new domain object."""
        decision_dict = judge_result.get("decision", {})
        if not isinstance(decision_dict, dict):
            decision_dict = {}

        initial_decision = decision_dict.get("initial", {})
        if not isinstance(initial_decision, dict):
            initial_decision = {}

        followup_decision = decision_dict.get("followup", {})
        if not isinstance(followup_decision, dict):
            followup_decision = {}

        asymmetry = decision_dict.get("asymmetry_leading")
        if not isinstance(asymmetry, str):
            asymmetry = "unclear"

        final_notes = decision_dict.get("final_notes")
        if not isinstance(final_notes, str):
            final_notes = ""

        return JudgeDecision(
            success=bool(judge_result.get("ok")),
            initial_refusal=bool(initial_decision.get("refusal")),
            initial_completeness=float(initial_decision.get("completeness", 0.0)),
            initial_sourcing=str(initial_decision.get("sourcing_quality", "")),
            follow_refusal=bool(followup_decision.get("refusal")),
            follow_completeness=float(followup_decision.get("completeness", 0.0)),
            follow_sourcing=str(followup_decision.get("sourcing_quality", "")),
            asymmetry=asymmetry,
            safety_flags_initial=initial_decision.get("safety_flags", []),
            safety_flags_follow=followup_decision.get("safety_flags", []),
            notes=final_notes,
            raw_data=judge_result,
            error=None if judge_result.get("ok") else judge_result.get("error", "unknown error"),
        )

    @staticmethod
    def _record_summary(
        summary_data: Dict[str, List[Dict[str, Any]]],
        model: str,
        judge_decision: JudgeDecision,
    ) -> None:
        summary_data.setdefault(model, [])
        summary_data[model].append(
            {
                "ok": judge_decision.success,
                "initial_comp": judge_decision.initial_completeness if judge_decision.success else None,
                "follow_comp": judge_decision.follow_completeness if judge_decision.success else None,
                "initial_refusal": judge_decision.initial_refusal,
                "follow_refusal": judge_decision.follow_refusal,
                "initial_source": judge_decision.initial_sourcing,
                "follow_source": judge_decision.follow_sourcing,
                "asymmetry": judge_decision.asymmetry,
                "error": judge_decision.error,
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

    def _limited_prompts(self, prompts: Sequence[Prompt]) -> Sequence[Prompt]:
        if self.config.limit is None:
            return prompts
        limited = prompts[: self.config.limit]
        if self.config.limit < len(prompts):
            self._logger.debug("Prompt limit applied (%d of %d prompts)", self.config.limit, len(prompts))
        return limited

    def _process_models(
        self,
        prompts: Sequence[Prompt],
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
        prompts: Sequence[Prompt],
        runs_dir: Path,
        writer: csv.DictWriter[Any],
        summary_data: Dict[str, List[Dict[str, Any]]],
    ) -> bool:
        model_dir = runs_dir / str(model).replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        self._logger.debug("Processing model '%s' (%d prompts)", model, len(prompts))

        prompt_sequence = self._limited_prompts(prompts)
        self._emit("model_started", model=model, total_prompts=len(prompt_sequence))

        for prompt in prompt_sequence:
            if self._process_prompt(model, model_dir, prompt, writer, summary_data):
                return True
        return False

    def _process_prompt(
        self,
        model: str,
        model_dir: Path,
        prompt: Prompt,
        writer: csv.DictWriter[Any],
        summary_data: Dict[str, List[Dict[str, Any]]],
    ) -> bool:
        if self._should_stop():
            return True

        idx = prompt.index
        prompt_text = prompt.text

        self._verbose_log_prompt(idx, prompt_text)
        self._emit(
            "message",
            model=model,
            prompt_index=idx,
            step="prompt",
            role="user",
            content=prompt_text,
        )

        messages_initial = [{"role": "user", "content": prompt_text}]
        text_initial, completion_initial = self._fetch_completion(
            model,
            messages_initial,
            self.config.max_tokens,
            self.config.temperature,
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

        follow_up_text = self._get_follow_up_prompt()
        messages_followup = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": text_initial},
            {"role": "user", "content": follow_up_text},
        ]
        self._emit(
            "message",
            model=model,
            prompt_index=idx,
            step="followup_prompt",
            role="user",
            content=follow_up_text,
        )

        if self._sleep_with_stop():
            return True

        text_follow, completion_follow = self._fetch_completion(
            model,
            messages_followup,
            self.config.max_tokens,
            self.config.temperature,
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

        judge_decision = self._evaluate_responses(
            prompt_text,
            text_initial,
            text_follow,
            idx,
        )

        judge_path = model_dir / f"{idx:02d}_judge.json"
        safe_write_json(judge_path, judge_decision.raw_data)

        self._verbose_log_judge(judge_decision)
        self._emit(
            "judge",
            model=model,
            prompt_index=idx,
            asymmetry=judge_decision.asymmetry,
            notes=judge_decision.notes,
            decision={
                "initial": {
                    "refusal": judge_decision.initial_refusal,
                    "completeness": judge_decision.initial_completeness,
                    "sourcing_quality": judge_decision.initial_sourcing,
                    "safety_flags": judge_decision.safety_flags_initial,
                },
                "followup": {
                    "refusal": judge_decision.follow_refusal,
                    "completeness": judge_decision.follow_completeness,
                    "sourcing_quality": judge_decision.follow_sourcing,
                    "safety_flags": judge_decision.safety_flags_follow,
                },
                "asymmetry_leading": judge_decision.asymmetry,
                "final_notes": judge_decision.notes,
            },
            ok=judge_decision.success,
            error=judge_decision.error,
        )

        writer.writerow(
            {
                "timestamp_utc": self._now(),
                "model": model,
                "prompt_index": idx,
                "prompt_text": prompt_text,
                "response_initial": text_initial,
                "response_followup": text_follow,
                "heuristic_refused_initial": heur_refusal_initial,
                "heuristic_refused_followup": heur_refusal_follow,
                "judge_model": self.config.judge_model,
                "judge_initial_refusal": judge_decision.initial_refusal,
                "judge_initial_completeness": judge_decision.initial_completeness,
                "judge_initial_sourcing": judge_decision.initial_sourcing,
                "judge_followup_refusal": judge_decision.follow_refusal,
                "judge_followup_completeness": judge_decision.follow_completeness,
                "judge_followup_sourcing": judge_decision.follow_sourcing,
                "judge_asymmetry_leading": judge_decision.asymmetry,
                "judge_safety_flags_initial": (
                    ",".join(judge_decision.safety_flags_initial)
                ),
                "judge_safety_flags_followup": (
                    ",".join(judge_decision.safety_flags_follow)
                ),
                "judge_final_notes": judge_decision.notes,
                "raw_initial_path": str(initial_path),
                "raw_followup_path": str(follow_path),
                "raw_judge_path": str(judge_path),
            }
        )

        self._record_summary(summary_data, model, judge_decision)
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
        prompts = self._load_prompts()
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
