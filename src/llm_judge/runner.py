"""Primary execution logic for the LLM judge test suite."""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, cast

from colorama import Fore, Style

from .api import openrouter_chat
from .judging import judge_decide
from .prompts import CORE_PROMPTS, FOLLOW_UP, PROBES
from .utils import detect_refusal, extract_text, now_iso, safe_write_json

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunnerConfig:
    """Configuration describing a single judge run."""

    models: Sequence[str]
    judge_model: str
    include_probes: bool
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


class LLMJudgeRunner:
    """Object-oriented orchestration of the LLM judge workflow."""

    def __init__(
        self,
        config: RunnerConfig,
        *,
        prompt_loader: Callable[[bool], Iterable[str]] | None = None,
        chat_client: Callable[..., Dict[str, Any]] | None = None,
        judge_client: Callable[..., Dict[str, Any]] | None = None,
        now_provider: Callable[[], str] | None = None,
        sleep_func: Callable[[float], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self._prompt_loader = prompt_loader or self._default_prompt_loader
        self._chat_client = chat_client or openrouter_chat
        self._judge_client = judge_client or judge_decide
        self._now = now_provider or now_iso
        self._sleep = sleep_func or time.sleep
        self._logger = logger or LOGGER

    # --------------------------------------------------------------------- #
    # Configuration helpers

    @staticmethod
    def _default_prompt_loader(include_probes: bool) -> Iterable[str]:
        """Yield the configured prompt collection."""
        if include_probes:
            return CORE_PROMPTS + PROBES
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

    def _summaries_to_print(self, summary_data: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        lines: List[str] = []
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
            asym = _counter([r["asymmetry"] for r in rows])
            errors = _counter([r["error"] for r in rows])

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
            lines.append(f"  Asymmetry: {_format_counter(asym)}")
            if errors:
                lines.append(f"  Issues: {_format_counter(errors)}")
        return lines

    # --------------------------------------------------------------------- #
    # Public API

    def run(self) -> RunArtifacts:
        """Execute the configured suite across all models."""
        prompts = list(self._prompt_loader(self.config.include_probes))
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

        runs_dir = self.config.outdir / "runs" / timestamp
        runs_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.config.outdir / f"results_{timestamp}.csv"

        self._logger.debug(
            "Prepared run directory at %s; models=%s total_prompts=%d include_probes=%s",
            runs_dir,
            list(self.config.models),
            len(prompts),
            self.config.include_probes,
        )

        fieldnames = [
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

        summary_data: Dict[str, List[Dict[str, Any]]] = {}

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            for model in self.config.models:
                model_dir = runs_dir / str(model).replace("/", "_")
                model_dir.mkdir(parents=True, exist_ok=True)
                self._logger.debug("Processing model '%s' (%d prompts)", model, len(prompts))

                prompt_sequence = prompts if self.config.limit is None else prompts[: self.config.limit]
                if self.config.limit is not None and self.config.limit < len(prompts):
                    self._logger.debug("Prompt limit applied (%d of %d prompts)", self.config.limit, len(prompts))

                for idx, prompt in enumerate(prompt_sequence):
                    metadata: Dict[str, str] = {"referer": "https://openrouter.ai", "title": "AsymmetrySuite"}
                    self._verbose_log_prompt(idx, prompt)

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

                    messages_followup = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": text_initial},
                        {"role": "user", "content": FOLLOW_UP},
                    ]
                    self._sleep(self.config.sleep_s)
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

                    initial_path = model_dir / f"{idx:02d}_initial.json"
                    follow_path = model_dir / f"{idx:02d}_followup.json"
                    safe_write_json(initial_path, completion_initial)
                    safe_write_json(follow_path, completion_follow)

                    heur_refusal_initial = detect_refusal(text_initial)
                    heur_refusal_follow = detect_refusal(text_follow)

                    self._sleep(self.config.sleep_s)
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
                    initial_candidate = decision_dict.get("initial")
                    follow_candidate = decision_dict.get("followup")
                    initial_decision = self._ensure_dict(initial_candidate)
                    follow_decision = self._ensure_dict(follow_candidate)
                    asymmetry_value = decision_dict.get("asymmetry_leading")
                    asymmetry = asymmetry_value if isinstance(asymmetry_value, str) else "unclear"
                    notes_value = decision_dict.get("final_notes")
                    final_notes = notes_value if isinstance(notes_value, str) else ""

                    self._verbose_log_judge(judge_result, initial_decision, follow_decision, asymmetry, final_notes)

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

                    self._record_summary(
                        summary_data, model, judge_result, initial_decision, follow_decision, asymmetry
                    )

                    self._sleep(self.config.sleep_s)

        print(self._color(f"[OK] CSV: {csv_path}", Fore.GREEN + Style.BRIGHT))
        print(self._color(f"[OK] Raw JSON per-call under: {runs_dir}", Fore.GREEN + Style.BRIGHT))
        summary_lines = self._summaries_to_print(summary_data)
        if summary_lines:
            print(self._color("[OK] Summary", Fore.GREEN + Style.BRIGHT))
            for line in summary_lines:
                print(line)

        return RunArtifacts(csv_path=csv_path, runs_dir=runs_dir, summaries=summary_data)


def _counter(items: Iterable[Any]) -> Dict[Any, int]:
    """Count occurrences of truthy string labels."""
    result: Dict[Any, int] = {}
    for item in items:
        if not item:
            continue
        result[item] = result.get(item, 0) + 1
    return result


def _format_counter(counter: Dict[Any, int]) -> str:
    if not counter:
        return "n/a"
    return ", ".join(f"{key}×{value}" for key, value in sorted(counter.items(), key=lambda kv: kv[1], reverse=True))


def run_suite(
    *,
    models: Sequence[str],
    judge_model: str,
    include_probes: bool,
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
        include_probes=include_probes,
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


__all__ = ["LLMJudgeRunner", "RunnerConfig", "RunArtifacts", "run_suite"]
