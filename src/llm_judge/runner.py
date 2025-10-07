"""Primary execution loop for the LLM judge test suite."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from collections import Counter
from typing import Any, Dict, Iterable, List, cast

from .api import openrouter_chat
from .judging import judge_decide
from .prompts import CORE_PROMPTS, PROBES, FOLLOW_UP
from .utils import detect_refusal, now_iso, safe_write_json, extract_text
from colorama import Fore, Style

LOGGER = logging.getLogger(__name__)


def _snippet(text: str, limit: int = 220) -> str:
    """Create a compact one-line snippet of text for verbose output."""
    if not text:
        return "(empty)"
    collapsed = " ".join(text.strip().split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


def _color(text: str, code: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{code}{text}{Style.RESET_ALL}"


def _ensure_dict(value: Any) -> Dict[str, Any]:
    return cast(Dict[str, Any], value) if isinstance(value, dict) else {}


def _verbose_log_prompt(verbose: bool, idx: int, prompt: str, use_color: bool) -> None:
    if verbose:
        LOGGER.info(
            "%s",
            _color(f"[Prompt {idx:02d}] {_snippet(prompt, limit=180)}", Fore.MAGENTA + Style.BRIGHT, use_color),
        )


def _verbose_log_response(verbose: bool, label: str, text: str, use_color: bool) -> None:
    if verbose:
        color = {
            "Initial": Fore.GREEN,
            "Follow-up": Fore.BLUE,
            "Judge": Fore.YELLOW,
        }.get(label, Fore.WHITE)
        LOGGER.info(
            "%s",
            _color(f"  {label} ▶ {_snippet(text)}", color, use_color),
        )


def _verbose_log_judge(
    verbose: bool,
    judge_result: Dict[str, Any],
    initial_decision: Dict[str, Any],
    follow_decision: Dict[str, Any],
    asymmetry: str,
    final_notes: str,
    use_color: bool,
) -> None:
    if not verbose:
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
        LOGGER.info("%s", _color(details, Fore.YELLOW, use_color))
        if final_notes:
            LOGGER.info("%s", _color(f"  Notes ▶ {_snippet(final_notes)}", Fore.LIGHTYELLOW_EX, use_color))
    else:
        LOGGER.info(
            "%s",
            _color(f"  Judge ▶ error={judge_result.get('error')}", Fore.RED + Style.BRIGHT, use_color),
        )


def _summaries_to_print(summary_data: Dict[str, List[Dict[str, Any]]], use_color: bool) -> List[str]:
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

        init_sources: Counter[str] = Counter(
            source for source in (r["initial_source"] for r in rows) if isinstance(source, str) and source
        )
        follow_sources: Counter[str] = Counter(
            source for source in (r["follow_source"] for r in rows) if isinstance(source, str) and source
        )
        asym: Counter[str] = Counter(
            label for label in (r["asymmetry"] for r in rows) if isinstance(label, str) and label
        )
        errors: Counter[str] = Counter(err for err in (r["error"] for r in rows) if isinstance(err, str) and err)

        def fmt_counter(counter: Counter[str]) -> str:
            if not counter:
                return "n/a"
            return ", ".join(f"{k}×{v}" for k, v in counter.most_common())

        header = _color(f"Model {model}", Fore.CYAN + Style.BRIGHT, use_color)
        lines.append(f"{header} • prompts {total} (ok {ok_count}, issues {fail_count})")
        lines.append(
            "  Initial: comp {:.2f}, refusal {:d}%, sourcing {}".format(
                avg_init,
                int(round(init_refusal_rate * 100)),
                fmt_counter(init_sources),
            )
        )
        lines.append(
            "  Follow : comp {:.2f}, refusal {:d}%, sourcing {}".format(
                avg_follow,
                int(round(follow_refusal_rate * 100)),
                fmt_counter(follow_sources),
            )
        )
        lines.append(f"  Asymmetry: {fmt_counter(asym)}")
        if errors:
            lines.append(f"  Issues: {fmt_counter(errors)}")
    return lines


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


def _fetch_completion(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    metadata: Dict[str, str],
    step: str,
    prompt_index: int,
    use_color: bool,
) -> tuple[str, Dict[str, Any]]:
    try:
        payload = openrouter_chat(
            model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            metadata=metadata,
            step=step,
            prompt_index=prompt_index,
            use_color=use_color,
        )
        text = extract_text(payload)
    except Exception as exc:  # pragma: no cover - network failures
        text = f"[ERROR] {exc}"
        payload = {"error": str(exc)}
    return text, payload


def _iter_prompts(include_probes: bool) -> Iterable[str]:
    """Yield the core prompts plus probes if requested."""
    if include_probes:
        return CORE_PROMPTS + PROBES
    return CORE_PROMPTS


def run_suite(
    models: List[str],
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
) -> None:
    """Execute the full suite across the requested models, persisting CSV and JSON artifacts."""
    prompts = list(_iter_prompts(include_probes))
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    runs_dir = outdir / "runs" / timestamp
    runs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"results_{timestamp}.csv"
    LOGGER.debug(
        "Prepared run directory at %s; models=%s total_prompts=%d include_probes=%s",
        runs_dir,
        models,
        len(prompts),
        include_probes,
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

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        summary_data: Dict[str, List[Dict[str, Any]]] = {}

        for model in models:
            model_dir = runs_dir / model.replace("/", "_")
            model_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.debug("Processing model '%s' (%d prompts)", model, len(prompts))

            prompt_sequence = prompts if limit is None else prompts[:limit]
            if limit is not None and limit < len(prompts):
                LOGGER.debug("Prompt limit applied (%d of %d prompts)", limit, len(prompts))

            for idx, prompt in enumerate(prompt_sequence):
                metadata: Dict[str, str] = {"referer": "https://openrouter.ai", "title": "AsymmetrySuite"}
                _verbose_log_prompt(verbose, idx, prompt, use_color)

                # Initial call.
                messages_initial = [{"role": "user", "content": prompt}]
                text_initial, completion_initial = _fetch_completion(
                    model,
                    messages_initial,
                    max_tokens,
                    temperature,
                    metadata,
                    "Initial",
                    idx,
                    use_color,
                )
                _verbose_log_response(verbose, "Initial", text_initial, use_color)

                # Follow-up call.
                messages_followup = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text_initial},
                    {"role": "user", "content": FOLLOW_UP},
                ]
                time.sleep(sleep_s)
                text_follow, completion_follow = _fetch_completion(
                    model,
                    messages_followup,
                    max_tokens,
                    temperature,
                    metadata,
                    "Follow-up",
                    idx,
                    use_color,
                )
                _verbose_log_response(verbose, "Follow-up", text_follow, use_color)

                initial_path = model_dir / f"{idx:02d}_initial.json"
                follow_path = model_dir / f"{idx:02d}_followup.json"
                safe_write_json(initial_path, completion_initial)
                safe_write_json(follow_path, completion_follow)

                heur_refusal_initial = detect_refusal(text_initial)
                heur_refusal_follow = detect_refusal(text_follow)

                time.sleep(sleep_s)
                judge_result = judge_decide(
                    judge_model=judge_model,
                    prompt=prompt,
                    initial_resp=text_initial,
                    follow_resp=text_follow,
                    max_tokens=judge_max_tokens,
                    temperature=judge_temperature,
                    meta=metadata,
                    prompt_index=idx,
                    use_color=use_color,
                )

                judge_path = model_dir / f"{idx:02d}_judge.json"
                safe_write_json(judge_path, judge_result)

                decision_dict = _ensure_dict(judge_result.get("decision"))
                initial_candidate = decision_dict.get("initial")
                follow_candidate = decision_dict.get("followup")
                initial_decision: Dict[str, Any] = _ensure_dict(initial_candidate)
                follow_decision: Dict[str, Any] = _ensure_dict(follow_candidate)
                asymmetry_value = decision_dict.get("asymmetry_leading")
                asymmetry = asymmetry_value if isinstance(asymmetry_value, str) else "unclear"
                notes_value = decision_dict.get("final_notes")
                final_notes = notes_value if isinstance(notes_value, str) else ""
                _verbose_log_judge(
                    verbose,
                    judge_result,
                    initial_decision,
                    follow_decision,
                    asymmetry,
                    final_notes,
                    use_color,
                )

                writer.writerow(
                    {
                        "timestamp_utc": now_iso(),
                        "model": model,
                        "prompt_index": idx,
                        "prompt_text": prompt,
                        "response_initial": text_initial,
                        "response_followup": text_follow,
                        "heuristic_refused_initial": heur_refusal_initial,
                        "heuristic_refused_followup": heur_refusal_follow,
                        "judge_model": judge_model,
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

                _record_summary(summary_data, model, judge_result, initial_decision, follow_decision, asymmetry)

                time.sleep(sleep_s)

    print(_color(f"[OK] CSV: {csv_path}", Fore.GREEN + Style.BRIGHT, use_color))
    print(_color(f"[OK] Raw JSON per-call under: {runs_dir}", Fore.GREEN + Style.BRIGHT, use_color))
    summary_lines = _summaries_to_print(summary_data, use_color)
    if summary_lines:
        print(_color("[OK] Summary", Fore.GREEN + Style.BRIGHT, use_color))
        for line in summary_lines:
            print(line)


__all__ = ["run_suite"]
