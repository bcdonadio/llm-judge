#!/usr/bin/env python3
"""Command-line interface for the llm-judge project."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

from llm_judge import LLMJudgeRunner, RunnerConfig
from colorama import Fore, Style, init as colorama_init


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run LLM asymmetry tests via OpenRouter with LLM-as-a-judge.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="OpenRouter model slugs to test (e.g., xai/grok-4-fast openai/gpt-4o-mini qwen/qwen-2.5-72b-instruct)",
    )
    parser.add_argument(
        "--judge-model",
        default="openai/gpt-5-mini",
        help="Judge model slug (default: openai/gpt-5-mini). Use any OpenRouter-accessible judge.",
    )
    parser.add_argument(
        "--include-probes",
        action="store_true",
        help="Include diagnostic probes A-D",
    )
    parser.add_argument(
        "--outdir",
        default="results",
        help="Output directory (defaults to ./results).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8000,
        help="Token cap for test-model responses",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=6000,
        help="Token cap for judge responses",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for test-models",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for judge (keep at 0.0-0.2)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep seconds between calls (rate-limit friendly)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of prompts evaluated per model (debugging aid).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show structured summaries of prompts and responses.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging to trace HTTP calls and internal flow.",
    )
    return parser


def configure_logging(debug: bool, verbose: bool) -> bool:
    """Configure root logger; return whether colored output is enabled."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    use_color = sys.stderr.isatty() and os.getenv("NO_COLOR") is None
    handler = logging.StreamHandler()

    if use_color:
        colorama_init()

        class ColorFormatter(logging.Formatter):
            COLORS = {
                logging.DEBUG: Style.DIM + Fore.BLUE,
                logging.INFO: Fore.CYAN,
                logging.WARNING: Fore.YELLOW,
                logging.ERROR: Fore.RED,
                logging.CRITICAL: Style.BRIGHT + Fore.RED,
            }

            def format(self, record: logging.LogRecord) -> str:
                color = self.COLORS.get(record.levelno, "")
                message = super().format(record)
                if color:
                    return f"{color}{message}{Style.RESET_ALL}"
                return message

        formatter: logging.Formatter = ColorFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    else:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=[handler])
    return use_color


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    use_color = configure_logging(args.debug, args.verbose)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        config = RunnerConfig(
            models=args.models,
            judge_model=args.judge_model,
            include_probes=args.include_probes,
            outdir=outdir,
            max_tokens=args.max_tokens,
            judge_max_tokens=args.judge_max_tokens,
            temperature=args.temperature,
            judge_temperature=args.judge_temperature,
            sleep_s=args.sleep,
            limit=args.limit,
            verbose=args.verbose,
            use_color=use_color,
        )
        runner = LLMJudgeRunner(config)
        runner.run()
    except KeyboardInterrupt:
        print("\n[Interrupted] Exiting.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
