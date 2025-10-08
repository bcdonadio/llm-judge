# LLM Judge

![LLM Judge logo](/llm-judge.png "LLM Judge logo")

[![codecov](https://codecov.io/gh/bcdonadio/llm-judge/branch/master/graph/badge.svg?token=YASCIBXVSB)](https://codecov.io/gh/bcdonadio/llm-judge)

LLM Judge is a command-line test harness that measures alignment and bias in LLM responses to politically sensitive topics. It queries OpenRouter-hosted models, gathers their initial and follow-up answers, and scores them using a configured judge model.

## Features

- Runs a curated prompt suite plus optional probes against one or more OpenRouter models.
- Captures raw completions and judge decisions as JSON artifacts.
- Produces a timestamped CSV summary with heuristic refusals and judge scores.
- Ships with strict linting, formatting, type checking, and tests (Black, Flake8, Mypy, Pyright, Pytest).

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. To install tooling and development extras:

```bash
make install
```

Alternatively, run `uv sync --extra dev --extra test`.

## Usage

Set an OpenRouter API key and invoke the CLI:

```bash
export OPENROUTER_API_KEY=sk-your-key
uv run python judge.py \
  --models x-ai/grok-4-fast openai/gpt-4o-mini qwen/qwen3-next-80b-a3b-thinking \
  --judge-model openai/gpt-5-mini \
  --include-probes \
  --verbose
```

Results are written under `results/`, e.g. `results/results_<timestamp>.csv` with raw artifacts in `results/runs/<timestamp>/`.

For local checks:

```bash
make fmt     # format with Black
make lint    # Black --check + Flake8
make type    # Mypy + Pyright
make test    # Pytest suite
make check   # run lint + type + test
```

## Project Layout

- `judge.py` – CLI entry point.
- `src/llm_judge/` – package containing API client, prompt definitions (YAML-backed), judge configuration, utilities, and runner.
- `tests/` – pytest suites for helpers, prompt loading, and judge configuration.
- `pyproject.toml` – project metadata and tool configuration (Black, Mypy, Pyright, Pytest, uv).

## License

Released under the MIT License. See [LICENSE](LICENSE).
