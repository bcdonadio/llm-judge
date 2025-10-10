# LLM Judge

![LLM Judge logo](/llm-judge.png "LLM Judge logo")

[![codecov](https://codecov.io/gh/bcdonadio/llm-judge/branch/master/graph/badge.svg?token=YASCIBXVSB)](https://codecov.io/gh/bcdonadio/llm-judge)

LLM Judge is a command-line test harness that measures alignment and bias in LLM responses to politically sensitive topics. It queries OpenRouter-hosted models, gathers their initial and follow-up answers, and scores them using a configured judge model. A Flask + Svelte web dashboard is included for streaming runs with live charts and chat-style transcripts.

## Features

- Runs the curated prompt suite against one or more OpenRouter models.
- Captures raw completions and judge decisions as JSON artifacts.
- Produces a timestamped CSV summary with heuristic refusals and judge scores.
- Real-time dashboard with server-sent events (SSE) showing prompts, answers, judge notes, and rolling scoreboards.
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
  --models qwen/qwen3-next-80b-a3b-instruct mistral/mistral-large-latest \
  --judge-model x-ai/grok-4-fast \
  --verbose
```

Results are written under `results/`, e.g. `results/results_<timestamp>.csv` with raw artifacts in `results/runs/<timestamp>/`.

By default the CLI and dashboard target `qwen/qwen3-next-80b-a3b-instruct` as the evaluated model and score with `x-ai/grok-4-fast`.

### Web dashboard

A live control panel is bundled in `webui/`. It streams judge runs over SSE, renders a chat-style prompt/response timeline, and keeps per-model scoreboards.

```bash
make web  # install web deps, build with Vite, start Gunicorn + gevent on :5000
```

Override defaults as needed:

```bash
make web GUNICORN_BIND=127.0.0.1:8000 GUNICORN_WORKERS=2
```

The command rebuilds Svelte assets with Vite and serves the Flask application via Gunicorn using gevent workers for concurrency.

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
- `src/llm_judge/` – package containing API client, prompt definitions (YAML-backed), judge configuration, utilities, runner, and the Flask web app.
- `webui/` – Svelte/Vite front-end compiled to `webui/dist` for the dashboard.
- `tests/` – pytest suites for helpers, prompt loading, and judge configuration.
- `pyproject.toml` – project metadata and tool configuration (Black, Mypy, Pyright, Pytest, uv).

## License

Released under the MIT License. See [LICENSE](LICENSE).
