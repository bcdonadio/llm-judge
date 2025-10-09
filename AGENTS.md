# LLM Judge Agent Guidelines

This document follows the [agents.md](https://agents.md) convention to describe the autonomous and semi-autonomous workflows inside this repository.

## Purpose

The `llm-judge` project evaluates how language models respond to a politically sensitive prompt suite using the OpenRouter API. It records raw completions, runs a judging model, and aggregates results for analysis. The automation here assumes careful handling of external APIs and generated artifacts.

## Roles

### CLI Operator (`judge.py`)

- Launches the test run via command-line arguments.
- Ensures `OPENROUTER_API_KEY` is set before execution.
- Chooses model slugs and judge model.
- Defaults to evaluating `qwen/qwen3-next-80b-a3b-instruct` with `x-ai/grok-4-fast` as the judge; override via CLI flags or the web dashboard.
- Uses `--debug` when deeper request/response tracing is required; otherwise stays at info-level logging.
- Uses `--verbose` for color-coded summaries of prompts/responses during manual spot checks.
- Enables `--limit 1` for live API runs unless performing a full suite; this is the default method to validate connectivity without exhausting quotas.
- Confirms output directories when using shared or production environments.

### Runner Agent (`llm_judge.runner.LLMJudgeRunner`)

- Coordinates prompt iteration, issues initial and follow-up calls, and saves JSON artifacts.
- Pauses between requests (`sleep` argument) to respect rate limits.
- Emits a CSV summary with judge metadata; no deletion of prior runs.

### Web Dashboard (`llm_judge.webapp` + `webui/`)

- Serves a Flask backend with SSE endpoints, powered by Gunicorn + gevent for concurrent streaming.
- Streams prompt/response and judge updates to the Svelte-based interface in `webui/`.
- Provides run, pause, resume, and cancel controls plus a live scoreboard aggregated from runner events.
- Requires `make web` (or `npm run build` under `webui/`) to compile Vite assets before serving.

### Judge Agent (`llm_judge.judging.judge_decide`)

- Loads reusable instructions, schema, and system prompt from YAML.
- Validates judge responses, attempting to parse JSON strictly. On failure it returns an error payload rather than raising.
- Avoids altering or storing OpenRouter payloads beyond the local run directory.

### Support Utilities

- `llm_judge.api.openrouter_chat` handles authenticated HTTP calls with strict timeouts.
- `llm_judge.prompts` loads prompt definitions from YAML to keep text out of code.
- `llm_judge.utils` provides safe JSON writing, refusal heuristics, and timestamp formatting.

## Safety & Compliance

- **Credentials:** Never commit or log API keys. Prefer environment variables or the user's secrets store.
- **Outputs:** Generated directories (`results/runs/<timestamp>`, `results/results_<timestamp>.csv`) may contain sensitive model text. Treat them as confidential and avoid pushing to public repos.
- **Rate Limits:** Adjust `--sleep` if encountering throttling. Do not bypass OpenRouter usage policies.
- **Network:** This project depends on outbound HTTP requests to `https://openrouter.ai`. When running offline, mock or skip those calls.
- **Content Sensitivity:** Prompts explore politically sensitive subject matter. Handle outputs respectfully and within applicable laws and policies.

## Development Workflow

- Install dependencies with `make install` (uv extra `dev`).
- Validate changes locally: run `make fmt` to apply formatting and `make check` (runs `fmt-check`, lint, type, and tests) before opening a PR.
- Build web assets with `make web-build` (or `npm run build` in `webui/`) before serving the dashboard.
- When adding prompts or judge instructions, edit the YAML resources (`prompts.yaml`, `judge_config.yaml`) and update tests if structures change.
- Keep new files ASCII unless referencing non-ASCII content from sources.
- Respect `.flake8` exclusions to avoid linting generated artifacts.

## Extending the System

- New prompt suites can be added by extending `prompts.yaml` and corresponding tests.
- Additional judges or scoring heuristics belong in dedicated modules under `src/llm_judge`.
- Integrations (dashboards, databases) must sanitize data and adhere to MIT license terms.

## Contact

For license details see [LICENSE](LICENSE). Contributions should follow the safety and workflow guidance above.
