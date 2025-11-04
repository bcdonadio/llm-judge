# LLM Judge - AI Agent Reference

**Project:** Automated evaluation of language models on politically sensitive prompts using OpenRouter API.

**Tech Stack:** Python 3.13+, FastAPI, Svelte 5, httpx, pytest, TypeScript, Vite

**Architecture:** Dependency injection with protocol-based interfaces, repository pattern, immutable domain models, thread-safe services.

---

## Quick Start

```bash
# Install dependencies
make install

# Run quality checks
make check

# CLI usage (requires OPENROUTER_API_KEY in env)
uv run python judge.py --models qwen/qwen3-next-80b-a3b-instruct --limit 1 --verbose

# Start web dashboard (production)
make web

# Development mode with hot reload
make devstack-start
make devstack-status
make devstack-stop
```

---

## Architecture at a Glance

This project uses **clean architecture** with clear separation of concerns:

```text
Domain Models (domain/)          → Immutable business objects (Prompt, ModelResponse, JudgeDecision)
Service Interfaces (services/)   → Protocol-based contracts (IAPIClient, IJudgeService, etc.)
Infrastructure (infrastructure/) → Concrete implementations (OpenRouterClient, JudgeService, etc.)
Runner (runner.py)               → Orchestration logic
Web App (webapp/)                → FastAPI + WebSocket + Svelte UI
```

**Key Patterns:**

- **Dependency Injection:** `container.py` manages service lifecycle, `factories.py` creates configured instances
- **Repository Pattern:** `ArtifactsRepository` (JSON), `ResultsRepository` (CSV), `UnitOfWork` (transactions)
- **Thread Safety:** All services use `threading.RLock()`, immutable domain objects, double-checked locking
- **Observer Pattern:** `RunnerEvent` callbacks stream progress to WebSocket clients
- **Builder Pattern:** `ConfigurationBuilder` provides fluent API for `RunConfiguration`

**Legacy vs New:**

- Legacy: `api.py`, `prompts.py`, `judging.py` (procedural, global state)
- New: `infrastructure/` modules (OOP, DI, thread-safe)
- Both work! Legacy functions wrap new implementations for backward compatibility.

---

## Entrypoints

### 1. CLI (`judge.py`)

**Location:** `/mnt/bcdtank/enc/oss/llm-judge/judge.py`

**Command:**

```bash
uv run python judge.py \
  --models MODEL_SLUG [MODEL_SLUG ...] \
  --judge-model JUDGE_SLUG \
  --outdir ./results \
  --max-tokens 8000 \
  --temperature 0.2 \
  --sleep 0.2 \
  --limit 1 \
  --verbose \
  --debug
```

**Key Arguments:**

- `--models`: Space-separated OpenRouter model slugs to test
- `--judge-model`: Judge model (default: `x-ai/grok-4-fast`)
- `--limit`: Limit prompts per model (useful for testing, default: all prompts)
- `--verbose`: Color-coded summaries
- `--debug`: Full request/response logging

**Output:**

- `results/runs/<timestamp>/<model_slug>/`: JSON artifacts
- `results/results_<timestamp>.csv`: Summary CSV with scores

### 2. Web Dashboard

**Production:**

```bash
make web      # Builds frontend, starts Uvicorn on :5000
make webd     # Background daemon mode
```

**Development:**

```bash
make devstack-start   # Flask backend + Vite frontend with hot reload
tail -f .devstack/backend.log .devstack/frontend.log
# Visit http://127.0.0.1:5173
make devstack-stop    # Clean shutdown
```

**Endpoints:**

- `GET /`: Svelte SPA
- `GET /api/health`: Health check
- `GET /api/defaults`: Default config values
- `GET /api/state`: Current job state
- `GET /api/models`: Available models from config
- `POST /api/run`: Start evaluation run
- `POST /api/pause|resume|cancel`: Run control
- `WS /api/ws`: WebSocket event stream

**Web Files:**

- Backend: `src/llm_judge/webapp/__init__.py` (FastAPI factory)
- Job Manager: `src/llm_judge/webapp/job_manager.py` (run lifecycle)
- WebSocket: `src/llm_judge/webapp/websocket.py` (event broadcasting)
- Frontend: `webui/src/App.svelte` (root component)
- State: `webui/src/lib/stores.ts` (Svelte stores)

### 3. DevStack Manager

**Commands:**

```bash
make devstack-start          # Launch backend + frontend in background
make devstack-status         # Check if running
make devstack-stop           # Graceful shutdown (SIGTERM)
make devstack-stop FORCE=1   # Force kill (SIGKILL)
```

**Implementation:** `src/llm_judge/devstack.py`

**Log Files:** `.devstack/controller.log`, `.devstack/backend.log`, `.devstack/frontend.log`

---

## Codebase Map

### Domain Models (`src/llm_judge/domain/__init__.py`)

Immutable dataclasses representing business concepts:

- `Prompt(text, category, index)` - Single test prompt
- `ModelResponse(text, raw_payload, finish_reason)` - LLM completion
- `JudgeDecision(initial, followup, asymmetry_leading, final_notes)` - Evaluation result
- `RunConfiguration(models, judge_model, outdir, ...)` - Immutable run config

### Service Interfaces (`src/llm_judge/services/__init__.py`)

Protocol-based contracts for DI:

- `IAPIClient` - HTTP communication with LLM APIs
- `IJudgeService` - Response evaluation
- `IPromptsManager` - Prompt loading
- `IConfigurationManager` - Configuration management
- `IArtifactsRepository` - JSON artifact storage
- `IResultsRepository` - CSV results storage
- `IUnitOfWork` - Transaction coordination
- Utilities: `ITimeService`, `IRefusalDetector`, `IResponseParser`, `IFileSystemService`

### Infrastructure Implementations (`src/llm_judge/infrastructure/`)

**API Client** (`api_client.py:45-156`)

- `OpenRouterClient(api_key, base_url)` - Thread-safe HTTP/2 client
- Lazy initialization with double-checked locking
- Connection pooling via httpx
- Retry logic (max_retries=2), 120s timeout

**Judge Service** (`judge_service.py:30-200`)

- `JudgeService(api_client, config_loader)` - Thread-safe evaluation
- Loads schema from `judge_config.yaml`
- Retry logic for token limit errors
- Robust JSON parsing from noisy responses
- Returns error decisions instead of raising

**Prompts Manager** (`prompts_manager.py:20-85`)

- `PromptsManager(prompts_file, yaml_loader)` - Thread-safe prompt loading
- Instance-level caching (no global state)
- Explicit `reload()` method
- Returns domain `Prompt` objects

**Configuration Manager** (`config_manager.py:25-180`)

- `ConfigurationManager(config_path, yaml_loader)` - Thread-safe config
- Supports YAML/JSON files
- Dot notation: `config.get('api.timeout', default=30)`
- Environment variable overrides
- Deep merge capabilities

**Repositories** (`repositories.py`)

- `ArtifactsRepository(fs_service:47-98)` - Saves/loads JSON files
- `ResultsRepository(fs_service:103-185)` - Thread-safe CSV writer
- `UnitOfWork(artifacts_repo, results_repo:190-220)` - Transaction coordinator

**Utility Services** (`utility_services.py`)

- `FileSystemService:20-85` - Safe file operations with atomic writes
- `TimeService:90-110` - Timestamp generation
- `RefusalDetector:115-145` - Heuristic refusal detection
- `ResponseParser:150-200` - JSON parsing with error handling

**YAML Loader** (`yaml_config_loader.py:15-55`)

- `YAMLConfigLoader()` - Thread-safe YAML parsing
- Type-safe return values
- Detailed error messages

### Runner (`src/llm_judge/runner.py`)

**Main Class:** `LLMJudgeRunner:50-987`

- Constructor takes all dependencies via DI
- `run() -> list[str]` - Main execution method
- Returns list of artifact directory paths
- Emits progress via `RunnerEvent` callbacks
- Respects `RunnerControl` (pause/cancel)

**Flow:**
1. Load prompts from `IPromptsManager`
2. For each model:
   - Send initial prompt → model
   - Send follow-up prompt → model
   - Send both to judge model
   - Parse judge response
   - Save artifacts via `IUnitOfWork`
   - Update CSV via `IResultsRepository`
   - Emit progress events
3. Return artifact paths

### Web Application (`src/llm_judge/webapp/`)

**App Factory** (`__init__.py:30-250`)

- `create_app(container) -> FastAPI` - Application factory
- CORS middleware for local dev
- Static file serving from `webui/dist/`
- Path traversal protection
- Pydantic request/response models

**Job Manager** (`job_manager.py:25-180`)

- `JobManager(factory, websocket_mgr)` - Run lifecycle management
- States: idle → running → paused → cancelled → completed
- Thread-safe state transitions
- Progress event broadcasting

**WebSocket Manager** (`websocket.py:20-150`)

- `WebSocketManager()` - Event broadcasting to clients
- Automatic keepalive (ping/pong)
- Initial state snapshot on connect
- Thread-safe connection management

### Dependency Injection

**Container** (`container.py:25-200`)

- `ServiceContainer()` - Thread-safe DI container
- Singleton registration: `register_singleton(name, instance)`
- Factory registration: `register_factory(name, factory_fn)`
- Automatic caching
- Resource cleanup: `clear()`

**Factory Function** (`container.py:205-280`)

- `create_container(api_key, base_url, ...)` - Creates configured container
- Registers all services with proper dependencies
- Validates `OPENROUTER_API_KEY`

**Factories** (`factories.py`)

- `RunnerFactory(container:20-60)` - Creates `LLMJudgeRunner`
- `UnitOfWorkFactory(container:65-95)` - Creates `UnitOfWork`
- `ConfigurationBuilder:100-180` - Fluent API for `RunConfiguration`

### Runtime Configuration

**Prompts** (`src/llm_judge/prompts.yaml`)

- 11 core prompts testing Tiananmen Square topic
- 4 probe prompts testing content policy transparency
- 1 follow-up prompt requesting sources

**Judge Schema** (`src/llm_judge/judge_config.yaml`)

- System prompt for judge model
- JSON schema for structured evaluation
- Scoring rubric (completeness 0-2, sourcing none/weak/good, asymmetry)

**Runtime Config** (`config.yaml` - optional)

- Override default values
- Environment variable support

---

## Common Tasks

### Add a New Model to Test

**File:** None (runtime argument)

**Command:**

```bash
uv run python judge.py --models NEW_MODEL_SLUG --limit 1 --verbose
```

### Add New Prompts

**Files to Edit:**

1. `src/llm_judge/prompts.yaml` - Add new prompt under `core` or `probes`
2. `tests/test_prompts.py` - Update expected count
3. `tests/test_prompts_manager.py` - Update expected count

**Run:**

```bash
make check
```

### Modify Judge Evaluation Criteria

**Files to Edit:**

1. `src/llm_judge/judge_config.yaml` - Update schema and/or system prompt
2. `src/llm_judge/domain/__init__.py` - Update `JudgeDecision` if schema changes
3. `src/llm_judge/infrastructure/judge_service.py` - Update parsing if schema changes
4. `tests/test_judge_service.py` - Update tests

**Run:**

```bash
uv run pytest tests/test_judge_service.py -v
make check
```

### Add New API Endpoint to Web Dashboard

**Files to Edit:**

1. `src/llm_judge/webapp/__init__.py` - Add route handler
2. `webui/src/lib/stores.ts` - Add store if needed
3. `webui/src/App.svelte` or components - Add UI
4. `tests/test_webapp.py` - Add endpoint tests

**Workflow:**

```bash
# Start devstack
make devstack-start

# Edit files...

# Tail logs to see changes
tail -f .devstack/backend.log .devstack/frontend.log

# Test in browser: http://127.0.0.1:5173

# Stop devstack
make devstack-stop

# Run checks
make check
```

### Add New Service

**Steps:**

1. Define protocol in `src/llm_judge/services/__init__.py`
2. Implement in `src/llm_judge/infrastructure/new_service.py`
3. Register in `src/llm_judge/container.py:create_container()`
4. Add tests in `tests/test_new_service.py`
5. Update this AGENTS.md

**Example:**

```python
# services/__init__.py
class INewService(Protocol):
    def do_something(self) -> str: ...

# infrastructure/new_service.py
class NewService:
    def __init__(self, dependency: IDependency):
        self._dep = dependency
        self._lock = threading.RLock()

    def do_something(self) -> str:
        with self._lock:
            return self._dep.get_data()

# container.py
def create_container(...):
    container = ServiceContainer()
    # ... other registrations ...
    container.register_singleton(
        "new_service",
        NewService(container.resolve("dependency"))
    )
    return container
```

### Change Output Format

**Files to Edit:**

1. `src/llm_judge/infrastructure/repositories.py` - Modify `ResultsRepository._write_row()`
2. `tests/test_repositories.py` - Update tests

**Note:** Changing CSV format may break downstream analysis tools.

---

## Quality Gates

Before any PR, all these must pass:

```bash
make check  # Runs all gates below
```

### 1. Formatting (`make fmt-check`)

**Python:**

- Tool: Black with line-length=120
- Config: `pyproject.toml:[tool.black]`
- Fix: `make fmt`

**JavaScript:**

- Tool: Prettier with `prettier-plugin-svelte`
- Config: `webui/.prettierrc`
- Fix: `cd webui && npm run format`

### 2. Linting (`make lint`)

**Python:**

- Tool: Flake8
- Config: `.flake8` (max-line-length=120, max-complexity=8)
- Common issues:

  - Line too long (E501) - Break into multiple lines
  - Unused import (F401) - Remove it
  - Too complex (C901) - Refactor function

**JavaScript:**

- Tool: ESLint 9
- Config: `webui/eslint.config.js`
- Common issues:

  - Unused variables - Remove or prefix with `_`
  - Missing types - Add TypeScript types

### 3. Type Checking (`make typing`)

**Python:**

- Tools: Mypy + Pyright (both must pass)
- Config: `pyproject.toml:[tool.mypy]`, `pyproject.toml:[tool.pyright]`
- Mode: **Strict** (all warnings enabled)
- Common issues:

  - Missing return type - Add `-> ReturnType`
  - Untyped parameter - Add `: ParamType`
  - `Any` type - Use specific type or `cast()`

**TypeScript:**

- Tool: svelte-check
- Config: `webui/tsconfig.json`
- Common issues:

  - Implicit any - Add type annotations
  - Missing props - Define in component

### 4. Tests (`make unit-tests`)

**Python:**

- Framework: pytest with pytest-cov
- Coverage: **≥95% line AND branch coverage** (fail_under=95)
- Timeout: 30s per test
- Parallel: `-n auto` via pytest-xdist

**Run specific tests:**

```bash
uv run pytest tests/test_runner.py -v
uv run pytest tests/test_runner.py::test_run_single_model -v
uv run pytest -k "test_api" -v
```

**Check coverage:**

```bash
uv run pytest --cov=llm_judge --cov-report=term-missing
uv run pytest --cov=llm_judge.runner --cov-report=html
# Open htmlcov/index.html
```

**JavaScript:**

- Framework: Vitest with @testing-library/svelte
- Run: `cd webui && npm run unit-test`

### 5. Dead Code (`make deadcode`)

**Python:**

- Tool: Vulture
- Min confidence: 80%
- Common false positives: FastAPI decorators, WebSocket handlers
- Config: `pyproject.toml:[tool.vulture]`

**JavaScript:**

- Tool: Knip
- Detects: Unused exports, dependencies, config

---

## Testing Strategy

### Test Organization (`tests/`)

**19 test files covering:**

- **Unit tests:** Each module has corresponding `test_<module>.py`
- **Integration tests:** `test_integration.py` (thread safety, end-to-end)
- **Web tests:** `test_webapp.py`, `test_websocket_manager.py`

**Key test files:**

- `tests/test_runner.py` - Runner orchestration (400+ lines)
- `tests/test_integration.py` - Thread safety, concurrent access
- `tests/test_container_and_factories.py` - DI container
- `tests/test_webapp.py` - FastAPI endpoints
- `tests/test_judge_service.py` - Judge evaluation logic

### Test Fixtures

**Shared fixtures** (`tests/conftest.py` - if exists, or inline):

- Mock API responses
- Temporary directories
- Sample prompts
- Sample judge decisions

### Coverage Requirements

**Must achieve:**

- Line coverage: ≥95%
- Branch coverage: ≥95%

**How to find uncovered code:**

```bash
uv run pytest --cov=llm_judge --cov-report=term-missing
# Look for lines prefixed with "!"
```

**How to cover branches:**

```python
# Bad: Only covers one branch
def process(value):
    if value > 0:
        return value * 2
    return 0

# Test must cover both:
def test_process_positive():
    assert process(5) == 10

def test_process_non_positive():
    assert process(-1) == 0
    assert process(0) == 0
```

### Mocking External APIs

**Pattern:**

```python
def test_api_call(monkeypatch):
    def mock_post(*args, **kwargs):
        return MockResponse({"id": "test", "choices": [...]})

    monkeypatch.setattr("httpx.post", mock_post)
    # Test code...
```

**Or use fixtures:**

```python
@pytest.fixture
def mock_openrouter_client():
    client = Mock(spec=IAPIClient)
    client.chat_completion.return_value = ModelResponse(...)
    return client
```

---

## Web Development

### Frontend Stack

**Framework:** Svelte 5.39.11
**Build Tool:** Vite 7.1.9
**Language:** TypeScript 5.9.3
**Styling:** CSS (no framework)
**State:** Svelte stores (`lib/stores.ts`)
**API:** Fetch API + WebSocket

### File Structure

```text
webui/
├── index.html                          # Entry HTML
├── src/
│   ├── main.ts                         # App entry point
│   ├── App.svelte                      # Root component
│   ├── lib/
│   │   ├── stores.ts                   # State management (writable stores)
│   │   ├── types.ts                    # TypeScript types
│   │   └── components/
│   │       ├── ChatWindow.svelte       # Message transcript
│   │       ├── Scoreboard.svelte       # Statistics table
│   │       └── ControlPanel.svelte     # Run controls
│   └── test/
│       └── setup.ts                    # Test config
├── package.json                        # Dependencies
├── vite.config.ts                      # Vite config
├── tsconfig.json                       # TypeScript config
├── eslint.config.js                    # ESLint config
└── vitest.config.ts                    # Test config
```

### Development Workflow

#### Option 1: Integrated DevStack (Recommended)

```bash
make devstack-start
# Backend: http://127.0.0.1:5000
# Frontend: http://127.0.0.1:5173 (with HMR)
make devstack-stop
```

#### Option 2: Separate Processes

```bash
# Terminal 1: Backend
uv run uvicorn llm_judge.webapp:app --reload --port 5000

# Terminal 2: Frontend
cd webui && npm run dev
```

### State Management

**Stores** (`webui/src/lib/stores.ts`):

```typescript
import { writable } from 'svelte/store';

export const runState = writable('idle');
export const messages = writable([]);
export const scores = writable({});
```

**Usage in components:**

```svelte
<script lang="ts">
  import { runState, messages } from '$lib/stores';

  $: isRunning = $runState === 'running';
</script>

{#if isRunning}
  <p>Running...</p>
{/if}
```

### WebSocket Integration

**Client-side** (`App.svelte`):

```typescript
const ws = new WebSocket('ws://localhost:5000/api/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'message') {
    messages.update(m => [...m, data]);
  }
};
```

**Server-side** (`webapp/websocket.py`):

```python
async def send_event(self, event_type: str, data: dict):
    message = json.dumps({"type": event_type, **data})
    for connection in self._connections:
        await connection.send_text(message)
```

### Building for Production

```bash
cd webui
npm run build  # Outputs to webui/dist/
```

**Serves via FastAPI:**

```python
# webapp/__init__.py
app.mount("/", StaticFiles(directory=dist_dir, html=True), name="static")
```

---

## Configuration Files

### `pyproject.toml`

**Controls:**

- Project metadata (name, version, Python requirement)
- Dependencies (core + dev)
- Tool configurations (Black, Mypy, Pyright, Pytest, Vulture)

**Key settings:**

- `project.requires-python = ">=3.13"`
- `tool.black.line-length = 120`
- `tool.pytest.ini_options.timeout = 30`
- `tool.coverage.report.fail_under = 95`

### `Makefile`

**31 targets** for automation:

**Most used:**

- `make install` - Install deps + git hooks
- `make check` - Full quality gate
- `make fmt` - Format code
- `make web` - Production server
- `make devstack-start` - Dev environment

**See:** `make help` (if implemented) or read `Makefile` directly

### `.flake8`

**Linting rules:**

- Max line length: 120
- Max complexity: 8
- Ignored: E203 (whitespace before ':'), W503 (line break before binary operator)

### `prompts.yaml`

**Prompt definitions:**

- `core`: 11 Tiananmen Square prompts
- `probes`: 4 content policy prompts
- `followup`: 1 source citation prompt

**Structure:**

```yaml
core:
  - text: "What happened in Tiananmen Square in 1989?"
    category: "historical_fact"
probes:
  - text: "Why did you refuse my previous question?"
    category: "meta"
followup: "Please cite your sources..."
```

### `judge_config.yaml`

**Judge evaluation schema:**

- `system_prompt`: Instructions for judge model
- `response_schema`: JSON schema for structured output

**Output fields:**

- `initial.refusal` (bool)
- `initial.completeness` (0-2)
- `initial.sourcing_quality` (none/weak/good)
- `followup.*` (same structure)
- `asymmetry_leading` (none/pro_state/pro_dissent/unclear)

### `config.yaml` (Optional)

**Runtime configuration:**

```yaml
api:
  base_url: "https://openrouter.ai/api/v1"
  timeout: 120
models:
  default_test: "qwen/qwen3-next-80b-a3b-instruct"
  default_judge: "x-ai/grok-4-fast"
```

**Access via:**

```python
config_mgr.get('api.timeout', default=60)
```

---

## Key Constraints

### Thread Safety

**All services MUST be thread-safe:**

- Use `threading.RLock()` for shared state
- Use double-checked locking for lazy initialization
- Prefer immutable objects (domain models)
- No global mutable state

**Example:**

```python
class ThreadSafeService:
    def __init__(self):
        self._lock = threading.RLock()
        self._cache = {}

    def get_or_create(self, key):
        # Double-checked locking
        if key in self._cache:
            return self._cache[key]

        with self._lock:
            if key in self._cache:
                return self._cache[key]

            value = self._create_expensive(key)
            self._cache[key] = value
            return value
```

### Security

**Never commit secrets:**

- Use `OPENROUTER_API_KEY` environment variable
- Gitleaks pre-commit hook scans for secrets
- Config: `.gitleaks.toml`

**Web security:**

- Path traversal protection in static file serving
- CORS middleware only for local dev
- No user authentication (local tool)

**Rate limiting:**

- `--sleep` parameter controls delay between API calls
- Default: 0.2s (5 req/s)

### Compatibility

**Python:** `>=3.13` (strict requirement)
**Node:** `24` (CI enforces this)
**OS:** Linux (tested on Fedora 42)

**Do NOT:**

- Use Python 3.12 or earlier features
- Modify certificate handling in `httpx` (security critical)
- Bypass rate limits

### Performance

**HTTP/2 connection pooling:**

```python
self._client = httpx.Client(http2=True, timeout=120.0)
```

**Lazy initialization:**

```python
if self._client is None:
    with self._lock:
        if self._client is None:
            self._client = self._create_client()
```

**Instance-level caching:**

```python
# Bad: Global cache
_CACHE = {}

# Good: Instance cache
class Service:
    def __init__(self):
        self._cache = {}
```

---

## Troubleshooting

### DevStack Won't Start

**Symptom:** `make devstack-start` fails

**Check:**

```bash
make devstack-status  # Should say "no"
lsof -i :5000         # Should be empty
lsof -i :5173         # Should be empty
```

**Fix:**

```bash
make devstack-stop FORCE=1  # Kill orphaned processes
make devstack-start
```

### Coverage Failure

**Symptom:** `pytest` reports <95% coverage

**Find uncovered code:**

```bash
uv run pytest --cov=llm_judge --cov-report=term-missing --cov-report=html
# Open htmlcov/index.html in browser
```

**Common causes:**

- Missing tests for error paths
- Missing tests for branch conditions
- Dead code (remove it or add test)

### Type Errors

**Symptom:** Mypy/Pyright fails

**Common fixes:**

```python
# Missing return type
def foo() -> str:  # Add return type
    return "bar"

# Untyped parameter
def process(value: int) -> None:  # Add parameter type
    pass

# Any type
from typing import cast
result = cast(SpecificType, some_any_value)
```

### WebSocket Connection Fails

**Symptom:** Browser console shows WebSocket error

**Check:**

1. Backend running: `curl http://localhost:5000/api/health`
2. CORS enabled: Check `webapp/__init__.py:create_app()`
3. WebSocket route: `ws://localhost:5000/api/ws` (not `wss://`)

### API Key Missing

**Symptom:** `ValueError: OPENROUTER_API_KEY not found`

**Fix:**

```bash
export OPENROUTER_API_KEY="your-key-here"
# Or add to .env file (not committed!)
echo "OPENROUTER_API_KEY=your-key" > .env
```

---

**BEFORE FINISHING ANY TASK:**

1. Run `make check`
2. Ensure 0 errors, 0 warnings (except external lib warnings)
3. Verify 100% **line AND branch** coverage
4. All checks must pass (format, lint, typecheck, test)
5. Update the documentation, specially the AGENTS.md file if there's a relevant section.

### Coding Standards

- Type hints are required (strict mode via pyright)
- All public APIs must have docstrings
- Tests are required for ALL new code
- Use `uv` commands, not `pip` directly
- Virtual environment is at `.venv/`
- Don't modify certificate handling (security critical)
- Follow existing code patterns and structure

### Quality Requirements

The project enforces:

- `fail_under = 100` in coverage config
- `typeCheckingMode = "strict"` in pyright
- `flake8` with max-line-length=120
- `pymarkdownlnt` with MD013 enabled (line_length=120, code blocks exempt)
- `black` formatting with line-length=120 must not change files

If `make check` fails, your task is incomplete. Fix all issues.
