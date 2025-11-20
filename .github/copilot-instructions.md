# Copilot Instructions for LLM Judge

## Overview

LLM Judge is a command-line test harness that measures alignment and bias in LLM
responses to politically sensitive topics. It queries OpenRouter-hosted models,
gathers their initial and follow-up answers, and scores them using a configured
judge model. The project includes a Flask + Svelte web dashboard for streaming
runs with live charts and chat-style transcripts.

**When working on this repository:**

- Always maintain backward compatibility unless explicitly requested otherwise
- Use dependency injection patterns - never create services directly, use the container
- Ensure thread safety - use `threading.RLock()` for all shared state
- Keep immutable domain objects - never add mutable state to domain models
- Write tests for all code changes - maintain 95%+ coverage
- Follow existing architectural patterns (see below)

## Project Information

- **Languages**: Python 3.13+ (strict requirement), TypeScript, Svelte
- **Frameworks**: Flask (backend), Vite + Svelte (frontend)
- **Architecture**: Object-oriented with dependency injection, thread-safe services
- **Size**: ~40 Python files, ~15 TypeScript/Svelte files
- **Key Constraint**: Must be thread-safe - use `threading.RLock()` for all shared state

## Getting Started

### Installation

**Always run these commands before starting development:**

```bash
# Install all dependencies (Python + Node.js)
make install

# This automatically:
# - Installs Python dependencies via: uv sync --extra dev
# - Installs Node.js dependencies via: cd webui && npm install
# - Sets up git hooks for gitleaks and format checking
```

### Quality Checks

**Before committing any code, run:**

```bash
make check  # Runs all validation: format, lint, type, tests, deadcode
```

**Individual commands:**

```bash
make fmt              # Auto-format Python (Black) and TypeScript (Prettier)
make format-check     # Check formatting without changes
make lint             # Run Flake8 (Python) and ESLint (TypeScript)
make type             # Run Mypy + Pyright (Python) and svelte-check (TypeScript)
make unit-tests       # Run pytest with coverage (must be ≥95%)
make deadcode         # Check for unused code with Vulture and Knip
```

### Development Workflow

**For iterative development with live reload:**

```bash
make devstack-start   # Starts Flask backend + Vite frontend with hot reload
make devstack-status  # Check if devstack is running
make devstack-stop    # Stop all devstack processes
```

**Development servers:**
- Backend: http://127.0.0.1:5000 (Flask with reload)
- Frontend: http://127.0.0.1:5173 (Vite with HMR)
- Logs: `.devstack/backend.log` and `.devstack/frontend.log`

**For production web server:**

```bash
make web-build  # Build Svelte assets to webui/dist/
make web        # Start Uvicorn server on :5000
make webd       # Start in background (daemon mode)
```

## Architecture Overview

### Core Architectural Patterns

**Dependency Injection (REQUIRED)**
- Never instantiate services directly - always use `ServiceContainer`
- All services implement Protocol interfaces from `services/`
- Example:
  ```python
  # Bad - Don't do this
  client = OpenRouterClient(api_key)
  
  # Good - Use the container
  container = create_container(api_key)
  client = container.resolve("api_client")
  ```

**Thread Safety (CRITICAL)**
- All services must be thread-safe
- Use `threading.RLock()` for shared state
- Use double-checked locking for lazy initialization
- Example:
  ```python
  class ThreadSafeService:
      def __init__(self):
          self._lock = threading.RLock()
          self._cache = {}
      
      def get(self, key):
          if key in self._cache:  # First check
              return self._cache[key]
          with self._lock:
              if key in self._cache:  # Double-check
                  return self._cache[key]
              value = self._compute(key)
              self._cache[key] = value
              return value
  ```

**Immutable Domain Models (REQUIRED)**
- Domain objects in `domain/` must be immutable dataclasses
- Never add mutable state to domain models
- Use `frozen=True` for all domain dataclasses

### Directory Structure

**Backend** (`src/llm_judge/`):
- `domain/` - Immutable data transfer objects (Prompt, ModelResponse, JudgeDecision)
- `services/` - Protocol interfaces defining contracts (IAPIClient, IJudgeService, etc.)
- `infrastructure/` - Concrete implementations of services
- `webapp/` - Flask application with SSE endpoints
- `runner.py` - Main test execution orchestration
- `container.py` - Dependency injection container
- `prompts.yaml` - Test prompt definitions
- `judge_config.yaml` - Judge evaluation schema

**Frontend** (`webui/`):
- `src/` - Svelte components and TypeScript code
- `src/lib/components/` - ChatWindow, ControlPanel, Scoreboard
- `src/lib/stores.ts` - State management
- `dist/` - Build output (served by Flask)

## Coding Standards

### Python Code Requirements

**Formatting:**
- Use Black with line-length=120, target-version=py313
- Run `make fmt` to auto-format before committing

**Linting:**
- Flake8 enforces max-line-length=120, max-complexity=8
- No unused imports or variables

**Type Checking:**
- Strict mode enabled for both Mypy and Pyright
- All functions must have return type annotations
- All parameters must have type annotations
- Example:
  ```python
  # Bad - Missing type hints
  def process_data(value):
      return value * 2
  
  # Good - Full type hints
  def process_data(value: int) -> int:
      return value * 2
  ```

**Testing:**
- Pytest with ≥95% line AND branch coverage (strict requirement)
- Write tests for all new code
- Use protocol interfaces for easy mocking
- Example:
  ```python
  @pytest.fixture
  def mock_api_client():
      client = Mock(spec=IAPIClient)
      client.chat_completion.return_value = ModelResponse(...)
      return client
  ```

### TypeScript Code Requirements

**Formatting:**
- Use Prettier with prettier-plugin-svelte
- Run `cd webui && npm run format` to auto-format

**Linting:**
- ESLint 9 configuration in `webui/eslint.config.js`
- No unused variables (prefix with `_` if intentionally unused)

**Type Checking:**
- Use svelte-check for type validation
- Avoid implicit `any` types

## Common Development Tasks

### Adding New Prompts

1. Edit `src/llm_judge/prompts.yaml`
2. Add prompt under `core` or `probes` section
3. Update expected count in `tests/test_prompts.py` and `tests/test_prompts_manager.py`
4. Run `make check` to verify

### Modifying Judge Criteria

1. Edit `src/llm_judge/judge_config.yaml` (update schema/system prompt)
2. Update `JudgeDecision` dataclass in `src/llm_judge/domain/__init__.py` if schema changes
3. Update parsing logic in `src/llm_judge/infrastructure/judge_service.py` if needed
4. Update tests in `tests/test_judge_service.py`
5. Run `make check` to verify

### Adding New API Endpoint

1. Add route handler in `src/llm_judge/webapp/__init__.py`
2. Create/update stores in `webui/src/lib/stores.ts` if needed
3. Update UI components in `webui/src/` as needed
4. Add endpoint tests in `tests/test_webapp.py`
5. Run `make check` to verify

### Adding New Service

**Follow this pattern:**

1. Define protocol interface in `src/llm_judge/services/__init__.py`:
   ```python
   class INewService(Protocol):
       def do_something(self, param: str) -> str: ...
   ```

2. Implement in `src/llm_judge/infrastructure/new_service.py`:
   ```python
   class NewService:
       def __init__(self, dependency: IDependency):
           self._dep = dependency
           self._lock = threading.RLock()
       
       def do_something(self, param: str) -> str:
           with self._lock:
               return self._dep.process(param)
   ```

3. Register in `src/llm_judge/container.py`:
   ```python
   container.register_singleton(
       "new_service",
       NewService(container.resolve("dependency"))
   )
   ```

4. Add tests in `tests/test_new_service.py`
5. Run `make check` to verify

## Important Constraints and Requirements

### Environment Requirements

- **Python**: 3.13+ (strict requirement - do not use Python 3.12 features)
- **Node.js**: 24+ (CI enforces this)
- **API Key**: Set `OPENROUTER_API_KEY` environment variable for API access

### Security Requirements

- Never commit secrets or API keys
- Gitleaks pre-commit hook scans for secrets
- Use environment variables for all credentials
- Path traversal protection in static file serving
- CORS middleware only for local development

### Performance Requirements

- Use HTTP/2 connection pooling: `httpx.Client(http2=True)`
- Implement lazy initialization with double-checked locking
- Use instance-level caching (not global state)
- Respect rate limits: default 0.2s sleep between API calls

### Testing Requirements

- ≥95% line coverage (enforced by pytest-cov)
- ≥95% branch coverage (test both paths of conditionals)
- 30s timeout per test
- All tests must be thread-safe
- Use mocks for external API calls

### Critical Constraints

**DO:**
- Use dependency injection container for all services
- Ensure thread safety with `threading.RLock()`
- Keep domain models immutable (`frozen=True`)
- Write comprehensive tests
- Follow existing architectural patterns
- Run `make check` before submitting changes

**DO NOT:**
- Create services directly without container
- Add mutable state to domain objects
- Use global state or caches
- Modify certificate handling in httpx
- Bypass rate limits
- Use Python 3.12 or earlier features
- Remove or modify working tests

## Configuration Files

### Key Configuration Files

- **pyproject.toml** - Python project metadata, dependencies, tool settings
- **Makefile** - Build automation (single source of truth for commands)
- **.flake8** - Python linting rules (max-line-length=120, max-complexity=8)
- **webui/package.json** - Node.js dependencies and scripts
- **prompts.yaml** - 11 core prompts + 4 probe prompts + 1 follow-up
- **judge_config.yaml** - Judge model system prompt and JSON schema

### Tool Configurations

All tool configurations are in `pyproject.toml`:
- `[tool.black]` - line-length=120, target-version=py313
- `[tool.mypy]` - Strict mode enabled
- `[tool.pyright]` - typeCheckingMode="strict"
- `[tool.pytest.ini_options]` - timeout=30, testpaths=tests
- `[tool.coverage.report]` - fail_under=95

## Additional Resources

For more detailed information about specific components, workflows, or troubleshooting:
- See `README.md` for user documentation
- See `AGENTS.md` for comprehensive technical details
- See `Makefile` for all available commands
- Check `.github/workflows/code-checks.yml` for CI pipeline

---

**Remember: Always run `make check` before finalizing any changes. All checks must pass.**
