# Copilot Instructions for LLM Judge

## High Level Details

### Repository Purpose

LLM Judge is a command-line test harness that measures alignment and bias in LLM
responses to politically sensitive topics. It queries OpenRouter-hosted models,
gathers their initial and follow-up answers, and scores them using a configured
judge model. The project includes a Flask + Svelte web dashboard for streaming
runs with live charts and chat-style transcripts.

### Repository Information

- **Size**: Medium-sized project with ~40 Python source files and ~15
  TypeScript/Svelte files
- **Type**: Full-stack application with Python backend and TypeScript/Svelte
  frontend
- **Languages**: Python 3.13+, TypeScript, Svelte
- **Frameworks**: Flask (backend), Vite + Svelte (frontend)
- **Target Runtimes**: Python 3.13+, Node.js 24+
- **Architecture**: Object-oriented with dependency injection, thread-safe
  services, and clean separation of concerns

## Build Instructions

### Prerequisites

- Python 3.13+
- Node.js 24+
- uv (Python package manager)
- npm (Node.js package manager)

### Bootstrap and Installation

```bash
# Install all dependencies (Python + Node.js)
make install

# This runs:
# - uv sync --extra dev (Python dependencies)
# - cd webui && npm install (Node.js dependencies)
```

### Build Commands

#### Python Backend

```bash
# Format code
make fmt

# Check formatting
make format-check

# Run linters
make lint

# Run type checking (Mypy + Pyright)
make type

# Run unit tests
make unit-tests

# Run all validation checks
make check
```

#### Web Frontend

```bash
# Build frontend assets
make web-build

# Start development server (with backend)
make webdev

# Start production web server
make web

# Start background web server
make webd
```

### Development Workflow

```bash
# Start development stack (backend + frontend with live reload)
make devstack-start

# Check dev stack status
make devstack-status

# Stop development stack
make devstack-stop
```

### Testing

```bash
# Run Python tests with coverage
make unit-tests

# Run frontend tests
cd webui && npm run unit-test

# Run all tests
make test
```

## Project Layout

### Major Architectural Elements

#### Backend Structure (`src/llm_judge/`)

- **Domain Layer** (`domain/`): Immutable data transfer objects (Prompt,
  ModelResponse, JudgeDecision, etc.)
- **Service Layer** (`services/`): Protocol interfaces defining contracts
  (IAPIClient, IPromptsManager, etc.)
- **Infrastructure Layer** (`infrastructure/`): Concrete implementations of
  services
- **Web Application** (`webapp/`): Flask application with SSE endpoints
- **Core Files**:
  - `runner.py`: Main test execution logic
  - `api.py`: OpenRouter API client
  - `judging.py`: Judge evaluation logic
  - `prompts.py`: Prompt management
  - `container.py`: Dependency injection container

#### Frontend Structure (`webui/`)

- **Source** (`src/`): Svelte components and TypeScript code
- **Components**: ChatWindow, ControlPanel, Scoreboard
- **Stores**: State management with Svelte stores
- **Build Output** (`dist/`): Compiled frontend assets

#### Configuration Files

- `pyproject.toml`: Python project configuration, dependencies, and tool
  settings
- `Makefile`: Build automation and development commands
- `.flake8`: Python linting configuration
- `webui/package.json`: Node.js dependencies and scripts
- `src/llm_judge/prompts.yaml`: Test prompt definitions
- `src/llm_judge/judge_config.yaml`: Judge configuration and schema

### Key Files and Locations

#### Root Level

- `judge.py`: CLI entry point for running tests
- `README.md`: Project documentation
- `AGENTS.md`: Agent guidelines and workflows
- `pyproject.toml`: Python project configuration
- `Makefile`: Build and development commands

#### Configuration

- Python formatting: Black (line-length: 120, target-version: py313)
- Python linting: Flake8 (max-line-length: 120, max-complexity: 8)
- Python typing: Mypy + Pyright (strict mode)
- Python testing: Pytest with coverage (95% minimum)
- Frontend formatting: Prettier
- Frontend linting: ESLint
- Frontend testing: Vitest

### Validation Pipeline

#### Pre-commit Hooks

- Gitleaks: Detects secrets in staged files
- Format check: Ensures code is properly formatted

#### CI/CD Workflow (`.github/workflows/code-checks.yml`)

1. **Static Analysis**:
   - Format check (Black + Prettier)
   - Linting (Flake8 + ESLint)
   - Type checking (Mypy + Pyright + Svelte-check)
2. **Unit Tests**:
   - Python tests with coverage reporting
   - Frontend tests with coverage
   - Coverage upload to Codecov
3. **Artifacts**: Test results and coverage reports

### Development Dependencies

#### Python Dependencies

- **Core**: PyYAML, openai, httpx, colorama, Flask, gevent, gunicorn
- **Development**: black, flake8, mypy, pyright, pytest, pytest-cov, vulture

#### Node.js Dependencies

- **Core**: svelte
- **Development**: vite, eslint, prettier, svelte-check, vitest,
  @testing-library

### Key Architectural Patterns

#### Dependency Injection

- All services use protocol-based interfaces
- ServiceContainer manages dependencies and lifetimes
- Factory pattern for complex object creation

#### Thread Safety

- All shared state protected by `threading.RLock()`
- Double-checked locking for expensive initializations
- Concurrent-safe API clients and managers

#### Repository Pattern

- Clean data access layer with Unit of Work
- Separate repositories for artifacts and results
- Transaction-like semantics for data operations

## Important Notes for Development

### Environment Setup

- Always run `make install` before development
- Set `OPENROUTER_API_KEY` environment variable for API access
- Use `make devstack-start` for iterative development

### Code Quality Standards

- All code must pass `make check` before submission
- Maintain 95%+ test coverage
- Use dependency injection container for service creation
- Follow the established architectural patterns

### Testing Requirements

- Write tests for all new functionality
- Use protocol interfaces for easy mocking
- Test thread safety for concurrent operations
- Maintain coverage thresholds

### Common Workflows

1. **Adding new prompts**: Edit `src/llm_judge/prompts.yaml`
2. **Modifying judge logic**: Update `src/llm_judge/judge_config.yaml`
3. **Backend changes**: Use dependency injection and maintain thread safety
4. **Frontend changes**: Build assets with `make web-build` before testing
5. **API changes**: Update both backend and frontend interfaces

### Trust These Instructions

This document provides the canonical build and development workflow. Only search
for additional information if these instructions are incomplete or found to be
in error. The Makefile is the single source of truth for build commands.
