# LLM Judge Architecture Documentation

This document describes the object-oriented architecture of the LLM Judge project, including design patterns, thread-safety guarantees, and best practices.

## Table of Contents

- [Overview](#overview)
- [Core Principles](#core-principles)
- [Architecture Layers](#architecture-layers)
- [Design Patterns](#design-patterns)
- [Thread Safety](#thread-safety)
- [Dependency Injection](#dependency-injection)
- [Migration Guide](#migration-guide)
- [Usage Examples](#usage-examples)

## Overview

The LLM Judge codebase has been refactored from a procedural, global-state-based design to a fully object-oriented architecture with:

- **Zero global state**: All state is encapsulated in classes
- **Complete thread-safety**: All services use proper locking mechanisms
- **Dependency injection**: Protocol-based interfaces enable testability
- **SOLID principles**: Clean separation of concerns throughout
- **Repository pattern**: Clean data access layer
- **Error handling**: Custom exception hierarchy with structured context
- **Backward compatibility**: Legacy code continues to work

## Core Principles

### 1. **No Global State**

All module-level global variables have been eliminated. Instead:

```python
# ❌ Old approach (global state)
_client = None
_http_client = None

def get_client():
    global _client
    if _client is None:
        _client = OpenAI(...)
    return _client

# ✅ New approach (encapsulated state)
class OpenRouterClient:
    def __init__(self, api_key: str):
        self._client: Optional[OpenAI] = None
        self._lock = threading.RLock()

    def _ensure_client(self) -> OpenAI:
        with self._lock:
            if self._client is None:
                self._client = OpenAI(...)
            return self._client
```

### 2. **Thread Safety**

All shared state is protected using `threading.RLock()`:

```python
class PromptsManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._prompts_cache: Optional[Dict[str, Any]] = None

    def _load_prompts(self) -> Dict[str, Any]:
        with self._lock:
            if self._prompts_cache is not None:
                return self._prompts_cache
            # Load and cache prompts
            self._prompts_cache = data
            return data
```

### 3. **Dependency Injection**

Services depend on abstract `Protocol` interfaces rather than concrete implementations:

```python
# Define interface
class IAPIClient(Protocol):
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        metadata: Dict[str, str],
    ) -> ModelResponse: ...

# Inject dependency
class JudgeService:
    def __init__(self, api_client: IAPIClient):
        self._api_client = api_client
```

## Architecture Layers

### Domain Layer (`src/llm_judge/domain/`)

Immutable data transfer objects representing core business concepts:

- **`Prompt`**: A prompt with text, category, and index
- **`ModelResponse`**: API response with text, payload, and finish reason
- **`JudgeDecision`**: Evaluation results with completeness, refusal flags, etc.
- **`RunConfiguration`**: Immutable configuration for a test run
- **`RunArtifacts`**: Results of a completed run

```python
@dataclass(frozen=True)
class ModelResponse:
    text: str
    raw_payload: Dict[str, Any]
    finish_reason: Optional[str] = None
```

### Service Layer (`src/llm_judge/services/`)

Protocol interfaces defining contracts for all services:

- **`IAPIClient`**: HTTP communication with LLM APIs
- **`IPromptsManager`**: Prompt loading and management
- **`IJudgeService`**: Response evaluation logic
- **`IConfigurationManager`**: Configuration management
- **`IArtifactsRepository`**: Artifact persistence
- **`IResultsRepository`**: Results CSV persistence
- **`IUnitOfWork`**: Transaction coordination

### Infrastructure Layer (`src/llm_judge/infrastructure/`)

Concrete implementations of service interfaces:

#### API Client (`api_client.py`)

```python
class OpenRouterClient(IAPIClient):
    """Thread-safe HTTP client with connection pooling."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 120,
        max_retries: int = 2,
    ):
        self._lock = threading.RLock()
        self._client: Optional[OpenAI] = None
        self._http_client: Optional[httpx.Client] = None

    def chat_completion(...) -> ModelResponse:
        client = self._ensure_client()
        with self._lock:
            response = client.chat.completions.create(...)
        return ModelResponse(...)
```

**Features:**

- Lazy initialization with double-checked locking
- HTTP/2 connection pooling via `httpx`
- Proper resource cleanup with context manager
- Structured logging with request/response details

#### Prompts Manager (`prompts_manager.py`)

```python
class PromptsManager(IPromptsManager):
    """Thread-safe prompt loading without global cache."""

    def get_core_prompts(self) -> List[Prompt]:
        """Returns domain objects, not raw strings."""
        data = self._load_prompts()
        return [Prompt(text=t, category="core", index=i)
                for i, t in enumerate(data["core_prompts"])]
```

**Features:**

- Thread-safe instance-level caching
- Explicit `reload()` method for cache invalidation
- Returns immutable domain objects

#### Judge Service (`judge_service.py`)

```python
class JudgeService(IJudgeService):
    """Thread-safe evaluation service."""

    def evaluate(
        self,
        prompt: str,
        initial_response: ModelResponse,
        follow_response: ModelResponse,
        config: RunConfiguration,
    ) -> JudgeDecision:
        # Build messages, request judgment, parse response
        # Returns structured JudgeDecision with error handling
```

**Features:**

- Retry logic with exponential backoff for token limits
- Robust JSON parsing from noisy responses
- Error decisions instead of raising exceptions
- Structured logging

#### Configuration Manager (`config_manager.py`)

```python
class ConfigurationManager(IConfigurationManager):
    """Thread-safe configuration with YAML/JSON and env vars."""

    def get(self, key: str, default: Any = None) -> Any:
        """Supports dot notation: config.get('api.timeout')"""

    def set(self, key: str, value: Any) -> None:
        """Thread-safe updates with lock protection."""

    def merge(self, updates: Dict[str, Any]) -> None:
        """Deep merge configuration updates."""
```

**Features:**

- Dot notation for nested keys
- Environment variable overrides
- Deep merge capabilities
- Thread-safe singleton pattern

### Repository Layer (`repositories.py`)

Implements the Repository and Unit of Work patterns:

```python
class ArtifactsRepository(IArtifactsRepository):
    """Stores completion and judge artifacts as JSON."""

    def save_completion(
        self,
        model: str,
        prompt_index: int,
        step: str,
        data: Dict[str, Any],
    ) -> Path:
        """Returns path to saved artifact."""

class ResultsRepository(IResultsRepository):
    """Thread-safe CSV writer with buffering."""

    def add_result(self, row: Dict[str, Any]) -> None:
        """Append row to CSV."""

class UnitOfWork(IUnitOfWork):
    """Coordinates repositories in a transaction."""

    def __enter__(self) -> UnitOfWork:
        return self

    def __exit__(self, *args) -> None:
        if no_exception:
            self.commit()
        self.results.close()
```

### Factory Layer (`factories.py`)

Factory classes for complex object creation:

```python
class RunnerFactory:
    """Creates LLMJudgeRunner with DI."""

    def __init__(self, container: ServiceContainer):
        self._container = container

    def create_runner(
        self,
        config: RunnerConfig,
        control: Optional[RunnerControl] = None,
        progress_callback: Optional[Callable] = None,
    ) -> LLMJudgeRunner:
        # Resolve services from container
        api_client = self._container.resolve(IAPIClient)
        judge_service = self._container.resolve(IJudgeService)
        prompts_manager = self._container.resolve(IPromptsManager)

        return LLMJudgeRunner(
            config,
            api_client=api_client,
            judge_service=judge_service,
            prompts_manager=prompts_manager,
            control=control,
            progress_callback=progress_callback,
        )
```

## Design Patterns

### 1. **Dependency Injection Container**

The `ServiceContainer` manages service lifetimes and dependencies:

```python
# Register services
container.register_singleton(IAPIClient, lambda: OpenRouterClient(api_key))
container.register_factory(IUnitOfWork, lambda: create_unit_of_work())

# Resolve services
api_client = container.resolve(IAPIClient)
```

### 2. **Factory Pattern**

Factories encapsulate complex object creation:

```python
factory = RunnerFactory(container)
runner = factory.create_runner(config)
```

### 3. **Builder Pattern**

Fluent API for configuration:

```python
config = (
    ConfigurationBuilder()
    .with_models(["model-1", "model-2"])
    .with_judge_model("judge")
    .with_outdir(Path("./results"))
    .with_max_tokens(1000)
    .build()
)
```

### 4. **Repository Pattern**

Clean data access layer:

```python
artifacts_repo = ArtifactsRepository(run_dir, fs_service)
path = artifacts_repo.save_completion(model, idx, "initial", data)
```

### 5. **Unit of Work Pattern**

Transaction-like semantics:

```python
with UnitOfWork(...) as uow:
    uow.artifacts.save_completion(...)
    uow.results.add_result(...)
    uow.commit()  # Auto-commits on success
```

### 6. **Protocol Pattern**

Interface segregation:

```python
class IAPIClient(Protocol):
    def chat_completion(...) -> ModelResponse: ...

class IPromptsManager(Protocol):
    def get_core_prompts(self) -> List[Prompt]: ...
```

## Thread Safety

### Locking Strategy

All shared state uses `threading.RLock()` (reentrant locks):

```python
class ThreadSafeService:
    def __init__(self):
        self._lock = threading.RLock()
        self._cache = {}

    def get(self, key):
        with self._lock:
            return self._cache.get(key)

    def set(self, key, value):
        with self._lock:
            self._cache[key] = value
```

### Double-Checked Locking

For expensive initializations:

```python
def _ensure_client(self) -> OpenAI:
    with self._lock:
        if self._client is None:
            self._http_client = httpx.Client(...)
            self._client = OpenAI(...)
        return self._client
```

### Concurrent Access Guarantees

- **OpenRouterClient**: Safe for concurrent `chat_completion()` calls
- **PromptsManager**: Safe for concurrent `get_core_prompts()` and `reload()`
- **JudgeService**: Safe for concurrent `evaluate()` calls
- **ConfigurationManager**: Safe for concurrent reads and writes
- **Repositories**: Safe for concurrent saves and writes
- **UnitOfWork**: Each instance is isolated (not shared across threads)

## Dependency Injection

### Creating the Container

```python
from llm_judge.container import create_container

# With defaults
container = create_container()

# With custom config
container = create_container({"config_file": "path/to/config.yaml"})
```

### Resolving Services

```python
# Resolve by protocol interface
api_client = container.resolve(IAPIClient)
prompts_manager = container.resolve(IPromptsManager)
judge_service = container.resolve(IJudgeService)

# Cleanup when done
container.clear()
```

### Using with Webapp

```python
from llm_judge.webapp import create_app

# DI-based webapp
app = create_app(container=container)
```

## Migration Guide

### From Legacy API Client

**Before:**

```python
from llm_judge.api import openrouter_chat

response = openrouter_chat(
    model="test-model",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=100,
    temperature=0.7,
    metadata={"title": "Test"},
)
from llm_judge.infrastructure.utility_services import ResponseParser

text = ResponseParser().extract_text(response)
```

**After:**

```python
from llm_judge.container import create_container
from llm_judge.services import IAPIClient

container = create_container()
api_client = container.resolve(IAPIClient)

response = api_client.chat_completion(
    model="test-model",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=100,
    temperature=0.7,
    metadata={"title": "Test"},
)
text = response.text  # ModelResponse object
```

### From Legacy Prompts

**Before:**

```python
from llm_judge.prompts import CORE_PROMPTS, FOLLOW_UP

for prompt in CORE_PROMPTS:
    # Use prompt string
```

**After:**

```python
from llm_judge.container import create_container
from llm_judge.services import IPromptsManager

container = create_container()
prompts_manager = container.resolve(IPromptsManager)

prompts = prompts_manager.get_core_prompts()
for prompt in prompts:
    # Use prompt.text (Prompt domain object)

follow_up = prompts_manager.get_follow_up()
```

### From Legacy Judge

**Before:**

```python
from llm_judge.judging import judge_decide

result = judge_decide(
    judge_model="judge",
    prompt="Question",
    initial_resp="Answer 1",
    follow_resp="Answer 2",
    max_tokens=2000,
    temperature=0.0,
    meta={"title": "Test"},
)
ok = result["ok"]
decision = result["decision"]
```

**After:**

```python
from llm_judge.container import create_container
from llm_judge.services import IJudgeService
from llm_judge.domain import ModelResponse, RunConfiguration

container = create_container()
judge_service = container.resolve(IJudgeService)

config = RunConfiguration(
    models=["test"],
    judge_model="judge",
    outdir=Path("./results"),
    max_tokens=1000,
    judge_max_tokens=2000,
    temperature=0.7,
    judge_temperature=0.0,
    sleep_s=0.1,
)

initial = ModelResponse(text="Answer 1", raw_payload={})
follow = ModelResponse(text="Answer 2", raw_payload={})

decision = judge_service.evaluate(
    prompt="Question",
    initial_response=initial,
    follow_response=follow,
    config=config,
)

# decision is a JudgeDecision domain object
if decision.success:
    print(f"Completeness: {decision.initial_completeness}")
```

### From Legacy Runner

**Before:**

```python
from llm_judge.runner import run_suite

artifacts = run_suite(
    models=["model-1"],
    judge_model="judge",
    outdir=Path("./results"),
    max_tokens=1000,
    judge_max_tokens=2000,
    temperature=0.7,
    judge_temperature=0.0,
    sleep_s=0.1,
    limit=1,
    verbose=True,
)
```

**After:**

```python
from llm_judge.container import create_container
from llm_judge.factories import RunnerFactory
from llm_judge.runner import RunnerConfig

container = create_container()
factory = RunnerFactory(container)

config = RunnerConfig(
    models=["model-1"],
    judge_model="judge",
    outdir=Path("./results"),
    max_tokens=1000,
    judge_max_tokens=2000,
    temperature=0.7,
    judge_temperature=0.0,
    sleep_s=0.1,
    limit=1,
    verbose=True,
)

runner = factory.create_runner(config)
artifacts = runner.run()

container.clear()
```

## Usage Examples

### Complete Example with DI

```python
import os
from pathlib import Path
from llm_judge.container import create_container
from llm_judge.factories import RunnerFactory
from llm_judge.runner import RunnerConfig
from llm_judge.logging_config import configure_logging

# Setup logging
configure_logging(level="INFO", log_file=Path("llm-judge.log"))

# Set API key
os.environ['OPENROUTER_API_KEY'] = 'your-key-here'

# Create DI container
container = create_container()

# Create runner via factory
factory = RunnerFactory(container)
config = RunnerConfig(
    models=["qwen/qwen3-next-80b-a3b-instruct"],
    judge_model="x-ai/grok-4-fast",
    outdir=Path("./results"),
    max_tokens=1000,
    judge_max_tokens=2000,
    temperature=0.7,
    judge_temperature=0.0,
    sleep_s=0.5,
    limit=5,
    verbose=True,
    use_color=True,
)

# Run with error handling
from llm_judge.exceptions import APIError, JudgingError

try:
    runner = factory.create_runner(config)
    artifacts = runner.run()

    print(f"Results: {artifacts.csv_path}")
    print(f"Runs: {artifacts.runs_dir}")

except APIError as e:
    print(f"API Error: {e} (context: {e.context})")
except JudgingError as e:
    print(f"Judging Error: {e}")
finally:
    container.clear()
```

### Using with Unit of Work

```python
from llm_judge.infrastructure.repositories import UnitOfWork
from llm_judge.infrastructure.utility_services import FileSystemService

fs_service = FileSystemService()

with UnitOfWork(
    run_directory=Path("./run"),
    csv_path=Path("./results.csv"),
    csv_fieldnames=["model", "score"],
    fs_service=fs_service,
) as uow:
    # Save artifacts
    path1 = uow.artifacts.save_completion(
        model="test-model",
        prompt_index=0,
        step="initial",
        data={"content": "Test"},
    )

    # Add results
    uow.results.add_result({"model": "test-model", "score": "5"})

    # Commit (or auto-commits on __exit__ if no exception)
    uow.commit()
```

### Custom Configuration

```python
from llm_judge.infrastructure.config_manager import ConfigurationManager

# From file
config_manager = ConfigurationManager(Path("config.yaml"))

# Get values (supports dot notation)
timeout = config_manager.get("api.timeout", default=30)
models = config_manager.get("models", default=[])

# Set values
config_manager.set("api.max_retries", 3)

# Merge updates
config_manager.merge({
    "api": {"timeout": 60},
    "new_section": {"key": "value"}
})

# Reload from disk
config_manager.reload()
```

## Best Practices

### 1. **Always Use DI Container**

Don't instantiate services directly. Use the container:

```python
# ❌ Don't do this
client = OpenRouterClient(api_key="...")

# ✅ Do this
container = create_container()
client = container.resolve(IAPIClient)
```

### 2. **Clean Up Resources**

Always clear the container when done:

```python
try:
    container = create_container()
    # Use services
finally:
    container.clear()
```

### 3. **Use Context Managers**

For resources that need cleanup:

```python
# UnitOfWork
with UnitOfWork(...) as uow:
    # Work with repositories
    uow.commit()

# OpenRouterClient
with OpenRouterClient(...) as client:
    response = client.chat_completion(...)
```

### 4. **Handle Errors with Custom Exceptions**

Use the exception hierarchy:

```python
from llm_judge.exceptions import (
    APIError,
    JudgingError,
    ConfigurationError,
)

try:
    response = api_client.chat_completion(...)
except APIError as e:
    logger.error(f"API failed: {e.message}, context: {e.context}")
except JudgingError as e:
    logger.error(f"Judge failed: {e.message}")
```

### 5. **Use Structured Logging**

```python
from llm_judge.logging_config import configure_logging, get_logger

configure_logging(level="INFO")
logger = get_logger(__name__)

logger.info("Processing prompt", extra={
    "model": "test-model",
    "prompt_index": 0,
})
```

## Testing

The architecture supports easy testing via dependency injection:

```python
from unittest.mock import MagicMock

# Mock the API client
mock_client = MagicMock(spec=IAPIClient)
mock_client.chat_completion.return_value = ModelResponse(
    text="test", raw_payload={}
)

# Inject mock
judge_service = JudgeService(api_client=mock_client)

# Test without hitting real API
decision = judge_service.evaluate(...)
```

See `tests/test_integration.py` for comprehensive thread-safety tests.

## Conclusion

The refactored architecture provides:

- **Maintainability**: Clear separation of concerns
- **Testability**: Easy mocking via protocols
- **Thread-Safety**: Guaranteed concurrent access safety
- **Flexibility**: Easy to swap implementations
- **Performance**: Connection pooling and efficient caching
- **Reliability**: Structured error handling

All while maintaining **100% backward compatibility** with legacy code.
