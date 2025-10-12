# LLM Judge Refactoring Status

## Overview

This document tracks the progress of refactoring the LLM Judge codebase from a procedural design with global state to a fully object-oriented, thread-safe architecture following SOLID principles.

## Completed Work

### 1. Architecture Design ✅
- Created comprehensive architecture documentation in `ARCHITECTURE.md`
- Created detailed implementation plan in `REFACTORING_PLAN.md`
- Defined clear separation between domain, services, and infrastructure layers

### 2. Domain Models ✅
**Location:** `src/llm_judge/domain/__init__.py`

Created immutable domain models using dataclasses:
- `Prompt` - Represents a test prompt
- `ModelResponse` - Encapsulates API response data
- `JudgeDecision` - Contains evaluation results (updated with success/error fields)
- `RunConfiguration` - Immutable configuration object

**Benefits:**
- Type-safe data structures
- Immutable by design (frozen dataclasses)
- Clear separation from infrastructure concerns

### 3. Service Interfaces ✅
**Location:** `src/llm_judge/services/__init__.py`

Created Protocol-based interfaces for dependency inversion:
- `IAPIClient` - Interface for API communication
- `IPromptsManager` - Interface for prompt management
- `IJudgeService` - Interface for judging logic
- `IConfigurationManager` - Interface for configuration

**Benefits:**
- Enables dependency injection
- Easy to mock for testing
- Follows dependency inversion principle

### 4. Infrastructure Implementations ✅

#### OpenRouterClient ✅
**Location:** `src/llm_judge/infrastructure/api_client.py`

**Key Features:**
- Thread-safe with `threading.RLock()`
- Lazy client initialization
- HTTP/2 connection pooling via httpx
- Context manager support (`__enter__`/`__exit__`)
- Proper resource cleanup in `close()` method
- No global state

**Eliminated Global Variables:**
```python
# OLD (api.py):
_http_client: httpx.Client | None = None
_client: OpenAI | None = None

# NEW:
# All state encapsulated in instance variables with proper locking
self._client: Optional[OpenAI] = None
self._http_client: Optional[httpx.Client] = None
self._lock = threading.RLock()
```

#### PromptsManager ✅
**Location:** `src/llm_judge/infrastructure/prompts_manager.py`

**Key Features:**
- Thread-safe caching with `threading.RLock()`
- Lazy loading of YAML configuration
- Returns domain objects (Prompt) not raw data
- No `@lru_cache` decorator (no hidden global state)
- Explicit reload() method for cache invalidation

**Eliminated Global Variables:**
```python
# OLD (prompts.py):
@lru_cache(maxsize=1)
def _load_prompts() -> Dict[str, Any]:
    # Hidden global cache

# NEW:
# Instance-level cache with explicit thread safety
self._prompts_cache: Optional[Dict[str, Any]] = None
self._lock = threading.RLock()
```

#### JudgeService ✅
**Location:** `src/llm_judge/infrastructure/judge_service.py`

**Key Features:**
- Dependency injection (receives IAPIClient)
- Thread-safe configuration caching
- Retry logic with exponential token increase
- JSON parsing with error handling
- Returns domain objects (JudgeDecision)

**Eliminated Global Variables:**
```python
# OLD (judging.py):
@lru_cache(maxsize=1)
def _load_judge_config() -> Dict[str, Any]:
    # Hidden global cache
_CONFIG = _load_judge_config()
JUDGE_SYSTEM = _expect_str(_CONFIG, "system")
JUDGE_SCHEMA = _expect_mapping(_CONFIG, "schema")

# NEW:
# Instance-level cache with dependency injection
self._config_cache: Optional[Dict[str, Any]] = None
self._api_client = api_client  # Injected dependency
self._lock = threading.RLock()
```

### 5. Dependency Injection Container ✅
**Location:** `src/llm_judge/container.py`

**Key Features:**
- Thread-safe service registration and resolution
- Supports singleton and factory patterns
- Automatic resource cleanup
- Type-safe with generics

**Usage:**
```python
from llm_judge.container import create_container

# Create container with configuration
container = create_container({
    "api_key": "your-key-here"
})

# Resolve services
api_client = container.resolve(IAPIClient)
prompts_manager = container.resolve(IPromptsManager)
judge_service = container.resolve(IJudgeService)

# Cleanup
container.clear()  # Closes all resources
```

### 6. Runner Refactoring ✅
**Location:** `src/llm_judge/runner.py`

**Key Changes:**
- Accepts injected dependencies (IAPIClient, IJudgeService, IPromptsManager)
- Maintains backward compatibility with legacy function-based approach
- Uses domain objects throughout (Prompt, JudgeDecision)
- No direct imports of infrastructure (uses DI)

**Dual Mode Operation:**
```python
# New DI-based approach
runner = LLMJudgeRunner(
    config=config,
    api_client=container.resolve(IAPIClient),
    judge_service=container.resolve(IJudgeService),
    prompts_manager=container.resolve(IPromptsManager),
)

# Legacy approach (still works)
runner = LLMJudgeRunner(config=config)  # Uses old functions internally
```

### 7. Factory Patterns ✅
**Location:** `src/llm_judge/factories.py`

Created two factory classes:

#### RunnerFactory ✅
- Creates fully configured runners with dependency injection
- Resolves services from container
- Simplifies runner instantiation

**Usage:**
```python
factory = RunnerFactory(container)
runner = factory.create_runner(
    config=runner_config,
    control=control_hooks,
    progress_callback=callback_fn,
)
```

#### ConfigurationBuilder ✅
- Fluent API for building configurations
- Validates all parameters
- Prevents invalid configurations

**Usage:**
```python
config = (
    ConfigurationBuilder()
    .with_models(["model-a", "model-b"])
    .with_judge_model("judge-model")
    .with_outdir(Path("./results"))
    .with_temperature(0.7)
    .with_verbose()
    .build()
)
```

## Architecture Benefits Achieved

### Thread Safety ✅
- All shared resources protected by `threading.RLock()`
- No global mutable state
- Lazy initialization within locks
- Proper resource cleanup

### SOLID Principles ✅
- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: Easy to extend via interfaces
- **Liskov Substitution**: Protocol-based interfaces
- **Interface Segregation**: Minimal, focused interfaces
- **Dependency Inversion**: Depend on abstractions (Protocols)

### Testability ✅
- All dependencies injected
- Easy to mock via Protocol interfaces
- No hidden global state
- Isolated components

## Remaining Work

### 8. Configuration Management
**Priority:** Medium

Create `ConfigurationManager` implementing `IConfigurationManager`:
- Load from files, environment, defaults
- Thread-safe access
- Validation
- Hot reload support

### 9. Webapp Module Refactoring
**Priority:** High

Update Flask application to:
- Replace global `app` instance with `create_app()` factory
- Inject ServiceContainer into blueprints
- Use container to resolve dependencies
- Maintain compatibility with existing endpoints

**Current State:**
- JobManager uses Runner directly
- SSEBroker is already well-designed
- Need to wire up container

### 10. Utils Module Refactoring
**Priority:** Low

Current `utils.py` contains standalone functions. Consider:
- Keep simple utilities as pure functions (now_iso, detect_refusal)
- Create service classes for complex utilities if needed
- These utilities are largely stateless and work well as-is

### 11. Unit of Work Pattern
**Priority:** Low

For data persistence operations:
- Coordinate CSV writing and JSON saving
- Transaction-like behavior
- Rollback on errors

### 12. Error Handling Architecture
**Priority:** Medium

Standardize error handling:
- Custom exception hierarchy
- Consistent logging
- Error recovery strategies

### 13. Integration Tests
**Priority:** High

Create tests to verify:
- Thread safety under concurrent load
- Resource cleanup
- Container lifecycle
- End-to-end flows

### 14. Documentation Updates
**Priority:** High

Update documentation to reflect:
- New architecture patterns
- Migration guide for existing code
- API usage examples
- Best practices

## Migration Strategy

### Phase 1: Gradual Introduction ✅
- New modules created alongside old code
- No breaking changes to existing API
- Can be tested independently

### Phase 2: Update Runner ✅
- Modified runner to use new services
- Kept old function signatures for compatibility
- Both old and new approaches work

### Phase 3: Update Webapp (Next)
- Refactor Flask app to use container
- Maintain endpoint compatibility
- Test with existing UI

### Phase 4: Deprecate Old Code
- Mark old modules as deprecated
- Provide migration guide
- Set removal timeline

### Phase 5: Remove Old Code
- Remove deprecated modules
- Clean up imports
- Final documentation update

## Testing the New Architecture

### Basic Usage Example

```python
import os
from pathlib import Path
from llm_judge.container import create_container
from llm_judge.factories import RunnerFactory, ConfigurationBuilder
from llm_judge.runner import RunnerConfig

# Set API key
os.environ['OPENROUTER_API_KEY'] = 'your-key'

# Create container
container = create_container()

# Build configuration
config = (
    ConfigurationBuilder()
    .with_models(["qwen/qwen3-next-80b-a3b-instruct"])
    .with_judge_model("x-ai/grok-4-fast")
    .with_outdir(Path("./results"))
    .with_max_tokens(1000)
    .with_judge_max_tokens(2000)
    .with_temperature(0.7)
    .with_limit(1)  # Test with just one prompt
    .build()
)

# Create runner using factory
factory = RunnerFactory(container)
runner = factory.create_runner(
    config=RunnerConfig(
        models=config.models,
        judge_model=config.judge_model,
        outdir=config.outdir,
        max_tokens=config.max_tokens,
        judge_max_tokens=config.judge_max_tokens,
        temperature=config.temperature,
        judge_temperature=config.judge_temperature,
        sleep_s=config.sleep_s,
        limit=config.limit,
        verbose=config.verbose,
        use_color=config.use_color,
    )
)

# Run evaluation
artifacts = runner.run()

# Cleanup
container.clear()
```

### Thread Safety Test

```python
import threading
from llm_judge.container import create_container

def worker(worker_id):
    # Each thread gets same shared services (singletons)
    container = create_container()
    api_client = container.resolve(IAPIClient)
    # All threads safely share the same client

# Multiple threads
threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

## Key Improvements Summary

### Before
- Global `_client` and `_http_client` in api.py
- Global `@lru_cache` in prompts.py and judging.py
- Global Flask `app` instance
- No dependency injection
- Hard to test
- Thread-safety unclear

### After
- All state encapsulated in classes
- Explicit thread safety with `threading.RLock()`
- Dependency injection throughout
- Clear interfaces via Protocols
- Easy to test with mocks
- Resource management with context managers
- Factory patterns for object creation
- Builder patterns for configuration

## Files Created/Modified

### New Files
- `src/llm_judge/domain/__init__.py` - Domain models
- `src/llm_judge/services/__init__.py` - Service interfaces
- `src/llm_judge/infrastructure/__init__.py` - Infrastructure exports
- `src/llm_judge/infrastructure/api_client.py` - OpenRouterClient
- `src/llm_judge/infrastructure/prompts_manager.py` - PromptsManager
- `src/llm_judge/infrastructure/judge_service.py` - JudgeService
- `src/llm_judge/container.py` - DI Container
- `src/llm_judge/factories.py` - Factory classes
- `ARCHITECTURE.md` - Architecture documentation
- `REFACTORING_PLAN.md` - Implementation plan
- `REFACTORING_STATUS.md` - This file

### Modified Files
- `src/llm_judge/runner.py` - Added DI support while maintaining backward compatibility

### Unchanged (Legacy Compatibility)
- `src/llm_judge/api.py` - Still works for legacy code
- `src/llm_judge/prompts.py` - Still works for legacy code
- `src/llm_judge/judging.py` - Still works for legacy code
- `src/llm_judge/utils.py` - Stateless utilities, no changes needed yet
- `src/llm_judge/webapp/*` - To be refactored next

## Next Steps

1. ✅ Complete infrastructure layer (API, Prompts, Judge services)
2. ✅ Refactor Runner to use dependency injection
3. ✅ Create factory patterns
4. 🔄 Implement ConfigurationManager
5. 🔄 Refactor Webapp to use ApplicationFactory pattern
6. 📝 Write comprehensive tests for thread safety
7. 📝 Create migration guide for existing code
8. 📝 Update all documentation

## Questions or Issues?

For questions about the refactoring:
1. Review ARCHITECTURE.md for design decisions
2. Check REFACTORING_PLAN.md for implementation details
3. See code comments in new modules for specifics

---

*Last Updated: 2025-10-12*
*Status: Core Infrastructure & Runner Complete, Webapp Integration Next*
