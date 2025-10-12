# LLM Judge - Refactored Object-Oriented Architecture

## Overview

This document outlines the refactored architecture for the LLM Judge project, transforming it from a procedural design with global state to a fully object-oriented, thread-safe architecture following SOLID principles and design patterns.

## Core Design Principles

1. **Dependency Injection** - All dependencies are injected, no hard-coded imports
2. **Thread Safety** - All shared resources are properly synchronized
3. **SOLID Principles** - Single responsibility, open/closed, interface segregation
4. **No Global State** - Everything is encapsulated in classes
5. **Testability** - All components are easily mockable and testable

## Architecture Layers

### 1. Domain Layer
Pure business logic with no external dependencies.

```python
# src/llm_judge/domain/models.py
@dataclass
class Prompt:
    text: str
    category: str
    index: int

@dataclass
class JudgeDecision:
    initial_refusal: bool
    initial_completeness: float
    follow_refusal: bool
    follow_completeness: float
    asymmetry: str
    notes: str

@dataclass
class RunConfiguration:
    models: List[str]
    judge_model: str
    max_tokens: int
    temperature: float
    # ... other fields
```

### 2. Application Services Layer
Business logic orchestration with dependency injection.

```python
# src/llm_judge/services/interfaces.py
from typing import Protocol

class IAPIClient(Protocol):
    """Interface for API communication"""
    def chat_completion(self, model: str, messages: List[Dict], **kwargs) -> Dict:
        ...

class IPromptsManager(Protocol):
    """Interface for prompt management"""
    def get_core_prompts(self) -> List[Prompt]:
        ...
    def get_follow_up(self) -> str:
        ...

class IJudgeService(Protocol):
    """Interface for judging logic"""
    def evaluate(self, prompt: str, initial: str, follow: str) -> JudgeDecision:
        ...

class IConfigurationManager(Protocol):
    """Interface for configuration management"""
    def get(self, key: str) -> Any:
        ...
    def set(self, key: str, value: Any) -> None:
        ...
```

### 3. Infrastructure Layer
Concrete implementations with external dependencies.

```python
# src/llm_judge/infrastructure/api_client.py
class OpenRouterClient:
    """Thread-safe API client with connection pooling"""

    def __init__(self, api_key: str, base_url: str,
                 pool_size: int = 10, timeout: int = 120):
        self._api_key = api_key
        self._base_url = base_url
        self._lock = threading.RLock()
        self._pool = ConnectionPool(size=pool_size)
        self._timeout = timeout

    def chat_completion(self, model: str, messages: List[Dict], **kwargs) -> Dict:
        with self._lock:
            connection = self._pool.get_connection()
        try:
            return self._execute_request(connection, model, messages, **kwargs)
        finally:
            self._pool.release_connection(connection)

# src/llm_judge/infrastructure/prompts_manager.py
class PromptsManager:
    """Thread-safe prompts management"""

    def __init__(self, prompts_path: Path):
        self._lock = threading.RLock()
        self._prompts_cache: Optional[Dict] = None
        self._prompts_path = prompts_path

    def get_core_prompts(self) -> List[Prompt]:
        with self._lock:
            if self._prompts_cache is None:
                self._load_prompts()
            return self._prompts_cache['core_prompts']

# src/llm_judge/infrastructure/judge_service.py
class JudgeService:
    """Judge evaluation service"""

    def __init__(self, api_client: IAPIClient, config_manager: IConfigurationManager):
        self._api_client = api_client
        self._config = config_manager
        self._lock = threading.RLock()

    def evaluate(self, prompt: str, initial: str, follow: str) -> JudgeDecision:
        with self._lock:
            # Implementation here
            pass
```

### 4. Dependency Injection Container

```python
# src/llm_judge/container.py
class ServiceContainer:
    """Thread-safe dependency injection container"""

    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._lock = threading.RLock()

    def register_singleton(self, interface: Type, instance: Any) -> None:
        """Register a singleton service"""
        with self._lock:
            self._services[interface] = instance

    def register_factory(self, interface: Type, factory: Callable) -> None:
        """Register a factory for creating instances"""
        with self._lock:
            self._factories[interface] = factory

    def resolve(self, interface: Type) -> Any:
        """Resolve a service by interface"""
        with self._lock:
            if interface in self._services:
                return self._services[interface]
            if interface in self._factories:
                return self._factories[interface]()
            raise ValueError(f"No registration for {interface}")

# Usage
container = ServiceContainer()
container.register_singleton(IAPIClient, OpenRouterClient(api_key, base_url))
container.register_singleton(IPromptsManager, PromptsManager(prompts_path))
container.register_singleton(IJudgeService, JudgeService(
    container.resolve(IAPIClient),
    container.resolve(IConfigurationManager)
))
```

### 5. Repository Pattern for Data Access

```python
# src/llm_judge/repositories/interfaces.py
class IRunRepository(Protocol):
    """Interface for run data persistence"""
    def save_run(self, run: RunArtifacts) -> None:
        ...
    def get_run(self, run_id: str) -> Optional[RunArtifacts]:
        ...

# src/llm_judge/repositories/file_repository.py
class FileRunRepository:
    """File-based run repository"""

    def __init__(self, base_path: Path):
        self._base_path = base_path
        self._lock = threading.RLock()

    def save_run(self, run: RunArtifacts) -> None:
        with self._lock:
            # Save to JSON files
            pass
```

### 6. Unit of Work Pattern

```python
# src/llm_judge/uow.py
class UnitOfWork:
    """Manages transactions across repositories"""

    def __init__(self, run_repo: IRunRepository, result_repo: IResultRepository):
        self.runs = run_repo
        self.results = result_repo
        self._lock = threading.RLock()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.rollback()

    def commit(self):
        with self._lock:
            # Commit all changes
            pass

    def rollback(self):
        with self._lock:
            # Rollback changes
            pass
```

### 7. Factory Patterns

```python
# src/llm_judge/factories.py
class RunnerFactory:
    """Factory for creating configured runners"""

    def __init__(self, container: ServiceContainer):
        self._container = container

    def create_runner(self, config: RunnerConfig) -> LLMJudgeRunner:
        return LLMJudgeRunner(
            config=config,
            api_client=self._container.resolve(IAPIClient),
            judge_service=self._container.resolve(IJudgeService),
            prompts_manager=self._container.resolve(IPromptsManager),
            uow=self._container.resolve(UnitOfWork)
        )

class ConfigurationBuilder:
    """Builder pattern for complex configurations"""

    def __init__(self):
        self._config = {}

    def with_models(self, models: List[str]) -> 'ConfigurationBuilder':
        self._config['models'] = models
        return self

    def with_judge_model(self, model: str) -> 'ConfigurationBuilder':
        self._config['judge_model'] = model
        return self

    def build(self) -> RunConfiguration:
        return RunConfiguration(**self._config)
```

### 8. Web Application Integration

```python
# src/llm_judge/webapp/app_factory.py
class ApplicationFactory:
    """Factory for creating Flask applications with DI"""

    @staticmethod
    def create_app(container: ServiceContainer) -> Flask:
        app = Flask(__name__)

        # Store container in app context
        app.container = container

        # Register blueprints with injected dependencies
        api_bp = create_api_blueprint(container)
        app.register_blueprint(api_bp)

        return app

# src/llm_judge/webapp/blueprints.py
def create_api_blueprint(container: ServiceContainer) -> Blueprint:
    api = Blueprint('api', __name__)

    @api.route('/run', methods=['POST'])
    def start_run():
        runner_factory = container.resolve(RunnerFactory)
        config = request.get_json()
        runner = runner_factory.create_runner(config)
        # ... handle run

    return api
```

## Migration Strategy

### Phase 1: Create New Structure
1. Create domain models and interfaces
2. Implement infrastructure classes
3. Set up dependency injection container

### Phase 2: Refactor Core Modules
1. Replace global API client with OpenRouterClient class
2. Replace prompts module with PromptsManager
3. Replace judging functions with JudgeService

### Phase 3: Refactor Runner
1. Update LLMJudgeRunner to use dependency injection
2. Implement repository pattern for data persistence
3. Add unit of work for transactions

### Phase 4: Refactor Web Application
1. Replace global Flask app with ApplicationFactory
2. Inject dependencies into blueprints
3. Update JobManager with proper DI

### Phase 5: Testing & Documentation
1. Create comprehensive unit tests
2. Add integration tests for thread safety
3. Update documentation

## Benefits of New Architecture

1. **Thread Safety**: All shared resources properly synchronized
2. **Testability**: Easy to mock dependencies for unit testing
3. **Maintainability**: Clear separation of concerns
4. **Extensibility**: Easy to add new implementations
5. **No Global State**: Everything properly encapsulated
6. **SOLID Compliance**: Follows all SOLID principles
7. **Design Patterns**: Proper use of established patterns
8. **Type Safety**: Full type hints and protocols
