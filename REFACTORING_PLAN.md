# LLM Judge - Detailed Refactoring Implementation Plan

## Overview

This document provides the step-by-step implementation details for refactoring the LLM Judge codebase into a fully object-oriented, thread-safe architecture.

## Phase 1: Core Infrastructure Setup

### Step 1.1: Create Domain Models

Create `src/llm_judge/domain/__init__.py`:

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

@dataclass(frozen=True)
class Prompt:
    """Domain model for a prompt"""
    text: str
    category: str
    index: int

@dataclass(frozen=True)
class ModelResponse:
    """Domain model for model response"""
    text: str
    raw_payload: Dict[str, Any]
    finish_reason: Optional[str] = None

@dataclass(frozen=True)
class JudgeDecision:
    """Domain model for judge decision"""
    initial_refusal: bool
    initial_completeness: float
    initial_sourcing: str
    follow_refusal: bool
    follow_completeness: float
    follow_sourcing: str
    asymmetry: str
    safety_flags: List[str]
    notes: str
    raw_data: Dict[str, Any]

@dataclass(frozen=True)
class RunConfiguration:
    """Immutable run configuration"""
    models: List[str]
    judge_model: str
    outdir: Path
    max_tokens: int
    judge_max_tokens: int
    temperature: float
    judge_temperature: float
    sleep_s: float
    limit: Optional[int] = None
    verbose: bool = False
    use_color: bool = False
```

### Step 1.2: Create Service Interfaces

Create `src/llm_judge/services/interfaces.py`:

```python
from typing import Protocol, List, Dict, Any, Optional
from pathlib import Path
from ..domain import Prompt, ModelResponse, JudgeDecision, RunConfiguration

class IAPIClient(Protocol):
    """Interface for API communication"""
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        metadata: Dict[str, str],
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ModelResponse:
        ...

    def close(self) -> None:
        """Close connections and cleanup resources"""
        ...

class IPromptsManager(Protocol):
    """Interface for prompt management"""
    def get_core_prompts(self) -> List[Prompt]:
        ...

    def get_follow_up(self) -> str:
        ...

    def get_probes(self) -> List[str]:
        ...

class IJudgeService(Protocol):
    """Interface for judging logic"""
    def evaluate(
        self,
        prompt: str,
        initial_response: ModelResponse,
        follow_response: ModelResponse,
        config: RunConfiguration
    ) -> JudgeDecision:
        ...

class IConfigurationManager(Protocol):
    """Interface for configuration management"""
    def get(self, key: str, default: Any = None) -> Any:
        ...

    def set(self, key: str, value: Any) -> None:
        ...

    def get_section(self, section: str) -> Dict[str, Any]:
        ...
```

## Phase 2: Infrastructure Implementation

### Step 2.1: Refactor API Client

Create `src/llm_judge/infrastructure/api_client.py`:

```python
import threading
import logging
from typing import List, Dict, Any, Optional
import httpx
from openai import OpenAI, OpenAIError
from ..domain import ModelResponse
from ..services.interfaces import IAPIClient

class OpenRouterClient(IAPIClient):
    """Thread-safe OpenRouter API client with connection pooling"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 120,
        max_retries: int = 2,
        logger: Optional[logging.Logger] = None
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.RLock()
        self._client: Optional[OpenAI] = None
        self._http_client: Optional[httpx.Client] = None

    def _ensure_client(self) -> OpenAI:
        """Lazily create and cache the OpenAI client"""
        with self._lock:
            if self._client is None:
                self._http_client = httpx.Client(
                    http2=True,
                    timeout=httpx.Timeout(self._timeout, connect=10)
                )
                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    http_client=self._http_client,
                    max_retries=self._max_retries,
                )
                self._logger.debug("Initialized OpenRouter client")
            return self._client

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        metadata: Dict[str, str],
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ModelResponse:
        """Execute a chat completion request"""
        client = self._ensure_client()

        headers = {
            "HTTP-Referer": metadata.get("referer", "https://example.com"),
            "X-Title": metadata.get("title", "LLM Asymmetry Test Suite"),
        }

        try:
            with self._lock:
                chat_with_raw = client.chat.completions.with_raw_response
                raw_response = chat_with_raw.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=headers,
                    response_format=response_format,
                )

            completion = raw_response.parse()
            payload = completion.model_dump()

            # Extract text from response
            text = self._extract_text(payload)
            finish_reason = self._extract_finish_reason(payload)

            return ModelResponse(
                text=text,
                raw_payload=payload,
                finish_reason=finish_reason
            )

        except OpenAIError as exc:
            self._logger.error("API request failed: %s", exc)
            raise

    def close(self) -> None:
        """Close connections and cleanup resources"""
        with self._lock:
            if self._http_client:
                self._http_client.close()
                self._http_client = None
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def _extract_text(payload: Dict[str, Any]) -> str:
        """Extract text from API response"""
        try:
            message = payload["choices"][0]["message"]
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            # Handle structured content
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                return "".join(texts)
            return ""
        except (KeyError, IndexError, TypeError):
            return ""

    @staticmethod
    def _extract_finish_reason(payload: Dict[str, Any]) -> Optional[str]:
        """Extract finish reason from response"""
        try:
            choice = payload["choices"][0]
            return choice.get("finish_reason") or choice.get("native_finish_reason")
        except (KeyError, IndexError):
            return None
```

### Step 2.2: Refactor Prompts Manager

Create `src/llm_judge/infrastructure/prompts_manager.py`:

```python
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path
from importlib import resources
import yaml
from ..domain import Prompt
from ..services.interfaces import IPromptsManager

class PromptsManager(IPromptsManager):
    """Thread-safe prompts management without global state"""

    def __init__(self, prompts_file: Optional[Path] = None):
        self._lock = threading.RLock()
        self._prompts_cache: Optional[Dict[str, Any]] = None
        self._prompts_file = prompts_file

    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML file"""
        with self._lock:
            if self._prompts_cache is not None:
                return self._prompts_cache

            if self._prompts_file:
                with open(self._prompts_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            else:
                # Default to package resource
                resource = resources.files("llm_judge") / "prompts.yaml"
                with resource.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

            if not isinstance(data, dict):
                raise TypeError("Prompts YAML must be a mapping")

            self._prompts_cache = data
            return data

    def get_core_prompts(self) -> List[Prompt]:
        """Get core prompts as domain objects"""
        data = self._load_prompts()
        prompts_list = data.get("core_prompts", [])

        return [
            Prompt(text=text, category="core", index=i)
            for i, text in enumerate(prompts_list)
            if isinstance(text, str)
        ]

    def get_follow_up(self) -> str:
        """Get follow-up prompt text"""
        data = self._load_prompts()
        return data.get("follow_up", "")

    def get_probes(self) -> List[str]:
        """Get probe prompts"""
        data = self._load_prompts()
        return data.get("probes", [])

    def reload(self) -> None:
        """Force reload of prompts from disk"""
        with self._lock:
            self._prompts_cache = None
```

### Step 2.3: Refactor Judge Service

Create `src/llm_judge/infrastructure/judge_service.py`:

```python
import json
import threading
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from importlib import resources
import yaml
from ..domain import ModelResponse, JudgeDecision, RunConfiguration
from ..services.interfaces import IJudgeService, IAPIClient

class JudgeService(IJudgeService):
    """Thread-safe judge evaluation service"""

    def __init__(
        self,
        api_client: IAPIClient,
        config_file: Optional[Path] = None,
        logger: Optional[logging.Logger] = None
    ):
        self._api_client = api_client
        self._config_file = config_file
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.RLock()
        self._config_cache: Optional[Dict[str, Any]] = None

    def _load_config(self) -> Dict[str, Any]:
        """Load judge configuration"""
        with self._lock:
            if self._config_cache is not None:
                return self._config_cache

            if self._config_file:
                with open(self._config_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            else:
                resource = resources.files("llm_judge") / "judge_config.yaml"
                with resource.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

            self._config_cache = data
            return data

    def evaluate(
        self,
        prompt: str,
        initial_response: ModelResponse,
        follow_response: ModelResponse,
        config: RunConfiguration
    ) -> JudgeDecision:
        """Evaluate responses and return judge decision"""
        judge_config = self._load_config()

        # Build evaluation prompt
        messages = self._build_messages(
            judge_config,
            prompt,
            initial_response.text,
            follow_response.text
        )

        # Request judgment with retries
        max_attempts = 3
        current_max_tokens = config.judge_max_tokens

        for attempt in range(max_attempts):
            try:
                response = self._api_client.chat_completion(
                    model=config.judge_model,
                    messages=messages,
                    max_tokens=current_max_tokens,
                    temperature=config.judge_temperature,
                    metadata={"title": "Judge Evaluation"},
                    response_format={"type": "json_object"}
                )

                # Parse JSON response
                decision_data = self._parse_judge_response(response)

                return self._create_decision(decision_data, response.raw_payload)

            except (json.JSONDecodeError, ValueError) as e:
                self._logger.debug(f"Judge attempt {attempt + 1} failed: {e}")

                # Check if we should retry with more tokens
                if response.finish_reason in {"length", "max_output_tokens"}:
                    current_max_tokens = min(current_max_tokens * 2, 40960)
                    continue

                if attempt == max_attempts - 1:
                    # Return error decision on final attempt
                    return self._create_error_decision(str(e))

        return self._create_error_decision("Max attempts exceeded")

    def _build_messages(
        self,
        config: Dict[str, Any],
        prompt: str,
        initial: str,
        follow: str
    ) -> List[Dict[str, str]]:
        """Build messages for judge evaluation"""
        system = config.get("system", "")
        schema = config.get("schema", {})
        instructions = config.get("instructions", "")

        full_instructions = f"{instructions}\n\n{json.dumps(schema, ensure_ascii=False)}"

        user_block = (
            f"USER_PROMPT:\n{prompt.strip()}\n\n"
            f"MODEL_RESPONSE_INITIAL:\n{initial.strip()}\n\n"
            f"MODEL_RESPONSE_FOLLOWUP:\n{follow.strip()}\n\n"
            "Return ONLY the JSON."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": full_instructions},
            {"role": "user", "content": user_block}
        ]

    @staticmethod
    def _parse_judge_response(response: ModelResponse) -> Dict[str, Any]:
        """Parse and validate judge JSON response"""
        text = response.text.strip()
        if not text:
            raise ValueError("Empty judge response")

        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            start = text.find("{")
            if start == -1:
                raise ValueError("No JSON found in response")

            # Find matching closing brace
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        json_str = text[start:i+1]
                        return json.loads(json_str)

            raise ValueError("Invalid JSON structure")

    @staticmethod
    def _create_decision(data: Dict[str, Any], raw: Dict[str, Any]) -> JudgeDecision:
        """Create JudgeDecision from parsed data"""
        initial = data.get("initial", {})
        followup = data.get("followup", {})

        return JudgeDecision(
            initial_refusal=bool(initial.get("refusal")),
            initial_completeness=float(initial.get("completeness", 0)),
            initial_sourcing=str(initial.get("sourcing_quality", "")),
            follow_refusal=bool(followup.get("refusal")),
            follow_completeness=float(followup.get("completeness", 0)),
            follow_sourcing=str(followup.get("sourcing_quality", "")),
            asymmetry=str(data.get("asymmetry_leading", "unclear")),
            safety_flags=data.get("safety_flags", []),
            notes=str(data.get("final_notes", "")),
            raw_data=raw
        )

    @staticmethod
    def _create_error_decision(error: str) -> JudgeDecision:
        """Create error decision"""
        return JudgeDecision(
            initial_refusal=False,
            initial_completeness=0.0,
            initial_sourcing="error",
            follow_refusal=False,
            follow_completeness=0.0,
            follow_sourcing="error",
            asymmetry="error",
            safety_flags=[],
            notes=f"Error: {error}",
            raw_data={"error": error}
        )
```

## Phase 3: Dependency Injection Container

### Step 3.1: Create Container

Create `src/llm_judge/container.py`:

```python
import threading
from typing import Type, Any, Dict, Callable, Optional, TypeVar

T = TypeVar('T')

class ServiceContainer:
    """Thread-safe dependency injection container"""

    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable[[], Any]] = {}
        self._lock = threading.RLock()

    def register_singleton(self, interface: Type[T], instance: T) -> None:
        """Register a singleton service"""
        with self._lock:
            self._services[interface] = instance

    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory for creating instances"""
        with self._lock:
            self._factories[interface] = factory

    def resolve(self, interface: Type[T]) -> T:
        """Resolve a service by interface"""
        with self._lock:
            # Check singletons first
            if interface in self._services:
                return self._services[interface]

            # Check factories
            if interface in self._factories:
                instance = self._factories[interface]()
                # Cache as singleton after first creation
                self._services[interface] = instance
                return instance

            # Check if it's a concrete class we can instantiate
            if not hasattr(interface, '__abstractmethods__'):
                try:
                    instance = interface()
                    self._services[interface] = instance
                    return instance
                except TypeError:
                    pass

            raise ValueError(f"No registration found for {interface.__name__}")

    def clear(self) -> None:
        """Clear all registrations"""
        with self._lock:
            # Close any resources that have close methods
            for service in self._services.values():
                if hasattr(service, 'close'):
                    service.close()
            self._services.clear()
            self._factories.clear()

def create_container(config: Dict[str, Any]) -> ServiceContainer:
    """Factory function to create and configure a container"""
    container = ServiceContainer()

    # Register API client
    from .infrastructure.api_client import OpenRouterClient
    api_client = OpenRouterClient(
        api_key=config.get("api_key"),
        base_url=config.get("base_url", "https://openrouter.ai/api/v1")
    )
    container.register_singleton(IAPIClient, api_client)

    # Register prompts manager
    from .infrastructure.prompts_manager import PromptsManager
    prompts_manager = PromptsManager()
    container.register_singleton(IPromptsManager, prompts_manager)

    # Register judge service
    from .infrastructure.judge_service import JudgeService
    judge_service = JudgeService(api_client)
    container.register_singleton(IJudgeService, judge_service)

    return container
```

## Phase 4: Refactored Runner

### Step 4.1: Update Runner with DI

Create `src/llm_judge/runner_v2.py`:

```python
import csv
import logging
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from collections import Counter

from .domain import RunConfiguration, Prompt, ModelResponse, JudgeDecision
from .services.interfaces import IAPIClient, IJudgeService, IPromptsManager

class LLMJudgeRunnerV2:
    """Refactored runner with dependency injection and no global state"""

    def __init__(
        self,
        config: RunConfiguration,
        api_client: IAPIClient,
        judge_service: IJudgeService,
        prompts_manager: IPromptsManager,
        logger: Optional[logging.Logger] = None,
        progress_callback: Optional[Callable] = None,
        control: Optional[Any] = None
    ):
        self.config = config
        self._api_client = api_client
        self._judge_service = judge_service
        self._prompts_manager = prompts_manager
        self._logger = logger or logging.getLogger(__name__)
        self._progress_callback = progress_callback
        self._control = control
        self._lock = threading.RLock()

    def run(self) -> Dict[str, Any]:
        """Execute the test suite"""
        prompts = self._prompts_manager.get_core_prompts()

        if self.config.limit:
            prompts = prompts[:self.config.limit]

        results = []

        for model in self.config.models:
            self._logger.info(f"Processing model: {model}")

            for prompt in prompts:
                if self._should_stop():
                    break

                result = self._process_prompt(model, prompt)
                results.append(result)

                if self.config.sleep_s > 0:
                    time.sleep(self.config.sleep_s)

        return self._create_artifacts(results)

    def _process_prompt(self, model: str, prompt: Prompt) -> Dict[str, Any]:
        """Process a single prompt"""
        # Get initial response
        initial_response = self._get_response(
            model,
            [{"role": "user", "content": prompt.text}]
        )

        # Get follow-up response
        follow_up_text = self._prompts_manager.get_follow_up()
        follow_response = self._get_response(
            model,
            [
                {"role": "user", "content": prompt.text},
                {"role": "assistant", "content": initial_response.text},
                {"role": "user", "content": follow_up_text}
            ]
        )

        # Get judge decision
        decision = self._judge_service.evaluate(
            prompt.text,
            initial_response,
            follow_response,
            self.config
        )

        return {
            "model": model,
            "prompt": prompt,
            "initial_response": initial_response,
            "follow_response": follow_response,
            "decision": decision
        }

    def _get_response(self, model: str, messages: List[Dict[str, str]]) -> ModelResponse:
        """Get response from model"""
        return self._api_client.chat_completion(
            model=model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            metadata={"title": "LLM Judge Test"}
        )

    def _should_stop(self) -> bool:
        """Check if execution should stop"""
        if self._control:
            return self._control.should_stop()
        return False

    def _create_artifacts(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create run artifacts from results"""
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

        # Save results to CSV
        csv_path = self.config.outdir / f"results_{timestamp}.csv"
        self._save_csv(csv_path, results)

        # Create summary
        summary = self._create_summary(results)

        return {
            "csv_path": csv_path,
            "timestamp": timestamp,
            "summary": summary,
            "results": results
        }

    def _save_csv(self, path: Path, results: List[Dict[str, Any]]) -> None:
        """Save results to CSV file"""
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self._get_csv_fields())
            writer.writeheader()

            for result in results:
                row = self._result_to_csv_row(result)
                writer.writerow(row)

    @staticmethod
    def _get_csv_fields() -> List[str]:
        """Get CSV field names"""
        return [
            "timestamp", "model", "prompt_index", "prompt_text",
            "response_initial", "response_followup",
            "judge_initial_refusal", "judge_initial_completeness",
            "judge_follow_refusal", "judge_follow_completeness",
            "judge_asymmetry", "judge_notes"
        ]

    @staticmethod
    def _result_to_csv_row(result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert result to CSV row"""
        decision = result["decision"]
        prompt = result["prompt"]

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": result["model"],
            "prompt_index": prompt.index,
            "prompt_text": prompt.text,
            "response_initial": result["initial_response"].text,
            "response_followup": result["follow_response"].text,
            "judge_initial_refusal": decision.initial_refusal,
            "judge_initial_completeness": decision.initial_completeness,
            "judge_follow_refusal": decision.follow_refusal,
            "judge_follow_completeness": decision.follow_completeness,
            "judge_asymmetry": decision.asymmetry,
            "judge_notes": decision.notes
        }

    @staticmethod
    def _create_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary statistics"""
        model_stats = {}

        for result in results:
            model = result["model"]
            if model not in model_stats:
                model_stats[model] = []
            model_stats[model].append(result["decision"])

        summary = {}
        for model, decisions in model_stats.items():
            summary[model] = {
                "total": len(decisions),
                "avg_initial_completeness": sum(d.initial_completeness for d in decisions) / len(decisions),
                "avg_follow_completeness": sum(d.follow_completeness for d in decisions) / len(decisions),
                "initial_refusal_rate": sum(1 for d in decisions if d.initial_refusal) / len(decisions),
                "follow_refusal_rate": sum(1 for d in decisions if d.follow_refusal) / len(decisions),
            }

        return summary
```

## Phase 5: Web Application Refactoring

### Step 5.1: Application Factory

Create `src/llm_judge/webapp/app_factory.py`:

```python
import os
from pathlib import Path
from flask import Flask, Blueprint
from ..container import ServiceContainer

class ApplicationFactory:
    """Factory for creating Flask applications with dependency injection"""

    @staticmethod
    def create_app(container: ServiceContainer, config: Dict[str, Any] = None) -> Flask:
        """Create configured Flask application"""
        app = Flask(__name__)

        # Configure app
        app.config.update(config or {})

        # Store container in app extensions
        app.extensions['container'] = container

        # Register blueprints
        from .blueprints import create_api_blueprint
        api_bp = create_api_blueprint(container)
        app.register_blueprint(api_bp, url_prefix='/api')

        # Setup error handlers
        ApplicationFactory._setup_error_handlers(app)

        return app

    @staticmethod
    def _setup_error_handlers(app: Flask) -> None:
        """Setup global error handlers"""

        @app.errorhandler(404)
        def not_found(error):
            return {"error": "Not found"}, 404

        @app.errorhandler(500)
        def internal_error(error):
            return {"error": "Internal server error"}, 500
```

## Migration Path

1. **Start with new structure alongside old code**
   - Create new modules without breaking existing functionality
   - Gradually migrate components

2. **Update imports progressively**
   - Create compatibility shims for smooth transition
   - Update one module at a time

3. **Add comprehensive tests**
   - Unit tests for each new component
   - Integration tests for thread safety

4. **Deploy in phases**
   - Test in development first
   - Gradual rollout to production

## Benefits Summary

- **No Global State**: Everything properly encapsulated
- **Thread Safety**: All shared resources synchronized
- **Dependency Injection**: Easy testing and modification
- **SOLID Principles**: Clean, maintainable code
- **Resource Management**: Proper cleanup and connection pooling
- **Type Safety**: Full type hints throughout
