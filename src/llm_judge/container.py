"""Thread-safe dependency injection container."""

import os
import threading
from typing import Type, Any, Dict, Callable, TypeVar, Optional

T = TypeVar("T")


class ServiceContainer:
    """Thread-safe dependency injection container."""

    def __init__(self):
        self._services: Dict[Type[Any], Any] = {}
        self._factories: Dict[Type[Any], Callable[[], Any]] = {}
        self._lock = threading.RLock()

    def register_singleton(self, interface: Type[T], instance: T) -> None:
        """Register a singleton service."""
        with self._lock:
            self._services[interface] = instance

    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory for creating instances."""
        with self._lock:
            self._factories[interface] = factory

    def resolve(self, interface: Type[T]) -> T:
        """Resolve a service by interface."""
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
            if not hasattr(interface, "__abstractmethods__"):
                try:
                    instance = interface()  # type: ignore[call-arg]
                    self._services[interface] = instance
                    return instance
                except TypeError:
                    pass

            raise ValueError(f"No registration found for {interface.__name__}")

    def clear(self) -> None:
        """Clear all registrations and close resources."""
        with self._lock:
            # Close any resources that have close methods
            for service in self._services.values():
                if hasattr(service, "close"):
                    try:
                        service.close()
                    except Exception:  # pragma: no cover
                        pass
            self._services.clear()
            self._factories.clear()


def create_container(config: Optional[Dict[str, Any]] = None) -> ServiceContainer:
    """Factory function to create and configure a container."""
    container = ServiceContainer()
    config = config or {}

    # Register configuration manager first
    from .infrastructure.config_manager import ConfigurationManager
    from .services import IConfigurationManager

    config_manager = ConfigurationManager(
        config_file=config.get("config_file"), auto_reload=config.get("auto_reload", False)
    )
    # Merge provided config into manager
    if config:
        config_manager.merge(config)
    container.register_singleton(IConfigurationManager, config_manager)

    # Get API key from config or environment
    api_key = config.get("api_key") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY must be provided in config or environment")

    # Register API client
    from .infrastructure.api_client import OpenRouterClient
    from .services import IAPIClient

    api_client = OpenRouterClient(api_key=api_key, base_url=config.get("base_url", "https://openrouter.ai/api/v1"))
    container.register_singleton(IAPIClient, api_client)

    # Register prompts manager
    from .infrastructure.prompts_manager import PromptsManager
    from .services import IPromptsManager

    prompts_manager = PromptsManager(prompts_file=config.get("prompts_file"))
    container.register_singleton(IPromptsManager, prompts_manager)

    # Register judge service
    from .infrastructure.judge_service import JudgeService
    from .services import IJudgeService

    judge_service = JudgeService(api_client=api_client, config_file=config.get("judge_config_file"))
    container.register_singleton(IJudgeService, judge_service)

    # Register utility services
    from .infrastructure.utility_services import (
        TimeService,
        RefusalDetector,
        ResponseParser,
        FileSystemService,
    )
    from .services import (
        ITimeService,
        IRefusalDetector,
        IResponseParser,
        IFileSystemService,
    )

    container.register_singleton(ITimeService, TimeService())
    container.register_singleton(IRefusalDetector, RefusalDetector())
    container.register_singleton(IResponseParser, ResponseParser())
    container.register_singleton(IFileSystemService, FileSystemService())

    return container


__all__ = ["ServiceContainer", "create_container"]
