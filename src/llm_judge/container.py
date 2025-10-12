"""Thread-safe dependency injection container."""

import os
import threading
from typing import Any, Dict, Callable, Optional


class ServiceContainer:
    """Thread-safe dependency injection container."""

    def __init__(self) -> None:
        self._services: Dict[object, Any] = {}
        self._factories: Dict[object, Callable[[], Any]] = {}
        self._lock = threading.RLock()

    def register_singleton(self, interface: object, instance: Any) -> None:
        """Register a singleton service."""
        with self._lock:
            self._services[interface] = instance

    def register_factory(self, interface: object, factory: Callable[[], Any]) -> None:
        """Register a factory for creating instances."""
        with self._lock:
            self._factories[interface] = factory

    def resolve(self, interface: object) -> Any:
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
            if isinstance(interface, type) and not getattr(interface, "__abstractmethods__", False):
                try:
                    instance = interface()
                    self._services[interface] = instance
                    return instance
                except TypeError:
                    pass

            name = getattr(interface, "__name__", repr(interface))
            raise ValueError(f"No registration found for {name}")

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
    config_dict: Dict[str, Any] = dict(config or {})

    # Register configuration manager first
    from .infrastructure.config_manager import ConfigurationManager
    from .services import IConfigurationManager

    config_manager = ConfigurationManager(
        config_file=config_dict.get("config_file"), auto_reload=config_dict.get("auto_reload", False)
    )
    # Merge provided config into manager
    if config_dict:
        config_manager.merge(config_dict)
    container.register_singleton(IConfigurationManager, config_manager)

    # Get API key from config or environment
    api_key = config_dict.get("api_key") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY must be provided in config or environment")

    # Register API client
    from .infrastructure.api_client import OpenRouterClient
    from .services import IAPIClient

    api_client = OpenRouterClient(api_key=api_key, base_url=config_dict.get("base_url", "https://openrouter.ai/api/v1"))
    container.register_singleton(IAPIClient, api_client)

    # Register prompts manager
    from .infrastructure.prompts_manager import PromptsManager
    from .services import IPromptsManager

    prompts_manager = PromptsManager(prompts_file=config_dict.get("prompts_file"))
    container.register_singleton(IPromptsManager, prompts_manager)

    # Register judge service
    from .infrastructure.judge_service import JudgeService
    from .services import IJudgeService

    judge_service = JudgeService(api_client=api_client, config_file=config_dict.get("judge_config_file"))
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
