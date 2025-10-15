"""Service interfaces for dependency injection."""

from typing import Protocol, List, Dict, Any, Optional
from pathlib import Path

from ..domain import Prompt, ModelResponse, JudgeDecision, RunConfiguration


class IAPIClient(Protocol):
    """Interface for API communication."""

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        metadata: Dict[str, str],
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Execute a chat completion request."""
        ...

    def list_models(self) -> List[Dict[str, Any]]:
        """Retrieve the catalog of available models."""
        ...

    def close(self) -> None:
        """Close connections and cleanup resources."""
        ...


class IPromptsManager(Protocol):
    """Interface for prompt management."""

    def get_core_prompts(self) -> List[Prompt]:
        """Get core prompts as domain objects."""
        ...

    def get_follow_up(self) -> str:
        """Get follow-up prompt text."""
        ...

    def get_probes(self) -> List[str]:
        """Get probe prompts."""
        ...


class IJudgeService(Protocol):
    """Interface for judging logic."""

    def evaluate(
        self, prompt: str, initial_response: ModelResponse, follow_response: ModelResponse, config: RunConfiguration
    ) -> JudgeDecision:
        """Evaluate responses and return judge decision."""
        ...


class IConfigurationManager(Protocol):
    """Interface for configuration management."""

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        ...

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        ...

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        ...

    def reload(self) -> None:
        """Reload configuration from source."""
        ...


class ITimeService(Protocol):
    """Interface for time-related operations."""

    def now_iso(self) -> str:
        """Get current UTC timestamp in ISO-8601 format with Z suffix."""
        ...


class IRefusalDetector(Protocol):
    """Interface for detecting refusals in text."""

    def is_refusal(self, text: str) -> bool:
        """Determine if text represents a refusal."""
        ...


class IResponseParser(Protocol):
    """Interface for parsing API responses."""

    def extract_text(self, payload: Dict[str, Any]) -> str:
        """Extract text content from API response payload."""
        ...


class IFileSystemService(Protocol):
    """Interface for file system operations."""

    def write_json(self, path: Path, data: Any) -> None:
        """Write data as JSON to path, creating directories as needed."""
        ...

    def create_temp_dir(self, prefix: str = "llm-judge-") -> Path:
        """Create and return a temporary directory."""
        ...


class IArtifactsRepository(Protocol):
    """Interface for storing run artifacts (JSON files)."""

    def save_completion(self, model: str, prompt_index: int, step: str, data: Dict[str, Any]) -> Path:
        """Save a completion artifact and return its path."""
        ...

    def save_judge_decision(self, model: str, prompt_index: int, data: Dict[str, Any]) -> Path:
        """Save a judge decision artifact and return its path."""
        ...

    def get_run_directory(self) -> Path:
        """Get the current run directory."""
        ...


class IResultsRepository(Protocol):
    """Interface for storing CSV results."""

    def add_result(self, row: Dict[str, Any]) -> None:
        """Add a result row to the collection."""
        ...

    def get_csv_path(self) -> Path:
        """Get the path to the CSV file."""
        ...

    def flush(self) -> None:
        """Flush any pending writes."""
        ...


class IUnitOfWork(Protocol):
    """Interface for coordinating repository operations."""

    @property
    def artifacts(self) -> IArtifactsRepository:
        """Access the artifacts repository."""
        ...

    @property
    def results(self) -> IResultsRepository:
        """Access the results repository."""
        ...

    def commit(self) -> None:
        """Commit all pending changes."""
        ...

    def rollback(self) -> None:
        """Rollback any uncommitted changes."""
        ...

    def __enter__(self) -> "IUnitOfWork":
        """Enter context manager."""
        ...

    def __exit__(self, *args: Any) -> None:
        """Exit context manager."""
        ...


__all__ = [
    "IAPIClient",
    "IPromptsManager",
    "IJudgeService",
    "IConfigurationManager",
    "ITimeService",
    "IRefusalDetector",
    "IResponseParser",
    "IFileSystemService",
    "IArtifactsRepository",
    "IResultsRepository",
    "IUnitOfWork",
]
