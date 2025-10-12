"""Custom exception hierarchy for llm-judge."""

from typing import Any, Dict, Optional


class LLMJudgeException(Exception):
    """Base exception for all llm-judge errors."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Initialize exception with message and optional context.

        Args:
            message: Human-readable error message
            context: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        """Return string representation with context if available."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message


class ConfigurationError(LLMJudgeException):
    """Raised when configuration is invalid or missing."""

    pass


class APIError(LLMJudgeException):
    """Base class for API-related errors."""

    pass


class APIConnectionError(APIError):
    """Raised when unable to connect to API."""

    pass


class APITimeoutError(APIError):
    """Raised when API request times out."""

    pass


class APIRateLimitError(APIError):
    """Raised when API rate limit is exceeded."""

    def __init__(self, message: str, retry_after: Optional[int] = None, context: Optional[Dict[str, Any]] = None):
        """Initialize with retry information.

        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
            context: Additional context
        """
        super().__init__(message, context)
        self.retry_after = retry_after


class APIResponseError(APIError):
    """Raised when API returns an error response."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize with response information.

        Args:
            message: Error message
            status_code: HTTP status code
            response_body: Raw response body
            context: Additional context
        """
        super().__init__(message, context)
        self.status_code = status_code
        self.response_body = response_body


class JudgingError(LLMJudgeException):
    """Raised when judging process fails."""

    pass


class JudgeParsingError(JudgingError):
    """Raised when judge response cannot be parsed."""

    def __init__(self, message: str, raw_response: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """Initialize with raw response.

        Args:
            message: Error message
            raw_response: Unparseable response text
            context: Additional context
        """
        super().__init__(message, context)
        self.raw_response = raw_response


class PromptLoadError(LLMJudgeException):
    """Raised when prompts cannot be loaded."""

    pass


class RepositoryError(LLMJudgeException):
    """Base class for repository/persistence errors."""

    pass


class ArtifactSaveError(RepositoryError):
    """Raised when artifact cannot be saved."""

    def __init__(self, message: str, file_path: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """Initialize with file information.

        Args:
            message: Error message
            file_path: Path where save failed
            context: Additional context
        """
        super().__init__(message, context)
        self.file_path = file_path


class ValidationError(LLMJudgeException):
    """Raised when input validation fails."""

    def __init__(
        self, message: str, field: Optional[str] = None, value: Any = None, context: Optional[Dict[str, Any]] = None
    ):
        """Initialize with validation details.

        Args:
            message: Error message
            field: Field that failed validation
            value: Invalid value
            context: Additional context
        """
        super().__init__(message, context)
        self.field = field
        self.value = value


class RunnerError(LLMJudgeException):
    """Raised when runner encounters an error."""

    pass


class RunnerCancelledError(RunnerError):
    """Raised when a run is cancelled by user."""

    pass


__all__ = [
    "LLMJudgeException",
    "ConfigurationError",
    "APIError",
    "APIConnectionError",
    "APITimeoutError",
    "APIRateLimitError",
    "APIResponseError",
    "JudgingError",
    "JudgeParsingError",
    "PromptLoadError",
    "RepositoryError",
    "ArtifactSaveError",
    "ValidationError",
    "RunnerError",
    "RunnerCancelledError",
]
