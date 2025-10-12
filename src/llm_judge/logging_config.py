"""Logging configuration and utilities for llm-judge."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import threading

from .exceptions import LLMJudgeException


class StructuredFormatter(logging.Formatter):
    """Formatter that adds structured context to log records."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured context."""
        # Add exception context if available
        if record.exc_info and isinstance(record.exc_info[1], LLMJudgeException):
            exc = record.exc_info[1]
            if exc.context:
                # Add context as extra fields
                for key, value in exc.context.items():
                    setattr(record, f"ctx_{key}", value)

        return super().format(record)


class ThreadSafeLogger:
    """Thread-safe wrapper for Python logger with structured logging support."""

    def __init__(self, name: str):
        """Initialize with logger name."""
        self._logger = logging.getLogger(name)
        self._lock = threading.RLock()

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log debug message with context."""
        with self._lock:
            self._logger.debug(msg, extra=kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log info message with context."""
        with self._lock:
            self._logger.info(msg, extra=kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log warning message with context."""
        with self._lock:
            self._logger.warning(msg, extra=kwargs)

    def error(self, msg: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log error message with context."""
        with self._lock:
            self._logger.error(msg, exc_info=exc_info, extra=kwargs)

    def critical(self, msg: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log critical message with context."""
        with self._lock:
            self._logger.critical(msg, exc_info=exc_info, extra=kwargs)

    def exception(self, msg: str, **kwargs: Any) -> None:
        """Log exception with context."""
        with self._lock:
            self._logger.exception(msg, extra=kwargs)


def configure_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
) -> None:
    """Configure logging for llm-judge.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        format_string: Custom format string (uses default if None)
        include_timestamp: Whether to include timestamps in logs
    """
    # Remove existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Set level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric_level)

    # Default format
    if format_string is None:
        if include_timestamp:
            format_string = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(funcName)s:%(lineno)d - %(message)s"
            )
        else:
            format_string = (
                "%(name)s - %(levelname)s - "
                "%(funcName)s:%(lineno)d - %(message)s"
            )

    formatter = StructuredFormatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    root.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        root.addHandler(file_handler)


def get_logger(name: str) -> ThreadSafeLogger:
    """Get a thread-safe logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Thread-safe logger instance
    """
    return ThreadSafeLogger(name)


class LogContext:
    """Context manager for adding structured context to logs."""

    def __init__(self, logger: ThreadSafeLogger, **context: Any):
        """Initialize with context fields.

        Args:
            logger: Logger to add context to
            **context: Key-value pairs to add as context
        """
        self._logger = logger
        self._context = context
        self._old_context: Dict[str, Any] = {}

    def __enter__(self) -> "LogContext":
        """Enter context and save old values."""
        # In a full implementation, we'd use contextvars here
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context and restore old values."""
        pass


__all__ = [
    "StructuredFormatter",
    "ThreadSafeLogger",
    "configure_logging",
    "get_logger",
    "LogContext",
]
