from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

from llm_judge.exceptions import (
    APIConnectionError,
    APIRateLimitError,
    APIResponseError,
    APITimeoutError,
    ArtifactSaveError,
    ConfigurationError,
    JudgeParsingError,
    LLMJudgeException,
    PromptLoadError,
    RunnerCancelledError,
    ValidationError,
)
from llm_judge.logging_config import LogContext, StructuredFormatter, configure_logging, get_logger


def test_llmjudgeexception_str_with_context() -> None:
    exc = LLMJudgeException("Failure", {"step": "chat", "code": 42})
    assert "step=chat" in str(exc)
    assert exc.context["code"] == 42


def test_exception_specialisations() -> None:
    assert str(ConfigurationError("Missing")) == "Missing"
    rate = APIRateLimitError("slow down", retry_after=15)
    assert rate.retry_after == 15
    resp = APIResponseError("bad", status_code=500, response_body="{}")
    assert resp.status_code == 500
    parsed = JudgeParsingError("no json", raw_response="oops")
    assert parsed.raw_response == "oops"
    validation = ValidationError("invalid", field="temperature", value=9)
    assert validation.field == "temperature"
    # ensure subclasses can be instantiated without raising
    APIConnectionError("down")
    APITimeoutError("slow")
    PromptLoadError("missing")
    artifact = ArtifactSaveError("failed", file_path="/tmp/file")
    assert artifact.file_path == "/tmp/file"
    RunnerCancelledError("stopped")


def test_configure_logging_and_threadsafe_logger(tmp_path: Path) -> None:
    log_file = tmp_path / "logs.txt"
    configure_logging(level="DEBUG", log_file=log_file, include_timestamp=False)

    logger = get_logger("llm_judge.tests")
    logger.debug("debug message", request_id=1)
    logger.info("info message", user="alice")
    logger.warning("warn message")
    logger.error("error message")
    logger.critical("critical message")
    logger.exception("exception message")

    assert log_file.exists()


def test_structured_formatter_adds_exception_context() -> None:
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    formatter = StructuredFormatter("%(levelname)s:%(message)s %(ctx_step)s")
    handler.setFormatter(formatter)
    logger = logging.getLogger("structured-test")
    logger.handlers = [handler]
    logger.setLevel(logging.ERROR)

    try:
        raise APIRateLimitError("rate", context={"step": "judge"})
    except APIRateLimitError:
        logger.exception("hit limit")

    handler.flush()
    output = stream.getvalue()
    assert "judge" in output


def test_log_context_manager_returns_self() -> None:
    logger = get_logger("context-test")
    with LogContext(logger, request_id="abc") as ctx:
        assert ctx is not None
