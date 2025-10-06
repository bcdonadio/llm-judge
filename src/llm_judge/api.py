"""API client utilities for interacting with OpenRouter via the OpenAI SDK."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, cast

import httpx
from openai import OpenAI
from openai import OpenAIError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_http_client: httpx.Client | None = None
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Return a cached OpenAI client configured for OpenRouter."""
    global _client, _http_client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY environment variable.")

    _http_client = httpx.Client(http2=True, timeout=httpx.Timeout(120, connect=10))
    _client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        http_client=_http_client,
        max_retries=2,
    )
    logging.getLogger("llm_judge.api").debug("Initialized OpenRouter client with persistent HTTP/2 connection pool.")
    return _client


def openrouter_chat(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    metadata: Dict[str, str],
    response_format: Dict[str, Any] | None = None,
    step: str | None = None,
    prompt_index: int | None = None,
    use_color: bool = False,
) -> Dict[str, Any]:
    """Send a chat completion request to OpenRouter and return the JSON response."""
    client = _get_client()
    headers = {
        "HTTP-Referer": metadata.get("referer", "https://example.com"),
        "X-Title": metadata.get("title", "LLM Asymmetry Test Suite"),
    }

    logger = logging.getLogger("llm_judge.api")
    if step is not None and prompt_index is not None:
        log_message = f"[Request {prompt_index:02d}] {step} → {model} (max_tokens={max_tokens})"
        if use_color:
            from colorama import Fore, Style

            log_message = f"{Style.BRIGHT}{Fore.MAGENTA}{log_message}{Style.RESET_ALL}"
        logger.info(log_message)
    else:
        logger.debug(
            "POST %s/chat/completions model=%s messages=%d max_tokens=%d temperature=%.2f headers=%s",
            OPENROUTER_BASE_URL,
            model,
            len(messages),
            max_tokens,
            temperature,
            headers,
        )
    chat_with_raw = cast(Any, client.chat.completions.with_raw_response)
    try:
        raw_response = chat_with_raw.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=headers,
            response_format=response_format,
        )
    except OpenAIError as exc:
        logging.getLogger("llm_judge.api").debug("Request to OpenRouter failed: %s", exc, exc_info=True)
        raise

    http_response = raw_response.http_response
    elapsed = http_response.elapsed.total_seconds() if http_response.elapsed else None
    logging.getLogger("llm_judge.api").debug(
        "Received response status=%s latency=%s model=%s",
        http_response.status_code,
        f"{elapsed:.3f}s" if elapsed is not None else "unknown",
        model,
    )
    logger = logging.getLogger("llm_judge.api")
    if logger.isEnabledFor(logging.DEBUG):
        try:
            preview = http_response.text[:400].replace("\n", " ")
            logger.debug("Response preview (truncated): %s", preview)
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Failed to obtain response preview.", exc_info=True)

    completion = raw_response.parse()
    return cast(Dict[str, Any], completion.model_dump())


__all__ = ["OPENROUTER_BASE_URL", "openrouter_chat"]
