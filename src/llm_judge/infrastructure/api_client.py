"""Thread-safe OpenRouter API client implementation."""

import threading
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import httpx
from openai import OpenAI, OpenAIError

from ..domain import ModelResponse
from ..services import IAPIClient


def _filter_model_entry(entry: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None
    sanitized: Dict[str, Any] = {}
    items_iter = cast(Iterable[Tuple[Any, Any]], entry.items())
    for key_any, value_any in items_iter:
        if isinstance(key_any, str):
            sanitized[key_any] = value_any
    return sanitized


class OpenRouterClient(IAPIClient):
    """Thread-safe OpenRouter API client with connection pooling."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 120,
        max_retries: int = 2,
        logger: Optional[logging.Logger] = None,
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
        """Lazily create and cache the OpenAI client."""
        with self._lock:
            if self._client is None:
                self._http_client = httpx.Client(http2=True, timeout=httpx.Timeout(self._timeout, connect=10))
                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    http_client=self._http_client,
                    max_retries=self._max_retries,
                )
                self._logger.debug("Initialized OpenRouter client with persistent HTTP/2 connection pool.")
            return self._client

    def list_models(self) -> List[Dict[str, Any]]:
        """Fetch available models from OpenRouter."""
        with self._lock:
            self._ensure_client()
            if self._http_client is None:
                raise RuntimeError("OpenRouter HTTP client not initialised.")
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            }
            try:
                response = self._http_client.get(f"{self._base_url}/models", headers=headers)
                response.raise_for_status()
                payload = cast(Dict[str, Any], response.json())
            except Exception as exc:
                self._logger.error("Failed to fetch OpenRouter model catalog: %s", exc)
                raise

        raw_data = payload.get("data")
        if not isinstance(raw_data, list):
            raise ValueError("Unexpected response payload for model catalog.")

        raw_entries: List[Any] = list(cast(Iterable[Any], raw_data))
        filtered: List[Dict[str, Any]] = []
        for entry in raw_entries:
            cleaned = _filter_model_entry(entry)
            if cleaned:
                filtered.append(cleaned)
        return filtered

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
        client = self._ensure_client()

        headers = {
            "HTTP-Referer": metadata.get("referer", "https://example.com"),
            "X-Title": metadata.get("title", "LLM Asymmetry Test Suite"),
        }

        # Extract optional parameters for logging
        step = kwargs.get("step")
        prompt_index = kwargs.get("prompt_index")
        use_color = kwargs.get("use_color", False)

        # Log request if step and index provided
        if step is not None and prompt_index is not None:
            log_message = f"[Request {prompt_index:02d}] {step} → {model} (max_tokens={max_tokens})"
            if use_color:
                try:
                    from colorama import Fore, Style

                    log_message = f"{Style.BRIGHT}{Fore.MAGENTA}{log_message}{Style.RESET_ALL}"
                except ImportError:
                    pass
            self._logger.info(log_message)
        else:
            self._logger.debug(
                "POST %s/chat/completions model=%s messages=%d max_tokens=%d temperature=%.2f",
                self._base_url,
                model,
                len(messages),
                max_tokens,
                temperature,
            )

        try:
            with self._lock:
                client_any = cast(Any, client)
                chat_with_raw = client_any.chat.completions.with_raw_response
                raw_response = chat_with_raw.create(
                    model=model,
                    messages=cast(Any, messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=headers,
                    response_format=cast(Any, response_format),
                )

            completion = raw_response.parse()
            payload = completion.model_dump()

            # Extract text and finish reason
            text = self._extract_text(payload)
            finish_reason = self._extract_finish_reason(payload)

            # Log response details
            http_response = raw_response.http_response
            elapsed = http_response.elapsed.total_seconds() if http_response.elapsed else None
            self._logger.debug(
                "Received response status=%s latency=%s model=%s",
                http_response.status_code,
                f"{elapsed:.3f}s" if elapsed is not None else "unknown",
                model,
            )

            return ModelResponse(text=text, raw_payload=payload, finish_reason=finish_reason)

        except OpenAIError as exc:
            self._logger.error("API request failed: %s", exc, exc_info=True)
            raise

    def close(self) -> None:
        """Close connections and cleanup resources."""
        with self._lock:
            if self._http_client:
                self._http_client.close()
                self._http_client = None
            self._client = None
            self._logger.debug("Closed OpenRouter client connections")

    def __enter__(self) -> "OpenRouterClient":
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.close()

    @staticmethod
    def _extract_text(payload: Dict[str, Any]) -> str:  # noqa: C901
        """Extract text content from API response."""
        try:
            message = payload["choices"][0]["message"]
            content = message.get("content", "")

            # Handle string content
            if isinstance(content, str) and content:
                return content

            # Handle structured content (list of segments)
            if isinstance(content, list):
                segments = cast(List[Dict[str, Any] | str], content)
                texts: List[str] = []
                for item in segments:
                    if isinstance(item, dict):
                        item_dict: Dict[str, Any] = item
                        text_value = item_dict.get("text")
                        if isinstance(text_value, str):
                            texts.append(text_value)
                            continue
                        if isinstance(text_value, bytes):
                            try:
                                texts.append(text_value.decode("utf-8"))
                            except Exception:  # pragma: no cover - defensive
                                continue
                            continue
                        continue
                    texts.append(item)
                combined = "".join(texts)
                if combined:
                    return combined

            # Handle reasoning field as fallback
            reasoning = message.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                return reasoning

            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                call_list = cast(List[Dict[str, Any]], tool_calls)
                for call in call_list:
                    function = call.get("function")
                    if not isinstance(function, dict):
                        continue
                    function_any = cast(Any, function)
                    arguments = function_any.get("arguments")
                    if isinstance(arguments, str) and arguments.strip():
                        return arguments

            return ""
        except (KeyError, IndexError, TypeError):
            return ""

    @staticmethod
    def _extract_finish_reason(payload: Dict[str, Any]) -> Optional[str]:
        """Extract finish reason from response."""
        try:
            choice = payload["choices"][0]
        except (KeyError, IndexError):
            return None

        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str):
            return finish_reason

        native_reason = choice.get("native_finish_reason")
        if isinstance(native_reason, str):
            return native_reason

        return None


__all__ = ["OpenRouterClient"]
