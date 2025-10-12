"""Thread-safe judge evaluation service implementation."""

import json
import threading
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from importlib import resources

import yaml

from ..domain import ModelResponse, JudgeDecision, RunConfiguration
from ..services import IJudgeService, IAPIClient


class JudgeService(IJudgeService):
    """Thread-safe judge evaluation service."""

    def __init__(
        self, api_client: IAPIClient, config_file: Optional[Path] = None, logger: Optional[logging.Logger] = None
    ):
        self._api_client = api_client
        self._config_file = config_file
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.RLock()
        self._config_cache: Optional[Dict[str, Any]] = None

    def _load_config(self) -> Dict[str, Any]:
        """Load judge configuration."""
        with self._lock:
            if self._config_cache is not None:
                return self._config_cache

            if self._config_file:
                with open(self._config_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            else:
                resource = resources.files("llm_judge") / "judge_config.yaml"
                with resource.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

            if not isinstance(data, dict):
                raise TypeError("Judge configuration must be a mapping")

            self._config_cache = data
            return data

    def evaluate(
        self, prompt: str, initial_response: ModelResponse, follow_response: ModelResponse, config: RunConfiguration
    ) -> JudgeDecision:
        """Evaluate responses and return judge decision."""
        judge_config = self._load_config()

        # Build evaluation messages
        messages = self._build_messages(judge_config, prompt, initial_response.text, follow_response.text)

        # Request judgment with retries
        max_attempts = 3
        current_max_tokens = config.judge_max_tokens
        response = None

        for attempt in range(max_attempts):
            try:
                response = self._api_client.chat_completion(
                    model=config.judge_model,
                    messages=messages,
                    max_tokens=current_max_tokens,
                    temperature=config.judge_temperature,
                    metadata={"title": "Judge Evaluation"},
                    response_format={"type": "json_object"},
                )

                # Parse JSON response
                decision_data = self._parse_judge_response(response)

                return self._create_decision(decision_data, response.raw_payload)

            except (json.JSONDecodeError, ValueError) as e:
                self._logger.debug("Judge attempt %d failed: %s", attempt + 1, e)

                # Check if we should retry with more tokens
                if response and response.finish_reason in {"length", "max_output_tokens"}:
                    current_max_tokens = min(current_max_tokens * 2, 40960)
                    continue

                if attempt == max_attempts - 1:
                    # Return error decision on final attempt
                    return self._create_error_decision(str(e))

        return self._create_error_decision("Max attempts exceeded")

    def _build_messages(self, config: Dict[str, Any], prompt: str, initial: str, follow: str) -> list[Dict[str, str]]:
        """Build messages for judge evaluation."""
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
            {"role": "user", "content": user_block},
        ]

    @staticmethod
    def _parse_judge_response(response: ModelResponse) -> Dict[str, Any]:  # noqa: C901
        """Parse and validate judge JSON response."""
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
                        json_str = text[start : i + 1]
                        return json.loads(json_str)

            raise ValueError("Invalid JSON structure")

    @staticmethod
    def _create_decision(data: Dict[str, Any], raw: Dict[str, Any]) -> JudgeDecision:
        """Create JudgeDecision from parsed data."""
        initial = data.get("initial", {})
        followup = data.get("followup", {})

        # Extract safety flags
        initial_flags = initial.get("safety_flags", [])
        follow_flags = followup.get("safety_flags", [])

        # Ensure they are lists
        if not isinstance(initial_flags, list):
            initial_flags = []
        if not isinstance(follow_flags, list):
            follow_flags = []

        return JudgeDecision(
            success=True,
            initial_refusal=bool(initial.get("refusal")),
            initial_completeness=float(initial.get("completeness", 0)),
            initial_sourcing=str(initial.get("sourcing_quality", "")),
            follow_refusal=bool(followup.get("refusal")),
            follow_completeness=float(followup.get("completeness", 0)),
            follow_sourcing=str(followup.get("sourcing_quality", "")),
            asymmetry=str(data.get("asymmetry_leading", "unclear")),
            safety_flags_initial=initial_flags,
            safety_flags_follow=follow_flags,
            notes=str(data.get("final_notes", "")),
            raw_data=raw,
            error=None,
        )

    @staticmethod
    def _create_error_decision(error: str) -> JudgeDecision:
        """Create error decision."""
        return JudgeDecision(
            success=False,
            initial_refusal=False,
            initial_completeness=0.0,
            initial_sourcing="error",
            follow_refusal=False,
            follow_completeness=0.0,
            follow_sourcing="error",
            asymmetry="error",
            safety_flags_initial=[],
            safety_flags_follow=[],
            notes=f"Error: {error}",
            raw_data={"error": error},
            error=error,
        )


__all__ = ["JudgeService"]
