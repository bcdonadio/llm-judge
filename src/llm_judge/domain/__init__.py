"""Domain models for LLM Judge."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass(frozen=True)
class Prompt:
    """Domain model for a prompt."""

    text: str
    category: str
    index: int


@dataclass(frozen=True)
class ModelResponse:
    """Domain model for model response."""

    text: str
    raw_payload: Dict[str, Any]
    finish_reason: Optional[str] = None


@dataclass(frozen=True)
class JudgeDecision:
    """Domain model for judge decision."""

    success: bool
    initial_refusal: bool
    initial_completeness: float
    initial_sourcing: str
    follow_refusal: bool
    follow_completeness: float
    follow_sourcing: str
    asymmetry: str
    safety_flags_initial: List[str]
    safety_flags_follow: List[str]
    notes: str
    raw_data: Dict[str, Any]
    error: Optional[str] = None


@dataclass(frozen=True)
class RunConfiguration:
    """Immutable run configuration."""

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


__all__ = [
    "Prompt",
    "ModelResponse",
    "JudgeDecision",
    "RunConfiguration",
]
