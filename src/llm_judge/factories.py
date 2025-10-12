"""Factory functions for creating configured service instances."""

import time
from pathlib import Path
from typing import List, Optional, Callable

from .container import ServiceContainer
from .domain import RunConfiguration
from .runner import LLMJudgeRunner, RunnerConfig, RunnerControl, RunnerEvent, CSV_FIELDNAMES
from .services import IAPIClient, IJudgeService, IPromptsManager, IFileSystemService, IUnitOfWork
from .infrastructure import UnitOfWork


class RunnerFactory:
    """Factory for creating configured LLMJudgeRunner instances."""

    def __init__(self, container: ServiceContainer):
        self._container = container

    def create_runner(
        self,
        config: RunnerConfig,
        control: Optional[RunnerControl] = None,
        progress_callback: Optional[Callable[[RunnerEvent], None]] = None,
    ) -> LLMJudgeRunner:
        """Create a runner with dependencies resolved from the container.

        Args:
            config: Runner configuration
            control: Optional control hooks for pause/cancel
            progress_callback: Optional callback for progress events

        Returns:
            Configured LLMJudgeRunner instance
        """
        api_client = self._container.resolve(IAPIClient)
        judge_service = self._container.resolve(IJudgeService)
        prompts_manager = self._container.resolve(IPromptsManager)

        return LLMJudgeRunner(
            config=config,
            api_client=api_client,
            judge_service=judge_service,
            prompts_manager=prompts_manager,
            control=control,
            progress_callback=progress_callback,
        )


class UnitOfWorkFactory:
    """Factory for creating Unit of Work instances for run persistence."""

    def __init__(self, container: ServiceContainer):
        self._container = container

    def create_unit_of_work(
        self, outdir: Path, timestamp: str | None = None, csv_fieldnames: List[str] | None = None
    ) -> IUnitOfWork:
        """Create a unit of work for a specific run.

        Args:
            outdir: Base output directory
            timestamp: Run timestamp (defaults to current time)
            csv_fieldnames: CSV column names (defaults to CSV_FIELDNAMES)

        Returns:
            Configured unit of work instance
        """
        if timestamp is None:
            timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

        if csv_fieldnames is None:
            csv_fieldnames = list(CSV_FIELDNAMES)

        fs_service = self._container.resolve(IFileSystemService)

        run_directory = outdir / "runs" / timestamp
        csv_path = outdir / f"results_{timestamp}.csv"

        return UnitOfWork(
            run_directory=run_directory, csv_path=csv_path, csv_fieldnames=csv_fieldnames, fs_service=fs_service
        )


class ConfigurationBuilder:
    """Builder for creating RunConfiguration instances."""

    def __init__(self):
        self._models: List[str] = []
        self._judge_model: str = "x-ai/grok-4-fast"
        self._outdir: Path = Path("./results")
        self._max_tokens: int = 2048
        self._judge_max_tokens: int = 8192
        self._temperature: float = 0.7
        self._judge_temperature: float = 0.3
        self._sleep_s: float = 1.0
        self._limit: int | None = None
        self._verbose: bool = False
        self._use_color: bool = False

    def with_models(self, models: List[str]) -> "ConfigurationBuilder":
        """Set the models to evaluate."""
        self._models = models
        return self

    def with_judge_model(self, judge_model: str) -> "ConfigurationBuilder":
        """Set the judge model."""
        self._judge_model = judge_model
        return self

    def with_outdir(self, outdir: Path) -> "ConfigurationBuilder":
        """Set the output directory."""
        self._outdir = outdir
        return self

    def with_max_tokens(self, max_tokens: int) -> "ConfigurationBuilder":
        """Set max tokens for model responses."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self._max_tokens = max_tokens
        return self

    def with_judge_max_tokens(self, judge_max_tokens: int) -> "ConfigurationBuilder":
        """Set max tokens for judge responses."""
        if judge_max_tokens <= 0:
            raise ValueError("judge_max_tokens must be positive")
        self._judge_max_tokens = judge_max_tokens
        return self

    def with_temperature(self, temperature: float) -> "ConfigurationBuilder":
        """Set temperature for model responses."""
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        self._temperature = temperature
        return self

    def with_judge_temperature(self, judge_temperature: float) -> "ConfigurationBuilder":
        """Set temperature for judge responses."""
        if not 0 <= judge_temperature <= 2:
            raise ValueError("judge_temperature must be between 0 and 2")
        self._judge_temperature = judge_temperature
        return self

    def with_sleep(self, sleep_s: float) -> "ConfigurationBuilder":
        """Set sleep duration between requests."""
        if sleep_s < 0:
            raise ValueError("sleep_s must be non-negative")
        self._sleep_s = sleep_s
        return self

    def with_limit(self, limit: int | None) -> "ConfigurationBuilder":
        """Set prompt limit."""
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive or None")
        self._limit = limit
        return self

    def with_verbose(self, verbose: bool = True) -> "ConfigurationBuilder":
        """Enable verbose output."""
        self._verbose = verbose
        return self

    def with_color(self, use_color: bool = True) -> "ConfigurationBuilder":
        """Enable colored output."""
        self._use_color = use_color
        return self

    def build(self) -> RunConfiguration:
        """Build the configuration."""
        if not self._models:
            raise ValueError("At least one model must be specified")

        return RunConfiguration(
            models=self._models,
            judge_model=self._judge_model,
            outdir=self._outdir,
            max_tokens=self._max_tokens,
            judge_max_tokens=self._judge_max_tokens,
            temperature=self._temperature,
            judge_temperature=self._judge_temperature,
            sleep_s=self._sleep_s,
            limit=self._limit,
            verbose=self._verbose,
            use_color=self._use_color,
        )


__all__ = ["RunnerFactory", "UnitOfWorkFactory", "ConfigurationBuilder"]
