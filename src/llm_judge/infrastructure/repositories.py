"""Repository implementations for data persistence."""

import csv
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..services import IArtifactsRepository, IResultsRepository, IUnitOfWork, IFileSystemService


class ArtifactsRepository(IArtifactsRepository):
    """Repository for storing run artifacts as JSON files."""

    def __init__(self, run_directory: Path, fs_service: IFileSystemService):
        """Initialize artifacts repository.

        Args:
            run_directory: Base directory for this run's artifacts
            fs_service: File system service for JSON operations
        """
        self._run_dir = run_directory
        self._fs_service = fs_service
        self._lock = threading.RLock()

    def save_completion(self, model: str, prompt_index: int, step: str, data: Dict[str, Any]) -> Path:
        """Save a completion artifact and return its path."""
        with self._lock:
            model_dir = self._run_dir / model.replace("/", "_")
            model_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{prompt_index:02d}_{step}.json"
            path = model_dir / filename

            self._fs_service.write_json(path, data)
            return path

    def save_judge_decision(self, model: str, prompt_index: int, data: Dict[str, Any]) -> Path:
        """Save a judge decision artifact and return its path."""
        with self._lock:
            model_dir = self._run_dir / model.replace("/", "_")
            model_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{prompt_index:02d}_judge.json"
            path = model_dir / filename

            self._fs_service.write_json(path, data)
            return path

    def get_run_directory(self) -> Path:
        """Get the current run directory."""
        return self._run_dir


class ResultsRepository(IResultsRepository):
    """Repository for storing CSV results."""

    def __init__(self, csv_path: Path, fieldnames: List[str]):
        """Initialize results repository.

        Args:
            csv_path: Path to the CSV file
            fieldnames: List of CSV column names
        """
        self._csv_path = csv_path
        self._fieldnames = fieldnames
        self._lock = threading.RLock()
        self._buffer: List[Dict[str, Any]] = []
        self._writer: Optional[csv.DictWriter[Any]] = None
        self._file_handle: Optional[Any] = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure CSV file and writer are initialized."""
        if self._initialized:
            return

        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_handle = self._csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file_handle, fieldnames=self._fieldnames)
        self._writer.writeheader()
        self._initialized = True

    def add_result(self, row: Dict[str, Any]) -> None:
        """Add a result row to the collection."""
        with self._lock:
            self._ensure_initialized()
            if self._writer:
                self._writer.writerow(row)
                self._buffer.append(row)

    def get_csv_path(self) -> Path:
        """Get the path to the CSV file."""
        return self._csv_path

    def flush(self) -> None:
        """Flush any pending writes."""
        with self._lock:
            if self._file_handle:
                self._file_handle.flush()

    def close(self) -> None:
        """Close the CSV file handle."""
        with self._lock:
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None
                self._writer = None
                self._initialized = False


class UnitOfWork(IUnitOfWork):
    """Unit of Work pattern for coordinating repository operations."""

    def __init__(
        self,
        run_directory: Path,
        csv_path: Path,
        csv_fieldnames: List[str],
        fs_service: IFileSystemService
    ):
        """Initialize unit of work.

        Args:
            run_directory: Directory for artifacts
            csv_path: Path to CSV results file
            csv_fieldnames: CSV column names
            fs_service: File system service
        """
        self._artifacts_repo = ArtifactsRepository(run_directory, fs_service)
        self._results_repo = ResultsRepository(csv_path, csv_fieldnames)
        self._committed = False

    @property
    def artifacts(self) -> IArtifactsRepository:
        """Access the artifacts repository."""
        return self._artifacts_repo

    @property
    def results(self) -> IResultsRepository:
        """Access the results repository."""
        return self._results_repo

    def commit(self) -> None:
        """Commit all pending changes."""
        self._results_repo.flush()
        self._committed = True

    def rollback(self) -> None:
        """Rollback any uncommitted changes.

        Note: This is a no-op for file-based storage. In a real implementation
        with transactional storage, this would undo changes.
        """
        pass

    def __enter__(self) -> "UnitOfWork":
        """Enter context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager and auto-commit if no exception."""
        exc_type = args[0] if len(args) > 0 else None
        try:
            if exc_type is None and not self._committed:
                self.commit()
        finally:
            self._results_repo.close()


__all__ = [
    "ArtifactsRepository",
    "ResultsRepository",
    "UnitOfWork",
]
