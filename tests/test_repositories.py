# pyright: reportPrivateUsage=false
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from llm_judge.infrastructure.repositories import ArtifactsRepository, ResultsRepository, UnitOfWork


class DummyFS:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def write_json(self, path: Path, data: Any) -> None:
        self.calls.append({"path": path, "data": data})
        path.write_text("{}", encoding="utf-8")

    def create_temp_dir(self, prefix: str = "tmp-") -> Path:  # pragma: no cover
        return Path(prefix)


def test_artifacts_repository_saves_files(tmp_path: Path) -> None:
    fs = DummyFS()
    repo = ArtifactsRepository(tmp_path, fs)
    path = repo.save_completion("model/x", 1, "initial", {"ok": True})
    assert path.exists()
    judge_path = repo.save_judge_decision("model/x", 1, {"judge": "y"})
    assert judge_path.exists()


def test_results_repository_writes_and_closes(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    repo = ResultsRepository(csv_path, ["a", "b"])
    repo.add_result({"a": 1, "b": 2})
    repo.flush()
    repo.close()
    repo.flush()  # no-op after close

    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"a": "1", "b": "2"}]


def test_unit_of_work_context(tmp_path: Path) -> None:
    fs = DummyFS()
    uow = UnitOfWork(tmp_path / "run", tmp_path / "res.csv", ["c"], fs)
    with uow as unit:
        unit.results.add_result({"c": 5})
        unit.rollback()

    assert (tmp_path / "res.csv").exists()


def test_results_repository_handles_missing_writer(tmp_path: Path) -> None:
    csv_path = tmp_path / "lazy.csv"
    repo = ResultsRepository(csv_path, ["a"])
    repo._initialized = True
    repo._writer = None
    repo._file_handle = None
    repo.add_result({"a": 1})


def test_results_repository_get_path_and_double_close(tmp_path: Path) -> None:
    csv_path = tmp_path / "paths.csv"
    repo = ResultsRepository(csv_path, ["x"])
    assert repo.get_csv_path() is csv_path
    repo.close()
    repo.close()
