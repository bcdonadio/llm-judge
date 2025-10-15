from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from llm_judge.container import (
    ServiceContainer,
    create_container,
)
from llm_judge.domain import ModelResponse
from llm_judge.factories import (
    ConfigurationBuilder,
    RunnerFactory,
    UnitOfWorkFactory,
)
from llm_judge.runner import RunnerConfig
from llm_judge.services import (
    IAPIClient,
    IFileSystemService,
    IJudgeService,
    IPromptsManager,
)


class RunnerStubAPIClient:
    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        metadata: dict[str, str],
        response_format: dict[str, Any] | None = None,
        **_: Any,
    ) -> ModelResponse:
        return ModelResponse(text="", raw_payload={"model": model, "messages": messages})

    def close(self) -> None:  # pragma: no cover
        pass

    def list_models(self) -> List[Dict[str, Any]]:  # pragma: no cover
        return []


class RunnerStubJudgeService:
    def evaluate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return args, kwargs


class RunnerStubPromptsManager:
    def get_core_prompts(self) -> Any:  # pragma: no cover
        return []


class RunnerStubFS:
    def write_json(self, path: Path, data: Any) -> None:
        path.write_text(json.dumps(data))

    def create_temp_dir(self, prefix: str = "tmp-") -> Path:  # pragma: no cover
        return Path(prefix)


class Closeable:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_service_container_register_and_resolve() -> None:
    container = ServiceContainer()
    sentinel = object()
    container.register_singleton("svc", sentinel)
    assert container.resolve("svc") is sentinel

    factory_counter = {"count": 0}

    def factory() -> Dict[str, int]:
        factory_counter["count"] += 1
        return {"created": factory_counter["count"]}

    container.register_factory("factory", factory)
    first = container.resolve("factory")
    second = container.resolve("factory")
    assert first is second
    assert first["created"] == 1

    class Concrete:
        def __init__(self) -> None:
            self.value = 5

    instance = container.resolve(Concrete)
    assert isinstance(instance, Concrete)

    class NeedsArg:
        def __init__(self, value: int) -> None:
            self.value = value

    with pytest.raises(ValueError):
        container.resolve(NeedsArg)

    with pytest.raises(ValueError):
        container.resolve("missing")

    closable = Closeable()
    container.register_singleton("closable", closable)
    container.clear()
    assert closable.closed is True


def test_create_container_registers_dependencies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "abc123")

    calls: Dict[str, Any] = {}

    class StubConfigManager:
        def __init__(self, *, config_file: Any, auto_reload: bool) -> None:
            calls["config_manager"] = {"config_file": config_file, "auto_reload": auto_reload}
            self._data: Dict[str, Any] = {}

        def merge(self, data: Dict[str, Any]) -> None:
            self._data.update(data)

    class StubAPIClient(Closeable):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            calls["api_client"] = kwargs

        def list_models(self) -> List[Dict[str, Any]]:
            return []

    class StubPromptsManager:
        def __init__(self, **kwargs: Any) -> None:
            calls["prompts_manager"] = kwargs

    class StubJudgeService:
        def __init__(self, **kwargs: Any) -> None:
            calls["judge_service"] = kwargs

    class StubSimple:
        def __init__(self) -> None:
            pass

    monkeypatch.setattr("llm_judge.infrastructure.config_manager.ConfigurationManager", StubConfigManager)
    monkeypatch.setattr("llm_judge.infrastructure.api_client.OpenRouterClient", StubAPIClient)
    monkeypatch.setattr("llm_judge.infrastructure.prompts_manager.PromptsManager", StubPromptsManager)
    monkeypatch.setattr("llm_judge.infrastructure.judge_service.JudgeService", StubJudgeService)
    monkeypatch.setattr("llm_judge.infrastructure.utility_services.TimeService", StubSimple)
    monkeypatch.setattr("llm_judge.infrastructure.utility_services.RefusalDetector", StubSimple)
    monkeypatch.setattr("llm_judge.infrastructure.utility_services.ResponseParser", StubSimple)
    monkeypatch.setattr("llm_judge.infrastructure.utility_services.FileSystemService", StubSimple)

    container = create_container({"config_file": str(tmp_path / "cfg.yaml"), "prompts_file": "extras.yaml"})

    from llm_judge.services import (
        IAPIClient,
        IConfigurationManager,
        IJudgeService,
        IPromptsManager,
        IFileSystemService,
    )

    assert isinstance(container.resolve(IConfigurationManager), StubConfigManager)
    assert isinstance(container.resolve(IAPIClient), StubAPIClient)
    assert isinstance(container.resolve(IPromptsManager), StubPromptsManager)
    assert isinstance(container.resolve(IJudgeService), StubJudgeService)
    assert isinstance(container.resolve(IFileSystemService), StubSimple)

    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(RuntimeError):
        create_container({})


def test_runner_factory_and_unit_of_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    container = ServiceContainer()
    container.register_singleton(IAPIClient, RunnerStubAPIClient())
    container.register_singleton(IJudgeService, RunnerStubJudgeService())
    container.register_singleton(IPromptsManager, RunnerStubPromptsManager())
    container.register_singleton(IFileSystemService, RunnerStubFS())

    runner_factory = RunnerFactory(container)
    config = RunnerConfig(
        models=("a",),
        judge_model="judge",
        outdir=tmp_path,
        max_tokens=10,
        judge_max_tokens=20,
        temperature=0.1,
        judge_temperature=0.2,
        sleep_s=0.0,
    )
    runner = runner_factory.create_runner(config)
    assert runner.config.judge_model == "judge"

    uow_factory = UnitOfWorkFactory(container)

    def fake_strftime(*_: Any) -> str:
        return "20240101T000000Z"

    monkeypatch.setattr("time.strftime", fake_strftime)
    unit_default = uow_factory.create_unit_of_work(tmp_path)
    assert unit_default.artifacts.get_run_directory() == tmp_path / "runs" / "20240101T000000Z"

    unit = uow_factory.create_unit_of_work(tmp_path, timestamp="20240202T000000Z", csv_fieldnames=["a"])
    assert unit.artifacts.get_run_directory() == tmp_path / "runs" / "20240202T000000Z"


def test_configuration_builder_builds_and_validates(tmp_path: Path) -> None:
    builder = ConfigurationBuilder()
    with pytest.raises(ValueError):
        builder.build()

    cfg = (
        builder.with_models(["x"])
        .with_judge_model("judge")
        .with_outdir(tmp_path)
        .with_max_tokens(100)
        .with_judge_max_tokens(200)
        .with_temperature(0.5)
        .with_judge_temperature(0.3)
        .with_sleep(0.1)
        .with_limit(5)
        .with_verbose()
        .with_color()
        .build()
    )
    assert cfg.models == ["x"]
    assert cfg.use_color is True

    with pytest.raises(ValueError):
        builder.with_max_tokens(0)
    with pytest.raises(ValueError):
        builder.with_judge_max_tokens(0)
    with pytest.raises(ValueError):
        builder.with_temperature(-1)
    with pytest.raises(ValueError):
        builder.with_judge_temperature(3)
    with pytest.raises(ValueError):
        builder.with_sleep(-0.1)
    with pytest.raises(ValueError):
        builder.with_limit(0)
