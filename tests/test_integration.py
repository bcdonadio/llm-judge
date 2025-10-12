"""Integration tests for thread-safety and concurrent operations."""

import concurrent.futures
import threading
import time
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch

import pytest

from llm_judge.container import create_container
from llm_judge.domain import Prompt, ModelResponse, JudgeDecision
from llm_judge.factories import RunnerFactory
from llm_judge.infrastructure.api_client import OpenRouterClient
from llm_judge.infrastructure.config_manager import ConfigurationManager
from llm_judge.infrastructure.judge_service import JudgeService
from llm_judge.infrastructure.prompts_manager import PromptsManager
from llm_judge.infrastructure.repositories import (
    ArtifactsRepository,
    ResultsRepository,
    UnitOfWork,
)
from llm_judge.infrastructure.utility_services import FileSystemService
from llm_judge.runner import RunnerConfig
from llm_judge.services import (
    IAPIClient,
    IPromptsManager,
    IJudgeService,
    IConfigurationManager,
)


class TestOpenRouterClientThreadSafety:
    """Test thread-safety of OpenRouterClient."""

    @pytest.fixture
    def mock_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set up mock API key."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")

    def test_concurrent_initialization(self, mock_api_key: None) -> None:
        """Test that concurrent initialization of client is thread-safe."""
        client = OpenRouterClient(api_key="test-key-123")
        results: List[bool] = []
        errors: List[Exception] = []

        def init_client() -> None:
            try:
                # Trigger lazy initialization via chat_completion which calls _ensure_client
                with patch("openai.OpenAI") as mock_openai_class:
                    mock_instance = MagicMock()
                    mock_openai_class.return_value = mock_instance
                    with patch("httpx.Client"):
                        try:
                            # This will fail but triggers initialization
                            client.chat_completion(
                                model="test",
                                messages=[],
                                max_tokens=10,
                                temperature=0.7,
                                metadata={},
                            )
                        except Exception:
                            pass  # Expected to fail in test
                results.append(True)
            except Exception as e:
                errors.append(e)

        # Run 10 threads concurrently trying to initialize
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(init_client) for _ in range(10)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        assert all(results)

    def test_concurrent_chat_completion(self, mock_api_key: None, tmp_path: Path) -> None:
        """Test concurrent chat completion requests."""
        client = OpenRouterClient(api_key="test-key-123")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model_dump.return_value = {"test": "data"}

        with patch.object(client, "_ensure_client") as mock_ensure:
            mock_openai = MagicMock()
            raw_resp = MagicMock()
            raw_resp.parse.return_value = mock_response
            raw_resp.http_response.status_code = 200
            raw_resp.http_response.elapsed = None
            mock_openai.chat.completions.with_raw_response.create.return_value = raw_resp
            mock_ensure.return_value = mock_openai

            results: List[ModelResponse] = []
            errors: List[Exception] = []

            def make_request(idx: int) -> None:
                try:
                    response = client.chat_completion(
                        model="test-model",
                        messages=[{"role": "user", "content": f"Test {idx}"}],
                        max_tokens=100,
                        temperature=0.7,
                        metadata={"title": "Test"},
                    )
                    results.append(response)
                except Exception as e:
                    errors.append(e)

            # Run 20 concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(make_request, i) for i in range(20)]
                concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 20
        assert all(isinstance(r, ModelResponse) for r in results)


class TestPromptsManagerThreadSafety:
    """Test thread-safety of PromptsManager."""

    def test_concurrent_load_prompts(self, tmp_path: Path) -> None:
        """Test concurrent loading of prompts."""
        # Create a test prompts file
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text(
            """
core_prompts:
  - "Test prompt 1"
  - "Test prompt 2"
follow_up: "Follow up"
"""
        )

        manager = PromptsManager(prompts_file)
        results: List[List[Prompt]] = []
        errors: List[Exception] = []

        def load_prompts() -> None:
            try:
                prompts = manager.get_core_prompts()
                results.append(prompts)
            except Exception as e:
                errors.append(e)

        # Run 15 threads concurrently loading prompts
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(load_prompts) for _ in range(15)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 15
        # All should return the same prompts
        assert all(len(p) == 2 for p in results)

    def test_concurrent_reload(self, tmp_path: Path) -> None:
        """Test concurrent reload operations."""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text(
            """
core_prompts:
  - "Original"
"""
        )

        manager = PromptsManager(prompts_file)
        # Load initial prompts
        initial = manager.get_core_prompts()
        assert len(initial) == 1

        # Update file
        prompts_file.write_text(
            """
core_prompts:
  - "Updated"
  - "New"
"""
        )

        results: List[int] = []
        errors: List[Exception] = []

        def reload_prompts() -> None:
            try:
                manager.reload()
                prompts = manager.get_core_prompts()
                results.append(len(prompts))
            except Exception as e:
                errors.append(e)

        # Run concurrent reloads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(reload_prompts) for _ in range(10)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0
        # After reload, all should see 2 prompts
        assert all(count == 2 for count in results)


class TestJudgeServiceThreadSafety:
    """Test thread-safety of JudgeService."""

    @pytest.fixture
    def mock_api_client(self) -> MagicMock:
        """Create a mock API client."""
        mock_client = MagicMock(spec=IAPIClient)
        # Create response with proper JSON structure
        response_json = (
            '{"initial": {"refusal": false, "completeness": 5, '
            '"sourcing_quality": "good"}, "followup": {"refusal": false, '
            '"completeness": 5, "sourcing_quality": "good"}, '
            '"asymmetry_leading": "none", "final_notes": "test"}'
        )
        mock_response = ModelResponse(
            text=response_json,
            raw_payload={"test": "data"},
            finish_reason="stop",
        )
        mock_client.chat_completion.return_value = mock_response
        return mock_client

    def test_concurrent_evaluate(self, mock_api_client: MagicMock, tmp_path: Path) -> None:
        """Test concurrent judge evaluations."""
        # Create judge config
        config_file = tmp_path / "judge_config.yaml"
        config_file.write_text(
            """
instructions: "Test instructions"
schema:
  decision: string
  reason: string
  score: integer
system: "Test system prompt"
"""
        )

        service = JudgeService(
            api_client=mock_api_client,
            config_file=config_file,
        )

        results: List[JudgeDecision] = []
        errors: List[Exception] = []

        def evaluate_response(idx: int) -> None:
            try:
                from llm_judge.domain import RunConfiguration

                config = RunConfiguration(
                    models=[f"model-{idx}"],
                    judge_model="judge-model",
                    outdir=tmp_path,
                    max_tokens=1000,
                    judge_max_tokens=2000,
                    temperature=0.7,
                    judge_temperature=0.0,
                    sleep_s=0.1,
                )

                initial_response = ModelResponse(
                    text=f"Response {idx}",
                    raw_payload={},
                )
                follow_response = ModelResponse(
                    text=f"Follow {idx}",
                    raw_payload={},
                )

                decision = service.evaluate(
                    prompt=f"Prompt {idx}",
                    initial_response=initial_response,
                    follow_response=follow_response,
                    config=config,
                )
                results.append(decision)
            except Exception as e:
                errors.append(e)

        # Run 25 concurrent evaluations
        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(evaluate_response, i) for i in range(25)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 25
        assert all(isinstance(d, JudgeDecision) for d in results)
        assert all(d.success for d in results)


class TestConfigurationManagerThreadSafety:
    """Test thread-safety of ConfigurationManager."""

    def test_concurrent_get_operations(self, tmp_path: Path) -> None:
        """Test concurrent read operations."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
api:
  timeout: 30
  retries: 3
models:
  - model1
  - model2
"""
        )

        manager = ConfigurationManager(config_file)
        results: List[Any] = []
        errors: List[Exception] = []

        def read_config(key: str) -> None:
            try:
                value = manager.get(key)
                results.append(value)
            except Exception as e:
                errors.append(e)

        # Run concurrent reads
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for _ in range(10):
                futures.append(executor.submit(read_config, "api.timeout"))
                futures.append(executor.submit(read_config, "models"))
            concurrent.futures.wait(futures)

        assert len(errors) == 0
        assert len(results) == 20

    def test_concurrent_set_and_get(self, tmp_path: Path) -> None:
        """Test concurrent set and get operations."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: initial")

        manager = ConfigurationManager(config_file)
        errors: List[Exception] = []
        final_values: List[str] = []

        def set_and_get(idx: int) -> None:
            try:
                manager.set(f"key{idx}", f"value{idx}")
                # Small delay then try to get
                time.sleep(0.01)
                value = manager.get(f"key{idx}", default=None)
                if value:
                    final_values.append(value)
            except Exception as e:
                errors.append(e)

        # Run concurrent sets and gets
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(set_and_get, i) for i in range(10)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0
        # All sets should have succeeded
        assert len(final_values) == 10


class TestRepositoriesThreadSafety:
    """Test thread-safety of repository classes."""

    def test_artifacts_repository_concurrent_saves(self, tmp_path: Path) -> None:
        """Test concurrent artifact saves."""
        fs_service = FileSystemService()
        repo = ArtifactsRepository(run_directory=tmp_path, fs_service=fs_service)

        results: List[Path] = []
        errors: List[Exception] = []

        def save_artifact(idx: int) -> None:
            try:
                path = repo.save_completion(
                    model=f"model-{idx % 3}",
                    prompt_index=idx,
                    step="test",
                    data={"content": f"Test {idx}"},
                )
                results.append(path)
            except Exception as e:
                errors.append(e)

        # Run 30 concurrent saves
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(save_artifact, i) for i in range(30)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 30
        # All files should exist
        assert all(p.exists() for p in results)

    def test_results_repository_concurrent_writes(self, tmp_path: Path) -> None:
        """Test concurrent CSV writes."""
        csv_path = tmp_path / "results.csv"
        repo = ResultsRepository(
            csv_path=csv_path,
            fieldnames=["model", "prompt_id", "score"],
        )

        errors: List[Exception] = []
        lock = threading.Lock()
        write_count = 0

        def write_result(idx: int) -> None:
            nonlocal write_count
            try:
                repo.add_result(
                    {
                        "model": f"model-{idx}",
                        "prompt_id": f"prompt-{idx}",
                        "score": str(idx % 10),
                    }
                )
                with lock:
                    write_count += 1
            except Exception as e:
                errors.append(e)

        # Run 50 concurrent writes
        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(write_result, i) for i in range(50)]
            concurrent.futures.wait(futures)

        repo.flush()
        repo.close()

        assert len(errors) == 0
        assert write_count == 50
        # Verify CSV content
        assert csv_path.exists()
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 51  # header + 50 rows


class TestUnitOfWorkThreadSafety:
    """Test thread-safety of UnitOfWork."""

    def test_concurrent_unit_of_work_usage(self, tmp_path: Path) -> None:
        """Test concurrent UnitOfWork instances."""
        errors: List[Exception] = []
        completed: List[int] = []

        def run_transaction(idx: int) -> None:
            try:
                run_dir = tmp_path / f"run-{idx}"
                csv_path = run_dir / "results.csv"
                fs_service = FileSystemService()

                with UnitOfWork(
                    run_directory=run_dir,
                    csv_path=csv_path,
                    csv_fieldnames=["model", "score"],
                    fs_service=fs_service,
                ) as uow:
                    # Save some artifacts
                    uow.artifacts.save_completion(
                        model="test-model",
                        prompt_index=0,
                        step="initial",
                        data={"content": f"Test {idx}"},
                    )

                    # Add results
                    uow.results.add_result({"model": "test-model", "score": str(idx)})

                    uow.commit()
                    completed.append(idx)
            except Exception as e:
                errors.append(e)

        # Run 10 concurrent transactions
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_transaction, i) for i in range(10)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(completed) == 10


class TestServiceContainerThreadSafety:
    """Test thread-safety of ServiceContainer."""

    def test_concurrent_service_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test concurrent service resolution from container."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        container = create_container()
        results: Dict[str, List[Any]] = {"clients": [], "managers": [], "services": []}
        errors: List[Exception] = []

        def resolve_services() -> None:
            try:
                client = container.resolve(IAPIClient)
                manager = container.resolve(IPromptsManager)
                service = container.resolve(IJudgeService)

                results["clients"].append(client)
                results["managers"].append(manager)
                results["services"].append(service)
            except Exception as e:
                errors.append(e)

        # Run 15 threads concurrently resolving services
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(resolve_services) for _ in range(15)]
            concurrent.futures.wait(futures)

        container.clear()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results["clients"]) == 15
        assert len(results["managers"]) == 15
        assert len(results["services"]) == 15

        # Singletons should all be the same instance
        assert len(set(id(c) for c in results["clients"])) == 1
        assert len(set(id(m) for m in results["managers"])) == 1


class TestEndToEndConcurrency:
    """End-to-end integration tests with concurrent operations."""

    def test_concurrent_runner_factory_creation(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test concurrent runner creation via factory."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        container = create_container()
        factory = RunnerFactory(container)

        results: List[Any] = []
        errors: List[Exception] = []

        def create_runner(idx: int) -> None:
            try:
                config = RunnerConfig(
                    models=[f"model-{idx}"],
                    judge_model="judge-model",
                    outdir=tmp_path / f"run-{idx}",
                    max_tokens=1000,
                    judge_max_tokens=2000,
                    temperature=0.7,
                    judge_temperature=0.0,
                    sleep_s=0.1,
                )
                runner = factory.create_runner(config)
                results.append(runner)
            except Exception as e:
                errors.append(e)

        # Create 10 runners concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_runner, i) for i in range(10)]
            concurrent.futures.wait(futures)

        container.clear()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 10

    def test_stress_test_all_components(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Stress test with high concurrency across all components."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        # Create prompts file
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text(
            """
core_prompts:
  - "Test 1"
  - "Test 2"
follow_up: "Follow"
"""
        )

        # Create config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
api:
  timeout: 30
models:
  - model1
  - model2
"""
        )

        container = create_container({"config_file": str(config_file)})

        errors: List[Exception] = []
        operations_completed = 0
        lock = threading.Lock()

        def mixed_operations(idx: int) -> None:
            nonlocal operations_completed
            try:
                # Resolve services
                config_manager = container.resolve(IConfigurationManager)
                prompts_manager = PromptsManager(prompts_file)

                # Perform operations
                _ = config_manager.get("api.timeout")
                _ = prompts_manager.get_core_prompts()

                # Create artifacts
                fs_service = FileSystemService()
                artifacts_repo = ArtifactsRepository(
                    run_directory=tmp_path / f"artifacts-{idx}",
                    fs_service=fs_service,
                )
                artifacts_repo.save_completion(
                    model=f"model-{idx}",
                    prompt_index=0,
                    step="test",
                    data={"test": "data"},
                )

                with lock:
                    operations_completed += 1

            except Exception as e:
                errors.append(e)

        # Run 50 threads doing mixed operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(mixed_operations, i) for i in range(50)]
            concurrent.futures.wait(futures)

        container.clear()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert operations_completed == 50
