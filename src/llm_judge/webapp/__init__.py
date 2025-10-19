"""FastAPI application providing the llm-judge control panel."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import logging
import os
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, cast

import yaml
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from .job_manager import JobManager
from .websocket import WebSocketManager
from ..runner import RunnerConfig, RunnerControl, RunnerEvent, LLMJudgeRunner
from ..infrastructure.utility_services import FileSystemService
from ..services import IAPIClient

if TYPE_CHECKING:
    from ..container import ServiceContainer as ServiceContainerType
    from ..factories import RunnerFactory as RunnerFactoryType
else:
    ServiceContainerType = Any  # type: ignore[assignment]
    RunnerFactoryType = Any  # type: ignore[assignment]

try:
    from ..factories import RunnerFactory as _RuntimeRunnerFactory
except ImportError:
    has_di_support = False
    _runner_factory: Optional[Type[Any]] = None
else:
    has_di_support = True
    _runner_factory = _RuntimeRunnerFactory


# Pydantic models for request/response validation
class RunRequest(BaseModel):
    """Request model for starting a run."""

    models: Optional[List[str]] = None
    judge_model: Optional[str] = None
    limit: Optional[int] = None
    max_tokens: Optional[int] = None
    judge_max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    judge_temperature: Optional[float] = None
    sleep_s: Optional[float] = None
    verbose: Optional[bool] = None


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = "ok"


class ModelsResponse(BaseModel):
    """Response model for models list."""

    models: List[Dict[str, Any]]


class RunResponse(BaseModel):
    """Response model for starting a run."""

    status: str
    config: Dict[str, Any]


class ActionResponse(BaseModel):
    """Response model for pause/resume/cancel actions."""

    status: str


class ErrorResponse(BaseModel):
    """Response model for errors."""

    detail: str
    hint: Optional[str] = None


def _invalid_path_response() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(detail="Invalid path").model_dump(),
    )


def _missing_frontend_response() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            detail="Frontend assets not found.",
            hint="Run `npm install && npm run build` inside webui/.",
        ).model_dump(exclude_none=True),
    )


class ListenerEndpoint(BaseModel):
    host: str
    port: int = Field(ge=0)


class ListenSettings(BaseModel):
    frontend: ListenerEndpoint
    proxy: ListenerEndpoint


class InferenceSettings(BaseModel):
    endpoint: str
    key: str


class RetrySettings(BaseModel):
    max: int = Field(ge=0)
    timeout: int = Field(gt=0)


class DefaultRunSettings(BaseModel):
    max_rounds: int = Field(gt=0)
    judge: str
    models: List[str] = Field(min_length=1)


class ApplicationSettings(BaseModel):
    listen: ListenSettings
    inference: InferenceSettings
    retries: RetrySettings
    log_level: str
    defaults: DefaultRunSettings


def _load_yaml_config(config_path: Path) -> ApplicationSettings:
    if not config_path.is_file():
        raise RuntimeError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:  # pragma: no cover - yaml error formatting
        raise RuntimeError(f"Failed to parse configuration file: {config_path}") from exc

    if not isinstance(loaded, dict):
        raise RuntimeError("Configuration must be a mapping of keys to values")

    config_data = cast(Dict[str, Any], loaded)
    try:
        return ApplicationSettings(**config_data)
    except ValidationError as exc:  # pragma: no cover - delegated validation details
        raise RuntimeError(f"Invalid configuration file: {config_path}") from exc


def _configure_logging_level(level_name: str) -> None:
    numeric_level = getattr(logging, level_name.upper(), None)
    if not isinstance(numeric_level, int):
        raise RuntimeError(f"Invalid log level specified in configuration: {level_name}")
    logging.getLogger().setLevel(numeric_level)


def _settings_to_app_config(settings: ApplicationSettings) -> Dict[str, Any]:
    return {
        "OPENROUTER_BASE_URL": settings.inference.endpoint,
        "OPENROUTER_TIMEOUT": settings.retries.timeout,
        "OPENROUTER_MAX_RETRIES": settings.retries.max,
        "DEFAULT_MODELS": list(settings.defaults.models),
        "DEFAULT_JUDGE_MODEL": settings.defaults.judge,
        "DEFAULT_LIMIT": settings.defaults.max_rounds,
        "LISTEN_FRONTEND_HOST": settings.listen.frontend.host,
        "LISTEN_FRONTEND_PORT": settings.listen.frontend.port,
        "LISTEN_PROXY_HOST": settings.listen.proxy.host,
        "LISTEN_PROXY_PORT": settings.listen.proxy.port,
    }


def _settings_to_job_defaults(settings: ApplicationSettings) -> Dict[str, Any]:
    return {
        "models": list(settings.defaults.models),
        "judge_model": settings.defaults.judge,
        "limit": settings.defaults.max_rounds,
    }


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_model_list(value: Any, fallback: List[str]) -> List[str]:
    if isinstance(value, list):
        return [str(model) for model in value]  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]
    return list(fallback)


def _resolve_api_client(
    container: Optional[ServiceContainerType],
    *,
    base_url: str,
    timeout: Optional[int],
    max_retries: Optional[int],
) -> Tuple[Optional[IAPIClient], bool]:
    if container is not None and has_di_support:
        try:
            resolved_client = container.resolve(IAPIClient)
            return cast(IAPIClient, resolved_client), False
        except Exception:  # pragma: no cover - defensive
            pass

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None, False

    from ..infrastructure.api_client import OpenRouterClient

    client_kwargs: Dict[str, Any] = {}
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    if max_retries is not None:
        client_kwargs["max_retries"] = max_retries

    client = OpenRouterClient(api_key=api_key, base_url=base_url, **client_kwargs)
    return client, True


def _load_supported_models(
    container: Optional[ServiceContainerType],
    *,
    base_url: str,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> List[Dict[str, Any]]:
    logger = logging.getLogger(__name__)
    api_client, close_client = _resolve_api_client(
        container,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )

    if api_client is None:
        logger.info("Skipping OpenRouter model preload: OPENROUTER_API_KEY not configured.")
        return []

    try:
        models = api_client.list_models()
        logger.info("Loaded %d OpenRouter models", len(models))
        return models
    except Exception:  # pragma: no cover - defensive
        logger.warning("Unable to preload OpenRouter models", exc_info=True)
        return []
    finally:
        if close_client:
            try:
                api_client.close()
            except Exception:  # pragma: no cover - defensive
                logger.debug("Failed to close temporary OpenRouter client", exc_info=True)


def _setup_app_config(
    frontend_dist: Path,
    config: Dict[str, Any] | None,
) -> Tuple[Dict[str, Any], str, Path]:
    """Configure application settings and paths."""
    app_config: Dict[str, Any] = {}
    app_config.setdefault("FRONTEND_DIST", str(frontend_dist))
    openrouter_base_url = app_config.setdefault("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if config:
        app_config.update(config)
        if "OPENROUTER_BASE_URL" in config:
            openrouter_base_url = config["OPENROUTER_BASE_URL"]

    outdir_value = app_config.get("RUNS_OUTDIR")
    if outdir_value is None:
        outdir_path = FileSystemService().create_temp_dir()
        app_config["RUNS_OUTDIR"] = str(outdir_path)
    else:
        outdir_path = Path(outdir_value)

    return app_config, openrouter_base_url, outdir_path


def _setup_runner_factory(
    container: Optional[ServiceContainerType],
) -> Optional[Callable[[RunnerConfig, Callable[[RunnerEvent], None], RunnerControl], LLMJudgeRunner]]:
    """Create runner factory with or without DI."""
    if container is None or not has_di_support:
        return None

    if _runner_factory is None:
        raise RuntimeError("Dependency injection support is not available.")

    typed_container = container
    factory: RunnerFactoryType = _runner_factory(typed_container)

    def di_runner_factory(
        config: RunnerConfig,
        progress_callback: Callable[[RunnerEvent], None],
        control: RunnerControl,
    ) -> LLMJudgeRunner:
        return factory.create_runner(
            config=config,
            control=control,
            progress_callback=progress_callback,
        )

    return di_runner_factory


def _setup_websocket_manager() -> WebSocketManager:
    """Initialize the WebSocketManager."""
    return WebSocketManager()


def _register_health_routes(app: FastAPI) -> None:
    """Register health and config endpoints."""

    @app.get("/api/health", response_model=HealthResponse, tags=["health"])
    async def api_health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(status="ok")

    @app.get("/api/defaults", tags=["config"])
    async def api_defaults() -> Dict[str, Any]:
        """Get default configuration values."""
        manager: JobManager = cast(JobManager, app.state.job_manager)
        return manager.defaults()

    @app.get("/api/state", tags=["state"])
    async def api_state() -> Dict[str, Any]:
        """Get current job state snapshot."""
        manager: JobManager = cast(JobManager, app.state.job_manager)
        return manager.snapshot()

    @app.get("/api/models", response_model=ModelsResponse, tags=["config"])
    async def api_models() -> ModelsResponse:
        """Get list of supported models."""
        return ModelsResponse(models=app.state.job_manager.get_supported_models())


def _register_control_routes(app: FastAPI) -> None:
    """Register run control endpoints."""

    _register_run_endpoint(app)
    _register_pause_endpoint(app)
    _register_resume_endpoint(app)
    _register_cancel_endpoint(app)


def _register_run_endpoint(app: FastAPI) -> None:
    @app.post("/api/run", response_model=RunResponse, tags=["control"])
    async def api_start_run(request: RunRequest) -> RunResponse:
        """Start a new run with the provided configuration."""
        manager = app.state.job_manager
        payload = request.model_dump(exclude_none=True)
        try:
            cfg = manager.start_run(payload)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid configuration provided")
        except RuntimeError:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="A run is already in progress")
        return RunResponse(status="started", config=cfg)


def _register_pause_endpoint(app: FastAPI) -> None:
    @app.post("/api/pause", response_model=ActionResponse, tags=["control"])
    async def api_pause() -> ActionResponse:
        """Pause the current run."""
        manager = app.state.job_manager
        if manager.pause():
            return ActionResponse(status="paused")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No active run to pause.")


def _register_resume_endpoint(app: FastAPI) -> None:
    @app.post("/api/resume", response_model=ActionResponse, tags=["control"])
    async def api_resume() -> ActionResponse:
        """Resume a paused run."""
        manager = app.state.job_manager
        if manager.resume():
            return ActionResponse(status="running")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No paused run to resume.")


def _register_cancel_endpoint(app: FastAPI) -> None:
    @app.post("/api/cancel", response_model=ActionResponse, tags=["control"])
    async def api_cancel() -> ActionResponse:
        """Cancel the current run."""
        manager = app.state.job_manager
        if manager.cancel():
            return ActionResponse(status="cancelling")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No active run to cancel.")


def _register_websocket_routes(app: FastAPI) -> None:
    """Register WebSocket endpoints."""

    @app.websocket("/api/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """WebSocket endpoint for real-time event streaming.

        Clients connect to this endpoint to receive real-time updates about run progress.
        The connection sends an initial state snapshot followed by live event updates.
        """
        ws_manager: WebSocketManager = app.state.websocket_manager
        job_manager: JobManager = app.state.job_manager

        await ws_manager.connect(websocket)

        try:
            # Get initial state snapshot
            snapshot = job_manager.snapshot()
            status_payload = snapshot.get("status", {})
            history = snapshot.get("history", [])

            # Prepare initial events
            initial_events: List[Dict[str, Any]] = [{"type": "status", "payload": status_payload}]
            initial_events.extend(history)

            # Add summary and artifacts if available
            if "summary" in status_payload:
                initial_events.append({"type": "summary", "payload": {"summary": status_payload["summary"]}})
            if "artifacts" in status_payload:
                initial_events.append({"type": "artifacts", "payload": status_payload["artifacts"]})

            # Send events (this will block and handle keepalive)
            await ws_manager.send_events(websocket, initial_events)

        except Exception as exc:  # pragma: no cover
            logging.getLogger(__name__).debug("WebSocket connection error: %s", exc)


def _register_frontend_routes(app: FastAPI, dist_dir: Path) -> None:
    """Register frontend static file serving routes."""
    if not dist_dir.exists():
        _register_frontend_missing_route(app)
        return

    _mount_frontend_assets(app, dist_dir)
    _register_frontend_handler(app, dist_dir)


def _register_frontend_missing_route(app: FastAPI) -> None:
    @app.get("/", include_in_schema=False)
    async def frontend_missing() -> JSONResponse:
        """Fallback when frontend is not built."""
        payload = ErrorResponse(
            detail="Frontend assets not found.",
            hint="Run `npm install && npm run build` inside webui/.",
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload.model_dump(exclude_none=True),
        )


def _mount_frontend_assets(app: FastAPI, dist_dir: Path) -> None:
    """Mount the static assets directory for the SPA."""

    app.mount("/assets", StaticFiles(directory=dist_dir / "assets"), name="assets")


def _contains_encoded_traversal(raw_path: Optional[bytes]) -> bool:
    if not raw_path:
        return False
    lower = raw_path.lower()
    traversal_markers = (b"/../", b"..%2f", b"%2f..", b"%2e%2e")
    return any(marker in lower for marker in traversal_markers)


def _decode_frontend_path(full_path: str) -> Optional[str]:
    decoded = unquote(full_path)
    if decoded.startswith("/"):
        return None
    return decoded


def _is_traversal_path(decoded_path: str) -> bool:
    return ".." in PurePosixPath(decoded_path).parts


def _resolve_safe_path(dist_dir: Path, decoded_path: str) -> Optional[Path]:
    candidate = (dist_dir / decoded_path).resolve()
    try:
        candidate.relative_to(dist_dir)
    except ValueError:
        return None
    return candidate


def _normalise_raw_path(raw_path: Any) -> Optional[bytes]:
    if isinstance(raw_path, bytearray):
        return bytes(raw_path)
    if isinstance(raw_path, bytes):
        return raw_path
    return None


def _serve_frontend_request(dist_dir: Path, full_path: str, raw_path: Optional[bytes]) -> Response:
    if _contains_encoded_traversal(raw_path):
        return _invalid_path_response()

    decoded_path = _decode_frontend_path(full_path)
    if decoded_path is None or _is_traversal_path(decoded_path):
        return _invalid_path_response()

    requested_path = _resolve_safe_path(dist_dir, decoded_path)
    if requested_path is None:
        return _invalid_path_response()

    if requested_path.is_file():
        return FileResponse(requested_path)

    index_path = dist_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)

    return _missing_frontend_response()


def _register_frontend_handler(app: FastAPI, dist_dir: Path) -> None:
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend(full_path: str, request: Request) -> Response:
        """Serve frontend files with SPA fallback to index.html."""
        raw_path = _normalise_raw_path(request.scope.get("raw_path", b""))
        return _serve_frontend_request(dist_dir, full_path, raw_path)


def create_app(
    config: Dict[str, Any] | None = None,
    container: Optional[ServiceContainerType] = None,
) -> FastAPI:
    """Factory for the FastAPI web application.

    Args:
        config: Optional application configuration dictionary
        container: Optional ServiceContainer for dependency injection.
                   When provided, the app uses DI-based runner factory.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="LLM Judge API",
        description="API for evaluating language models with politically sensitive prompts",
        version="1.0.0",
    )

    # Load configuration settings
    project_root = Path(__file__).resolve().parents[3]
    settings_path = project_root / "config.yaml"
    settings = _load_yaml_config(settings_path)
    _configure_logging_level(settings.log_level)

    if not os.getenv("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = settings.inference.key

    frontend_dist = project_root / "webui" / "dist"

    base_config = _settings_to_app_config(settings)
    merged_config = dict(base_config)
    if config:
        merged_config.update(config)

    # Configure application
    app_config, openrouter_base_url, outdir_path = _setup_app_config(frontend_dist, merged_config)
    runner_factory_fn = _setup_runner_factory(container)
    websocket_manager = _setup_websocket_manager()

    # Store container in app state if provided
    if container is not None:
        app.state.service_container = container

    default_models_value = app_config.get("DEFAULT_MODELS", settings.defaults.models)
    models_list = _coerce_model_list(default_models_value, settings.defaults.models)
    limit_value = _coerce_int(app_config.get("DEFAULT_LIMIT")) or settings.defaults.max_rounds
    judge_value = app_config.get("DEFAULT_JUDGE_MODEL", settings.defaults.judge) or settings.defaults.judge

    job_defaults: Dict[str, Any] = _settings_to_job_defaults(settings)
    job_defaults["models"] = models_list
    job_defaults["judge_model"] = judge_value
    job_defaults["limit"] = limit_value

    # Create JobManager
    if runner_factory_fn is not None:
        manager = JobManager(outdir=outdir_path, runner_factory=runner_factory_fn, defaults=job_defaults)
    else:
        manager = JobManager(outdir=outdir_path, defaults=job_defaults)

    # Connect WebSocket manager to JobManager for event broadcasting
    manager.set_websocket_manager(websocket_manager)

    # Load and configure models
    timeout_arg = _coerce_int(app_config.get("OPENROUTER_TIMEOUT"))
    retries_arg = _coerce_int(app_config.get("OPENROUTER_MAX_RETRIES"))
    models_catalog = _load_supported_models(
        container,
        base_url=openrouter_base_url,
        timeout=timeout_arg,
        max_retries=retries_arg,
    )
    manager.set_supported_models(models_catalog)
    app_config["OPENROUTER_MODELS"] = models_catalog

    # Store state
    app.state.job_manager = manager
    app.state.websocket_manager = websocket_manager
    app.state.config = app_config
    app.state.settings = settings
    outdir_str = str(manager.outdir)
    logging.getLogger(__name__).info("[Artifacts] Using output directory: %s", outdir_str)

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    _register_health_routes(app)
    _register_control_routes(app)
    _register_websocket_routes(app)
    dist_dir = Path(app.state.config["FRONTEND_DIST"]).resolve()
    _register_frontend_routes(app, dist_dir)

    return app


# DEPRECATED: Global app instance for backward compatibility only.
# For production use, call create_app() directly.
# This will be removed in a future version.
app = create_app()


__all__ = ["create_app", "app"]
