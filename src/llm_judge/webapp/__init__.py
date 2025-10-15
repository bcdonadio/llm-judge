"""Flask application providing the llm-judge control panel."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, cast

from flask import Blueprint, Flask, Response, current_app, jsonify, request, send_from_directory

from .job_manager import JobManager
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

api_bp = Blueprint("api", __name__, url_prefix="/api")
frontend_bp = Blueprint("frontend", __name__)


def _load_dotenv(env_path: Path) -> None:
    """Load key=value pairs from a dotenv file without overriding existing values."""
    if not env_path.is_file():
        return

    if _load_dotenv_with_library(env_path):
        return

    _load_dotenv_manual(env_path)


def _load_dotenv_with_library(env_path: Path) -> bool:
    """Load dotenv via python-dotenv if available. Return True on success."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    load_dotenv(env_path, override=False)
    return True


def _load_dotenv_manual(env_path: Path) -> None:
    """Minimal .env parser for environments without python-dotenv installed."""
    for raw_line in env_path.read_text().splitlines():
        parsed = _parse_env_line(raw_line)
        if not parsed:
            continue
        key, value = parsed
        if key in os.environ:
            continue
        os.environ[key] = value


def _parse_env_line(raw_line: str) -> Optional[Tuple[str, str]]:
    """Parse a single dotenv line into a key/value pair."""
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].strip()
    if "=" not in line:
        return None

    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
        return None

    value = _strip_inline_comment(_strip_quotes(value.strip()))
    return key, value


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _strip_inline_comment(value: str) -> str:
    if "#" not in value:
        return value
    for idx, char in enumerate(value):
        if char != "#":
            continue
        if idx == 0 or value[idx - 1].isspace():
            return value[:idx].rstrip()
    return value


def _manager() -> JobManager:
    return cast(JobManager, current_app.config["JOB_MANAGER"])


def _load_supported_models(
    container: Optional[ServiceContainerType],
    *,
    base_url: str,
) -> List[Dict[str, Any]]:
    logger = logging.getLogger(__name__)
    api_client: IAPIClient | None = None
    close_client = False

    if container is not None and has_di_support:
        try:
            resolved_client = container.resolve(IAPIClient)
            api_client = cast(IAPIClient, resolved_client)
        except Exception:  # pragma: no cover - defensive
            api_client = None

    if api_client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.info("Skipping OpenRouter model preload: OPENROUTER_API_KEY not configured.")
            return []
        from ..infrastructure.api_client import OpenRouterClient

        api_client = OpenRouterClient(api_key=api_key, base_url=base_url)
        close_client = True

    assert api_client is not None

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


def _toggle_response(
    action: str,
    success_state: str,
    error_message: str,
    error_status: int = 400,
) -> Response:
    manager = _manager()
    method = getattr(manager, action)
    if method():
        return jsonify({"status": success_state})
    response = jsonify({"error": error_message})
    response.status_code = error_status
    return response


@api_bp.get("/health")
def api_health() -> Response:
    return jsonify({"status": "ok"})


@api_bp.get("/defaults")
def api_defaults() -> Response:
    return jsonify(_manager().defaults())


@api_bp.get("/state")
def api_state() -> Response:
    return jsonify(_manager().snapshot())


@api_bp.get("/models")
def api_models() -> Response:
    return jsonify({"models": _manager().get_supported_models()})


@api_bp.post("/run")
def api_start_run() -> Response:
    manager = _manager()
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    try:
        cfg = manager.start_run(payload)
    except ValueError:
        response = jsonify({"error": "Invalid configuration provided"})
        response.status_code = 400
        return response
    except RuntimeError:
        response = jsonify({"error": "A run is already in progress"})
        response.status_code = 409
        return response
    return jsonify({"status": "started", "config": cfg})


@api_bp.post("/pause")
def api_pause() -> Response:
    return _toggle_response("pause", "paused", "No active run to pause.")


@api_bp.post("/resume")
def api_resume() -> Response:
    return _toggle_response("resume", "running", "No paused run to resume.")


@api_bp.post("/cancel")
def api_cancel() -> Response:
    return _toggle_response("cancel", "cancelling", "No active run to cancel.")


@api_bp.get("/events")
def api_events() -> Response:
    manager = _manager()

    def generate() -> Any:
        yield from manager.event_stream()

    response = Response(generate(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@frontend_bp.route("/", defaults={"path": ""})
@frontend_bp.route("/<path:path>")
def serve_frontend(path: str) -> Response:
    dist_dir = Path(cast(str, current_app.config["FRONTEND_DIST"])).resolve()
    if not dist_dir.exists():
        response = jsonify(
            {
                "error": "Frontend assets not found.",
                "hint": "Run `npm install && npm run build` inside webui/.",
            }
        )
        response.status_code = 503
        return response

    if path:
        # Prevent path traversal attacks by ensuring resolved path is within dist_dir
        requested_path = (dist_dir / path).resolve()
        try:
            requested_path.relative_to(dist_dir)
        except ValueError:
            # Path is outside dist_dir - reject the request
            response = jsonify({"error": "Invalid path"})
            response.status_code = 400
            return response

        if requested_path.is_file():
            return send_from_directory(dist_dir, path)

    return send_from_directory(dist_dir, "index.html")


def create_app(
    config: Dict[str, Any] | None = None,
    container: Optional[ServiceContainerType] = None,
) -> Flask:
    """Factory for the Flask web application.

    Args:
        config: Optional Flask configuration dictionary
        container: Optional ServiceContainer for dependency injection.
                   When provided, the app uses DI-based runner factory.

    Returns:
        Configured Flask application instance
    """
    app = Flask(__name__)

    project_root = Path(__file__).resolve().parents[3]
    _load_dotenv(project_root / ".env")
    frontend_dist = project_root / "webui" / "dist"

    app_config = cast(Dict[str, Any], app.config)
    app_config.setdefault("FRONTEND_DIST", str(frontend_dist))
    openrouter_base_url = cast(str, app_config.setdefault("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))

    if config:
        app_config.update(config)
        if "OPENROUTER_BASE_URL" in config:
            openrouter_base_url = cast(str, app_config["OPENROUTER_BASE_URL"])

    outdir_value = app_config.get("RUNS_OUTDIR")
    if outdir_value is None:
        outdir_path = FileSystemService().create_temp_dir()
        app_config["RUNS_OUTDIR"] = str(outdir_path)
    else:
        outdir_path = Path(outdir_value)

    # Create runner factory with or without DI
    runner_factory_fn: Optional[
        Callable[[RunnerConfig, Callable[[RunnerEvent], None], RunnerControl], LLMJudgeRunner]
    ] = None

    if container is not None and has_di_support:
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

        runner_factory_fn = di_runner_factory

        # Store container in app extensions for potential access
        app.extensions = getattr(app, "extensions", {})
        app.extensions["service_container"] = container

    # Create JobManager with or without custom factory
    if runner_factory_fn is not None:
        manager = JobManager(outdir=outdir_path, runner_factory=runner_factory_fn)
    else:
        manager = JobManager(outdir=outdir_path)  # Uses default factory

    models_catalog = _load_supported_models(container, base_url=openrouter_base_url)
    manager.set_supported_models(models_catalog)
    app.config["OPENROUTER_MODELS"] = models_catalog

    app.config["JOB_MANAGER"] = manager
    outdir_str = str(manager.outdir)
    print(f"[Artifacts] Using output directory: {outdir_str}")
    app.logger.info("[Artifacts] Using output directory: %s", outdir_str)

    app.register_blueprint(api_bp)
    app.register_blueprint(frontend_bp)
    return app


# DEPRECATED: Global app instance for backward compatibility only.
# For production use, call create_app() directly.
# This will be removed in a future version.
app = create_app()


__all__ = ["create_app", "app"]
