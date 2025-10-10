"""Flask application providing the llm-judge control panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, cast

from flask import Blueprint, Flask, Response, current_app, jsonify, request, send_from_directory

from .job_manager import JobManager

api_bp = Blueprint("api", __name__, url_prefix="/api")
frontend_bp = Blueprint("frontend", __name__)


def _manager() -> JobManager:
    return cast(JobManager, current_app.config["JOB_MANAGER"])


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
    dist_dir = Path(current_app.config["FRONTEND_DIST"]).resolve()
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


def create_app(config: Dict[str, Any] | None = None) -> Flask:
    """Factory for the Flask web application."""
    app = Flask(__name__)

    project_root = Path(__file__).resolve().parents[3]
    frontend_dist = project_root / "webui" / "dist"

    app_config = cast(Dict[str, Any], app.config)
    app_config.setdefault("FRONTEND_DIST", str(frontend_dist))
    app_config.setdefault("RUNS_OUTDIR", str(project_root / "results"))

    if config:
        app_config.update(config)

    manager = JobManager(outdir=Path(app_config["RUNS_OUTDIR"]))
    app.config["JOB_MANAGER"] = manager

    app.register_blueprint(api_bp)
    app.register_blueprint(frontend_bp)
    return app


app = create_app()
