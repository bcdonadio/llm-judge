"""Flask application providing the llm-judge control panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from flask import Flask, Response, current_app, jsonify, request, send_from_directory

from .job_manager import JobManager


def create_app(config: Dict[str, Any] | None = None) -> Flask:
    """Factory for the Flask web application."""
    app = Flask(__name__)

    project_root = Path(__file__).resolve().parents[3]
    frontend_dist = project_root / "webui" / "dist"

    app.config.setdefault("FRONTEND_DIST", str(frontend_dist))
    app.config.setdefault("RUNS_OUTDIR", str(project_root / "results"))

    if config:
        app.config.update(config)

    manager = JobManager(outdir=Path(app.config["RUNS_OUTDIR"]))
    app.config["JOB_MANAGER"] = manager

    _register_routes(app)
    return app


def _register_routes(app: Flask) -> None:
    @app.get("/api/health")
    def health() -> Response:
        return jsonify({"status": "ok"})

    @app.get("/api/defaults")
    def defaults() -> Response:
        manager: JobManager = current_app.config["JOB_MANAGER"]
        return jsonify(manager.defaults())

    @app.get("/api/state")
    def state() -> Response:
        manager: JobManager = current_app.config["JOB_MANAGER"]
        return jsonify(manager.snapshot())

    @app.post("/api/run")
    def start_run() -> Response:
        manager: JobManager = current_app.config["JOB_MANAGER"]
        payload = request.get_json(silent=True) or {}
        try:
            cfg = manager.start_run(payload)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"status": "started", "config": cfg})

    @app.post("/api/pause")
    def pause_run() -> Response:
        manager: JobManager = current_app.config["JOB_MANAGER"]
        if manager.pause():
            return jsonify({"status": "paused"})
        return jsonify({"error": "No active run to pause."}), 400

    @app.post("/api/resume")
    def resume_run() -> Response:
        manager: JobManager = current_app.config["JOB_MANAGER"]
        if manager.resume():
            return jsonify({"status": "running"})
        return jsonify({"error": "No paused run to resume."}), 400

    @app.post("/api/cancel")
    def cancel_run() -> Response:
        manager: JobManager = current_app.config["JOB_MANAGER"]
        if manager.cancel():
            return jsonify({"status": "cancelling"})
        return jsonify({"error": "No active run to cancel."}), 400

    @app.get("/api/events")
    def events() -> Response:
        manager: JobManager = current_app.config["JOB_MANAGER"]

        def generate() -> Any:
            yield from manager.event_stream()

        response = Response(generate(), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path: str) -> Response:
        dist_dir = Path(current_app.config["FRONTEND_DIST"])
        if not dist_dir.exists():
            return (
                jsonify(
                    {
                        "error": "Frontend assets not found.",
                        "hint": "Run `npm install && npm run build` inside webui/.",
                    }
                ),
                503,
            )

        if path and (dist_dir / path).is_file():
            return send_from_directory(dist_dir, path)
        return send_from_directory(dist_dir, "index.html")


app = create_app()
