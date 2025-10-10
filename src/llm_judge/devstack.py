from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

DEFAULT_BACKEND_HOST = "127.0.0.1"
DEFAULT_BACKEND_PORT = 5000
DEFAULT_FRONTEND_HOST = "127.0.0.1"
DEFAULT_FRONTEND_PORT = 5173
DEFAULT_LOG_DIR = Path(".devstack")
DEFAULT_PID_FILE = DEFAULT_LOG_DIR / "devstack.pid"
DEFAULT_STATE_FILE = DEFAULT_LOG_DIR / "state.json"
DEFAULT_CONTROLLER_LOG = DEFAULT_LOG_DIR / "controller.log"
DEFAULT_BACKEND_LOG = DEFAULT_LOG_DIR / "backend.log"
DEFAULT_FRONTEND_LOG = DEFAULT_LOG_DIR / "frontend.log"
FRONTEND_DIRNAME = "webui"

SHUTDOWN_WAIT_SECONDS = 10.0
STATE_WAIT_SECONDS = 30.0


class FileLogger:
    """Very small helper that writes timestamped lines to a log file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self._handle.write(f"[{timestamp}] {message}\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "FileLogger":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass
class DevStackConfig:
    backend_host: str = DEFAULT_BACKEND_HOST
    backend_port: int = DEFAULT_BACKEND_PORT
    frontend_host: str = DEFAULT_FRONTEND_HOST
    frontend_port: int = DEFAULT_FRONTEND_PORT
    python_executable: str = sys.executable
    npm_command: str = os.environ.get("WEBUI_NPM", "npm")
    flask_app: str = "llm_judge.webapp:create_app"
    project_root: Path = Path(__file__).resolve().parents[2]
    log_dir: Path = DEFAULT_LOG_DIR
    pid_file: Path = DEFAULT_PID_FILE
    state_file: Path = DEFAULT_STATE_FILE
    controller_log: Path = DEFAULT_CONTROLLER_LOG
    backend_log: Path = DEFAULT_BACKEND_LOG
    frontend_log: Path = DEFAULT_FRONTEND_LOG


def _prepare_backend_env(config: DevStackConfig) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("FLASK_APP", config.flask_app)
    env.setdefault("FLASK_ENV", "development")
    env.setdefault("FLASK_DEBUG", "1")
    return env


def _install_shutdown_handlers(stop_flag: Dict[str, bool], controller: FileLogger) -> None:
    def handle_shutdown(signum: int, _frame: object) -> None:
        if stop_flag["requested"]:
            return
        stop_flag["requested"] = True
        sig_name = signal.Signals(signum).name
        controller.log(f"Received signal {sig_name}; beginning shutdown")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT, signal.SIGHUP):
        signal.signal(sig, handle_shutdown)


def _monitor_processes(
    backend_proc: subprocess.Popen[Any],
    frontend_proc: subprocess.Popen[Any],
    controller: FileLogger,
    stop_flag: Dict[str, bool],
) -> int:
    while True:
        if stop_flag["requested"]:
            return 0

        backend_exit = backend_proc.poll()
        if backend_exit is not None:
            controller.log(f"Backend process exited with code {backend_exit}; shutting down dev stack")
            return backend_exit or 0

        frontend_exit = frontend_proc.poll()
        if frontend_exit is not None:
            controller.log(f"Frontend process exited with code {frontend_exit}; shutting down dev stack")
            return frontend_exit or 0

        time.sleep(0.5)


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _cleanup_runtime_files(config: DevStackConfig) -> None:
    _safe_unlink(config.pid_file)
    _safe_unlink(config.state_file)


def _wait_for_process_shutdown(pid: int) -> bool:
    deadline = time.time() + SHUTDOWN_WAIT_SECONDS
    while time.time() < deadline:
        if not process_alive(pid):
            return True
        time.sleep(0.2)
    return not process_alive(pid)


def _launch_dev_servers(
    config: DevStackConfig,
    backend_env: Dict[str, str],
    backend_log_handle: Any,
    frontend_log_handle: Any,
    controller: FileLogger,
) -> Tuple[subprocess.Popen[Any], subprocess.Popen[Any]]:
    backend_cmd = [
        config.python_executable,
        "-m",
        "flask",
        "--app",
        config.flask_app,
        "run",
        "--debug",
        "--host",
        config.backend_host,
        "--port",
        str(config.backend_port),
    ]
    frontend_cmd = [
        config.npm_command,
        "run",
        "dev",
        "--",
        "--host",
        config.frontend_host,
        "--port",
        str(config.frontend_port),
    ]

    controller.log("Launching Flask development server")
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=config.project_root,
        env=backend_env,
        stdout=backend_log_handle,
        stderr=subprocess.STDOUT,
    )

    controller.log("Launching webui development server (Vite)")
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=config.project_root / FRONTEND_DIRNAME,
        stdout=frontend_log_handle,
        stderr=subprocess.STDOUT,
    )

    return backend_proc, frontend_proc


def _write_state_file(
    config: DevStackConfig,
    backend_proc: subprocess.Popen[Any],
    frontend_proc: subprocess.Popen[Any],
) -> None:
    state: Dict[str, Any] = {
        "controller_pid": os.getpid(),
        "process_group": os.getpid(),
        "backend_pid": backend_proc.pid,
        "frontend_pid": frontend_proc.pid,
        "backend_url": f"http://{config.backend_host}:{config.backend_port}",
        "frontend_url": f"http://{config.frontend_host}:{config.frontend_port}",
        "debugger_url": f"http://{config.backend_host}:{config.backend_port}/__debugger__",
        "logs": {
            "controller": str(config.controller_log),
            "backend": str(config.backend_log),
            "frontend": str(config.frontend_log),
        },
        "started_at": time.time(),
    }
    atomic_write(config.state_file, json.dumps(state, indent=2))


def _run_devstack(config: DevStackConfig, controller: FileLogger) -> int:
    ensure_frontend_dependencies(config, controller)
    backend_env = _prepare_backend_env(config)

    backend_proc: Optional[subprocess.Popen[Any]] = None
    frontend_proc: Optional[subprocess.Popen[Any]] = None

    with (
        config.backend_log.open("a", encoding="utf-8") as backend_log_handle,
        config.frontend_log.open("a", encoding="utf-8") as frontend_log_handle,
    ):
        try:
            backend_proc, frontend_proc = _launch_dev_servers(
                config,
                backend_env,
                backend_log_handle,
                frontend_log_handle,
                controller,
            )
            _write_state_file(config, backend_proc, frontend_proc)
            controller.log("State file written; dev stack ready")

            stop_flag: Dict[str, bool] = {"requested": False}
            _install_shutdown_handlers(stop_flag, controller)
            return _monitor_processes(backend_proc, frontend_proc, controller, stop_flag)
        finally:
            if frontend_proc is not None:
                terminate_process(frontend_proc, "frontend", controller)
            if backend_proc is not None:
                terminate_process(backend_proc, "backend", controller)


def _cleanup_controller_resources(config: DevStackConfig, controller: FileLogger, controller_pid: int) -> None:
    try:
        if config.pid_file.exists() and read_pid(config.pid_file) == controller_pid:  # pragma: no branch
            config.pid_file.unlink()
    except OSError:  # pragma: no cover - best-effort cleanup
        pass
    try:
        if config.state_file.exists():  # pragma: no branch
            config.state_file.unlink()
    except OSError:  # pragma: no cover - best-effort cleanup
        pass
    controller.log("Clean up complete; shutting down controller")
    controller.close()


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def read_pid(pid_path: Path) -> Optional[int]:
    try:
        content = pid_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if not content:
        return None
    try:
        return int(content)
    except ValueError:
        return None


def atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(data, encoding="utf-8")
    tmp_path.replace(path)


def ensure_frontend_dependencies(config: DevStackConfig, logger: FileLogger) -> None:
    node_modules = config.project_root / FRONTEND_DIRNAME / "node_modules"
    if node_modules.exists():
        logger.log("webui dependencies already present (node_modules exists)")
        return
    logger.log("node_modules not found; running npm install before starting dev server")
    frontend_log_path = config.frontend_log
    with frontend_log_path.open("a", encoding="utf-8") as frontend_log:
        result = subprocess.run(
            [config.npm_command, "install"],
            cwd=config.project_root / FRONTEND_DIRNAME,
            stdout=frontend_log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            "npm install failed while preparing webui dependencies. " f"Inspect {frontend_log_path} for details."
        )
    logger.log("npm install completed successfully")


def terminate_process(proc: subprocess.Popen[Any], name: str, logger: FileLogger) -> None:
    if proc.poll() is not None:
        return
    logger.log(f"Sending SIGTERM to {name} (pid {proc.pid})")
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=SHUTDOWN_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        logger.log(f"{name} did not exit after {SHUTDOWN_WAIT_SECONDS:.0f}s; sending SIGKILL")
        try:
            proc.kill()
        except ProcessLookupError:
            return
        proc.wait()


def kill_process_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def serve(config: DevStackConfig) -> int:
    controller = FileLogger(config.controller_log)
    controller.log(f"Dev stack controller starting in {config.project_root} (pid {os.getpid()}, pgid {os.getpid()})")
    controller_pid = os.getpid()

    # Ensure PID file contains the controller pid early for other commands.
    atomic_write(config.pid_file, f"{controller_pid}\n")

    # Use own process group so `kill -- -PID` works.
    try:
        os.setsid()
    except OSError:
        # Already session leader when started with start_new_session=True.
        pass

    try:
        return _run_devstack(config, controller)
    finally:
        _cleanup_controller_resources(config, controller, controller_pid)


def wait_for_state(state_path: Path, timeout: float) -> Optional[Dict[str, Any]]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            raw = state_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            time.sleep(0.2)
            continue
        if not raw.strip():
            time.sleep(0.2)
            continue
        try:
            return cast(Dict[str, Any], json.loads(raw))
        except json.JSONDecodeError:
            time.sleep(0.2)
            continue
    return None


def start_devstack(config: DevStackConfig) -> int:
    existing_pid = read_pid(config.pid_file)
    if existing_pid and process_alive(existing_pid):
        print(
            f"Dev stack already running with PID {existing_pid}. "
            f"Stop it first with `python -m llm_judge.devstack stop` or `kill -- -{existing_pid}`.",
            file=sys.stderr,
        )
        return 1

    for path in (config.controller_log, config.backend_log, config.frontend_log):
        path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        config.python_executable,
        "-m",
        "llm_judge.devstack",
        "serve",
        "--backend-host",
        config.backend_host,
        "--backend-port",
        str(config.backend_port),
        "--frontend-host",
        config.frontend_host,
        "--frontend-port",
        str(config.frontend_port),
        "--python-executable",
        config.python_executable,
        "--npm-command",
        config.npm_command,
        "--log-dir",
        str(config.log_dir),
        "--pid-file",
        str(config.pid_file),
        "--state-file",
        str(config.state_file),
        "--controller-log",
        str(config.controller_log),
        "--backend-log",
        str(config.backend_log),
        "--frontend-log",
        str(config.frontend_log),
        "--project-root",
        str(config.project_root),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    state = wait_for_state(config.state_file, STATE_WAIT_SECONDS)
    if state is None:
        print(
            "Timed out while waiting for dev stack to initialize. "
            f"Check {config.controller_log} for diagnostic output.",
            file=sys.stderr,
        )
        return 1

    frontend_url = state.get("frontend_url", f"http://{config.frontend_host}:{config.frontend_port}")
    backend_url = state.get("backend_url", f"http://{config.backend_host}:{config.backend_port}")
    debugger_url = state.get("debugger_url", f"{backend_url}/__debugger__")
    print("Development stack is running.")
    print(f"  controller pid       : {proc.pid}")
    print(f"  process group        : {proc.pid} (kill with `kill -- -{proc.pid}`)")
    print(f"  frontend (Svelte)    : {frontend_url}")
    print(f"  backend (Flask API)  : {backend_url}")
    print(f"  debugger             : {debugger_url}")
    print(f"  frontend log         : {config.frontend_log}")
    print(f"  backend log          : {config.backend_log}")
    print(f"  controller log       : {config.controller_log}")
    print("To tail the logs: `tail -f <logfile>`.")
    print(f"To stop the stack: `python -m llm_judge.devstack stop` or `kill -- -{proc.pid}`.")
    return 0


def stop_devstack(config: DevStackConfig, force: bool) -> int:
    pid = read_pid(config.pid_file)
    if pid is None:
        print("Dev stack is not running (no pid file).", file=sys.stderr)
        return 1
    if not process_alive(pid):
        print("Dev stack pid file exists but process is not running; cleaning up stale files.")
        _cleanup_runtime_files(config)
        return 0

    sig = signal.SIGKILL if force else signal.SIGTERM
    kill_process_group(pid, sig)

    if not _wait_for_process_shutdown(pid):
        print(
            f"Process group {pid} still alive after {SHUTDOWN_WAIT_SECONDS:.0f}s. "
            "You may need to run `kill -9 -- -PID` manually.",
            file=sys.stderr,
        )
        return 1

    _cleanup_runtime_files(config)
    print("Dev stack stopped.")
    return 0


def status_devstack(config: DevStackConfig) -> int:
    pid = read_pid(config.pid_file)
    running = pid is not None and process_alive(pid)
    print(f"Dev stack running: {'yes' if running else 'no'}")
    if pid is not None:
        print(f"controller pid: {pid}")
    if config.state_file.exists():
        try:
            state = json.loads(config.state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("State file exists but is not valid JSON.")
            return 1
        print("Known endpoints:")
        for key in ("frontend_url", "backend_url", "debugger_url"):
            value = state.get(key)
            if value:
                print(f"  {key.replace('_', ' ')}: {value}")
        print("Log files:")
        logs = state.get("logs", {})
        for name, path in logs.items():
            print(f"  {name}: {path}")
    return 0


def make_config(args: argparse.Namespace) -> DevStackConfig:
    return DevStackConfig(
        backend_host=args.backend_host,
        backend_port=args.backend_port,
        frontend_host=args.frontend_host,
        frontend_port=args.frontend_port,
        python_executable=args.python_executable,
        npm_command=args.npm_command,
        flask_app=args.flask_app,
        project_root=Path(args.project_root).resolve(),
        log_dir=Path(args.log_dir).resolve(),
        pid_file=Path(args.pid_file).resolve(),
        state_file=Path(args.state_file).resolve(),
        controller_log=Path(args.controller_log).resolve(),
        backend_log=Path(args.backend_log).resolve(),
        frontend_log=Path(args.frontend_log).resolve(),
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage the llm-judge development stack.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--backend-host", default=DEFAULT_BACKEND_HOST)
        subparser.add_argument("--backend-port", type=int, default=DEFAULT_BACKEND_PORT)
        subparser.add_argument("--frontend-host", default=DEFAULT_FRONTEND_HOST)
        subparser.add_argument("--frontend-port", type=int, default=DEFAULT_FRONTEND_PORT)
        subparser.add_argument("--python-executable", default=sys.executable)
        subparser.add_argument("--npm-command", default=os.environ.get("WEBUI_NPM", "npm"))
        subparser.add_argument("--flask-app", default="llm_judge.webapp:create_app")
        subparser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
        subparser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
        subparser.add_argument("--pid-file", default=str(DEFAULT_PID_FILE))
        subparser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE))
        subparser.add_argument("--controller-log", default=str(DEFAULT_CONTROLLER_LOG))
        subparser.add_argument("--backend-log", default=str(DEFAULT_BACKEND_LOG))
        subparser.add_argument("--frontend-log", default=str(DEFAULT_FRONTEND_LOG))

    start_parser = subparsers.add_parser("start", help="Start the development stack in the background.")
    add_shared_arguments(start_parser)

    stop_parser = subparsers.add_parser("stop", help="Stop the development stack.")
    add_shared_arguments(stop_parser)
    stop_parser.add_argument("--force", action="store_true", help="Use SIGKILL instead of SIGTERM.")

    status_parser = subparsers.add_parser("status", help="Show dev stack status.")
    add_shared_arguments(status_parser)

    serve_parser = subparsers.add_parser("serve", help=argparse.SUPPRESS)
    add_shared_arguments(serve_parser)

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = make_config(args)

    if args.command == "start":
        return start_devstack(config)
    if args.command == "stop":
        return stop_devstack(config, force=getattr(args, "force", False))
    if args.command == "status":
        return status_devstack(config)
    if args.command == "serve":
        return serve(config)
    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
