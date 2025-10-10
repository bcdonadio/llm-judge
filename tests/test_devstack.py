from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path
import argparse
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, cast

import pytest

from llm_judge import devstack


def _noop(*_args: object, **_kwargs: object) -> None:
    return None


def _noop_sleep(_seconds: float) -> None:
    return None


def _setsid_stub() -> int:
    return 0


def _always_false(_pid: int) -> bool:
    return False


def _always_true(_pid: int) -> bool:
    return True


class StubProc:
    _pid_counter = 1000

    def __init__(
        self,
        poll_results: List[Any],
        wait_exception: BaseException | Sequence[BaseException] | None = None,
        terminate_exception: BaseException | None = None,
        kill_exception: BaseException | None = None,
    ) -> None:
        StubProc._pid_counter += 1
        self.pid = StubProc._pid_counter
        self._poll_results = list(poll_results)
        self._last_poll: Any = None
        if wait_exception is None:
            self._wait_exceptions: List[BaseException] = []
        elif isinstance(wait_exception, Sequence):
            self._wait_exceptions = list(wait_exception)
        else:
            self._wait_exceptions = [wait_exception]
        self.terminate_exception = terminate_exception
        self.kill_exception = kill_exception
        self.terminate_called = False
        self.kill_called = False
        self.wait_calls = 0

    def poll(self) -> Any:
        if self._poll_results:
            self._last_poll = self._poll_results.pop(0)
            return self._last_poll
        return self._last_poll

    def terminate(self) -> None:
        if self.terminate_exception:
            raise self.terminate_exception
        self.terminate_called = True

    def kill(self) -> None:
        if self.kill_exception:
            raise self.kill_exception
        self.kill_called = True

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self._wait_exceptions:
            raise self._wait_exceptions.pop(0)
        return 0


@pytest.fixture
def base_config(tmp_path: Path) -> devstack.DevStackConfig:
    project_root = tmp_path
    (project_root / devstack.FRONTEND_DIRNAME).mkdir()
    log_dir = project_root / "logs"
    log_dir.mkdir()
    return devstack.DevStackConfig(
        project_root=project_root,
        log_dir=log_dir,
        pid_file=log_dir / "devstack.pid",
        state_file=log_dir / "state.json",
        controller_log=log_dir / "controller.log",
        backend_log=log_dir / "backend.log",
        frontend_log=log_dir / "frontend.log",
        python_executable=sys.executable,
        npm_command="npm",
    )


def make_logger(path: Path) -> devstack.FileLogger:
    return devstack.FileLogger(path)


def test_process_alive_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[int, str] = {}

    def fake_kill(pid: int, _sig: int) -> None:
        if pid == 1:
            raise ProcessLookupError
        if pid == 2:
            raise PermissionError
        calls[pid] = "ok"

    monkeypatch.setattr(devstack.os, "kill", fake_kill)

    assert not devstack.process_alive(1)
    assert devstack.process_alive(2)
    assert devstack.process_alive(3)
    assert calls[3] == "ok"


def test_read_pid_variants(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    assert devstack.read_pid(pid_file) is None
    pid_file.write_text("\n", encoding="utf-8")
    assert devstack.read_pid(pid_file) is None
    pid_file.write_text("abc", encoding="utf-8")
    assert devstack.read_pid(pid_file) is None
    pid_file.write_text("42\n", encoding="utf-8")
    assert devstack.read_pid(pid_file) == 42


def test_atomic_write(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    devstack.atomic_write(target, "first")
    devstack.atomic_write(target, "second")
    assert target.read_text(encoding="utf-8") == "second"


def test_file_logger_context_manager(tmp_path: Path) -> None:
    log_path = tmp_path / "ctx.log"
    with devstack.FileLogger(log_path) as logger:
        logger.log("hello")
    contents = log_path.read_text(encoding="utf-8")
    assert "hello" in contents


def test_ensure_frontend_dependencies_cached(
    base_config: devstack.DevStackConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_modules = base_config.project_root / devstack.FRONTEND_DIRNAME / "node_modules"
    node_modules.mkdir()
    invoked = False

    def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[Any]:
        nonlocal invoked
        invoked = True
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(devstack.subprocess, "run", fake_run)
    logger = make_logger(base_config.controller_log)
    devstack.ensure_frontend_dependencies(base_config, logger)
    logger.close()
    assert not invoked


def test_ensure_frontend_dependencies_installs(
    base_config: devstack.DevStackConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: Dict[str, Any] = {}

    def fake_run(cmd: List[str], cwd: Path, stdout: Any, stderr: Any, check: bool) -> subprocess.CompletedProcess[Any]:
        called["cmd"] = cmd
        called["cwd"] = cwd
        assert check is False
        stdout.write("installed\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(devstack.subprocess, "run", fake_run)
    logger = make_logger(base_config.controller_log)
    devstack.ensure_frontend_dependencies(base_config, logger)
    logger.close()
    assert called["cmd"][0] == base_config.npm_command
    assert called["cwd"] == base_config.project_root / devstack.FRONTEND_DIRNAME


def test_ensure_frontend_dependencies_failure(
    base_config: devstack.DevStackConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[Any]:
        return subprocess.CompletedProcess([], 1)

    monkeypatch.setattr(devstack.subprocess, "run", fake_run)
    logger = make_logger(base_config.controller_log)
    with pytest.raises(RuntimeError):
        devstack.ensure_frontend_dependencies(base_config, logger)
    logger.close()


def test_terminate_process_finished(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    proc = StubProc([0])
    logger = make_logger(tmp_path / "log")
    devstack.terminate_process(cast(subprocess.Popen[Any], proc), "proc", logger)
    logger.close()
    assert not proc.terminate_called


def test_terminate_process_graceful(tmp_path: Path) -> None:
    proc = StubProc([None])
    logger = make_logger(tmp_path / "log")
    devstack.terminate_process(cast(subprocess.Popen[Any], proc), "svc", logger)
    logger.close()
    assert proc.terminate_called
    assert proc.wait_calls == 1


def test_terminate_process_force_kill(tmp_path: Path) -> None:
    proc = StubProc(
        [None],
        wait_exception=[subprocess.TimeoutExpired(cmd=["x"], timeout=1)],
    )
    logger = make_logger(tmp_path / "log")
    devstack.terminate_process(cast(subprocess.Popen[Any], proc), "svc", logger)
    logger.close()
    assert proc.kill_called


def test_terminate_process_kill_missing(tmp_path: Path) -> None:
    proc = StubProc(
        [None],
        wait_exception=[subprocess.TimeoutExpired(cmd=["x"], timeout=1)],
        kill_exception=ProcessLookupError(),
    )
    logger = make_logger(tmp_path / "log")
    devstack.terminate_process(cast(subprocess.Popen[Any], proc), "svc", logger)
    logger.close()


def test_terminate_process_missing(tmp_path: Path) -> None:
    proc = StubProc([None], terminate_exception=ProcessLookupError())
    logger = make_logger(tmp_path / "log")
    devstack.terminate_process(cast(subprocess.Popen[Any], proc), "svc", logger)
    logger.close()
    assert not proc.kill_called


def test_kill_process_group_handles_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_killpg(_pgid: int, _sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(devstack.os, "killpg", fake_killpg)
    devstack.kill_process_group(1, signal.SIGTERM)


def test_wait_for_state_success(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"ok": True}), encoding="utf-8")
    result = devstack.wait_for_state(state_file, timeout=1)
    assert result == {"ok": True}


def test_wait_for_state_invalid_then_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_text("invalid", encoding="utf-8")

    def delayed_write() -> None:
        state_file.write_text(json.dumps({"retry": True}), encoding="utf-8")

    # Replace time.sleep so we can inject valid JSON after first read.
    def fake_sleep(_seconds: float) -> None:
        delayed_write()

    monkeypatch.setattr(devstack.time, "sleep", fake_sleep)
    result = devstack.wait_for_state(state_file, timeout=1)
    assert result == {"retry": True}


def test_wait_for_state_blank_then_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_text("   ", encoding="utf-8")

    def to_valid() -> None:
        state_file.write_text(json.dumps({"ok": True}), encoding="utf-8")

    def fake_sleep_blank(_seconds: float) -> None:
        to_valid()

    monkeypatch.setattr(devstack.time, "sleep", fake_sleep_blank)
    assert devstack.wait_for_state(state_file, timeout=1) == {"ok": True}


def test_wait_for_state_timeout(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    assert devstack.wait_for_state(state_file, timeout=0.1) is None


def test_serve_backend_exit(base_config: devstack.DevStackConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    backend_proc = StubProc([0])
    frontend_proc = StubProc([None, None])
    popen_calls: List[List[str]] = []

    def fake_popen(cmd: List[str], **kwargs: Any) -> StubProc:
        popen_calls.append(cmd)
        return {0: backend_proc, 1: frontend_proc}[len(popen_calls) - 1]

    monkeypatch.setattr(devstack.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(devstack, "ensure_frontend_dependencies", _noop)
    monkeypatch.setattr(devstack.os, "setsid", _setsid_stub)
    monkeypatch.setattr(devstack.time, "sleep", _noop_sleep)

    handlers: Dict[int, Any] = {}

    def fake_signal(sig: int, handler: Any) -> None:
        handlers[sig] = handler

    monkeypatch.setattr(devstack.signal, "signal", fake_signal)

    result = devstack.serve(base_config)
    assert result == 0
    assert backend_proc.terminate_called is False
    assert frontend_proc.terminate_called is True
    assert not base_config.pid_file.exists()
    assert not base_config.state_file.exists()
    assert devstack.signal.SIGTERM in handlers
    assert len(popen_calls) == 2


def test_serve_frontend_exit(base_config: devstack.DevStackConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    backend_proc = StubProc([None, None])
    frontend_proc = StubProc([1])
    stubs = [backend_proc, frontend_proc]

    def fake_popen(_cmd: List[str], **_kwargs: Any) -> StubProc:
        return stubs.pop(0)

    monkeypatch.setattr(devstack.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(devstack, "ensure_frontend_dependencies", _noop)
    monkeypatch.setattr(devstack.os, "setsid", _setsid_stub)
    monkeypatch.setattr(devstack.time, "sleep", _noop_sleep)

    handlers: Dict[int, Any] = {}

    def fake_signal(sig: int, handler: Any) -> None:
        handlers[sig] = handler

    monkeypatch.setattr(devstack.signal, "signal", fake_signal)

    result = devstack.serve(base_config)
    assert result == 1
    assert handlers


def test_serve_handles_signal(base_config: devstack.DevStackConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    backend_proc = StubProc([None, None])
    frontend_proc = StubProc([None, None])
    stubs = [backend_proc, frontend_proc]

    def fake_popen(_cmd: List[str], **_kwargs: Any) -> StubProc:
        return stubs.pop(0)

    monkeypatch.setattr(devstack.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(devstack, "ensure_frontend_dependencies", _noop)

    handlers: Dict[int, Any] = {}

    def fake_signal(sig: int, handler: Any) -> None:
        handlers[sig] = handler

    monkeypatch.setattr(devstack.signal, "signal", fake_signal)

    def setsid_failure() -> int:
        raise OSError()

    monkeypatch.setattr(devstack.os, "setsid", setsid_failure)

    sleep_calls = 0

    def fake_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 1:
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            handlers[signal.SIGTERM](signal.SIGTERM, None)

    monkeypatch.setattr(devstack.time, "sleep", fake_sleep)

    result = devstack.serve(base_config)
    assert result == 0
    assert backend_proc.terminate_called
    assert frontend_proc.terminate_called
    assert not stubs


def test_serve_cleanup_handles_errors(base_config: devstack.DevStackConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    backend_proc = StubProc([0])
    frontend_proc = StubProc([None])
    stubs = [backend_proc, frontend_proc]

    def fake_popen(_cmd: List[str], **_kwargs: Any) -> StubProc:
        return stubs.pop(0)

    monkeypatch.setattr(devstack.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(devstack, "ensure_frontend_dependencies", _noop)
    monkeypatch.setattr(devstack.os, "setsid", _setsid_stub)
    monkeypatch.setattr(devstack.time, "sleep", _noop_sleep)

    path_cls = type(base_config.pid_file)
    original_unlink = path_cls.unlink
    raised: List[Path] = []

    def fake_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self in {base_config.pid_file, base_config.state_file}:
            raised.append(self)
            raise OSError("cannot unlink")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(path_cls, "unlink", fake_unlink)
    devstack.serve(base_config)
    assert set(raised) == {base_config.pid_file, base_config.state_file}


def test_start_devstack_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = devstack.DevStackConfig(
        project_root=tmp_path,
        log_dir=tmp_path,
        pid_file=tmp_path / "pid",
        state_file=tmp_path / "state",
        controller_log=tmp_path / "controller",
        backend_log=tmp_path / "backend",
        frontend_log=tmp_path / "frontend",
    )
    config.pid_file.write_text("123\n", encoding="utf-8")

    def fake_alive(pid: int) -> bool:
        return pid == 123

    monkeypatch.setattr(devstack, "process_alive", fake_alive)
    assert devstack.start_devstack(config) == 1


def test_start_devstack_success(
    base_config: devstack.DevStackConfig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeProc:
        def __init__(self) -> None:
            self.pid = 4242

    popen_kwargs: Dict[str, Any] = {}

    def fake_popen(cmd: List[str], **kwargs: Any) -> FakeProc:
        popen_kwargs["cmd"] = cmd
        popen_kwargs["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(devstack.subprocess, "Popen", fake_popen)

    def fake_wait_for_state(_path: Path, _timeout: float) -> Dict[str, str]:
        return {"frontend_url": "http://x", "backend_url": "http://y", "debugger_url": "http://dbg"}

    monkeypatch.setattr(devstack, "wait_for_state", fake_wait_for_state)

    result = devstack.start_devstack(base_config)
    captured = capsys.readouterr()
    assert result == 0
    assert "Development stack is running" in captured.out
    assert popen_kwargs["kwargs"]["start_new_session"] is True
    assert "serve" in popen_kwargs["cmd"]


def test_start_devstack_timeout(
    base_config: devstack.DevStackConfig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def wait_timeout(_path: Path, _timeout: float) -> None:
        return None

    monkeypatch.setattr(devstack, "wait_for_state", wait_timeout)

    class FakeProc:
        pid = 5555

    def fake_popen(*_args: object, **_kwargs: object) -> FakeProc:
        return FakeProc()

    monkeypatch.setattr(devstack.subprocess, "Popen", fake_popen)

    result = devstack.start_devstack(base_config)
    captured = capsys.readouterr()
    assert result == 1
    assert "Timed out" in captured.err or "Timed out" in captured.out


def test_stop_devstack_no_pid(base_config: devstack.DevStackConfig, capsys: pytest.CaptureFixture[str]) -> None:
    if base_config.pid_file.exists():
        base_config.pid_file.unlink()
    result = devstack.stop_devstack(base_config, force=False)
    captured = capsys.readouterr()
    assert result == 1
    assert "not running" in captured.err or "not running" in captured.out


def test_stop_devstack_stale_pid(base_config: devstack.DevStackConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    base_config.pid_file.write_text("99\n", encoding="utf-8")
    base_config.state_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(devstack, "process_alive", _always_false)
    assert devstack.stop_devstack(base_config, force=False) == 0
    assert not base_config.pid_file.exists()
    assert not base_config.state_file.exists()


def test_stop_devstack_active(base_config: devstack.DevStackConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    base_config.pid_file.write_text("314\n", encoding="utf-8")
    base_config.state_file.write_text("{}", encoding="utf-8")
    alive_calls = iter([True, True, False])

    def fake_alive(_pid: int) -> bool:
        return next(alive_calls, False)

    captured_signal: Dict[str, Any] = {}

    def fake_killpg(pid: int, sig: int) -> None:
        captured_signal["pid"] = pid
        captured_signal["sig"] = sig

    monkeypatch.setattr(devstack, "process_alive", fake_alive)
    monkeypatch.setattr(devstack, "kill_process_group", fake_killpg)
    monkeypatch.setattr(devstack.time, "sleep", _noop_sleep)
    assert devstack.stop_devstack(base_config, force=False) == 0
    assert captured_signal["sig"] == signal.SIGTERM
    assert not base_config.pid_file.exists()


def test_stop_devstack_force(base_config: devstack.DevStackConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    base_config.pid_file.write_text("271\n", encoding="utf-8")
    alive_calls = iter([True, False])

    def alive_iter(_pid: int) -> bool:
        return next(alive_calls, False)

    monkeypatch.setattr(devstack, "process_alive", alive_iter)

    captured: Dict[str, Any] = {}

    def record_signal(pid: int, sig: int) -> None:
        captured.update({"pid": pid, "sig": sig})

    monkeypatch.setattr(devstack, "kill_process_group", record_signal)
    monkeypatch.setattr(devstack.time, "sleep", _noop_sleep)
    assert devstack.stop_devstack(base_config, force=True) == 0
    assert captured["sig"] == signal.SIGKILL


def test_stop_devstack_stale_cleanup_missing_file(
    base_config: devstack.DevStackConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_config.pid_file.write_text("55\n", encoding="utf-8")
    base_config.state_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(devstack, "process_alive", _always_false)
    path_cls = type(base_config.state_file)
    original_unlink = path_cls.unlink

    def fake_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == base_config.state_file:
            raise FileNotFoundError
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(path_cls, "unlink", fake_unlink)
    assert devstack.stop_devstack(base_config, force=False) == 0


def test_stop_devstack_still_alive_timeout(
    base_config: devstack.DevStackConfig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base_config.pid_file.write_text("808\n", encoding="utf-8")
    time_values = iter([0.0, 0.0, devstack.SHUTDOWN_WAIT_SECONDS + 1])
    monkeypatch.setattr(devstack, "process_alive", _always_true)

    def fake_time() -> float:
        return next(time_values)

    monkeypatch.setattr(devstack.time, "time", fake_time)
    monkeypatch.setattr(devstack.time, "sleep", _noop_sleep)
    monkeypatch.setattr(devstack, "kill_process_group", _noop)
    result = devstack.stop_devstack(base_config, force=False)
    captured = capsys.readouterr()
    assert result == 1
    assert "still alive" in captured.err or "still alive" in captured.out


def test_stop_devstack_final_unlink_error(
    base_config: devstack.DevStackConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_config.pid_file.write_text("909\n", encoding="utf-8")
    base_config.state_file.write_text("{}", encoding="utf-8")
    alive_calls = iter([True, False])

    def alive_iter_force(_pid: int) -> bool:
        return next(alive_calls, False)

    monkeypatch.setattr(devstack, "process_alive", alive_iter_force)
    monkeypatch.setattr(devstack, "kill_process_group", _noop)
    monkeypatch.setattr(devstack.time, "sleep", _noop_sleep)
    path_cls = type(base_config.state_file)
    original_unlink = path_cls.unlink

    def fake_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == base_config.state_file:
            raise OSError("cannot remove")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(path_cls, "unlink", fake_unlink)
    assert devstack.stop_devstack(base_config, force=False) == 0


def test_status_devstack_running(
    base_config: devstack.DevStackConfig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base_config.pid_file.write_text("777\n", encoding="utf-8")
    state: Dict[str, Any] = {
        "frontend_url": "http://front",
        "backend_url": "http://back",
        "debugger_url": "http://dbg",
        "logs": {"controller": "ctrl.log"},
    }
    base_config.state_file.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(devstack, "process_alive", _always_true)
    assert devstack.status_devstack(base_config) == 0
    captured = capsys.readouterr().out
    assert "frontend url" in captured
    assert "controller" in captured


def test_status_devstack_invalid_state(
    base_config: devstack.DevStackConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    base_config.pid_file.write_text("888\n", encoding="utf-8")
    base_config.state_file.write_text("not-json", encoding="utf-8")
    result = devstack.status_devstack(base_config)
    captured = capsys.readouterr().out
    assert result == 1
    assert "not valid JSON" in captured


def test_status_devstack_missing_values(
    base_config: devstack.DevStackConfig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base_config.pid_file.write_text("1001\n", encoding="utf-8")
    state: Dict[str, Any] = {
        "frontend_url": "",
        "backend_url": "http://back",
        "debugger_url": None,
        "logs": {},
    }
    base_config.state_file.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(devstack, "process_alive", _always_true)
    devstack.status_devstack(base_config)
    captured = capsys.readouterr().out
    assert "backend url" in captured
    assert "frontend url" not in captured


def test_status_devstack_not_running(base_config: devstack.DevStackConfig, capsys: pytest.CaptureFixture[str]) -> None:
    if base_config.pid_file.exists():
        base_config.pid_file.unlink()
    result = devstack.status_devstack(base_config)
    captured = capsys.readouterr().out
    assert result == 0
    assert "Dev stack running: no" in captured


def test_make_config() -> None:
    args = argparse.Namespace(
        backend_host="host",
        backend_port=1,
        frontend_host="fhost",
        frontend_port=2,
        python_executable="python",
        npm_command="npm",
        flask_app="app:create",
        project_root=".",
        log_dir=".logs",
        pid_file=".pid",
        state_file=".state",
        controller_log=".ctrl",
        backend_log=".back",
        frontend_log=".front",
    )
    config = devstack.make_config(args)
    assert config.backend_host == "host"
    assert config.frontend_port == 2
    assert config.pid_file.name == ".pid"


def test_parse_args_commands() -> None:
    start_args = devstack.parse_args(["start"])
    assert start_args.command == "start"
    stop_args = devstack.parse_args(["stop", "--force"])
    assert stop_args.command == "stop" and stop_args.force
    status_args = devstack.parse_args(["status"])
    assert status_args.command == "status"
    serve_args = devstack.parse_args(["serve"])
    assert serve_args.command == "serve"


def test_main_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[str, Any] = {}

    def fake_make_config(args: Any) -> str:
        return f"config-{args.command}"

    monkeypatch.setattr(devstack, "make_config", fake_make_config)

    def record_start(config: Any) -> None:
        calls.setdefault("start", config)

    def record_stop(config: Any, force: bool = False) -> None:
        calls.setdefault("stop", (config, force))

    def record_status(config: Any) -> None:
        calls.setdefault("status", config)

    def record_serve(config: Any) -> None:
        calls.setdefault("serve", config)

    def set_parse_result(command: str) -> None:
        def fake_parse(_argv: List[str] | None = None) -> SimpleNamespace:
            return SimpleNamespace(command=command)

        monkeypatch.setattr(devstack, "parse_args", fake_parse)

    monkeypatch.setattr(devstack, "start_devstack", record_start)
    monkeypatch.setattr(devstack, "stop_devstack", record_stop)
    monkeypatch.setattr(devstack, "status_devstack", record_status)
    monkeypatch.setattr(devstack, "serve", record_serve)

    set_parse_result("start")
    devstack.main([])
    assert calls["start"] == "config-start"

    set_parse_result("stop")
    devstack.main([])
    assert calls["stop"][0] == "config-stop"

    set_parse_result("status")
    devstack.main([])
    assert calls["status"] == "config-status"

    set_parse_result("serve")
    devstack.main([])
    assert calls["serve"] == "config-serve"

    set_parse_result("unknown")
    with pytest.raises(ValueError):
        devstack.main([])


def test_main_guard_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    exit_codes: List[int] = []
    monkeypatch.setitem(devstack.__dict__, "__name__", "__main__")

    def fake_exit(code: int = 0) -> None:
        exit_codes.append(code)

    monkeypatch.setattr(devstack.sys, "exit", fake_exit)

    def fake_main(argv: List[str] | None = None) -> int:
        return 0

    monkeypatch.setattr(devstack, "main", fake_main)
    exec("if __name__ == '__main__': sys.exit(main())", devstack.__dict__)
    assert exit_codes == [0]
