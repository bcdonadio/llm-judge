from __future__ import annotations

import logging
import math
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

import judge as cli
import llm_judge


def test_build_parser_parses_expected_arguments(tmp_path: Path) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--models",
            "model-a",
            "model-b",
            "--judge-model",
            "judge-x",
            "--outdir",
            str(tmp_path),
            "--max-tokens",
            "1234",
            "--judge-max-tokens",
            "4321",
            "--temperature",
            "0.7",
            "--judge-temperature",
            "0.2",
            "--sleep",
            "0.5",
            "--limit",
            "9",
            "--verbose",
        ]
    )
    assert args.models == ["model-a", "model-b"]
    assert args.judge_model == "judge-x"
    assert Path(args.outdir) == tmp_path
    assert args.max_tokens == 1234
    assert args.judge_max_tokens == 4321
    assert math.isclose(args.temperature, 0.7)
    assert math.isclose(args.judge_temperature, 0.2)
    assert math.isclose(args.sleep, 0.5)
    assert args.limit == 9
    assert args.verbose is True


def test_configure_logging_enables_color(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    class RecordingHandler:
        def __init__(self) -> None:
            self.formatter: logging.Formatter | None = None

        def setFormatter(self, formatter: logging.Formatter) -> None:
            self.formatter = formatter

    handler = RecordingHandler()

    monkeypatch.setattr(cli.logging, "StreamHandler", lambda: handler)

    def record_basic_config(**kwargs: Any) -> None:
        calls.setdefault("basicConfig", kwargs)

    monkeypatch.setattr(cli.logging, "basicConfig", record_basic_config)

    class DummyStderr:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(cli.sys, "stderr", DummyStderr())
    monkeypatch.delenv("NO_COLOR", raising=False)

    color_triggered = False

    def fake_colorama_init() -> None:
        nonlocal color_triggered
        color_triggered = True

    monkeypatch.setattr(cli, "colorama_init", fake_colorama_init)

    use_color = cli.configure_logging(debug=False, verbose=True)
    assert use_color is True
    assert color_triggered is True
    assert calls["basicConfig"]["level"] == logging.INFO
    assert calls["basicConfig"]["handlers"] == [handler]
    assert handler.formatter is not None

    formatter = handler.formatter
    colored_record = logging.LogRecord("cli", logging.INFO, __file__, 1, "colored", (), None)
    colored_output = formatter.format(colored_record)
    assert cli.Style.RESET_ALL in colored_output
    assert "colored" in colored_output

    plain_record = logging.LogRecord("cli", logging.NOTSET, __file__, 2, "plain", (), None)
    plain_output = formatter.format(plain_record)
    assert cli.Style.RESET_ALL not in plain_output
    assert plain_output.endswith("plain")


def test_configure_logging_respects_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")

    class DummyStderr:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(cli.sys, "stderr", DummyStderr())
    monkeypatch.setattr(cli, "colorama_init", lambda: pytest.fail("color init should not run when NO_COLOR is set"))

    def fake_basic_config(**kwargs: Any) -> None:
        pass

    monkeypatch.setattr(cli.logging, "basicConfig", fake_basic_config)

    use_color = cli.configure_logging(debug=True, verbose=False)
    assert use_color is False


def test_configure_logging_defaults_to_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, Any] = {}

    def fake_basic_config(**kwargs: Any) -> None:
        recorded.update(kwargs)

    monkeypatch.setattr(cli.logging, "basicConfig", fake_basic_config)

    class DummyStderr:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(cli.sys, "stderr", DummyStderr())
    monkeypatch.setattr(cli, "colorama_init", lambda: pytest.fail("color init should not run when TTY is absent"))

    use_color = cli.configure_logging(debug=False, verbose=False)
    assert use_color is False
    assert recorded["level"] == logging.WARNING


def test_main_invokes_run_suite_with_expected_arguments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class RecordingRunner:
        def __init__(self, config: Any) -> None:
            captured["config"] = config

        def run(self) -> None:
            captured["run_called"] = True

    monkeypatch.setattr(cli, "LLMJudgeRunner", RecordingRunner)

    def fake_configure_logging(debug: bool, verbose: bool) -> str:
        assert debug is True
        assert verbose is False
        return "color-enabled"

    monkeypatch.setattr(cli, "configure_logging", fake_configure_logging)

    outdir = tmp_path / "artifacts"
    argv = [
        "--models",
        "model-1",
        "--judge-model",
        "judge-1",
        "--outdir",
        str(outdir),
        "--sleep",
        "0.75",
        "--limit",
        "3",
        "--debug",
    ]

    exit_code = cli.main(argv)
    assert exit_code == 0
    config = captured["config"]
    assert config.models == ["model-1"]
    assert config.judge_model == "judge-1"
    assert config.outdir == outdir
    assert math.isclose(config.sleep_s, 0.75)
    assert config.limit == 3
    assert config.use_color == "color-enabled"
    assert captured["run_called"] is True
    assert outdir.is_dir()


def test_main_handles_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class RaisingRunner:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def run(self) -> None:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "LLMJudgeRunner", RaisingRunner)

    def fake_configure_logging_noop(*args: Any, **kwargs: Any) -> bool:
        return False

    monkeypatch.setattr(cli, "configure_logging", fake_configure_logging_noop)

    exit_code = cli.main(["--models", "model-1", "--outdir", str(tmp_path), "--judge-model", "judge-1"])
    captured = capsys.readouterr()

    assert exit_code == 130
    assert "[Interrupted] Exiting." in captured.out


def test_module_entry_point_executes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded: dict[str, Any] = {}

    class RecordingRunner:
        def __init__(self, config: Any) -> None:
            recorded["config"] = config

        def run(self) -> None:
            recorded["run_called"] = True

    monkeypatch.setattr(llm_judge, "LLMJudgeRunner", RecordingRunner)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(
        sys, "argv", ["judge.py", "--models", "run-model", "--judge-model", "judge-x", "--outdir", str(tmp_path)]
    )
    monkeypatch.setattr(sys, "exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit) as exc:
        runpy.run_path("judge.py", run_name="__main__")

    assert exc.value.code == 0
    config = recorded["config"]
    assert config.models == ["run-model"]
    assert config.judge_model == "judge-x"
    assert config.outdir == tmp_path
    assert recorded["run_called"] is True
