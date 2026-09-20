"""CLI tests: entry points, slash commands, exit codes.

Everything here is offline - the session dependencies are faked, so no model or
network call is ever made.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from essay_agent import __version__
from essay_agent import cli as cli_module
from essay_agent.agent import TaskResult
from essay_agent.cli import main
from essay_agent.console import ConsoleUI
from essay_agent.errors import TaskCancelled
from essay_agent.memory.lesson import LessonStore
from tests.conftest import make_settings
from tests.fakes import FakeLLM

PLAN_FILE = "lesson_plan.txt"


@dataclass
class FakeDeps:
    """Only the attributes the CLI touches."""

    ui: Any
    lesson_store: LessonStore
    llm: Any = None


class ScriptedSession:
    """Stands in for prompt_toolkit's session."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)

    def prompt(self, message: str = "") -> str:
        if not self.script:
            raise EOFError
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def install_fake_session(monkeypatch, tmp_path: Path) -> tuple[Any, FakeDeps]:
    """Point the CLI at a hermetic (settings, deps) pair."""
    settings = make_settings(tmp_path)
    deps = FakeDeps(
        ui=ConsoleUI(),
        lesson_store=LessonStore(settings.resolved_paths().ensure().lesson_dir, settings.memory),
        llm=FakeLLM(texts={"default": "- merged rule"}, responses={}),
    )
    monkeypatch.setattr(cli_module, "build_session", lambda *args, **kwargs: (settings, deps))
    return settings, deps


def script_session(monkeypatch, script: list[Any]) -> None:
    monkeypatch.setattr(
        cli_module, "_prompt_session", lambda *args, **kwargs: ScriptedSession(script)
    )


def task_result(**overrides: Any) -> TaskResult:
    payload: dict[str, Any] = {
        "run_id": "run-1",
        "run_dir": Path("/tmp/run-1"),
        "status": "completed",
        "state": {
            "verdict": {"verdict": "successful"},
            "exec_result": {"metrics": {"accuracy": 0.9}},
        },
        "report_path": "/tmp/run-1/result/report.md",
        "message": "",
        "published": {"result_dir": "/tmp/run-1/result", "code_dir": "/tmp/run-1/repro"},
    }
    payload.update(overrides)
    return TaskResult(**payload)


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


# --------------------------------------------------------------------- basics
def test_help_and_version(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0, result.output
    for word in ("agent", "run", "config", "lesson"):
        assert word in result.output

    version = runner.invoke(main, ["--version"])
    assert version.exit_code == 0
    assert __version__ in version.output


def test_bare_invocation_points_at_the_agent_command(runner: CliRunner) -> None:
    result = runner.invoke(main, [])
    assert result.exit_code == 0
    assert "essay agent" in result.output


# --------------------------------------------------------------------- config
def test_config_show_redacts_the_api_key(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ESSAY_AGENT_LLM__API_KEY", "sk-secret-1234")
    result = runner.invoke(main, ["config", "show"])
    assert result.exit_code == 0, result.output
    assert "***1234" in result.output
    assert "sk-secret-1234" not in result.output


def test_config_check_passes_with_a_key(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ESSAY_AGENT_LLM__API_KEY", "sk-test")
    result = runner.invoke(main, ["config", "check"])
    assert result.exit_code == 0, result.output
    assert "configuration looks usable" in result.output


def test_config_check_fails_without_a_key(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    for name in (
        "ESSAY_AGENT_LLM__API_KEY",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    result = runner.invoke(main, ["config", "check"])
    assert result.exit_code == 1
    assert "No API key" in result.output


def test_config_init_writes_a_starter_file(runner, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(main, ["config", "init"])
    assert result.exit_code == 0, result.output
    target = tmp_path / "essay-agent.yaml"
    assert target.is_file()
    body = target.read_text(encoding="utf-8")
    assert "provider:" in body
    assert "time_budget_seconds" in body

    again = runner.invoke(main, ["config", "init"])
    assert again.exit_code == 1
    assert "already exists" in again.output


# ------------------------------------------------------------------ one-shot
def test_run_command_reports_and_exits_zero(runner, monkeypatch, tmp_path) -> None:
    install_fake_session(monkeypatch, tmp_path)
    monkeypatch.setattr(cli_module, "run_task", lambda query, deps: task_result())
    result = runner.invoke(main, ["run", "Attention", "is", "all", "you", "need"])
    assert result.exit_code == 0, result.output
    assert "verifier verdict: successful" in result.output
    assert "published:" in result.output


def test_run_command_exit_codes(runner, monkeypatch, tmp_path) -> None:
    install_fake_session(monkeypatch, tmp_path)

    monkeypatch.setattr(
        cli_module,
        "run_task",
        lambda query, deps: task_result(status="aborted", message="no paper"),
    )
    aborted = runner.invoke(main, ["run", "nope"])
    assert aborted.exit_code == 1
    assert "task stopped: no paper" in aborted.output

    def cancel(query, deps):
        raise TaskCancelled("ctrl+c")

    monkeypatch.setattr(cli_module, "run_task", cancel)
    cancelled = runner.invoke(main, ["run", "nope"])
    assert cancelled.exit_code == 130
    assert "nothing was saved" in cancelled.output


# ----------------------------------------------------------------- interactive
def test_interactive_session_handles_commands_and_exits(runner, monkeypatch, tmp_path) -> None:
    install_fake_session(monkeypatch, tmp_path)
    script_session(monkeypatch, ["/help", "/config", "/lesson", "/nope", "/exit"])

    result = runner.invoke(main, ["agent"])

    assert result.exit_code == 0, result.output
    assert "unknown command: /nope" in result.output
    assert PLAN_FILE in result.output
    assert "bye" in result.output


def test_ctrl_c_during_a_task_keeps_the_session_alive(runner, monkeypatch, tmp_path) -> None:
    install_fake_session(monkeypatch, tmp_path)
    seen: list[str] = []

    def cancel(query, deps):
        seen.append(query)
        raise TaskCancelled("cancelled")

    monkeypatch.setattr(cli_module, "run_task", cancel)
    script_session(monkeypatch, ["A paper title", "/exit"])

    result = runner.invoke(main, ["agent"])

    assert result.exit_code == 0, result.output
    assert seen == ["A paper title"]
    assert "task cancelled - nothing was saved" in result.output
    assert "bye" in result.output


def test_ctrl_c_at_the_prompt_leaves_cleanly(runner, monkeypatch, tmp_path) -> None:
    install_fake_session(monkeypatch, tmp_path)
    script_session(monkeypatch, [KeyboardInterrupt()])
    result = runner.invoke(main, ["agent"])
    assert result.exit_code == 0
    assert "bye" in result.output


# --------------------------------------------------------------------- lesson
def test_lesson_show_and_clear_use_the_store(runner, monkeypatch, tmp_path) -> None:
    settings, _ = install_fake_session(monkeypatch, tmp_path)
    lesson_dir = settings.resolved_paths().lesson_dir
    lesson_dir.mkdir(parents=True, exist_ok=True)
    (lesson_dir / PLAN_FILE).write_text("- always pin the dataset split\n", encoding="utf-8")

    shown = runner.invoke(main, ["lesson", "show"])
    assert shown.exit_code == 0, shown.output
    assert "always pin the dataset split" in shown.output

    cleared = runner.invoke(main, ["lesson", "clear"])
    assert cleared.exit_code == 0
    assert not (lesson_dir / PLAN_FILE).exists()
    assert any(
        path.name.startswith("lesson_plan-cleared-") for path in (lesson_dir / "archive").iterdir()
    )


def test_lesson_consolidate_compresses_a_long_file(runner, monkeypatch, tmp_path) -> None:
    settings, _ = install_fake_session(monkeypatch, tmp_path)
    lesson_dir = settings.resolved_paths().lesson_dir
    lesson_dir.mkdir(parents=True, exist_ok=True)
    (lesson_dir / PLAN_FILE).write_text("- rule line\n" * 200, encoding="utf-8")

    result = runner.invoke(main, ["lesson", "consolidate"])

    assert result.exit_code == 0, result.output
    assert "->" in result.output
    assert "- merged rule" in (lesson_dir / PLAN_FILE).read_text(encoding="utf-8")
