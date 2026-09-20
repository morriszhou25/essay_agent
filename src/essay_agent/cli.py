"""Command-line entry points.

essay agent            interactive session (the main way to use the agent)
essay run "<request>"  one-shot, non-interactive
essay config ...       show / init / check the configuration
essay lesson ...       inspect and maintain the lesson memory
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console

from essay_agent import __version__
from essay_agent.agent import TaskResult, build_session, run_task
from essay_agent.config import Settings, load_settings
from essay_agent.errors import ConfigError, EssayAgentError, TaskCancelled
from essay_agent.memory.lesson import PHASES, LessonStore, phase_file

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def _console() -> Console:
    return Console(highlight=False)


def _load(ctx: click.Context, *, with_ui: bool = True):
    config_file = (ctx.obj or {}).get("config_file")
    verbose = bool((ctx.obj or {}).get("verbose"))
    try:
        settings = load_settings(config_file, overrides={"verbose": verbose})
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    if not with_ui:
        return settings, None
    return build_session(settings, interactive=True)


@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.option(
    "--config",
    "config_file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to a YAML config file (defaults to ./essay-agent.yaml when present).",
)
@click.option(
    "--verbose", is_flag=True, default=False, help="Also dump every LLM exchange to disk."
)
@click.version_option(__version__, "-V", "--version", prog_name="essay-agent")
@click.pass_context
def main(ctx: click.Context, config_file: Path | None, verbose: bool) -> None:
    """essay-agent - a lightweight paper-reproduction agent."""
    ctx.ensure_object(dict)
    ctx.obj["config_file"] = config_file
    ctx.obj["verbose"] = verbose
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        click.echo("\nRun `essay agent` to start the interactive session.")


@main.command(context_settings=CONTEXT_SETTINGS)
@click.pass_context
def agent(ctx: click.Context) -> None:
    """Start the interactive session (type /help for commands)."""
    settings, deps = _load(ctx)
    ui = deps.ui
    console = ui.console
    ui.banner(__version__, settings.llm.model, settings.llm.provider)
    session = _prompt_session(settings.resolved_paths().history_file)
    while True:
        try:
            line = session.prompt("essay> ")
        except KeyboardInterrupt:
            console.print()
            ui.info("bye")
            return
        except EOFError:
            ui.info("bye")
            return
        command = (line or "").strip()
        if not command:
            continue
        if command.startswith("/"):
            if _handle_slash_command(command, settings, deps):
                return
            continue
        try:
            result = run_task(command, deps)
        except TaskCancelled:
            ui.warn("task cancelled - nothing was saved")
            continue
        except EssayAgentError as exc:
            ui.error(str(exc))
            continue
        except KeyboardInterrupt:
            ui.warn("interrupted - nothing was saved")
            continue
        _report(ui, result)


def _prompt_session(history_file: Path):
    class _Fallback:
        def prompt(self, message: str) -> str:
            return input(message)

    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory

        history_file.parent.mkdir(parents=True, exist_ok=True)
        return PromptSession(history=FileHistory(str(history_file)))
    except Exception:
        return _Fallback()


def _handle_slash_command(command: str, settings: Settings, deps) -> bool:
    """Returns True when the session should end."""
    ui = deps.ui
    parts = command.split()
    name = parts[0].lower()
    if name in {"/exit", "/quit", "/q"}:
        ui.info("bye")
        return True
    if name == "/help":
        ui.help_text()
        return False
    if name == "/config":
        ui.console.print_json(json.dumps(settings.summary(), default=str, ensure_ascii=False))
        return False
    if name == "/lesson":
        if len(parts) > 1 and parts[1] in {"clear", "reset"}:
            deps.lesson_store.clear()
            ui.success("lesson memory archived and reset")
            return False
        _show_lessons(ui, deps.lesson_store)
        return False
    ui.warn(f"unknown command: {name} (try /help)")
    return False


def _report(ui, result: TaskResult) -> None:
    state = result.state or {}
    if result.status == "completed":
        ui.rule("done")
        verdict = (state.get("verdict") or {}).get("verdict")
        metrics = (state.get("exec_result") or {}).get("metrics") or {}
        if verdict:
            ui.info(f"verifier verdict: {verdict}")
        if metrics:
            ui.info("metrics: " + ", ".join(f"{k}={v:.4g}" for k, v in list(metrics.items())[:8]))
        if result.report_path:
            ui.info(f"report: {result.report_path}")
        published = result.published or {}
        if published.get("result_dir"):
            ui.info(f"published: {published['result_dir']}")
        if published.get("code_dir"):
            ui.info(f"code: {published['code_dir']}")
    elif result.status == "aborted":
        ui.warn(f"task stopped: {result.message}")
        ui.dim(f"partial files kept for inspection: {result.run_dir}")
    else:
        ui.error(f"task failed: {result.message}")
        ui.dim(f"artifacts kept at {result.run_dir}")


@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument("request_text", nargs=-1, required=True)
@click.pass_context
def run(ctx: click.Context, request_text: tuple[str, ...]) -> None:
    """Run one reproduction non-interactively. REQUEST is a title, URL, DOI or arXiv id."""
    _, deps = _load(ctx)
    query = " ".join(request_text).strip()
    try:
        result = run_task(query, deps)
    except TaskCancelled:
        deps.ui.warn("task cancelled - nothing was saved")
        sys.exit(130)
    except EssayAgentError as exc:
        deps.ui.error(str(exc))
        sys.exit(2)
    _report(deps.ui, result)
    sys.exit(0 if result.ok else 1)


@main.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=True)
@click.pass_context
def config(ctx: click.Context) -> None:
    """Inspect the effective configuration."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(config_show)


@config.command("show")
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Print the effective configuration (API keys are redacted)."""
    settings, _ = _load(ctx, with_ui=False)
    _console().print_json(json.dumps(settings.summary(), default=str, ensure_ascii=False))


@config.command("check")
@click.pass_context
def config_check(ctx: click.Context) -> None:
    """Validate the configuration without calling the model."""
    console = _console()
    settings, _ = _load(ctx, with_ui=False)
    problems: list[str] = []
    try:
        settings.llm.resolve_api_key()
        console.print(f"[green]v[/] api key present for provider {settings.llm.provider}")
    except ConfigError as exc:
        problems.append(str(exc))
    paths = settings.resolved_paths()
    for label, path in (
        ("workspace", paths.workspace_root),
        ("cache", paths.cache_dir),
        ("result", paths.result_dir),
        ("code", paths.code_dir),
        ("lesson", paths.lesson_dir),
    ):
        console.print(f"[green]v[/] {label}: {path}")
    console.print(
        f"[green]v[/] model: {settings.llm.model} (verifier: {settings.llm.effective_verifier_model})"
    )
    console.print(
        f"[green]v[/] budget: {settings.runtime.time_budget_seconds:.0f}s, rounds: "
        f"plan={settings.runtime.max_plan_rounds} exec={settings.runtime.max_execute_rounds}"
    )
    if settings.config_file:
        console.print(f"[green]v[/] config file: {settings.config_file}")
    if problems:
        for problem in problems:
            console.print(f"[red]x[/] {problem}")
        sys.exit(1)
    console.print("[green]configuration looks usable[/]")


@config.command("init")
@click.option("--force", is_flag=True, default=False, help="Overwrite an existing file.")
@click.pass_context
def config_init(ctx: click.Context, force: bool) -> None:
    """Write a starter essay-agent.yaml into the current directory."""
    target = Path.cwd() / "essay-agent.yaml"
    if target.exists() and not force:
        raise click.ClickException(f"{target} already exists (use --force)")
    settings, _ = _load(ctx, with_ui=False)
    target.write_text(_starter_config(settings), encoding="utf-8")
    _console().print(f"[green]v[/] wrote {target}")


def _starter_config(settings: Settings) -> str:
    return (
        "# essay-agent configuration\n"
        "# Environment variables (ESSAY_AGENT_*) override these values.\n\n"
        "llm:\n"
        f"  provider: {settings.llm.provider}\n"
        f"  model: {settings.llm.model}\n"
        "  # api_key: set ESSAY_AGENT_LLM__API_KEY instead of writing it here\n"
        "  temperature: 0.2\n\n"
        "runtime:\n"
        f"  time_budget_seconds: {settings.runtime.time_budget_seconds:.0f}\n"
        "  allow_synthetic_data: false\n\n"
        "search:\n"
        f"  max_results: {settings.search.max_results}\n"
        "  # contact_email: you@example.org\n"
    )


@main.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=True)
@click.pass_context
def lesson(ctx: click.Context) -> None:
    """Inspect and maintain the long-term lesson memory."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(lesson_show)


def _show_lessons(ui, store: LessonStore) -> None:
    stats = store.stats()
    for phase in PHASES:
        info = stats[phase]
        ui.info(
            f"{phase_file(phase)}: {info['chars']} chars, {info['blocks']} block(s) - {info['file']}"
        )
        text = store.read(phase).strip()
        if text:
            ui.console.print(f"[dim]{text[-1200:]}[/dim]")


@lesson.command("show")
@click.pass_context
def lesson_show(ctx: click.Context) -> None:
    """Show lesson memory size and the most recent entries."""
    _, deps = _load(ctx)
    _show_lessons(deps.ui, deps.lesson_store)


@lesson.command("consolidate")
@click.pass_context
def lesson_consolidate(ctx: click.Context) -> None:
    """Compress the lesson files with the LLM (also happens after every run)."""
    _, deps = _load(ctx)
    for phase in PHASES:
        report = deps.lesson_store.consolidate(phase, llm=deps.llm)
        deps.ui.info(report.describe())


@lesson.command("clear")
@click.pass_context
def lesson_clear(ctx: click.Context) -> None:
    """Archive and reset the lesson files."""
    _, deps = _load(ctx)
    deps.lesson_store.clear()
    deps.ui.success("lesson memory archived and reset")
