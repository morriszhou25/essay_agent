"""Entry point: run one task through the graph, with Ctrl+C safety.

Robustness rule #1: everything a task writes lives in its run workspace. If the
user presses Ctrl+C (or the process dies unexpectedly) the workspace is deleted
and nothing is published. Only a task that reaches ``finalize`` is published.
"""

from __future__ import annotations

import atexit
import contextlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from essay_agent.config import Settings, load_settings
from essay_agent.console import UI, build_ui
from essay_agent.errors import EssayAgentError, TaskCancelled
from essay_agent.graph import DEFAULT_RECURSION_LIMIT, build_graph
from essay_agent.nodes.base import Deps, build_deps
from essay_agent.workspace import RunWorkspace, new_run_id


@dataclass
class TaskResult:
    """What the CLI needs to report after a task."""

    run_id: str
    run_dir: Path
    status: str
    state: dict[str, Any] = field(default_factory=dict)
    report_path: str | None = None
    message: str = ""
    published: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "completed"


_ACTIVE: set[RunWorkspace] = set()
_LOCK = threading.Lock()


def _register(workspace: RunWorkspace) -> None:
    with _LOCK:
        _ACTIVE.add(workspace)


def _unregister(workspace: RunWorkspace) -> None:
    with _LOCK:
        _ACTIVE.discard(workspace)


@atexit.register
def _purge_unfinished_workspaces() -> None:
    """Last line of defence: an unfinished task must not leave files behind."""
    with _LOCK:
        pending = [workspace for workspace in _ACTIVE if not workspace.published]
    for workspace in pending:
        with contextlib.suppress(Exception):
            workspace.purge()


def run_task(
    query: str,
    deps: Deps,
    *,
    run_id: str | None = None,
    slug: str = "",
    checkpointer: Any = None,
    initial_state: dict[str, Any] | None = None,
) -> TaskResult:
    """Execute the full pipeline for one user request."""
    ui = deps.ui
    paths = deps.settings.resolved_paths().ensure()
    workspace = RunWorkspace.create(paths.workspace_root, run_id or new_run_id(), slug)
    deps.workspace_factory = lambda _run_id, _slug: workspace
    _register(workspace)

    state: dict[str, Any] = {
        "run_id": workspace.run_id,
        "run_dir": str(workspace.dir),
        "slug": slug,
        "query": query,
        "status": "running",
        "messages": [],
    }
    if initial_state:
        state.update(initial_state)

    graph = build_graph(deps, checkpointer=checkpointer)
    final: dict[str, Any] = dict(state)
    try:
        final = graph.invoke(
            state,
            config={
                "recursion_limit": DEFAULT_RECURSION_LIMIT,
                "configurable": {"thread_id": workspace.run_id},
            },
        )
    except (KeyboardInterrupt, TaskCancelled) as exc:
        workspace.purge()
        _unregister(workspace)
        ui.warn("Ctrl+C - the task was cancelled and every file it produced has been deleted")
        raise TaskCancelled(str(exc) or "cancelled by user") from None
    except EssayAgentError as exc:
        workspace.log(f"fatal: {type(exc).__name__}: {exc}")
        _unregister(workspace)
        return TaskResult(
            run_id=workspace.run_id,
            run_dir=workspace.dir,
            status="failed",
            state=final,
            message=str(exc),
        )
    except Exception as exc:  # unexpected: keep the staging area for debugging
        workspace.log(f"unexpected error: {type(exc).__name__}: {exc}")
        _unregister(workspace)
        raise
    _unregister(workspace)

    status = final.get("status") or "failed"
    return TaskResult(
        run_id=workspace.run_id,
        run_dir=workspace.dir,
        status=status,
        state=final,
        report_path=final.get("report_path"),
        message=final.get("error") or final.get("fetch_message") or "",
        published=final.get("publish") or {},
    )


def build_session(
    settings: Settings | None = None,
    *,
    interactive: bool = True,
    ui: UI | None = None,
    deps: Deps | None = None,
) -> tuple[Settings, Deps]:
    """Create (settings, deps) for a CLI session."""
    active_settings = settings or load_settings()
    active_ui = ui or build_ui(active_settings, interactive=interactive)
    if deps is None:
        deps = build_deps(active_settings, active_ui)
    else:
        deps.ui = active_ui
    return active_settings, deps
