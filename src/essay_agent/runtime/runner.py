"""Subprocess runner for generated reproduction scripts.

Responsibilities:

* stream the child's stdout, parse the JSON progress protocol and feed the sink
* guarantee the child dies (with its whole process tree) on timeout or Ctrl+C
* collect ``metrics.json`` and the figure files the script produced
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from essay_agent.config import RuntimeSettings
from essay_agent.errors import TaskCancelled
from essay_agent.runtime.progress import (
    NullSink,
    ProgressEvent,
    ProgressSink,
    build_sink,
    parse_event,
)

SinkFactory = Callable[[str, "int | None"], ProgressSink]
FIGURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".svg", ".pdf", ".csv")

# Environment every generated script is started with: plumbing defaults plus the known-good
# workarounds for runtimes that abort the child over a machine-specific conflict. Do not let a
# reproduction depend on how the user's shell happens to be configured.
#
# ``KMP_DUPLICATE_LIB_OK``: PyTorch ships one copy of Intel's OpenMP runtime and another library
# can lazily load a second ``libiomp5md.dll``, at which point the runtime aborts the whole process
# ("OMP: Error #15"). Measured on run-20260920-062827-3e6074: every execution round died at step 0
# with it, and the same script finished with exit code 0 once the variable was set. Intel documents
# the flag as unsafe ("may cause crashes or silently produce incorrect results"), so it is never
# silent: `ScriptRunner.env_fixes` states what was injected and why, and the run outcome records it.
CHILD_ENV_DEFAULTS: dict[str, str] = {
    "PYTHONUNBUFFERED": "1",
    "PYTHONIOENCODING": "utf-8",
    "MPLBACKEND": "Agg",
    "PYTHONDONTWRITEBYTECODE": "1",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
}

# The subset of the above that changes numerical or runtime behaviour, with the reason it is there.
ENV_WORKAROUNDS: dict[str, str] = {
    "KMP_DUPLICATE_LIB_OK": (
        "duplicate Intel OpenMP runtime: a lazily loaded library adds a second libiomp5md.dll and "
        "the runtime aborts the process (OMP: Error #15)"
    ),
}


@dataclass
class RunOutcome:
    """Everything the verifier needs to know about one script execution."""

    ok: bool
    exit_code: int | None = None
    duration: float = 0.0
    timed_out: bool = False
    cancelled: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)
    events: int = 0
    event_list: list[ProgressEvent] = field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""
    script: str = ""
    metrics_path: str | None = None
    error: str | None = None

    def describe(self) -> str:
        status = "ok" if self.ok else ("timeout" if self.timed_out else "failed")
        bits = [f"{status} in {self.duration:.1f}s", f"exit={self.exit_code}"]
        if self.metrics:
            bits.append(
                "metrics: " + ", ".join(f"{k}={v:.4g}" for k, v in list(self.metrics.items())[:6])
            )
        if self.figures:
            bits.append(f"{len(self.figures)} figure(s)")
        if self.error:
            bits.append(self.error)
        return " | ".join(bits)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill a child and everything it spawned."""
    if proc.poll() is not None:
        return
    with suppress(Exception):
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
        else:  # pragma: no cover - POSIX
            import signal

            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    with suppress(Exception):
        proc.kill()
    with suppress(Exception):
        proc.wait(timeout=10)


class _StreamReader(threading.Thread):
    """Line reader that never lets the child block on a full pipe."""

    def __init__(self, stream, sink_queue: queue.Queue) -> None:
        super().__init__(daemon=True)
        self.stream = stream
        self.queue = sink_queue

    def run(self) -> None:
        try:
            with suppress(Exception):
                for line in iter(self.stream.readline, ""):
                    self.queue.put(line.rstrip("\n"))
        finally:
            with suppress(Exception):
                self.stream.close()
            self.queue.put(None)


class ScriptRunner:
    """Runs generated scripts and turns their output into a :class:`RunOutcome`."""

    def __init__(
        self,
        settings: RuntimeSettings | None = None,
        *,
        python_executable: str | None = None,
        sink_factory: SinkFactory | None = None,
    ) -> None:
        self.settings = settings or RuntimeSettings()
        self.python = python_executable or self.settings.python_executable or _default_python()
        self._sink_factory = sink_factory

    @property
    def env_fixes(self) -> dict[str, str]:
        """The known-good workarounds every child gets, each with the reason it is needed."""
        return {key: reason for key, reason in ENV_WORKAROUNDS.items() if key in CHILD_ENV_DEFAULTS}

    # ------------------------------------------------------------------ public
    def run(
        self,
        script: Path,
        *,
        cwd: Path,
        timeout: float,
        label: str = "run",
        args: list[str] | None = None,
        env: dict[str, str | None] | None = None,
        metrics_file: str = "metrics.json",
        sink_factory: SinkFactory | None = None,
        total_steps: int | None = None,
    ) -> RunOutcome:
        script = Path(script)
        cwd = Path(cwd)
        cwd.mkdir(parents=True, exist_ok=True)
        if not script.is_file():
            return RunOutcome(ok=False, error=f"script not found: {script}", script=str(script))

        factory = sink_factory or self._sink_factory
        sink: ProgressSink
        if factory is None:
            sink = build_sink(_fallback_console(), label, total_steps, plain=True)
        else:
            sink = factory(label, total_steps)

        outcome = RunOutcome(ok=False, script=str(script))
        started = time.monotonic()
        env_map = self._build_env(env)
        creationflags = 0
        if os.name == "nt":  # pragma: no cover - platform specific
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )

        stdout_queue: queue.Queue = queue.Queue()
        stderr_queue: queue.Queue = queue.Queue()
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        opened = _enter(sink)
        try:
            proc = subprocess.Popen(
                [self.python, "-u", str(script), *(args or [])],
                cwd=str(cwd),
                env=env_map,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
        except OSError as exc:
            _exit(sink, opened)
            outcome.error = f"cannot start {self.python}: {exc}"
            return outcome

        readers = [
            _StreamReader(proc.stdout, stdout_queue),
            _StreamReader(proc.stderr, stderr_queue),
        ]
        for reader in readers:
            reader.start()

        stdout_done = stderr_done = False
        try:
            while True:
                if not stdout_done:
                    stdout_done = _drain(stdout_queue, stdout_lines, sink, events_counter=outcome)
                if not stderr_done:
                    stderr_done = _drain(stderr_queue, stderr_lines, None, events_counter=outcome)
                elapsed = time.monotonic() - started
                exited = proc.poll() is not None
                if exited and stdout_done and stderr_done:
                    break
                if not exited and elapsed > timeout:
                    outcome.timed_out = True
                    outcome.error = f"exceeded the {timeout:.0f}s timeout"
                    _kill_tree(proc)
                    break
                if exited and elapsed > timeout + 10:
                    # The process is gone but a pipe stayed open (a grandchild kept it).
                    outcome.error = outcome.error or "output streams did not close"
                    _kill_tree(proc)
                    break
                time.sleep(0.05)
            # Flush whatever the readers managed to buffer before the process died.
            for _ in range(50):
                drained_stdout = _drain(stdout_queue, stdout_lines, sink, events_counter=outcome)
                drained_stderr = _drain(stderr_queue, stderr_lines, None, events_counter=outcome)
                if drained_stdout and drained_stderr:
                    break
                if stdout_queue.empty() and stderr_queue.empty():
                    break
                time.sleep(0.01)
        except KeyboardInterrupt:
            outcome.cancelled = True
            outcome.error = "cancelled by user"
            _kill_tree(proc)
            _exit(sink, opened)
            raise TaskCancelled("run cancelled by Ctrl+C") from None
        finally:
            try:
                outcome.exit_code = proc.poll()
            except Exception:
                outcome.exit_code = None
            outcome.duration = time.monotonic() - started
            _exit(sink, opened)
            for reader in readers:
                reader.join(timeout=2)

        outcome.metrics = _collect_metrics(sink, cwd, metrics_file)
        if outcome.metrics:
            outcome.metrics_path = str(cwd / metrics_file)
        outcome.figures = _collect_figures(cwd)
        outcome.stdout_tail = "\n".join(stdout_lines[-60:])
        outcome.stderr_tail = "\n".join(stderr_lines[-60:])
        outcome.ok = (not outcome.timed_out) and outcome.exit_code == 0
        if not outcome.ok and not outcome.error:
            outcome.error = f"script exited with code {outcome.exit_code}"
        return outcome

    # ----------------------------------------------------------------- helpers
    def _build_env(self, extra: dict[str, str | None] | None) -> dict[str, str]:
        """The child's environment: this process's, plus ``extra``.

        A ``None`` value in ``extra`` *removes* the variable (including one of our own
        defaults). That is how a verified direct route takes back an inherited
        ``HTTP_PROXY``: setting it to "" would leave a value the HTTP libraries still parse.
        """
        env = dict(os.environ)
        env.update(CHILD_ENV_DEFAULTS)
        env.update(
            {
                "ESSAY_AGENT_DEVICE": self.settings.device,
                "ESSAY_AGENT_TIME_BUDGET_SECONDS": str(self.settings.time_budget_seconds),
                "ESSAY_AGENT_ALLOW_SYNTHETIC_DATA": str(self.settings.allow_synthetic_data).lower(),
            }
        )
        for key, value in (extra or {}).items():
            if value is None:
                env.pop(str(key), None)
            else:
                env[str(key)] = str(value)
        return env


def _default_python() -> str:
    import sys

    return sys.executable


def _fallback_console():
    from rich.console import Console

    return Console()


def _enter(sink: ProgressSink) -> bool:
    if hasattr(sink, "__enter__"):
        sink.__enter__()
        return True
    return False


def _exit(sink: ProgressSink, opened: bool) -> None:
    try:
        if opened:
            sink.__exit__(None, None, None)  # type: ignore[attr-defined]
        else:
            sink.close()
    except Exception:
        pass


def _drain(
    source: queue.Queue,
    sink_lines: list[str],
    sink: ProgressSink | None,
    *,
    events_counter: RunOutcome,
    limit: int = 500,
) -> bool:
    """Move lines from ``source`` into ``sink_lines``. Returns True when the stream ended."""
    drained = 0
    while drained < limit:
        try:
            line = source.get_nowait()
        except queue.Empty:
            return False
        drained += 1
        if line is None:
            return True
        sink_lines.append(line)
        if sink is None:
            continue
        event = parse_event(line)
        if event is not None:
            events_counter.events += 1
            events_counter.event_list.append(event)
            with suppress(Exception):
                sink.handle(event)
        else:
            with suppress(Exception):
                sink.log(line)
    return False


def _collect_metrics(sink: ProgressSink, cwd: Path, metrics_file: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if isinstance(sink, NullSink):
        metrics.update(sink.metrics)
    else:
        for event in getattr(sink, "events", []):
            if isinstance(event, ProgressEvent):
                if event.kind == "metric" and event.name and event.value is not None:
                    metrics[event.name] = event.value
                metrics.update(event.metrics)
        metrics.update(getattr(sink, "_metrics", {}) or {})
    path = cwd / metrics_file
    if path.is_file():
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            flat: dict[str, float] = {}
            for key, value in payload.items():
                if isinstance(value, (bool, int, float)):
                    flat[key] = float(value)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)) and not isinstance(sub_value, bool):
                            flat[f"{key}.{sub_key}"] = float(sub_value)
            metrics.update(flat)
    return metrics


def _collect_figures(cwd: Path) -> list[str]:
    figures: list[str] = []
    for base in (cwd / "figures", cwd / "figs", cwd):
        if not base.is_dir():
            continue
        for path in sorted(base.iterdir()):
            if path.is_file() and path.suffix.lower() in FIGURE_SUFFIXES:
                figures.append(str(path.relative_to(cwd)).replace("\\", "/"))
        if figures:
            break
    return figures
