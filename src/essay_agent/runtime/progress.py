"""Progress protocol shared by every generated script and the runner.

Generated reproduction scripts are required to print one JSON object per line::

    {"event": "progress", "step": 3, "total": 10, "epoch": 1}
    {"event": "metric",   "name": "accuracy", "value": 0.91}
    {"event": "estimate", "estimated_full_seconds": 142.0, "params": {"steps": 200}}
    {"event": "figure",   "path": "figures/fig1.png"}
    {"event": "done",     "metrics": {"accuracy": 0.93}}

Anything else on stdout is passed through to the run log. This module parses
those lines, tracks an ETA and renders a live bar.
"""

from __future__ import annotations

import json
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

EventKind = Literal["progress", "metric", "estimate", "figure", "log", "done", "error"]
_KINDS = {"progress", "metric", "estimate", "figure", "log", "done", "error"}


@dataclass
class ProgressEvent:
    """One parsed line of the child process protocol."""

    kind: EventKind
    step: int | None = None
    total: int | None = None
    epoch: int | None = None
    name: str | None = None
    value: float | None = None
    message: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    raw: str = ""

    @property
    def fraction(self) -> float | None:
        if self.step is None or not self.total:
            return None
        return min(1.0, max(0.0, self.step / self.total))


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_event(line: str) -> ProgressEvent | None:
    """Parse one stdout line. Returns ``None`` for anything that is not a protocol event."""
    if not line:
        return None
    stripped = line.strip()
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    kind = str(payload.get("event", payload.get("type", ""))).lower()
    if kind not in _KINDS:
        return None

    metrics: dict[str, float] = {}
    raw_metrics = payload.get("metrics")
    if isinstance(raw_metrics, dict):
        for key, value in raw_metrics.items():
            number = _as_float(value)
            if number is not None:
                metrics[str(key)] = number

    message = payload.get("message")
    if message is None and kind == "log":
        message = payload.get("text") or payload.get("msg")

    return ProgressEvent(
        kind=kind,  # type: ignore[arg-type]
        step=_as_int(payload.get("step")),
        total=_as_int(payload.get("total")),
        epoch=_as_int(payload.get("epoch")),
        name=payload.get("name") if isinstance(payload.get("name"), str) else None,
        value=_as_float(payload.get("value")),
        message=str(message) if message is not None else None,
        metrics=metrics,
        data=payload,
        raw=stripped,
    )


class EtaEstimator:
    """Estimates remaining time from observed step throughput.

    Uses the throughput between the oldest and newest sample in a sliding window,
    which is stable against a single slow step but still reacts to real slow-downs.
    """

    def __init__(self, window: int = 24, min_samples: int = 3) -> None:
        self.window = max(2, window)
        self.min_samples = max(2, min_samples)
        self._samples: deque[tuple[float, int]] = deque(maxlen=self.window)
        self._started_at: float | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._started_at = None

    def update(self, step: int | None, now: float) -> None:
        if step is None:
            return
        if self._started_at is None:
            self._started_at = now
        self._samples.append((now, step))

    @property
    def samples(self) -> int:
        return len(self._samples)

    def rate(self) -> float | None:
        """Steps per second, or ``None`` when there is not enough signal yet."""
        if len(self._samples) < self.min_samples:
            return None
        (t0, s0), (t1, s1) = self._samples[0], self._samples[-1]
        elapsed = t1 - t0
        steps = s1 - s0
        if elapsed <= 0 or steps <= 0:
            return None
        return steps / elapsed

    def eta(self, step: int | None, total: int | None, now: float) -> float | None:
        if step is None or not total:
            return None
        rate = self.rate()
        if not rate or rate <= 0:
            return None
        remaining = max(0, total - step)
        return remaining / rate

    def elapsed(self, now: float) -> float | None:
        return None if self._started_at is None else now - self._started_at


def format_duration(seconds: float | None) -> str:
    """``m:ss`` / ``h:mm:ss`` rendering used in the progress bar."""
    if seconds is None:
        return "--:--"
    seconds = max(0.0, float(seconds))
    if seconds < 3600:
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes:02d}:{secs:02d}"
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


@runtime_checkable
class ProgressSink(Protocol):
    """Where parsed events go."""

    def handle(self, event: ProgressEvent) -> None: ...

    def log(self, line: str) -> None: ...

    def close(self) -> None: ...


class NullSink:
    """Collects events without rendering (tests, non-interactive runs)."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []
        self.logs: list[str] = []
        self.closed = False

    def handle(self, event: ProgressEvent) -> None:
        self.events.append(event)

    def log(self, line: str) -> None:
        self.logs.append(line)

    def close(self) -> None:
        self.closed = True

    @property
    def metrics(self) -> dict[str, float]:
        found: dict[str, float] = {}
        for event in self.events:
            if event.kind == "metric" and event.name:
                found[event.name] = event.value if event.value is not None else float("nan")
            found.update(event.metrics)
        return found


class RichProgressSink:
    """Live bar + ETA rendered with rich."""

    def __init__(self, console, label: str = "run", total: int | None = None) -> None:
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        self.console = console
        self.label = label
        self.total = total
        self.estimator = EtaEstimator()
        self.progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%", justify="right"),
            TextColumn("eta {task.fields[eta]}", style="yellow"),
            TimeElapsedColumn(),
            TextColumn("{task.fields[note]}", style="dim"),
            console=console,
            transient=False,
            auto_refresh=False,
        )
        self._task_id = None
        self._metrics: dict[str, float] = {}
        self._closed = False

    # ------------------------------------------------------------- lifecycle
    def __enter__(self) -> RichProgressSink:
        self.progress.start()
        self._task_id = self.progress.add_task(
            self.label,
            total=self.total,
            eta="--:--",
            note="starting",
        )
        self.progress.refresh()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with suppress(Exception):  # pragma: no cover - rich internals
            self.progress.stop()

    # ---------------------------------------------------------------- events
    def _update(self, **fields) -> None:
        if self._task_id is None:
            return
        self.progress.update(self._task_id, **fields)
        self.progress.refresh()

    def handle(self, event: ProgressEvent) -> None:
        import time

        if event.kind == "progress":
            now = time.monotonic()
            self.estimator.update(event.step, now)
            if event.total:
                self.total = event.total
            eta = self.estimator.eta(event.step, event.total or self.total, now)
            note_bits = []
            if event.epoch is not None:
                note_bits.append(f"epoch {event.epoch}")
            if event.step is not None and not event.total:
                note_bits.append(f"step {event.step}")
            if self._metrics:
                note_bits.append(
                    " ".join(f"{k}={v:.4g}" for k, v in list(self._metrics.items())[:3])
                )
            self._update(
                completed=event.step if event.step is not None else None,
                total=event.total or self.total,
                eta=format_duration(eta),
                note=" | ".join(note_bits),
            )
        elif event.kind == "metric":
            if event.name and event.value is not None:
                self._metrics[event.name] = event.value
            self._metrics.update(event.metrics)
            self._update(note=" ".join(f"{k}={v:.4g}" for k, v in list(self._metrics.items())[:4]))
        elif event.kind == "estimate":
            seconds = event.data.get("estimated_full_seconds")
            self.console.print(
                f"[dim]{self.label}: estimated full-run time {format_duration(_as_float(seconds))}[/dim]"
            )
        elif event.kind in {"done", "error"}:
            self._metrics.update(event.metrics)
            if event.kind == "done":
                self._update(
                    completed=self.total,
                    eta="00:00",
                    note="done" + (f" | {self._format_metrics()}" if self._metrics else ""),
                )
            else:
                self._update(note=f"error: {event.message or 'see log'}")
        elif event.kind in {"log", "figure"} and event.message:
            self.log(event.message)

    def _format_metrics(self) -> str:
        return " ".join(f"{k}={v:.4g}" for k, v in list(self._metrics.items())[:4])

    def log(self, line: str) -> None:
        self.console.print(f"[dim]{line}[/dim]")


class PlainProgressSink:
    """Minimal, throttled text output for non-tty environments (CI, log files)."""

    def __init__(
        self, console, label: str = "run", total: int | None = None, every: float = 15.0
    ) -> None:
        import time as _time

        self.console = console
        self.label = label
        self.total = total
        self.every = every
        self.estimator = EtaEstimator()
        self._last = 0.0
        self._time = _time
        self._pct = 0

    def __enter__(self) -> PlainProgressSink:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def handle(self, event: ProgressEvent) -> None:
        now = self._time.monotonic()
        if event.kind == "progress":
            self.estimator.update(event.step, now)
            if event.total:
                self.total = event.total
            if event.step is None:
                return
            percent = int(100 * event.step / self.total) if self.total else 0
            if now - self._last < self.every and percent < 100 and percent - self._pct < 5:
                return
            self._last = now
            self._pct = percent
            eta = self.estimator.eta(event.step, self.total, now)
            self.console.print(
                f"[dim]{self.label}: {event.step}/{self.total or '?'} "
                f"({percent}%) eta {format_duration(eta)}[/dim]"
            )
        elif event.kind == "metric" and event.name:
            self.console.print(f"[dim]{self.label}: {event.name}={event.value:.4g}[/dim]")
        elif event.kind == "log" and event.message:
            self.log(event.message)

    def log(self, line: str) -> None:
        self.console.print(f"[dim]{line}[/dim]")

    def close(self) -> None:
        return None


def build_sink(
    console, label: str, total: int | None = None, *, plain: bool = False
) -> ProgressSink:
    """Return the best sink for the current console."""
    is_terminal = bool(getattr(console, "is_terminal", False))
    if plain or not is_terminal:
        return PlainProgressSink(console, label=label, total=total)
    return RichProgressSink(console, label=label, total=total)
