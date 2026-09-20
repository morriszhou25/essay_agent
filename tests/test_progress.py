"""The stdout progress protocol and the ETA estimator."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from essay_agent.runtime.progress import (
    EtaEstimator,
    NullSink,
    RichProgressSink,
    build_sink,
    format_duration,
    parse_event,
)


def test_parse_progress_event() -> None:
    event = parse_event('{"event": "progress", "step": 3, "total": 10, "epoch": 1}')
    assert event is not None
    assert (event.kind, event.step, event.total, event.epoch) == ("progress", 3, 10, 1)
    assert event.fraction == pytest.approx(0.3)


def test_parse_metric_estimate_and_done_events() -> None:
    metric = parse_event('{"event": "metric", "name": "accuracy", "value": 0.91}')
    assert metric is not None and metric.name == "accuracy"
    assert metric.value == pytest.approx(0.91)

    estimate = parse_event('{"event": "estimate", "estimated_full_seconds": 12.5}')
    assert estimate is not None and estimate.data["estimated_full_seconds"] == 12.5

    done = parse_event('{"event": "done", "metrics": {"accuracy": 0.9, "skip": "x"}}')
    assert done is not None and done.metrics == {"accuracy": 0.9}

    logged = parse_event('{"event": "log", "message": "loading"}')
    assert logged is not None and logged.message == "loading"


def test_parse_event_rejects_noise() -> None:
    assert parse_event("") is None
    assert parse_event("plain log line") is None
    assert parse_event("{not json}") is None
    assert parse_event('{"event": "unknown-kind"}') is None
    assert parse_event('["a", "list"]') is None


def test_eta_estimator_needs_history_then_reports() -> None:
    estimator = EtaEstimator(window=10, min_samples=3)
    estimator.update(0, 0.0)
    assert estimator.eta(0, 10, 0.0) is None
    for step, when in ((1, 1.0), (2, 2.0)):
        estimator.update(step, when)
    assert estimator.rate() == pytest.approx(1.0)
    assert estimator.eta(2, 12, 2.0) == pytest.approx(10.0)
    assert estimator.elapsed(5.0) == pytest.approx(5.0)


def test_eta_estimator_handles_unknown_total_and_reset() -> None:
    estimator = EtaEstimator()
    estimator.update(1, 0.0)
    assert estimator.eta(1, None, 1.0) is None
    estimator.reset()
    assert estimator.samples == 0 and estimator.elapsed(1.0) is None


def test_format_duration() -> None:
    assert format_duration(None) == "--:--"
    assert format_duration(0) == "00:00"
    assert format_duration(61) == "01:01"
    assert format_duration(3725) == "1:02:05"


def test_null_sink_accumulates_metrics_and_logs() -> None:
    sink = NullSink()
    sink.handle(parse_event('{"event": "metric", "name": "accuracy", "value": 0.5}'))
    sink.handle(parse_event('{"event": "done", "metrics": {"loss": 0.1}}'))
    sink.log("noise")
    sink.close()
    assert sink.metrics == {"accuracy": 0.5, "loss": 0.1}
    assert sink.logs == ["noise"]
    assert sink.closed is True


def test_build_sink_falls_back_to_plain_on_a_dumb_terminal() -> None:
    console = Console(file=io.StringIO(), force_terminal=False)
    sink = build_sink(console, "run", 10)
    assert sink.__class__.__name__ == "PlainProgressSink"


def test_rich_sink_renders_and_tracks_state() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=True, width=120)
    sink = RichProgressSink(console, label="reproduction", total=10)
    with sink:
        for step in range(1, 5):
            sink.handle(parse_event(f'{{"event": "progress", "step": {step}, "total": 10}}'))
        sink.handle(parse_event('{"event": "metric", "name": "accuracy", "value": 0.9}'))
        sink.handle(parse_event('{"event": "estimate", "estimated_full_seconds": 3.0}'))
        sink.handle(parse_event('{"event": "done", "metrics": {"accuracy": 0.9}}'))
    assert sink._metrics["accuracy"] == pytest.approx(0.9)
    assert sink._closed is True
    assert "reproduction" in buffer.getvalue()


def test_plain_sink_reports_progress(capsys) -> None:
    console = Console(file=io.StringIO(), force_terminal=False, width=100)
    sink = build_sink(console, "run", 10, plain=True)
    with sink:
        sink.handle(parse_event('{"event": "progress", "step": 5, "total": 10}'))
        sink.handle(parse_event('{"event": "metric", "name": "acc", "value": 0.5}'))
    assert sink._pct == 50
    assert capsys is not None
