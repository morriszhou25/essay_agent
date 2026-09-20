"""Generated-script execution: streaming, timeouts and artifact collection."""

from __future__ import annotations

from pathlib import Path

import pytest

from essay_agent.config import RuntimeSettings
from essay_agent.runtime.preflight import estimate_from_events
from essay_agent.runtime.progress import NullSink
from essay_agent.runtime.runner import ScriptRunner
from tests.fakes import SAMPLE_REPRO_SCRIPT


def _runner(**overrides) -> ScriptRunner:
    return ScriptRunner(RuntimeSettings(**overrides), sink_factory=lambda label, total: NullSink())


def _write(tmp_path: Path, source: str, name: str = "repro.py") -> Path:
    script = tmp_path / name
    script.write_text(source, encoding="utf-8")
    return script


def test_full_run_collects_metrics_and_figures(tmp_path: Path) -> None:
    script = _write(tmp_path, SAMPLE_REPRO_SCRIPT)
    outcome = _runner().run(
        script, cwd=tmp_path, timeout=60, label="test", args=["--out", str(tmp_path)]
    )
    assert outcome.ok is True, outcome.stderr_tail
    assert outcome.exit_code == 0
    assert outcome.metrics["accuracy"] == pytest.approx(0.912)
    assert outcome.metrics["baseline_accuracy"] == pytest.approx(0.884)
    assert outcome.figures == ["figures/fig1.png"]
    assert outcome.events > 0
    assert any(event.kind == "done" for event in outcome.event_list)
    assert outcome.duration > 0


def test_preflight_args_produce_an_estimate(tmp_path: Path) -> None:
    script = _write(tmp_path, SAMPLE_REPRO_SCRIPT)
    outcome = _runner().run(script, cwd=tmp_path, timeout=60, label="pre", args=["--preflight"])
    assert outcome.ok is True, outcome.stderr_tail
    estimate = estimate_from_events(outcome.event_list, outcome.duration)
    assert estimate.source == "script"
    assert estimate.estimated_full_seconds == pytest.approx(0.4)
    assert estimate.params == {"steps": 12}
    assert not (tmp_path / "metrics.json").exists()


def test_timeout_kills_the_child(tmp_path: Path) -> None:
    script = _write(tmp_path, "import time\nfor i in range(100):\n    time.sleep(0.5)\n", "slow.py")
    outcome = _runner().run(script, cwd=tmp_path, timeout=1.0, label="slow")
    assert outcome.ok is False
    assert outcome.timed_out is True
    assert outcome.duration < 10
    assert "timeout" in (outcome.error or "")


def test_failing_script_reports_the_exit_code(tmp_path: Path) -> None:
    script = _write(tmp_path, "import sys\nprint('about to fail')\nsys.exit(3)\n", "boom.py")
    outcome = _runner().run(script, cwd=tmp_path, timeout=20, label="boom")
    assert outcome.ok is False
    assert outcome.exit_code == 3
    assert "exited with code 3" in (outcome.error or "")
    assert "about to fail" in outcome.stdout_tail


def test_missing_script_is_reported(tmp_path: Path) -> None:
    outcome = _runner().run(tmp_path / "nope.py", cwd=tmp_path, timeout=5, label="x")
    assert outcome.ok is False
    assert "not found" in (outcome.error or "")


def test_unparseable_output_is_passed_to_the_sink_as_log(tmp_path: Path) -> None:
    script = _write(tmp_path, "print('hello there')\nprint('second line')\n")
    sink = NullSink()
    runner = ScriptRunner(RuntimeSettings(), sink_factory=lambda label, total: sink)
    outcome = runner.run(script, cwd=tmp_path, timeout=20, label="log")
    assert outcome.ok is True
    assert sink.logs == ["hello there", "second line"]
    assert outcome.events == 0
