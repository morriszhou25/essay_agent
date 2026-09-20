"""Time-budget arithmetic."""

from __future__ import annotations

import pytest

from essay_agent.runtime.preflight import (
    Estimate,
    estimate_from_events,
    needs_adjustment,
    scale_params,
    suggest_adjustment,
)
from essay_agent.runtime.progress import parse_event


def _events(*lines: str):
    return [parse_event(line) for line in lines]


def test_estimate_prefers_the_scripts_own_number() -> None:
    estimate = estimate_from_events(
        _events(
            '{"event": "progress", "step": 5, "total": 100}',
            '{"event": "estimate", "estimated_full_seconds": 42.0, "params": {"steps": 100}}',
        ),
        wall_seconds=3.0,
    )
    assert estimate.source == "script"
    assert estimate.estimated_full_seconds == pytest.approx(42.0)
    assert estimate.params == {"steps": 100}
    assert estimate.step_fraction == pytest.approx(0.05)


def test_estimate_extrapolates_from_progress() -> None:
    estimate = estimate_from_events(
        _events('{"event": "progress", "step": 10, "total": 100}'), wall_seconds=5.0
    )
    assert estimate.source == "extrapolated"
    assert estimate.estimated_full_seconds == pytest.approx(50.0)


def test_estimate_uses_the_fallback_total_when_the_script_never_prints_one() -> None:
    estimate = estimate_from_events(
        _events('{"event": "progress", "step": 5}'), wall_seconds=5.0, fallback_total_steps=50
    )
    assert estimate.source == "extrapolated"
    assert estimate.estimated_full_seconds == pytest.approx(50.0)


def test_estimate_unknown_without_signal() -> None:
    estimate = estimate_from_events([], wall_seconds=1.0)
    assert estimate.source == "unknown"
    assert estimate.is_known is False
    assert "no time estimate" in estimate.describe()


def test_describe_mentions_the_source() -> None:
    estimate = Estimate(estimated_full_seconds=90.0, source="script", step_fraction=0.25)
    assert "01:30" in estimate.describe()
    assert "reported by the pre-flight script" in estimate.describe()


def test_needs_adjustment_only_for_known_overruns() -> None:
    assert needs_adjustment(Estimate(estimated_full_seconds=120.0, source="script"), 60.0) is True
    assert needs_adjustment(Estimate(estimated_full_seconds=30.0, source="script"), 60.0) is False
    assert needs_adjustment(Estimate(), 60.0) is False


def test_scale_params_respects_floors_and_ignores_other_keys() -> None:
    scaled = scale_params({"steps": 1000, "epochs": 10, "lr": 0.1, "note": "x"}, 0.01)
    assert scaled["steps"] == 20
    assert scaled["epochs"] == 1
    assert scaled["lr"] == 0.1
    assert scaled["note"] == "x"


def test_scale_params_keeps_floats_float() -> None:
    assert scale_params({"epochs": 2.0}, 0.5)["epochs"] == pytest.approx(1.0)


def test_suggest_adjustment_computes_a_factor_and_notes() -> None:
    suggestion = suggest_adjustment(
        Estimate(estimated_full_seconds=1000.0, source="script"),
        budget_seconds=600.0,
        params={"steps": 500},
    )
    assert suggestion["factor"] == pytest.approx(0.54, abs=1e-6)
    assert suggestion["params"]["steps"] == 270
    assert any("scaled" in note for note in suggestion["notes"])


def test_suggest_adjustment_without_knobs_says_so() -> None:
    suggestion = suggest_adjustment(
        Estimate(estimated_full_seconds=1000.0, source="script"), budget_seconds=100.0, params={}
    )
    assert any("strategy" in note for note in suggestion["notes"])


def test_suggest_adjustment_with_unknown_estimate_is_a_no_op() -> None:
    suggestion = suggest_adjustment(Estimate(), budget_seconds=100.0, params={"steps": 10})
    assert suggestion["factor"] == 1.0
    assert suggestion["params"] == {"steps": 10}
