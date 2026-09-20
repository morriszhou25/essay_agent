"""Budget arithmetic for stage 4 (``execute``).

Before paying for the full run, the agent runs a deliberately cheap *pre-flight*
script that either reports ``estimated_full_seconds`` itself or lets us
extrapolate from the fraction of work it covered. If the projection busts the
time budget, the plan is scaled down until it fits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from essay_agent.runtime.progress import ProgressEvent, format_duration

EstimateSource = Literal["script", "extrapolated", "unknown"]

# Parameters that (roughly) scale linearly with wall-clock cost.
COST_PARAMS: tuple[str, ...] = (
    "steps",
    "max_steps",
    "num_steps",
    "train_steps",
    "epochs",
    "num_epochs",
    "n_epochs",
    "train_size",
    "train_samples",
    "subset_size",
    "dataset_size",
    "n_samples",
    "num_samples",
    "max_iter",
)

# Parameters with a floor we must respect to keep the reproduction meaningful.
_MINIMUMS: dict[str, float] = {
    "steps": 20,
    "max_steps": 20,
    "num_steps": 20,
    "train_steps": 20,
    "epochs": 1,
    "num_epochs": 1,
    "n_epochs": 1,
    "train_size": 200,
    "train_samples": 200,
    "subset_size": 200,
    "dataset_size": 200,
    "n_samples": 200,
    "num_samples": 200,
    "max_iter": 20,
}


@dataclass
class Estimate:
    """Projected cost of the full run."""

    estimated_full_seconds: float | None = None
    source: EstimateSource = "unknown"
    params: dict[str, Any] = field(default_factory=dict)
    step_fraction: float | None = None
    wall_seconds: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_known(self) -> bool:
        return self.estimated_full_seconds is not None and self.estimated_full_seconds > 0

    def describe(self) -> str:
        if not self.is_known:
            return (
                f"no time estimate available (pre-flight took {format_duration(self.wall_seconds)})"
            )
        origin = {
            "script": "reported by the pre-flight script",
            "extrapolated": "extrapolated from the pre-flight run",
            "unknown": "unknown",
        }[self.source]
        fraction = f", covering {self.step_fraction:.0%} of the work" if self.step_fraction else ""
        return (
            f"estimated full run {format_duration(self.estimated_full_seconds)} "
            f"({origin}{fraction})"
        )


def estimate_from_events(
    events: list[ProgressEvent],
    wall_seconds: float,
    *,
    fallback_total_steps: int | None = None,
) -> Estimate:
    """Derive an :class:`Estimate` from the pre-flight script's output.

    Priority: an explicit ``estimate`` event, then progress-based extrapolation,
    then a known total with no progress (unknown).
    """
    explicit: dict[str, Any] | None = None
    explicit_seconds: float | None = None
    last_step: int | None = None
    last_total: int | None = None
    for event in events:
        if event.kind == "estimate":
            value = event.data.get("estimated_full_seconds")
            if value is None:
                value = event.data.get("estimated_seconds")
            try:
                seconds = float(value) if value is not None else None
            except (TypeError, ValueError):
                seconds = None
            if seconds is not None and seconds > 0:
                explicit = dict(event.data)
                explicit_seconds = seconds
        elif event.kind == "progress" and event.step is not None:
            last_step = event.step
            last_total = event.total or last_total

    total = last_total or fallback_total_steps
    if explicit is not None and explicit_seconds is not None:
        params = explicit.get("params") if isinstance(explicit.get("params"), dict) else {}
        fraction = None
        if last_step is not None and total:
            fraction = min(1.0, last_step / total)
        return Estimate(
            estimated_full_seconds=explicit_seconds,
            source="script",
            params=dict(params),
            step_fraction=fraction,
            wall_seconds=wall_seconds,
            raw=explicit,
        )

    if last_step and total and last_step > 0 and wall_seconds > 0:
        fraction = min(1.0, last_step / total)
        if fraction > 0:
            return Estimate(
                estimated_full_seconds=wall_seconds / fraction,
                source="extrapolated",
                params={},
                step_fraction=fraction,
                wall_seconds=wall_seconds,
            )

    return Estimate(estimated_full_seconds=None, source="unknown", wall_seconds=wall_seconds)


def needs_adjustment(estimate: Estimate, budget_seconds: float) -> bool:
    """True when the projection exceeds the budget (unknown projections never block)."""
    if not estimate.is_known:
        return False
    return bool(estimate.estimated_full_seconds > budget_seconds)


def scale_params(params: dict[str, Any], factor: float) -> dict[str, Any]:
    """Scale every cost-driving parameter by ``factor``, respecting floors."""
    scaled: dict[str, Any] = dict(params or {})
    for key, value in list(scaled.items()):
        if key not in COST_PARAMS or isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        new_value = float(value) * factor
        floor = _MINIMUMS.get(key, 1)
        new_value = max(floor, new_value)
        scaled[key] = round(new_value) if isinstance(value, int) else round(new_value, 6)
    return scaled


def suggest_adjustment(
    estimate: Estimate,
    budget_seconds: float,
    params: dict[str, Any] | None = None,
    *,
    safety: float = 0.9,
) -> dict[str, Any]:
    """Work out how much to shrink the plan so it fits the budget.

    Returns a dict with ``factor`` (the multiplicative shrink), ``params``
    (scaled values, when the plan exposes numeric knobs) and ``notes``.
    """
    target = budget_seconds * safety
    notes: list[str] = []
    if not estimate.is_known:
        return {
            "factor": 1.0,
            "params": dict(params or {}),
            "notes": ["no estimate; nothing to scale"],
        }
    assert estimate.estimated_full_seconds is not None
    factor = target / estimate.estimated_full_seconds
    factor = max(0.05, min(1.0, factor))
    merged = dict(params or {})
    if estimate.params and not merged:
        merged = dict(estimate.params)
    scaled = scale_params(merged, factor)
    changed = {
        key: (merged.get(key), scaled.get(key))
        for key in scaled
        if key in COST_PARAMS and scaled.get(key) != merged.get(key)
    }
    if changed:
        notes.append(
            "scaled " + ", ".join(f"{k}: {old} -> {new}" for k, (old, new) in changed.items())
        )
    else:
        notes.append(
            "no numeric cost knobs were exposed by the plan; the strategy itself must change"
        )
    if factor < 1.0:
        notes.append(
            f"target {format_duration(target)} of the {format_duration(budget_seconds)} budget"
        )
    return {"factor": factor, "params": scaled, "notes": notes}
