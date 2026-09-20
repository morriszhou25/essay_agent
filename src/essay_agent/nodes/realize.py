"""Stage 3 - feasibility: can these cards be reproduced in this environment?"""

from __future__ import annotations

from collections.abc import Callable

from essay_agent.errors import LLMError
from essay_agent.nodes.base import (
    Deps,
    NodeReturn,
    abort_state,
    cards_of,
    collect_probe_targets,
    connectivity_report,
    environment_report,
    probe_report,
)
from essay_agent.prompts import realize as prompts
from essay_agent.schemas.dialogue import RealizationReport
from essay_agent.state import RunState


def make_realize_node(deps: Deps) -> Callable[[RunState], NodeReturn]:
    """Probe the requirements, let the model judge, and stop on blockers."""

    def realize(state: RunState) -> NodeReturn:
        ui = deps.ui
        workspace = deps.workspace(state)
        cards = cards_of(state)
        paper_title = (state.get("paper") or {}).get("title", "unknown paper")
        ui.step("stage 3/5 - feasibility check", f"{len(cards)} card(s)")

        targets = collect_probe_targets(cards)
        if targets:
            ui.dim(f"probing {len(targets)} requirement(s): " + ", ".join(targets[:6]))
        probe_text, results = probe_report(deps, targets)
        workspace.write_json("cards/probes.json", [result.to_dict() for result in results])
        for result in results:
            (ui.success if result.reachable else ui.warn)(result.describe())

        connectivity = connectivity_report(deps)
        workspace.write_json("logs/connectivity.json", connectivity.to_dict())
        environment = environment_report(deps, connectivity=connectivity)
        workspace.write_text("logs/environment.txt", environment)
        ui.dim(connectivity.lines()[0])
        if (hint := connectivity.hint()) is not None:
            ui.warn(hint)
        try:
            report: RealizationReport = deps.llm.json(
                prompts.SYSTEM,
                prompts.user_message(
                    paper_title=paper_title,
                    cards=cards,
                    probe_report=probe_text,
                    environment=environment,
                    time_budget_seconds=deps.settings.runtime.time_budget_seconds,
                    allow_synthetic_data=deps.settings.runtime.allow_synthetic_data,
                ),
                RealizationReport,
                label="feasibility",
            )
        except LLMError as exc:
            ui.warn(f"feasibility review failed ({exc}); continuing on the probe evidence alone")
            return {
                "feasibility": {
                    "checks": [],
                    "blockers": [],
                    "proceed": True,
                    "summary": f"verifier unavailable: {exc}",
                },
                "blockers": [],
                "connectivity": connectivity.to_dict(),
                "status": "running",
            }

        workspace.write_json("cards/feasibility.json", report.model_dump(mode="json"))
        blockers = list(report.blockers)
        for check in report.checks:
            if check.severity == "blocker" and not check.feasible:
                blocker = f"[{check.card_id}] " + "; ".join(
                    check.findings or [check.requirement or "blocked"]
                )
                if blocker not in blockers:
                    blockers.append(blocker)

        if not report.proceed or blockers:
            ui.error("the feasibility check found blocking problems:")
            for blocker in blockers:
                ui.error(f"  - {blocker}")
            ui.notice(
                "reproduction blocked",
                "The agent will not fabricate data or results to work around this.\n"
                "Continuing anyway only makes sense if you know the artifact is reachable "
                "by another route (mirror, local copy, credentials).",
            )
            if not deps.ui.confirm("continue anyway with the reduced scope?", default=False):
                return abort_state(
                    state,
                    "blocked by the feasibility check: " + "; ".join(blockers),
                    extra={"feasibility": report.model_dump(mode="json"), "blockers": blockers},
                )
            ui.warn("continuing against the feasibility verdict at the user's request")
        else:
            ui.success(f"feasible: {report.summary[:200]}")

        return {
            "feasibility": report.model_dump(mode="json"),
            "blockers": blockers,
            "connectivity": connectivity.to_dict(),
            "status": "running",
        }

    return realize
