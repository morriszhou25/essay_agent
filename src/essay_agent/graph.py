"""LangGraph wiring for the five-stage pipeline.

    START -> fetch -> plan -> verify_plan ->(replan loop)-> realize
          -> execute_plan -> preflight ->(adjust loop)-> run
          -> verify_execute ->(re-execute loop)-> interpret -> finalize -> END

Every loop is bounded by ``runtime.max_*_rounds``; when a loop exhausts its
rounds the work is released and the open issues travel with the state instead of
blocking the run forever.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from essay_agent.nodes import (
    Deps,
    make_adjust_node,
    make_execute_plan_node,
    make_fetch_node,
    make_finalize_node,
    make_interpret_node,
    make_plan_node,
    make_preflight_node,
    make_realize_node,
    make_reexecute_node,
    make_revise_plan_node,
    make_run_node,
    make_verify_execute_node,
    make_verify_plan_node,
)
from essay_agent.nodes.execute import preflight_needs_adjustment
from essay_agent.nodes.plan import plan_review_ok
from essay_agent.nodes.verify_execute import verdict_allows_report
from essay_agent.state import RunState

DEFAULT_RECURSION_LIMIT = 120

RouteMap = dict[str, str]


def route_fetch(state: RunState) -> str:
    return "plan" if state.get("fetch_status") == "ok" else "abort"


def route_after(next_node: str) -> Callable[[RunState], str]:
    """Continue only when the previous node left the run healthy."""

    def route(state: RunState) -> str:
        if state.get("status") in {"failed", "aborted"}:
            return "abort"
        return next_node

    return route


def route_verify_plan(state: RunState, max_rounds: int = 3) -> str:
    if plan_review_ok(state):
        return "realize"
    if int(state.get("plan_round") or 0) >= max_rounds:
        return "realize"
    return "revise_plan"


def route_realize(state: RunState) -> str:
    return "execute_plan" if state.get("status") == "running" else "abort"


def route_preflight(state: RunState) -> str:
    return "adjust" if preflight_needs_adjustment(state) else "run_full"


def make_verdict_router(max_rounds: int) -> Callable[[RunState], str]:
    def route_verdict(state: RunState) -> str:
        return "interpret" if verdict_allows_report(state, max_rounds) else "reexecute"

    return route_verdict


def build_graph(deps: Deps, *, checkpointer: Any = None):
    """Compile the agent graph. ``deps`` is closed over by every node."""
    runtime = deps.settings.runtime
    builder = StateGraph(RunState)

    builder.add_node("fetch", make_fetch_node(deps))
    builder.add_node("plan", make_plan_node(deps))
    builder.add_node("verify_plan", make_verify_plan_node(deps))
    builder.add_node("revise_plan", make_revise_plan_node(deps))
    builder.add_node("realize", make_realize_node(deps))
    builder.add_node("execute_plan", make_execute_plan_node(deps))
    builder.add_node("preflight", make_preflight_node(deps))
    builder.add_node("adjust", make_adjust_node(deps))
    builder.add_node("run_full", make_run_node(deps))
    builder.add_node("verify_execute", make_verify_execute_node(deps))
    builder.add_node("reexecute", make_reexecute_node(deps))
    builder.add_node("interpret", make_interpret_node(deps))
    builder.add_node("finalize", make_finalize_node(deps))
    builder.add_node("abort", _abort_node)

    builder.add_edge(START, "fetch")
    builder.add_conditional_edges("fetch", route_fetch, {"plan": "plan", "abort": "abort"})
    builder.add_conditional_edges(
        "plan", route_after("verify_plan"), {"verify_plan": "verify_plan", "abort": "abort"}
    )
    builder.add_conditional_edges(
        "verify_plan",
        lambda state: route_verify_plan(state, runtime.max_plan_rounds),
        {"revise_plan": "revise_plan", "realize": "realize"},
    )
    builder.add_edge("revise_plan", "verify_plan")
    builder.add_conditional_edges(
        "realize", route_realize, {"execute_plan": "execute_plan", "abort": "abort"}
    )
    builder.add_conditional_edges(
        "execute_plan", route_after("preflight"), {"preflight": "preflight", "abort": "abort"}
    )
    builder.add_conditional_edges(
        "preflight", route_preflight, {"adjust": "adjust", "run_full": "run_full"}
    )
    builder.add_edge("adjust", "preflight")
    builder.add_edge("run_full", "verify_execute")
    builder.add_conditional_edges(
        "verify_execute",
        make_verdict_router(runtime.max_execute_rounds),
        {"reexecute": "reexecute", "interpret": "interpret"},
    )
    builder.add_conditional_edges(
        "reexecute", route_after("preflight"), {"preflight": "preflight", "abort": "abort"}
    )
    builder.add_edge("interpret", "finalize")
    builder.add_edge("finalize", END)
    builder.add_edge("abort", END)

    return builder.compile(checkpointer=checkpointer)


def _abort_node(state: RunState) -> dict[str, Any]:
    """Terminal node for aborted/failed runs - keeps whatever the state recorded."""
    status = state.get("status") or "aborted"
    return {"status": "failed" if status == "failed" else "aborted"}
