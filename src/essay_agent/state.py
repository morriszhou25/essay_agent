"""The LangGraph state object.

Everything is JSON-friendly: pydantic models are stored as plain dicts so the
graph can be checkpointed and logged without a custom serialiser.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

FetchStatus = Literal["ok", "needs_choice", "not_found", "error", "aborted"]
RunStatus = Literal["running", "completed", "aborted", "failed", "cancelled"]


class RunState(TypedDict, total=False):
    """State carried between nodes."""

    # ---------------------------------------------------------------- identity
    run_id: str
    run_dir: str
    slug: str
    query: str
    started_at: str
    status: RunStatus
    error: str
    messages: list[str]

    # ------------------------------------------------------------ stage 1 fetch
    search_args: dict[str, Any]
    candidates: list[dict[str, Any]]
    search_report: dict[str, Any]
    match_decision: dict[str, Any]
    chosen_candidate: dict[str, Any] | None
    paper: dict[str, Any] | None
    paper_text: str
    fetch_status: FetchStatus
    fetch_message: str

    # ------------------------------------------------------------- stage 2 plan
    sections: list[dict[str, Any]]
    # One row per eligible section: key, name, card ids, status (carded /
    # no_testable_claim / failed / unanswered) and a detail note.
    coverage: list[dict[str, Any]]
    cards: list[dict[str, Any]]
    plan_round: int
    plan_review: dict[str, Any]
    plan_issues: list[str]
    plan_carryover: str

    # ---------------------------------------------------------- stage 3 realize
    feasibility: dict[str, Any]
    blockers: list[str]
    connectivity: dict[str, Any]

    # ---------------------------------------------------------- stage 4 execute
    # Which cards stage 4 was allowed to execute, and which were excluded (see CardScope).
    scope: dict[str, Any]
    exec_plan: dict[str, Any]
    exec_code_path: str
    preflight: dict[str, Any]
    adjust_round: int
    exec_round: int
    exec_result: dict[str, Any]
    verdict: dict[str, Any]
    verdict_history: list[dict[str, Any]]

    # -------------------------------------------------------- stage 5 interpret
    report_path: str

    # ------------------------------------------------------------- publication
    publish: dict[str, Any]
    lesson_report: list[str]
