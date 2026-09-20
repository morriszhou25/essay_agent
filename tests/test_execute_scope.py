"""Stage 4 only executes the cards stage 3 says are reproducible.

Measured on run-20260920-042507 (``A review of sparse expert models in deep learning``): stage 3
classified 6 of 24 cards as feasible and 18 as infeasible, yet the script writer was asked to
cover all 24 inside a ~250-line script and the execution verifier graded all 24 - its round-3
problem list demanded work on cards the feasibility check had just ruled out, and all three
rounds were spent on a target that could not be reached.

``card_scope`` makes that verdict the single input to what stage 4 may test, and reports the rest
as out of scope instead of grading them.
"""

from __future__ import annotations

import re
from typing import Any

from essay_agent.console import SilentUI
from essay_agent.nodes.base import EXEC_CARD_LIMIT, CardScope, card_scope
from essay_agent.nodes.execute import make_execute_plan_node
from essay_agent.nodes.interpret import make_interpret_node
from essay_agent.nodes.verify_execute import make_verify_execute_node
from essay_agent.schemas.dialogue import ExecPlan, ExecuteVerdict, ScopeCheck
from tests.fakes import FakeLLM
from tests.pipeline import build_deps, card_payload, plan_payload, script_with_estimate

FEASIBLE = ("c01", "c03", "c04")
INFEASIBLE = ("c02", "c05")
OUT_OF_SCOPE_HEADER = "## Cards out of scope"
# The out-of-scope block also starts with "## Cards", so the plan/code headers must be specific.
PLAN_CARDS_HEADER = "## Cards to reproduce"
CARDS_HEADER = "## Cards ("


def cards_for(ids: tuple[str, ...] = ("c01", "c02", "c03", "c04", "c05")) -> list[dict[str, Any]]:
    payloads = []
    for card_id in ids:
        payload = card_payload(card_id)
        payload["identity"]["section"] = f"{card_id[1:]} Section"
        payloads.append(payload)
    return payloads


def scope_check(card_id: str, feasible: bool) -> dict[str, Any]:
    return {
        "card_id": card_id,
        "feasible": feasible,
        "severity": "minor" if feasible else "blocker",
        "findings": ["dataset reachable" if feasible else "no dataset or metric in the section"],
        "dataset": "sklearn:iris" if feasible else None,
        "dataset_available": feasible if feasible else None,
    }


def feasibility(
    feasible: tuple[str, ...] = FEASIBLE, infeasible: tuple[str, ...] = INFEASIBLE
) -> dict[str, Any]:
    return {
        "checks": [scope_check(card_id, True) for card_id in feasible]
        + [scope_check(card_id, False) for card_id in infeasible],
        "blockers": [f"[{card_id}] not reproducible here" for card_id in infeasible],
        "proceed": True,
        "summary": f"{len(feasible)} of {len(feasible) + len(infeasible)} cards are reproducible",
    }


def state_for(
    cards: list[dict[str, Any]], feasibility_report: dict[str, Any] | None
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "run_id": "run-scope-test",
        "slug": "scope",
        "query": "sparse expert models",
        "paper": {"id": "p1", "title": "A review of sparse expert models"},
        "cards": cards,
        "coverage": [],
        "plan_round": 0,
        "exec_round": 1,
    }
    if feasibility_report is not None:
        state["feasibility"] = feasibility_report
    return state


def ids_in(text: str) -> list[str]:
    """Card ids in the order they appear, deduplicated."""
    return list(dict.fromkeys(re.findall(r"\bc\d{2}\b", text)))


def prompt_section(prompt: str, header: str) -> str:
    """The body of one ``## `` section of a prompt, up to the next heading."""
    lines = prompt.splitlines()
    start = next((index for index, line in enumerate(lines) if line.startswith(header)), None)
    if start is None:
        return ""
    body = []
    for line in lines[start:]:
        if body and line.startswith("## "):
            break
        body.append(line)
    return "\n".join(body)


# ------------------------------------------------------------------ card_scope
def test_scope_keeps_the_feasible_cards_and_names_the_blocked_ones() -> None:
    scope = card_scope(state_for(cards_for(), feasibility()))

    assert [card.card_id for card in scope.cards] == list(FEASIBLE)
    assert scope.excluded_blocked == list(INFEASIBLE)
    assert scope.excluded == list(INFEASIBLE)
    assert scope.excluded_budget == []
    assert scope.relaxed is False


def test_a_card_stage_three_never_classified_stays_in_scope() -> None:
    report = feasibility(feasible=("c01",), infeasible=("c02",))

    scope = card_scope(state_for(cards_for(), report))

    assert [card.card_id for card in scope.cards] == ["c01", "c03", "c04", "c05"]


def test_the_scope_is_capped_and_the_overflow_is_reported() -> None:
    ids = tuple(f"c{position:02d}" for position in range(1, 25))
    report = feasibility(feasible=ids, infeasible=())

    scope = card_scope(state_for(cards_for(ids), report))

    assert len(scope.cards) == EXEC_CARD_LIMIT
    assert scope.excluded_blocked == []
    assert len(scope.excluded_budget) == len(ids) - EXEC_CARD_LIMIT
    assert str(len(ids) - EXEC_CARD_LIMIT) in scope.describe()


def test_a_scope_that_would_be_empty_is_relaxed_instead_of_left_dead() -> None:
    report = feasibility(
        feasible=(), infeasible=tuple(f"c{position:02d}" for position in range(1, 25))
    )

    scope = card_scope(state_for(cards_for(tuple(f"c{p:02d}" for p in range(1, 25))), report))

    assert scope.relaxed is True
    assert scope.cards and scope.excluded_blocked == []
    assert "best effort" in scope.describe()


def test_the_scope_reads_scope_check_models_too() -> None:
    report = {
        "checks": [
            ScopeCheck(card_id="c01", feasible=True),
            ScopeCheck(card_id="c02", feasible=False, severity="blocker"),
        ],
        "blockers": [],
        "proceed": True,
        "summary": "mixed",
    }

    scope = card_scope(state_for(cards_for(), report))

    assert [card.card_id for card in scope.cards] == ["c01", "c03", "c04", "c05"]
    assert scope.excluded_blocked == ["c02"]


def test_without_a_feasibility_report_nothing_is_excluded() -> None:
    scope = card_scope(state_for(cards_for(), None))

    assert [card.card_id for card in scope.cards] == ["c01", "c02", "c03", "c04", "c05"]
    assert scope.excluded == []
    assert scope.context() == ""


def test_the_context_section_names_every_excluded_card() -> None:
    scope = card_scope(state_for(cards_for(), feasibility()))

    context = scope.context()

    assert context.startswith(OUT_OF_SCOPE_HEADER)
    assert set(ids_in(context)) == set(INFEASIBLE)
    assert "untested" in context
    assert scope.summary() == {
        "in_scope": list(FEASIBLE),
        "excluded_blocked": list(INFEASIBLE),
        "excluded_budget": [],
        "relaxed": False,
    }


def test_an_empty_context_is_the_empty_string() -> None:
    scope = CardScope(cards=[], excluded_blocked=[], excluded_budget=[])

    assert scope.context() == ""


# --------------------------------------------------------------- execute node
def run_execute_plan(
    settings, cards: list[dict[str, Any]], report: dict[str, Any] | None
) -> tuple[dict[str, Any], SilentUI, dict[str, str]]:
    captured: dict[str, str] = {}

    def capture_plan(_system: str, user: str) -> dict[str, Any]:
        captured["plan"] = user
        return plan_payload()

    def capture_code(_system: str, user: str) -> str:
        captured["code"] = user
        return script_with_estimate(0.2)

    llm = FakeLLM({ExecPlan: capture_plan}, texts={"repro_script_code1": capture_code})
    ui = SilentUI()
    deps = build_deps(settings, llm, ui=ui)
    state = state_for(cards, report)
    state.update(make_execute_plan_node(deps)(state))
    return state, ui, captured


def test_the_plan_prompt_only_carries_the_cards_in_scope(settings) -> None:
    _state, _ui, captured = run_execute_plan(settings, cards_for(), feasibility())

    section = prompt_section(captured["plan"], PLAN_CARDS_HEADER)

    assert ids_in(section) == list(FEASIBLE)
    assert ids_in(prompt_section(captured["code"], PLAN_CARDS_HEADER)) == list(FEASIBLE)


def test_the_plan_prompt_hands_over_the_excluded_cards_as_context(settings) -> None:
    _state, _ui, captured = run_execute_plan(settings, cards_for(), feasibility())

    context = prompt_section(captured["plan"], OUT_OF_SCOPE_HEADER)

    assert set(ids_in(context)) == set(INFEASIBLE)
    assert "out of scope" in context


def test_the_run_records_and_announces_what_was_excluded(settings) -> None:
    state, ui, _captured = run_execute_plan(settings, cards_for(), feasibility())

    assert state["scope"] == {
        "in_scope": list(FEASIBLE),
        "excluded_blocked": list(INFEASIBLE),
        "excluded_budget": [],
        "relaxed": False,
    }
    assert any(
        "excluded by the feasibility check" in message
        for kind, message in ui.events
        if kind == "dim"
    )


# -------------------------------------------------------- verify_execute node
def run_verify_execute(
    settings, cards: list[dict[str, Any]], report: dict[str, Any] | None
) -> tuple[dict[str, Any], SilentUI, dict[str, str]]:
    captured: dict[str, str] = {}

    def capture_verdict(_system: str, user: str) -> dict[str, Any]:
        captured["verify"] = user
        graded = ids_in(prompt_section(user, CARDS_HEADER))
        return {
            "verdict": "pending",
            "rationale": f"graded {len(graded)} card(s)",
            "problems": [f"{card_id}: not measured" for card_id in graded],
            "evidence": [],
            "per_card": dict.fromkeys(graded, "untested"),
        }

    llm = FakeLLM({ExecuteVerdict: capture_verdict})
    ui = SilentUI()
    deps = build_deps(settings, llm, ui=ui)
    state = state_for(cards, report)
    state["exec_plan"] = plan_payload()
    state["exec_result"] = {
        "ok": True,
        "exit_code": 0,
        "duration": 2.0,
        "timed_out": False,
        "metrics": {"accuracy": 0.912},
        "figures": ["fig1.png"],
        "stdout_tail": "",
        "error": None,
    }
    state.update(make_verify_execute_node(deps)(state))
    return state, ui, captured


def test_the_verifier_only_grades_the_cards_in_scope(settings) -> None:
    state, _ui, captured = run_verify_execute(settings, cards_for(), feasibility())

    graded = ids_in(prompt_section(captured["verify"], CARDS_HEADER))

    assert graded == list(FEASIBLE)
    assert set(state["verdict"]["per_card"]) == set(FEASIBLE)


def test_the_verifier_is_told_the_excluded_cards_are_not_failures(settings) -> None:
    state, _ui, captured = run_verify_execute(settings, cards_for(), feasibility())

    context = prompt_section(captured["verify"], OUT_OF_SCOPE_HEADER)

    assert set(ids_in(context)) == set(INFEASIBLE)
    assert "do not" in context.lower() or "not" in context.lower()
    assert not (set(ids_in(" ".join(state["verdict"]["problems"]))) & set(INFEASIBLE))


# ---------------------------------------------------------------- report node
def test_the_report_still_publishes_every_card(settings) -> None:
    captured: dict[str, str] = {}

    def capture_report(_system: str, user: str) -> str:
        captured["report"] = user
        return "# Report\n\nbody\n"

    llm = FakeLLM(texts={"report": capture_report})
    ui = SilentUI()
    deps = build_deps(settings, llm, ui=ui)
    state = state_for(cards_for(), feasibility())
    state["exec_plan"] = plan_payload()
    state["exec_result"] = {"metrics": {"accuracy": 0.912}, "figures": [], "ok": True}

    workspace = deps.workspace(state)
    state.update(make_interpret_node(deps)(state))

    prompt = captured["report"]
    assert ids_in(prompt_section(prompt, CARDS_HEADER)) == list(FEASIBLE)
    assert set(ids_in(prompt_section(prompt, OUT_OF_SCOPE_HEADER))) == set(INFEASIBLE)
    published = (workspace.dir / "result" / "cards.md").read_text(encoding="utf-8")
    assert set(ids_in(published)) == {"c01", "c02", "c03", "c04", "c05"}
