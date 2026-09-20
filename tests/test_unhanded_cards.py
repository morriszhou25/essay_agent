"""Cards the run never handed in are not graded.

The execution verifier used to grade every card it was given, so an in-scope card that the plan
never covered produced "c09 was never measured" as a *problem* - a re-execution round for work the
run deliberately did not attempt. ``unhanded_cards`` derives that set (in scope, minus the plan's
``cards_covered``) and the prompt marks it: `untested`, never a problem, never a failure.
"""

from __future__ import annotations

import re
from typing import Any

from essay_agent.console import SilentUI
from essay_agent.nodes.base import (
    UNHANDED_HEADER,
    cards_of,
    covered_card_ids,
    unhanded_cards,
    unhanded_context,
)
from essay_agent.nodes.verify_execute import make_verify_execute_node
from essay_agent.schemas.dialogue import ExecuteVerdict
from tests.fakes import FakeLLM
from tests.pipeline import build_deps, card_payload, plan_payload

ALL_CARDS = ("c01", "c02", "c03", "c04")
CARDS_HEADER = "## Cards ("
OUT_OF_SCOPE_HEADER = "## Cards out of scope"


def cards_fixture(ids: tuple[str, ...] = ALL_CARDS) -> list[dict[str, Any]]:
    payloads = []
    for card_id in ids:
        payload = card_payload(card_id)
        payload["identity"]["section"] = f"{card_id[1:]} Section"
        payloads.append(payload)
    return payloads


def feasibility(ids: tuple[str, ...] = ALL_CARDS, blocked: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "checks": [
            {"card_id": card_id, "feasible": True, "severity": "minor", "findings": []}
            for card_id in ids
        ]
        + [
            {"card_id": card_id, "feasible": False, "severity": "blocker", "findings": []}
            for card_id in blocked
        ],
        "blockers": [],
        "proceed": True,
        "summary": "fixture",
    }


def ids_in(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\bc\d{2}\b", text)))


def prompt_section(prompt: str, header: str) -> str:
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


def run_verify_execute(
    settings,
    cards: list[dict[str, Any]],
    plan: dict[str, Any],
    feasibility_report: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    captured: dict[str, str] = {}

    def capture(_system: str, user: str) -> dict[str, Any]:
        captured["verify"] = user
        graded = ids_in(prompt_section(user, CARDS_HEADER))
        return {
            "verdict": "pending",
            "rationale": "graded what was handed in",
            "problems": [f"{card_id}: no metric" for card_id in graded],
            "evidence": [],
            "per_card": dict.fromkeys(graded, "untested"),
        }

    llm = FakeLLM({ExecuteVerdict: capture})
    deps = build_deps(settings, llm, ui=SilentUI())
    state: dict[str, Any] = {
        "run_id": "run-unhanded",
        "slug": "unhanded",
        "query": "query",
        "paper": {"id": "p1", "title": "Fixture"},
        "cards": cards,
        "coverage": [],
        "plan_round": 0,
        "exec_round": 1,
        "exec_plan": plan,
        "exec_result": {
            "ok": True,
            "exit_code": 0,
            "duration": 1.0,
            "timed_out": False,
            "metrics": {"accuracy": 0.9},
            "figures": [],
            "stdout_tail": "",
            "error": None,
        },
    }
    if feasibility_report is not None:
        state["feasibility"] = feasibility_report
    state.update(make_verify_execute_node(deps)(state))
    return state, captured


# ------------------------------------------------------------------ the helper
def card_models(ids: tuple[str, ...] = ALL_CARDS) -> list[Any]:
    return cards_of({"cards": cards_fixture(ids)})


def test_unhanded_is_the_scope_minus_what_the_plan_covers() -> None:
    state = {"exec_plan": plan_payload(cards_covered=["c01", "c02"])}

    assert covered_card_ids(state) == ["c01", "c02"]
    assert unhanded_cards(state, card_models()) == ["c03", "c04"]


def test_a_plan_that_states_no_coverage_never_ungrades_anything() -> None:
    cards = card_models()

    assert unhanded_cards({"exec_plan": {"cards_covered": []}}, cards) == []
    assert unhanded_cards({"exec_plan": {}}, cards) == []
    assert unhanded_cards({}, cards) == []


def test_a_plan_covering_everything_leaves_nothing_unhanded() -> None:
    cards = card_models()

    assert unhanded_cards({"exec_plan": plan_payload(cards_covered=list(ALL_CARDS))}, cards) == []


def test_the_context_names_the_cards_and_forbids_grading_them() -> None:
    context = unhanded_context(["c03", "c04"])

    assert context.startswith(UNHANDED_HEADER)
    assert ids_in(context) == ["c03", "c04"]
    assert "untested" in context
    assert "problems" in context
    assert "do NOT grade" in context


def test_an_empty_unhanded_set_renders_nothing() -> None:
    assert unhanded_context([]) == ""


# ------------------------------------------------------------------ the node
def test_the_verifier_prompt_marks_the_unhanded_cards(settings) -> None:
    cards = cards_fixture()
    _state, captured = run_verify_execute(
        settings, cards, plan_payload(cards_covered=["c01", "c02"]), feasibility()
    )

    prompt = captured["verify"]

    assert ids_in(prompt_section(prompt, UNHANDED_HEADER)) == ["c03", "c04"]


def test_the_verdict_records_what_was_never_handed_in(settings) -> None:
    cards = cards_fixture()
    state, _captured = run_verify_execute(
        settings, cards, plan_payload(cards_covered=["c01", "c02"]), feasibility()
    )

    assert state["verdict"]["unhanded"] == ["c03", "c04"]


def test_a_fully_covering_plan_carries_no_unhanded_section(settings) -> None:
    cards = cards_fixture()
    state, captured = run_verify_execute(
        settings, cards, plan_payload(cards_covered=list(ALL_CARDS)), feasibility()
    )

    assert prompt_section(captured["verify"], UNHANDED_HEADER) == ""
    assert state["verdict"]["unhanded"] == []


def test_the_two_kinds_of_exclusion_do_not_overlap(settings) -> None:
    cards = cards_fixture(("c01", "c02", "c03", "c04", "c05"))
    state, captured = run_verify_execute(
        settings,
        cards,
        plan_payload(cards_covered=["c01", "c02"]),
        feasibility(ids=("c01", "c02", "c03", "c04"), blocked=("c05",)),
    )

    prompt = captured["verify"]
    out_of_scope = ids_in(prompt_section(prompt, OUT_OF_SCOPE_HEADER))
    unhanded = ids_in(prompt_section(prompt, UNHANDED_HEADER))

    # c05 was ruled out by stage 3, so it is not in scope and cannot be "unhanded" as well.
    assert unhanded == ["c03", "c04"]
    assert out_of_scope == ["c05"]
    assert not set(unhanded) & set(out_of_scope)
    assert state["verdict"]["unhanded"] == ["c03", "c04"]
