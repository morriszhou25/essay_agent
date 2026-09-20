"""Stage-4 card scope - only the cards stage 3 allows reach the executor and the verifier.

Measured on run-20260920-042507 (``A review of sparse expert models in deep learning``): stage 3
classified 6 of 24 cards as feasible and 18 as infeasible, but stages 4 and 4.1 ignored that
verdict. The script writer was asked to cover all 24 cards with a script of at most ~250 lines,
and the execution verifier graded all 24 - its round-3 problem list demanded work on
``c15/c16/c19/c21/c22/c23/c24``, i.e. on cards stage 3 had just ruled out. Three rounds were spent
chasing a target that could not be reached.

The shipped rule: a card reaches stage 4 when stage 3 did not rule it out (explicitly feasible, or
never classified), and at most ``EXEC_CARD_LIMIT`` cards reach it. The rest are excluded
explicitly, reported to the user, and handed to the verifier only as *context* - to be reported as
untested, never graded, never re-executed.

    python tests/experiment_execute_scope.py
"""

from __future__ import annotations

# ruff: noqa: E402 -- the package is imported after the src/ path shim below.
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(ROOT / "src"), str(ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from essay_agent.console import SilentUI
from essay_agent.nodes import base as base_nodes
from essay_agent.nodes.execute import make_execute_plan_node
from essay_agent.nodes.verify_execute import make_verify_execute_node
from essay_agent.schemas.dialogue import ExecPlan, ExecuteVerdict
from tests.conftest import make_settings
from tests.fakes import FakeLLM
from tests.pipeline import build_deps, card_payload, plan_payload, script_with_estimate

# The helper the shipped code must expose; None means the change has not landed yet.
card_scope = getattr(base_nodes, "card_scope", None)
EXEC_CARD_LIMIT = getattr(base_nodes, "EXEC_CARD_LIMIT", 10)

# Measured on the real run: cards.md 40,198 chars / 24 cards -> 1,674 chars per card, and
# deepseek-chat caps *output* (not input) at 8192 tokens.
CARD_CHARS = 1674
OUTPUT_TOKEN_CEILING = 8192
CHARS_PER_TOKEN = 4
CEILING_CHARS = OUTPUT_TOKEN_CEILING * CHARS_PER_TOKEN
# The real run's feasibility verdict, card for card.
FEASIBLE = ("c01", "c03", "c04", "c08", "c09", "c11")
INFEASIBLE = (
    "c02",
    "c05",
    "c06",
    "c07",
    "c10",
    "c12",
    "c13",
    "c14",
    "c15",
    "c16",
    "c17",
    "c18",
    "c19",
    "c20",
    "c21",
    "c22",
    "c23",
    "c24",
)
OUT_OF_SCOPE_HEADER = "## Cards out of scope"
PLAN_CARDS_HEADER = "## Cards to reproduce"
VERIFY_CARDS_HEADER = "## Cards ("

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ------------------------------------------------------------------- the fixture
def ids_of(count: int = 24) -> list[str]:
    return [f"c{position:02d}" for position in range(1, count + 1)]


def cards_fixture(ids: list[str] | None = None) -> list[dict[str, Any]]:
    payloads = []
    for card_id in ids if ids is not None else ids_of():
        payload = card_payload(card_id)
        payload["identity"]["section"] = f"{card_id[1:]} Section"
        payloads.append(payload)
    return payloads


def feasibility_fixture(
    feasible: tuple[str, ...] = FEASIBLE, infeasible: tuple[str, ...] = INFEASIBLE
) -> dict[str, Any]:
    checks = [
        {
            "card_id": card_id,
            "feasible": True,
            "severity": "minor",
            "findings": ["dataset reachable"],
            "dataset": "sklearn:iris",
            "dataset_available": True,
        }
        for card_id in feasible
    ]
    checks += [
        {
            "card_id": card_id,
            "feasible": False,
            "severity": "blocker",
            "findings": ["the survey section reports no dataset, metric or number"],
            "dataset": None,
            "dataset_available": None,
        }
        for card_id in infeasible
    ]
    return {
        "checks": checks,
        "blockers": [f"[{card_id}] not reproducible here" for card_id in infeasible],
        "proceed": True,
        "summary": "6 of 24 cards are reproducible in this environment",
    }


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


def ids_in(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\bc\d{2}\b", text)))


# -------------------------------------------------------------- the intended rule
def prototype_scope(
    cards: list[dict[str, Any]], feasibility: dict[str, Any] | None, limit: int
) -> tuple[list[str], list[str], list[str], bool]:
    """The rule as specified, independent of the shipped implementation."""
    verdicts = {
        str(check.get("card_id")): bool(check.get("feasible"))
        for check in (feasibility or {}).get("checks") or []
    }
    selected = [card for card in cards if verdicts.get(card["card_id"]) is not False]
    relaxed = not selected and bool(cards)
    if relaxed:
        selected, blocked = list(cards), []
    else:
        blocked = [card["card_id"] for card in cards if verdicts.get(card["card_id"]) is False]
    return (
        [card["card_id"] for card in selected[:limit]],
        blocked,
        [card["card_id"] for card in selected[limit:]],
        relaxed,
    )


# ------------------------------------------------------------------ the real nodes
def base_state(cards: list[dict[str, Any]], feasibility: dict[str, Any] | None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "run_id": "run-experiment-scope",
        "slug": "scope",
        "query": "sparse expert models",
        "paper": {"id": "p1", "title": "A review of sparse expert models in deep learning"},
        "cards": cards,
        "coverage": [
            {
                "key": card["identity"]["section"],
                "section": card["identity"]["section"],
                "cards": [card["card_id"]],
                "status": "carded",
                "detail": "",
                "passes": 1,
            }
            for card in cards
        ],
        "plan_round": 0,
        "exec_round": 1,
    }
    if feasibility is not None:
        state["feasibility"] = feasibility
    return state


def run_execute_plan(
    cards: list[dict[str, Any]], feasibility: dict[str, Any] | None
) -> tuple[dict[str, Any], SilentUI, dict[str, str]]:
    captured: dict[str, str] = {}
    settings = make_settings(ROOT / ".essay_agent" / "experiments" / "execute_scope")

    def capture_plan(_system: str, user: str) -> dict[str, Any]:
        captured["plan"] = user
        return plan_payload()

    def capture_code(_system: str, user: str) -> str:
        captured["code"] = user
        return script_with_estimate(0.2)

    llm = FakeLLM({ExecPlan: capture_plan}, texts={"repro_script_code1": capture_code})
    ui = SilentUI()
    deps = build_deps(settings, llm, ui=ui)
    state = base_state(cards, feasibility)
    returned = make_execute_plan_node(deps)(state)
    state.update(returned)
    captured["code_used"] = (returned.get("exec_code_path") or "").replace("\\\\", "/")
    return state, ui, captured


def run_verify_execute(
    cards: list[dict[str, Any]], feasibility: dict[str, Any] | None
) -> tuple[dict[str, Any], SilentUI, dict[str, str]]:
    captured: dict[str, str] = {}
    settings = make_settings(ROOT / ".essay_agent" / "experiments" / "execute_scope")

    def capture_verdict(_system: str, user: str) -> dict[str, Any]:
        captured["verify"] = user
        graded = ids_in(prompt_section(user, VERIFY_CARDS_HEADER))
        return {
            "verdict": "pending",
            "rationale": f"graded {len(graded)} card(s)",
            "problems": [f"{card_id}: untested" for card_id in graded],
            "evidence": [],
            "per_card": dict.fromkeys(graded, "untested"),
        }

    llm = FakeLLM({ExecuteVerdict: capture_verdict})
    ui = SilentUI()
    deps = build_deps(settings, llm, ui=ui)
    state = base_state(cards, feasibility)
    state["exec_plan"] = plan_payload()
    state["exec_result"] = {
        "ok": True,
        "exit_code": 0,
        "duration": 2.09,
        "timed_out": False,
        "metrics": {"c01_param_growth_factor": 32.0, "c01_criterion_ok": 1.0},
        "figures": ["fig_c01_param_vs_flops_by_experts.png"],
        "stdout_tail": '{"event": "progress", "step": 27, "total": 27}',
        "error": None,
    }
    returned = make_verify_execute_node(deps)(state)
    state.update(returned)
    return state, ui, captured


# ------------------------------------------------------------------------ parts
def part1_the_measured_problem() -> None:
    print("\n[1] the measured problem: stages 4 and 4.1 eat the whole card set")
    cards = cards_fixture()
    total = len(cards) * CARD_CHARS
    in_scope = len(FEASIBLE) * CARD_CHARS
    print(f"      24 cards: {total:,} chars of card markdown ({total / CEILING_CHARS:.0%} of the")
    print(
        f"      {OUTPUT_TOKEN_CEILING}-token output ceiling, and it is prompt text on every call)"
    )
    print(f"      the 6 feasible cards alone: {in_scope:,} chars ({in_scope / CEILING_CHARS:.0%})")
    check(
        "[1a] the full card set dwarfs the 6 cards the run can actually test",
        total > in_scope * 3,
        f"{total:,} vs {in_scope:,} chars",
    )
    check(
        "[1b] the real verifier demanded work on cards stage 3 had ruled out",
        len(INFEASIBLE) == 18 and "c19" in INFEASIBLE and "c15" in INFEASIBLE,
        "round-3 problems named c15/c16/c19/c21-c24",
    )


def part2_the_rule() -> None:
    print("\n[2] the rule: feasibility decides the scope")
    cards = cards_fixture()
    in_scope, blocked, over_budget, relaxed = prototype_scope(
        cards, feasibility_fixture(), EXEC_CARD_LIMIT
    )
    print(f"      in scope: {', '.join(in_scope)}")
    print(f"      blocked by stage 3: {len(blocked)} card(s)")
    print(f"      over the {EXEC_CARD_LIMIT}-card limit: {len(over_budget)}")
    check(
        "[2a] only the feasible cards are in scope",
        in_scope == list(FEASIBLE),
        f"{len(in_scope)} in scope",
    )
    check(
        "[2b] the infeasible cards are excluded, not silently dropped",
        blocked == list(INFEASIBLE),
        f"{len(blocked)} excluded",
    )
    check("[2c] nothing is over the limit here", over_budget == [] and not relaxed)
    check(
        "[2d] the shipped helper exists",
        card_scope is not None,
        "essay_agent.nodes.base.card_scope",
    )


def part3_execute_plan_scope() -> None:
    print("\n[3] the real execute_plan node with the run's 24 cards")
    cards = cards_fixture()
    state, ui, captured = run_execute_plan(cards, feasibility_fixture())
    plan_prompt = captured.get("plan", "")
    code_prompt = captured.get("code", "")
    in_scope_line = prompt_section(plan_prompt, PLAN_CARDS_HEADER)
    excluded = prompt_section(plan_prompt, OUT_OF_SCOPE_HEADER)
    print(f"      plan prompt: {len(plan_prompt):,} chars")
    print(f"      cards in the plan prompt: {len(ids_in(in_scope_line))}")
    print(f"      out-of-scope section: {len(ids_in(excluded))} card id(s)")
    check(
        "[3a] the plan prompt carries only the in-scope cards",
        ids_in(in_scope_line) == list(FEASIBLE),
        f"{ids_in(in_scope_line)}",
    )
    check(
        "[3b] the excluded cards are named as out of scope",
        set(ids_in(excluded)) == set(INFEASIBLE),
        f"{len(ids_in(excluded))} named",
    )
    check(
        "[3c] the exclusion says they must not be tested",
        "not" in excluded.lower()
        and ("untested" in excluded.lower() or "grade" in excluded.lower()),
        excluded.splitlines()[0] if excluded else "(missing)",
    )
    check(
        "[3d] the same scope reaches the code call",
        set(ids_in(prompt_section(code_prompt, PLAN_CARDS_HEADER))) == set(FEASIBLE),
        f"{len(ids_in(code_prompt))} card id(s) mentioned",
    )
    steps = [message for kind, message in ui.events if kind in ("step", "dim", "warn")]
    check(
        "[3e] the human sees the exclusion",
        any(str(len(INFEASIBLE)) in message or "18" in message for message in steps),
        " | ".join(steps[:3]),
    )
    check("[3f] the plan/code split still ran", bool(state.get("exec_code_path")))


def part4_verify_scope() -> None:
    print("\n[4] the real verify_execute node with the run's 24 cards")
    cards = cards_fixture()
    state, _ui, captured = run_verify_execute(cards, feasibility_fixture())
    prompt = captured.get("verify", "")
    graded = ids_in(prompt_section(prompt, VERIFY_CARDS_HEADER))
    excluded = prompt_section(prompt, OUT_OF_SCOPE_HEADER)
    per_card = state["verdict"].get("per_card") or {}
    print(f"      verifier prompt: {len(prompt):,} chars")
    print(f"      cards graded: {len(graded)} -> {graded}")
    print(f"      per_card entries: {len(per_card)}")
    check(
        "[4a] the verifier only grades the in-scope cards",
        graded == list(FEASIBLE),
        f"{len(graded)} graded",
    )
    check(
        "[4b] the excluded cards are handed over as context",
        set(ids_in(excluded)) == set(INFEASIBLE),
        f"{len(ids_in(excluded))} named",
    )
    check(
        "[4c] the verifier is told not to grade them",
        bool(excluded) and "out of scope" in prompt.lower(),
        excluded.splitlines()[0] if excluded else "(missing)",
    )
    check(
        "[4d] no problem is raised about a card stage 3 ruled out",
        not (set(ids_in(" ".join(state["verdict"]["problems"]))) & set(INFEASIBLE)),
        f"{len(state['verdict']['problems'])} problem(s)",
    )


def scope_of(
    cards: list[dict[str, Any]], feasibility: dict[str, Any] | None, limit: int | None = None
):
    """Call the shipped helper, or the prototype when the change has not landed."""
    state = base_state(cards, feasibility)
    if card_scope is None:
        in_scope, blocked, over_budget, relaxed = prototype_scope(
            cards, feasibility, limit or EXEC_CARD_LIMIT
        )
        return in_scope, blocked, over_budget, relaxed
    kwargs = {"limit": limit} if limit is not None else {}
    scope = card_scope(state, **kwargs)
    return (
        [card.card_id for card in scope.cards],
        list(scope.excluded_blocked),
        list(scope.excluded_budget),
        bool(scope.relaxed),
    )


def part5_edge_cases() -> None:
    print("\n[5] the limit and the fallbacks")
    all_feasible = feasibility_fixture(feasible=tuple(ids_of()), infeasible=())
    in_scope, blocked, over_budget, _relaxed = scope_of(cards_fixture(), all_feasible)
    print(f"      24 feasible cards -> {len(in_scope)} in scope, {len(over_budget)} over budget")
    check(
        "[5a] the scope is capped so the script can still fit the output budget",
        len(in_scope) == EXEC_CARD_LIMIT and len(over_budget) == 24 - EXEC_CARD_LIMIT,
        f"{len(in_scope)} + {len(over_budget)}",
    )
    check("[5b] a capped card is reported, not silently dropped", blocked == [])
    in_scope, blocked, over_budget, relaxed = scope_of(cards_fixture(), None)
    print(f"      no feasibility data -> {len(in_scope)} in scope, relaxed={relaxed}")
    check(
        "[5c] without a stage-3 verdict nothing is excluded on feasibility grounds",
        blocked == [] and not relaxed and len(in_scope) == EXEC_CARD_LIMIT,
        f"{len(in_scope)} in scope, {len(over_budget)} over budget",
    )
    dead = feasibility_fixture(feasible=(), infeasible=tuple(ids_of()))
    in_scope, blocked, _over, relaxed = scope_of(cards_fixture(), dead)
    print(f"      every card infeasible -> {len(in_scope)} in scope, relaxed={relaxed}")
    check(
        "[5d] a scope that would be empty degrades to a best effort instead of a dead end",
        bool(in_scope) and relaxed and blocked == [],
        f"{len(in_scope)} in scope",
    )


def main() -> int:
    print("stage-4 card scope: only feasible cards are executed and graded (offline, no key)")
    part1_the_measured_problem()
    part2_the_rule()
    part3_execute_plan_scope()
    part4_verify_scope()
    part5_edge_cases()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
