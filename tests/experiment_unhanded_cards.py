"""Cards the run never handed in must not be graded ("不交的不评").

The execution verifier is asked to grade every card it is given. But a card that the plan does
not cover produces no evidence at all - the run never handed it in - and the verifier then writes
"c09 was never measured" as a *problem*, which sends the pipeline into a re-execution round for
something the run deliberately did not attempt.

The shipped rule: the node derives the set of cards with no submitted evidence (in scope, but not
in the plan's `cards_covered`), and the prompt marks it explicitly - those cards are `untested`,
never problems, never failures. A plan that does not state its coverage changes nothing, so this
can only ever shrink the *graded* set, never the scope.

    python tests/experiment_unhanded_cards.py
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
from essay_agent.nodes.verify_execute import make_verify_execute_node
from essay_agent.schemas.dialogue import ExecuteVerdict
from tests.conftest import make_settings
from tests.fakes import FakeLLM
from tests.pipeline import build_deps, card_payload, plan_payload

# The helper the shipped code must expose; None means the change has not landed yet.
unhanded_cards = getattr(base_nodes, "unhanded_cards", None)
UNHANDED_HEADER = "## Cards with no submitted evidence"

# The run's in-scope cards (stage 3 judged these 6 reproducible) ...
IN_SCOPE = ("c01", "c03", "c04", "c08", "c09", "c11")
# ... and a plan that only reaches four of them, which is what "not handed in" means.
COVERED = ("c01", "c03", "c04", "c08")
UNHANDED = ("c09", "c11")
CARDS_HEADER = "## Cards ("

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ------------------------------------------------------------------- fixture
def cards_fixture(ids: tuple[str, ...] = IN_SCOPE) -> list[dict[str, Any]]:
    payloads = []
    for card_id in ids:
        payload = card_payload(card_id)
        payload["identity"]["section"] = f"{card_id[1:]} Section"
        payloads.append(payload)
    return payloads


def feasibility_fixture() -> dict[str, Any]:
    return {
        "checks": [
            {"card_id": card_id, "feasible": True, "severity": "minor", "findings": []}
            for card_id in IN_SCOPE
        ],
        "blockers": [],
        "proceed": True,
        "summary": "all six cards are reproducible here",
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


def prototype_unhanded(cards: list[dict[str, Any]], plan: dict[str, Any]) -> list[str]:
    """The rule: in scope, minus what the plan promises to cover."""
    covered = [str(card_id) for card_id in plan.get("cards_covered") or []]
    if not covered:
        return []
    return [card["card_id"] for card in cards if card["card_id"] not in set(covered)]


# ------------------------------------------------------------------ the real node
def strict_verifier(_system: str, user: str) -> dict[str, Any]:
    """A model that grades exactly the cards the prompt hands it as gradeable.

    This is the behaviour the prompt must control: a strict verifier used to see all six cards,
    and turning "no evidence" into a problem is what triggered the wasted rounds.
    """
    presented = ids_in(prompt_section(user, CARDS_HEADER))
    not_handed_in = set(ids_in(prompt_section(user, UNHANDED_HEADER)))
    graded = [card_id for card_id in presented if card_id not in not_handed_in]
    return {
        "verdict": "pending",
        "rationale": f"graded {len(graded)} handed-in card(s)",
        "problems": [f"{card_id}: no metric was written for it" for card_id in graded],
        "evidence": [],
        "per_card": dict.fromkeys(graded, "untested")
        | dict.fromkeys(sorted(not_handed_in), "untested (not handed in)"),
    }


def run_verify(
    cards: list[dict[str, Any]], plan: dict[str, Any]
) -> tuple[dict[str, Any], SilentUI, dict[str, str]]:
    captured: dict[str, str] = {}

    def capture(_system: str, user: str) -> dict[str, Any]:
        captured["verify"] = user
        return strict_verifier(_system, user)

    llm = FakeLLM({ExecuteVerdict: capture})
    ui = SilentUI()
    settings = make_settings(ROOT / ".essay_agent" / "experiments" / "unhanded")
    deps = build_deps(settings, llm, ui=ui)
    state: dict[str, Any] = {
        "run_id": "run-experiment-unhanded",
        "slug": "unhanded",
        "query": "sparse expert models",
        "paper": {"id": "p1", "title": "A review of sparse expert models"},
        "cards": cards,
        "coverage": [],
        "feasibility": feasibility_fixture(),
        "plan_round": 0,
        "exec_round": 1,
        "exec_plan": plan,
        "exec_result": {
            "ok": True,
            "exit_code": 0,
            "duration": 2.09,
            "timed_out": False,
            "metrics": {"c01_criterion_ok": 1.0},
            "figures": ["fig_c01.png"],
            "stdout_tail": "",
            "error": None,
        },
    }
    state.update(make_verify_execute_node(deps)(state))
    return state, ui, captured


# ------------------------------------------------------------------------ parts
def part1_the_measured_problem() -> None:
    print("\n[1] the problem: the verifier grades cards the run never handed in")
    cards = cards_fixture()
    _state, _ui, captured = run_verify(cards, plan_payload(cards_covered=list(COVERED)))
    graded = ids_in(prompt_section(captured["verify"], CARDS_HEADER))
    section = prompt_section(captured["verify"], UNHANDED_HEADER)
    print(f"      in scope:            {', '.join(IN_SCOPE)}")
    print(f"      the plan covers:     {', '.join(COVERED)}")
    print(f"      handed to the verifier for grading: {len(graded)} card(s)")
    print(f"      marked as not handed in: {ids_in(section) or '(nothing)'}")
    check(
        "[1a] the plan leaves in-scope cards with no evidence at all",
        set(IN_SCOPE) - set(COVERED) == set(UNHANDED),
        f"{len(UNHANDED)} of {len(IN_SCOPE)}",
    )
    check(
        "[1b] the prompt marks exactly those cards as not graded",
        ids_in(section) == list(UNHANDED),
        "before this change the section was missing, so a strict verifier graded them",
    )


def part2_the_rule() -> None:
    print("\n[2] the rule: in scope, minus what the plan covers")
    unhanded = prototype_unhanded(cards_fixture(), plan_payload(cards_covered=list(COVERED)))
    print(f"      not handed in: {', '.join(unhanded)}")
    check("[2a] the uncovered in-scope cards are the unhanded set", unhanded == list(UNHANDED))
    check(
        "[2b] a plan that states no coverage changes nothing",
        prototype_unhanded(cards_fixture(), plan_payload(cards_covered=[])) == [],
        "an empty cards_covered never un-grades anything",
    )
    check(
        "[2c] a plan covering everything leaves nothing unhanded",
        prototype_unhanded(cards_fixture(), plan_payload(cards_covered=list(IN_SCOPE))) == [],
    )
    check(
        "[2d] the shipped helper exists",
        unhanded_cards is not None,
        "essay_agent.nodes.base.unhanded_cards",
    )


def part3_the_real_node() -> None:
    print("\n[3] the real verify_execute node")
    state, _ui, captured = run_verify(cards_fixture(), plan_payload(cards_covered=list(COVERED)))
    prompt = captured["verify"]
    section = prompt_section(prompt, UNHANDED_HEADER)
    print(f"      verifier prompt: {len(prompt):,} chars")
    print(f"      unhanded section: {ids_in(section)}")
    check(
        "[3a] the uncovered cards are named in their own section",
        ids_in(section) == list(UNHANDED),
        f"{ids_in(section)}",
    )
    check(
        "[3b] the section says they are not to be graded",
        "not" in section.lower() and "untested" in section.lower(),
        section.splitlines()[0] if section else "(missing)",
    )
    check(
        "[3c] the section says they are not problems",
        "problem" in section.lower(),
        section.splitlines()[-1] if section else "(missing)",
    )
    graded = ids_in(prompt_section(prompt, CARDS_HEADER))
    per_card = state["verdict"]["per_card"]
    problems = " ".join(state["verdict"]["problems"])
    print(f"      per_card: {len(per_card)} entries; problems: {len(state['verdict']['problems'])}")
    check(
        "[3d] the in-scope cards are still all listed for the model",
        graded == list(IN_SCOPE),
        f"{graded}",
    )
    check(
        "[3e] no problem is raised for a card that was never handed in",
        not (set(ids_in(problems)) & set(UNHANDED)),
        problems or "(no problems)",
    )


def part4_edges() -> None:
    print("\n[4] the edges")
    _state, _ui, captured = run_verify(cards_fixture(), plan_payload(cards_covered=list(IN_SCOPE)))
    check(
        "[4a] a fully covering plan carries no unhanded section",
        prompt_section(captured["verify"], UNHANDED_HEADER) == "",
    )
    if unhanded_cards is not None:
        cards = base_nodes.cards_of({"cards": cards_fixture()})
        state = {"cards": cards_fixture(), "exec_plan": plan_payload(cards_covered=list(COVERED))}
        check(
            "[4b] the helper reads the plan out of the state",
            unhanded_cards(state, cards) == list(UNHANDED),
            f"{unhanded_cards(state, cards)}",
        )
        check(
            "[4c] an empty coverage list never un-grades anything",
            unhanded_cards({"cards": cards_fixture(), "exec_plan": {"cards_covered": []}}, cards)
            == [],
        )
    else:
        check("[4b] the helper reads the plan out of the state", False, "helper missing")
        check("[4c] an empty coverage list never un-grades anything", False, "helper missing")


def main() -> int:
    print("unhanded cards are not graded: the shipped behaviour, measured (offline, no key)")
    part1_the_measured_problem()
    part2_the_rule()
    part3_the_real_node()
    part4_edges()
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
