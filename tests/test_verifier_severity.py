"""The verifier's `problems` list only carries serious defects.

On the real run the execution verifier returned 5 problems, three of which were not defects: an
8.6% spread on a 6 ms wall-clock benchmark (inside the harness's own resolution), "no training was
performed" (our budget decision, which the verifier itself called acceptable), and a figure-naming
nitpick. All three landed in the re-execution task list. `problems` is now the action channel and
everything else goes to `observations`, which is recorded and reported but never acted on.
"""

from __future__ import annotations

from typing import Any

from essay_agent.console import SilentUI
from essay_agent.nodes.interpret import _history_line, make_interpret_node
from essay_agent.nodes.verify_execute import make_reexecute_node, make_verify_execute_node
from essay_agent.prompts import execute as prompts
from essay_agent.schemas.dialogue import ExecPlan, ExecuteVerdict
from tests.fakes import FakeLLM
from tests.pipeline import build_deps, card_payload, plan_payload, script_with_estimate

SERIOUS = "c03: the sparse arm does not match the dense model's per-token FLOPs"
NOISY = "c09: the wall-clock margin (0.00659s vs 0.00607s) is inside a 6 ms benchmark's resolution"
SMALL = "no training was performed; this is analytic accounting, our own budget decision"


def cards_fixture() -> list[dict[str, Any]]:
    return [card_payload("c01")]


def base_state() -> dict[str, Any]:
    return {
        "run_id": "run-severity",
        "slug": "severity",
        "query": "query",
        "paper": {"id": "p1", "title": "Fixture"},
        "cards": cards_fixture(),
        "coverage": [],
        "feasibility": {
            "checks": [
                {"card_id": "c01", "feasible": True, "severity": "minor", "findings": []},
            ],
            "blockers": [],
            "proceed": True,
            "summary": "fixture",
        },
        "plan_round": 0,
        "exec_round": 1,
        "exec_plan": plan_payload(cards_covered=["c01"]),
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


def run_verify(
    settings, *, problems: list[str], observations: list[str]
) -> tuple[dict[str, Any], SilentUI, Any]:
    llm = FakeLLM(
        {
            ExecuteVerdict: {
                "verdict": "pending",
                "rationale": "one setup defect; the rest are observations",
                "problems": list(problems),
                "observations": list(observations),
                "evidence": [],
                "per_card": {"c01": "untested"},
            }
        }
    )
    ui = SilentUI()
    deps = build_deps(settings, llm, ui=ui)
    state = base_state()
    state.update(make_verify_execute_node(deps)(state))
    return state, ui, deps


# ------------------------------------------------------------------- the schema
def test_the_verdict_carries_observations_separately() -> None:
    verdict = ExecuteVerdict(
        verdict="pending", rationale="r", problems=[SERIOUS], observations=[NOISY, SMALL]
    )

    assert verdict.problems == [SERIOUS]
    assert verdict.observations == [NOISY, SMALL]


def test_older_verdicts_without_observations_still_parse() -> None:
    verdict = ExecuteVerdict.model_validate(
        {
            "verdict": "successful",
            "rationale": "r",
            "problems": [],
            "per_card": {"c01": "supported"},
        }
    )

    assert verdict.observations == []


def test_the_prompt_defines_the_split_and_the_noise_rule() -> None:
    system = prompts.VERIFY_SYSTEM.lower()

    assert "observations" in system
    assert "noise" in system
    assert "inconclusive" in system
    assert "action channel" in system


# --------------------------------------------------------------------- the node
def test_the_node_records_and_shows_the_observations(settings) -> None:
    state, ui, _deps = run_verify(settings, problems=[SERIOUS], observations=[NOISY, SMALL])

    assert state["verdict"]["problems"] == [SERIOUS]
    assert state["verdict"]["observations"] == [NOISY, SMALL]
    shown = [message for kind, message in ui.events if kind == "dim"]
    assert any("observation (not a defect)" in message for message in shown)
    assert any("0.00659" in message for message in shown)


def test_observations_never_enter_the_lesson_memory(settings) -> None:
    state, _ui, deps = run_verify(settings, problems=[SERIOUS], observations=[NOISY, SMALL])

    staged = deps.lessons(state).read("execute")

    assert SERIOUS[:40] in staged
    assert "0.00659" not in staged
    assert "analytic accounting" not in staged


def test_a_noise_only_round_demands_no_work(settings) -> None:
    state, _ui, _deps = run_verify(settings, problems=[], observations=[NOISY, SMALL])

    assert state["verdict"]["problems"] == []
    assert state["verdict"]["observations"] == [NOISY, SMALL]


def test_the_reexecution_prompt_carries_only_the_problems(settings) -> None:
    state, _ui, _deps = run_verify(settings, problems=[SERIOUS], observations=[NOISY, SMALL])
    captured: dict[str, str] = {}

    def capture_plan(_system: str, user: str) -> dict[str, Any]:
        captured["reexec"] = user
        return plan_payload()

    llm = FakeLLM({ExecPlan: capture_plan}, texts={"code": script_with_estimate(0.2)})
    rerun_deps = build_deps(settings, llm, ui=SilentUI())
    make_reexecute_node(rerun_deps)(dict(state))

    prompt = captured["reexec"]

    assert SERIOUS[:40] in prompt
    assert "0.00659" not in prompt
    assert "analytic accounting" not in prompt


# ------------------------------------------------------------------- the report
def test_the_report_history_keeps_the_observations() -> None:
    line = _history_line(
        {"round": 2, "verdict": "pending", "rationale": "r", "observations": [NOISY, SMALL]}
    )

    assert "attempt 2: pending - r" in line
    assert "observations (not defects):" in line
    assert NOISY in line and SMALL in line


def test_the_report_prompt_sees_them(settings) -> None:
    captured: dict[str, str] = {}

    def capture_report(_system: str, user: str) -> str:
        captured["report"] = user
        return "# Report\n\nbody\n"

    llm = FakeLLM(texts={"report": capture_report})
    deps = build_deps(settings, llm, ui=SilentUI())
    state = base_state()
    state["verdict_history"] = [
        {"round": 1, "verdict": "pending", "rationale": "r", "observations": [NOISY]}
    ]

    make_interpret_node(deps)(state)

    assert "observations (not defects)" in captured["report"]
    assert "0.00659" in captured["report"]
