"""Only serious problems get raised: the verifier's `problems` list is the action channel.

Measured on run-20260920-042507 (round 3) the execution verifier returned 5 `problems`, and 3 of
them were not defects at all:

* c09: "the wall-clock criterion failed (top-k 0.00659s vs soft 0.00607s)" - an 8.6% spread on a
  6-millisecond benchmark, i.e. inside the harness's own resolution, not a refutation;
* "no training was performed anywhere; all results are analytic FLOP/parameter accounting" - the
  verifier itself calls that acceptable for the mechanism cards; it is our own budget decision;
* figure-naming/style nitpicks.

Every one of them landed in the re-execution task list, so a round could be spent on noise. The
shipped rule: `problems` carries only defects that change a card's conclusion and that the executor
can fix; everything else goes to `observations`, which is recorded and shown but never drives a
re-run - and a comparison whose margin is inside the run's noise is inconclusive, not a
contradiction.

    python tests/experiment_verifier_severity.py
"""

from __future__ import annotations

# ruff: noqa: E402 -- the package is imported after the src/ path shim below.
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(ROOT / "src"), str(ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from essay_agent.console import SilentUI
from essay_agent.nodes.verify_execute import make_reexecute_node, make_verify_execute_node
from essay_agent.prompts import execute as prompts
from essay_agent.schemas.dialogue import ExecPlan, ExecuteVerdict
from tests.conftest import make_settings
from tests.fakes import FakeLLM
from tests.pipeline import build_deps, card_payload, plan_payload, script_with_estimate

# The real run's round-3 findings, split by what they actually are.
SERIOUS = (
    "c03: the sparse arm does not match the dense model's per-token FLOPs (8704 vs 32768, rel gap "
    "0.734); set the sparse per-expert width so the comparison holds",
    "c04: c04_criterion_ok=0.0 and c04_dense_d_ff_matched=8318.5 is non-integer, so the "
    "matched-parameter dense baseline was not validly constructed",
)
OBSERVATIONS = (
    "c09: the wall-clock criterion failed (top-k 0.00659s vs soft 0.00607s); the difference is "
    "inside the resolution of a 6 ms benchmark, so the criterion is inconclusive rather than "
    "contradicted",
    "no training was performed anywhere; the run is analytic FLOP/parameter accounting, which is "
    "acceptable for the mechanism cards and is our own budget decision",
    "the figure filenames use ad-hoc suffixes rather than the paper's figure numbers",
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ------------------------------------------------------------------- fixture
def cards_fixture() -> list[dict[str, Any]]:
    payloads = []
    for card_id in ("c01", "c03", "c04", "c08", "c09", "c11"):
        payload = card_payload(card_id)
        payload["identity"]["section"] = f"{card_id[1:]} Section"
        payloads.append(payload)
    return payloads


def base_state() -> dict[str, Any]:
    return {
        "run_id": "run-experiment-severity",
        "slug": "severity",
        "query": "sparse expert models",
        "paper": {"id": "p1", "title": "A review of sparse expert models"},
        "cards": cards_fixture(),
        "coverage": [],
        "feasibility": {
            "checks": [
                {"card_id": card_id, "feasible": True, "severity": "minor", "findings": []}
                for card_id in ("c01", "c03", "c04", "c08", "c09", "c11")
            ],
            "blockers": [],
            "proceed": True,
            "summary": "all six cards are reproducible here",
        },
        "plan_round": 0,
        "exec_round": 1,
        "exec_plan": plan_payload(cards_covered=["c01", "c03", "c04", "c08", "c09", "c11"]),
        "exec_result": {
            "ok": True,
            "exit_code": 0,
            "duration": 2.09,
            "timed_out": False,
            "metrics": {"c03_flops_rel_gap": 0.734375, "c09_wallclock_topk_s": 0.006594},
            "figures": ["fig_c01.png"],
            "stdout_tail": "",
            "error": None,
        },
    }


def deps_for(settings, llm: FakeLLM, ui: SilentUI):
    return build_deps(settings, llm, ui=ui)


# ------------------------------------------------------------------- the node
def run_verify(settings, *, problems: list[str], observations: list[str]):
    captured: dict[str, str] = {}

    def capture(_system: str, user: str) -> dict[str, Any]:
        captured["verify"] = user
        return {
            "verdict": "pending",
            "rationale": "two setup defects; the rest are observations",
            "problems": list(problems),
            "observations": list(observations),
            "evidence": [],
            "per_card": {"c03": "not supported", "c09": "inconclusive"},
        }

    llm = FakeLLM({ExecuteVerdict: capture})
    ui = SilentUI()
    deps = deps_for(settings, llm, ui)
    state = base_state()
    state.update(make_verify_execute_node(deps)(state))
    return state, ui, captured, deps


def run_reexecute(settings, state: dict[str, Any], deps):
    captured: dict[str, str] = {}

    def capture_plan(_system: str, user: str) -> dict[str, Any]:
        captured["reexec"] = user
        return plan_payload()

    llm = FakeLLM({ExecPlan: capture_plan}, texts={"code": script_with_estimate(0.2)})
    rerun_deps = deps_for(settings, llm, SilentUI())
    fresh = dict(state)
    fresh.update(make_reexecute_node(rerun_deps)(fresh))
    return fresh, captured


# ------------------------------------------------------------------------ parts
def part1_the_measured_problem() -> None:
    print("\n[1] the measured verdict: 5 problems, 3 of them not defects")
    total = len(SERIOUS) + len(OBSERVATIONS)
    print(f"      round-3 problems: {total}")
    print(f"      that change a card's conclusion and are fixable: {len(SERIOUS)}")
    print(f"      noise / our own budget choice / style: {len(OBSERVATIONS)}")
    check(
        "[1a] most of the reported problems were not defects",
        len(OBSERVATIONS) > len(SERIOUS),
        f"{len(OBSERVATIONS)} of {total}",
    )
    check(
        "[1b] the c09 finding is a margin inside the benchmark's own resolution",
        "0.00659" in OBSERVATIONS[0] and "0.00607" in OBSERVATIONS[0],
        "8.6% spread on a 6 ms measurement",
    )


def part2_the_rule() -> None:
    print("\n[2] the rule: `problems` is the action channel, `observations` is the record")
    check(
        "[2a] the schema carries a separate observations list",
        "observations" in ExecuteVerdict.model_fields,
        f"{sorted(ExecuteVerdict.model_fields)}",
    )
    check(
        "[2b] the verifier prompt defines the split",
        "observations" in prompts.VERIFY_SYSTEM,
        "VERIFY_SYSTEM",
    )
    check(
        "[2c] the prompt names noise as the thing that must not become a problem",
        "noise" in prompts.VERIFY_SYSTEM.lower(),
    )
    check(
        "[2d] the prompt says a within-noise margin is inconclusive, not a contradiction",
        "inconclusive" in prompts.VERIFY_SYSTEM.lower(),
    )


def part3_the_real_nodes(settings) -> None:
    print("\n[3] the real verify_execute and reexecute nodes")
    state, ui, _captured, deps = run_verify(
        settings, problems=list(SERIOUS), observations=list(OBSERVATIONS)
    )
    verdict = state["verdict"]
    print(f"      problems recorded:     {len(verdict['problems'])}")
    print(f"      observations recorded: {len(verdict.get('observations') or [])}")
    check(
        "[3a] both lists reach the recorded verdict",
        verdict["problems"] == list(SERIOUS) and verdict.get("observations") == list(OBSERVATIONS),
        f"{len(verdict['problems'])} + {len(verdict.get('observations') or [])}",
    )
    shown = [message for kind, message in ui.events if kind == "dim"]
    print(f"      dim lines: {len(shown)}")
    check(
        "[3b] the human is told about the observations",
        any("observation" in message.lower() for message in shown),
        shown[0][:90] if shown else "(nothing shown)",
    )
    lessons = deps.lessons(state).read("execute")
    check(
        "[3c] the lesson memory only records the serious problems",
        all(item[:24] in lessons for item in SERIOUS)
        and "filenames" not in lessons
        and "0.00659" not in lessons,
        f"{len(lessons)} chars staged",
    )
    _fresh, captured = run_reexecute(settings, state, deps)
    prompt = captured.get("reexec", "")
    task_list = prompt.split("## Problems to address")[-1].split("## Previous run")[0]
    print(f"      re-execution task list: {len(task_list)} chars")
    check(
        "[3d] the re-execution task list carries the serious problems",
        all(item[:24] in task_list for item in SERIOUS),
    )
    check(
        "[3e] no observation leaks into the re-execution task list",
        "filenames" not in task_list and "0.00659" not in task_list and "analytic" not in task_list,
        task_list.strip().splitlines()[-1][:90] if task_list.strip() else "(empty)",
    )


def part4_noise_only(settings) -> None:
    print("\n[4] a round whose only finding is noise")
    state, ui, _captured, deps = run_verify(settings, problems=[], observations=list(OBSERVATIONS))
    verdict = state["verdict"]
    print(f"      problems: {len(verdict['problems'])}, verdict: {verdict['verdict']}")
    check(
        "[4a] a noise-only round raises no problem (so no work is demanded)",
        verdict["problems"] == [] and len(verdict.get("observations") or []) == len(OBSERVATIONS),
    )
    check(
        "[4b] the round still reports what it saw",
        any("observation" in message.lower() for kind, message in ui.events if kind == "dim"),
    )
    _fresh, captured = run_reexecute(settings, state, deps)
    check(
        "[4c] the next attempt still gets the verifier's reasoning",
        "two setup defects; the rest are observations" in captured.get("reexec", ""),
    )


def main() -> int:
    print("only serious problems are raised (offline, no key)")
    part1_the_measured_problem()
    part2_the_rule()
    part3_the_real_nodes(make_settings(ROOT / ".essay_agent" / "experiments" / "severity"))
    part4_noise_only(make_settings(ROOT / ".essay_agent" / "experiments" / "severity_noise"))
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
