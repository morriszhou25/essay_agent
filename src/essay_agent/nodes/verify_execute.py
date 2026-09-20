"""Stage 4.1 - verify the measured results, and re-execute when they do not hold up.

The verifier only ever returns ``successful`` / ``pending`` / ``unsuccessful``. On the
last permitted round its verdict is accepted as final and the report must state it.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from essay_agent.errors import LLMError
from essay_agent.nodes.base import (
    Deps,
    NodeReturn,
    card_scope,
    json_block,
    unhanded_cards,
    unhanded_context,
)
from essay_agent.nodes.execute import (
    _generate_script,
    _plan_summary,
    _write_script,
    log_tail,
)
from essay_agent.prompts import execute as prompts
from essay_agent.schemas.dialogue import ExecuteVerdict
from essay_agent.state import RunState


def make_verify_execute_node(deps: Deps) -> Callable[[RunState], NodeReturn]:
    """Judge the run against the cards' claims, assumptions and criteria."""

    def verify_execute(state: RunState) -> NodeReturn:
        ui = deps.ui
        workspace = deps.workspace(state)
        scope = card_scope(state)
        cards = scope.cards
        # Cards the plan does not cover: the run hands in no evidence for them, so they are
        # never graded (see `unhanded_cards`).
        unhanded = unhanded_cards(state, cards)
        result = state.get("exec_result") or {}
        round_no = int(state.get("exec_round") or 1)
        max_rounds = deps.settings.runtime.max_execute_rounds
        ui.step(f"stage 4.1/5 - execution verifier (round {round_no}/{max_rounds})")
        if unhanded:
            ui.dim(f"not handed in by the plan, so not graded: {', '.join(unhanded)}")

        duration = result.get("duration")
        run_summary = (
            f"- exit code: {result.get('exit_code')}\n"
            f"- ok: {result.get('ok')}\n"
            f"- timed out: {result.get('timed_out')}\n"
            f"- duration: {duration:.1f}s"
            if isinstance(duration, (int, float))
            else "- duration: n/a"
        )
        if result.get("error"):
            run_summary += f"\n- error: {result['error']}"
        metrics_json = json_block(result.get("metrics") or {})
        try:
            verdict: ExecuteVerdict = deps.llm.json(
                prompts.VERIFY_SYSTEM,
                prompts.verify_user_message(
                    cards=cards,
                    metrics=metrics_json,
                    figures=list(result.get("figures") or []),
                    plan_summary=_plan_summary(state),
                    run_summary=run_summary,
                    log_tail=log_tail(result),
                    round_no=round_no,
                    max_rounds=max_rounds,
                    lesson_context=deps.lesson_context("execute"),
                    excluded=scope.context(),
                    unhanded=unhanded_context(unhanded),
                ),
                ExecuteVerdict,
                role="verifier",
                label=f"exec_verdict:{round_no}",
            )
        except LLMError as exc:
            ui.warn(f"execution verifier unavailable ({exc}); recording 'pending'")
            verdict = ExecuteVerdict(
                verdict="pending",
                rationale=f"the verifier could not be reached: {exc}",
                problems=["verification could not be completed"],
            )

        payload = verdict.model_dump(mode="json")
        payload["round"] = round_no
        payload["duration"] = duration
        payload["unhanded"] = list(unhanded)
        workspace.write_json(f"code/verdict_round{round_no}.json", payload)
        # `problems` is the action channel and only carries serious defects; everything the
        # verifier noticed but did not treat as a defect is recorded and shown, never acted on.
        for item in verdict.observations[:5]:
            ui.dim(f"observation (not a defect): {item}")
        if len(verdict.observations) > 5:
            ui.dim(f"... and {len(verdict.observations) - 5} more observation(s)")

        history = list(state.get("verdict_history") or [])
        history.append(payload)
        final_round = round_no >= max_rounds

        if verdict.verdict != "successful" and not final_round:
            if verdict.problems:
                deps.lessons(state).append(
                    "execute",
                    f"execution verifier, attempt {round_no}: {verdict.verdict}",
                    verdict.problems,
                    tags=("execute", "verdict"),
                    meta=verdict.rationale[:400],
                )
            ui.verdict_notice(
                verdict.verdict, round_no, max_rounds, verdict.rationale, verdict.problems
            )
            ui.replan_notice(
                "execute", round_no, max_rounds, verdict.problems or [verdict.rationale]
            )
        else:
            ui.verdict_notice(
                verdict.verdict, round_no, max_rounds, verdict.rationale, verdict.problems
            )
            if final_round and verdict.verdict != "successful":
                ui.warn("this was the last permitted attempt - the report will state this verdict")
        return {"status": "running", "verdict": payload, "verdict_history": history}

    return verify_execute


def make_reexecute_node(deps: Deps) -> Callable[[RunState], NodeReturn]:
    """Ask the planner to fix the experiment, then run the pre-flight again."""

    def reexecute(state: RunState) -> NodeReturn:
        ui = deps.ui
        workspace = deps.workspace(state)
        scope = card_scope(state)
        cards = scope.cards
        verdict = state.get("verdict") or {}
        result = state.get("exec_result") or {}
        round_no = int(state.get("exec_round") or 1) + 1
        ui.step(f"stage 4.1/5 - re-execution {round_no}", "the planner is fixing the experiment")
        staged = deps.lessons(state).read("execute")
        user = prompts.reexecute_user_message(
            cards=cards,
            verdict=str(verdict.get("rationale") or verdict.get("verdict") or ""),
            problems=list(verdict.get("problems") or []),
            metrics=json_block(result.get("metrics") or {}),
            previous_params=json_block((state.get("exec_plan") or {}).get("params") or {}),
            run_summary=(
                f"ok={result.get('ok')} exit={result.get('exit_code')} "
                f"duration={result.get('duration')} error={result.get('error')}"
            ),
            plan_round_notes=state.get("plan_carryover") or "",
            lesson_context=(staged[-2500:] if staged else "") or deps.lesson_context("execute"),
            round_no=round_no,
            excluded=scope.context(),
        )
        try:
            script = _generate_script(
                deps, prompts.REEXECUTE_SYSTEM, user, label=f"repro_reexec{round_no}"
            )
        except LLMError as exc:
            ui.error(f"could not regenerate the script ({exc})")
            return {
                "status": "failed",
                "error": f"re-execution script generation failed: {exc}",
                "exec_round": round_no,
            }
        path = _write_script(deps, workspace, script, tag=f"reexec{round_no}")
        if script.notes:
            ui.dim(f"changes: {script.notes[:300]}")
        ui.success("re-execution script ready")
        return {
            "status": "running",
            "scope": scope.summary(),
            "exec_plan": script.plan.model_dump(mode="json"),
            "exec_code_path": str(path),
            "exec_round": round_no,
            "adjust_round": 0,
            "exec_result": {},
        }

    return reexecute


def verdict_allows_report(state: RunState, max_rounds: int) -> bool:
    """Routing helper: stop looping when the run succeeded or the rounds are exhausted."""
    verdict = (state.get("verdict") or {}).get("verdict")
    round_no = int(state.get("exec_round") or 1)
    return verdict == "successful" or round_no >= max_rounds


def verdict_json(state: RunState) -> str:
    return json.dumps(state.get("verdict") or {}, ensure_ascii=False, indent=2)
