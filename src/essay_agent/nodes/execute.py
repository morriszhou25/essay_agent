"""Stage 4 - write the lightweight reproduction, fit it to the budget, run it."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from essay_agent.errors import LLMError
from essay_agent.llm import looks_cut_off
from essay_agent.nodes.base import (
    Deps,
    NodeReturn,
    card_scope,
    cards_of,
    environment_report,
    json_block,
)
from essay_agent.prompts import execute as prompts
from essay_agent.runtime.preflight import (
    Estimate,
    estimate_from_events,
    needs_adjustment,
    suggest_adjustment,
)
from essay_agent.runtime.runner import RunOutcome
from essay_agent.schemas.dialogue import ExecPlan, ReproScript
from essay_agent.state import RunState
from essay_agent.workspace import RunWorkspace

SCRIPT_NAME = "repro.py"
STDOUT_LABEL = "[stdout]"
STDERR_LABEL = "[stderr]"
_COST_STEP_KEYS = ("steps", "max_steps", "num_steps", "train_steps")
_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_CODE_START = ("import ", "from ", "def ", "class ", "if __name__", "#!", '"""', "'''")
_CUT_OFF_SIGNS = ("was never closed", "unexpected EOF", "unterminated", "expected ':'")


# --------------------------------------------------------------------- helpers
def check_code(code: str) -> str | None:
    """Static sanity check on generated source. Returns an error string or ``None``."""
    if not code or len(code.strip()) < 120:
        return "the code is empty or too short to be a reproduction script"
    if "```" in code:
        return "the code contains markdown fences; return raw python only"
    try:
        compile(code, SCRIPT_NAME, "exec")
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"
    stripped = code.strip()
    if not any(
        stripped.startswith(prefix) or f"\n{prefix}" in stripped
        for prefix in ("import ", "from ", "def ", "class ", "if __name__")
    ):
        return "no python statements were found in the code"
    for marker in ("<placeholder>", "TODO: implement", "your code here", "... rest of code ..."):
        if marker in code:
            return f"the code still contains a placeholder ({marker!r})"
    if '"--preflight"' not in code and "'--preflight'" not in code:
        return "the script does not implement the --preflight mode"
    return None


def extract_code(reply: str) -> str:
    """Pull python source out of a plain-text reply (a fenced block or the bare text)."""
    text = (reply or "").strip()
    blocks = [block.strip() for block in _FENCE.findall(text)]
    if blocks:
        return max(blocks, key=len)
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(_CODE_START):
            return "\n".join(lines[index:]).strip()
    return text


def _looks_truncated(reply: str, error: str) -> bool:
    """True when the reply was cut off by the output limit rather than written badly."""
    if looks_cut_off(reply):
        return True
    return any(sign in error for sign in _CUT_OFF_SIGNS)


def _ask_code(
    deps: Deps, system: str, user: str, *, label: str, attempts: int
) -> tuple[str, list[str]]:
    """Ask for ``repro.py`` as plain text and static-check it, repairing up to ``attempts`` times.

    Plain text needs no JSON escaping, so a regex such as ``\\d`` in the code can no longer break
    the reply. A file that fails the static check goes back with the exact error; a file that looks
    cut off goes back with an instruction to write a *shorter* one instead of repeating it.
    """
    notes: list[str] = []
    error = "the previous answer was empty"
    for attempt in range(1, attempts + 1):
        message = user
        if attempt > 1:
            message = (
                f"{user}\n\n## Mandatory fix (attempt {attempt})\n"
                f"Your previous answer was rejected: {error}\n"
                "Return the COMPLETE corrected file as raw python text, nothing else."
            )
        reply = deps.llm.text(
            system,
            message,
            label=f"{label}_code{attempt}",
            max_tokens=deps.settings.llm.code_max_tokens,
        )
        code = extract_code(reply)
        failure = check_code(code)
        if failure is None:
            return code, notes
        error = failure
        notes.append(f"code attempt {attempt} failed a static check: {failure}")
        if _looks_truncated(reply, failure):
            error = (
                f"{failure}\nThe previous reply looks CUT OFF by the output limit. Do not repeat "
                "the same file: drop optional checks and helpers so the whole file fits, and keep "
                "only the comparison the cards need."
            )
        deps.ui.warn(f"generated code failed a static check ({failure}); asking for a fix")
    raise LLMError(f"no valid {SCRIPT_NAME} after {attempts} attempt(s): {error}")


def _generate_script(
    deps: Deps,
    system: str,
    user: str,
    *,
    label: str,
    plan_system: str = prompts.PLAN_ONLY_SYSTEM,
    code_attempts: int = 2,
) -> ReproScript:
    """Write the reproduction in two cheap steps, falling back to one structured call.

    The old single call asked for plan + code + params inside one JSON envelope. With a couple of
    dozen cards that reply is longer than the output limit, and the model also has to escape every
    backslash in the code, so it failed often and the failure was invisible until the run started.
    Instead:

    1. the plan alone as JSON - small, structured, easy to validate;
    2. the script as *plain text* - no escaping - with a static check and one repair round;
    3. only if either step fails, the old ``ReproScript`` call, whose result is still checked.
    """
    split_failure: Exception | None = None
    try:
        plan = deps.llm.json(plan_system, user, ExecPlan, label=f"{label}_plan")
        code, notes = _ask_code(
            deps,
            system,
            prompts.code_user_message(plan=plan, context=user),
            label=label,
            attempts=code_attempts,
        )
        return ReproScript(
            plan=plan, code=code, params=dict(plan.params), notes="\n".join(notes) or None
        )
    except LLMError as exc:
        split_failure = exc
    deps.ui.warn(
        f"the plan/code split failed ({split_failure}); falling back to one structured call"
    )
    script: ReproScript = deps.llm.json(system, user, ReproScript, label=label)
    failure = check_code(script.code)
    if failure is not None:
        # Never hand a broken script to the runner (the old code returned it silently).
        raise LLMError(f"the fallback script failed a static check: {failure}") from split_failure
    return script


def _write_script(deps: Deps, workspace: RunWorkspace, script: ReproScript, *, tag: str) -> Path:
    path = workspace.write_text(f"code/{SCRIPT_NAME}", script.code)
    workspace.write_json(
        f"code/plan_{tag}.json",
        {
            "plan": script.plan.model_dump(mode="json"),
            "params": script.params,
            "notes": script.notes,
        },
    )
    return path


def _total_steps(plan_payload: dict[str, Any], params: dict[str, Any]) -> int | None:
    for source in (params, plan_payload.get("params") or {}):
        for key in _COST_STEP_KEYS:
            value = source.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
    return None


def _dataset_env(state: RunState) -> dict[str, str | None]:
    """Environment the script needs to reach its datasets (verified route or mirror).

    Comes from the stage-3 connectivity report, so the script and the pre-flight
    agree on which route reaches the dataset host. A ``None`` value means "remove this
    variable" - see :func:`essay_agent.connectivity.route_env`.
    """
    return dict((state.get("connectivity") or {}).get("dataset_env") or {})


def _harvest(workspace: RunWorkspace, outcome: RunOutcome) -> list[str]:
    """Copy metrics + figures into the publishable ``result/`` tree."""
    result_dir = workspace.subdir("result")
    code_dir = workspace.subdir("code")
    figures: list[str] = []
    for candidate in (code_dir / "figures", code_dir / "figs"):
        if candidate.is_dir():
            target = result_dir / "figures"
            target.mkdir(parents=True, exist_ok=True)
            for item in sorted(candidate.iterdir()):
                if item.is_file():
                    shutil.copy2(item, target / item.name)
                    figures.append(item.name)
    source_metrics = code_dir / "metrics.json"
    if source_metrics.is_file():
        shutil.copy2(source_metrics, result_dir / "metrics.json")
    elif outcome.metrics:
        (result_dir / "metrics.json").write_text(
            json.dumps(outcome.metrics, indent=2), encoding="utf-8"
        )
    return figures


def log_tail(result: dict[str, Any], *, stdout_chars: int = 3000, stderr_chars: int = 1500) -> str:
    """The tail of the child's output, both streams labelled.

    A native abort (a duplicate OpenMP runtime, a missing DLL, a segfault) writes to stderr and
    nothing at all to stdout, so a verifier handed the stdout tail alone gets a run it cannot
    diagnose. A run with no error output keeps the previous shape.
    """
    stdout = (result.get("stdout_tail") or "").strip()
    stderr = (result.get("stderr_tail") or "").strip()
    if not stderr:
        return stdout[-stdout_chars:]
    return f"{STDOUT_LABEL}\n{stdout[-stdout_chars:]}\n\n{STDERR_LABEL}\n{stderr[-stderr_chars:]}"


def _error_lines(stderr_tail: str, limit: int = 3) -> list[str]:
    """The last non-empty lines of the child's error output, for the console."""
    lines = [line.strip() for line in (stderr_tail or "").splitlines() if line.strip()]
    return lines[-limit:]


def _plan_summary(state: RunState) -> str:
    plan = state.get("exec_plan") or {}
    if not plan:
        return "(no plan recorded)"
    bits = [
        f"approach: {plan.get('approach', 'n/a')}",
        f"metrics: {', '.join(plan.get('metrics') or []) or 'n/a'}",
        f"figures: {', '.join(plan.get('figures') or []) or 'n/a'}",
        f"cards covered: {', '.join(plan.get('cards_covered') or []) or 'n/a'}",
        f"success signal: {plan.get('success_signal', 'n/a')}",
    ]
    return "\n".join(f"- {bit}" for bit in bits)


# ----------------------------------------------------------------------- nodes
def make_execute_plan_node(deps: Deps) -> Callable[[RunState], NodeReturn]:
    """Write ``repro.py`` plus the executable plan."""

    def execute_plan(state: RunState) -> NodeReturn:
        ui = deps.ui
        workspace = deps.workspace(state)
        cards = cards_of(state)
        scope = card_scope(state, cards=cards)
        cards = scope.cards
        title = (state.get("paper") or {}).get("title", "unknown paper")
        coverage = state.get("coverage") or []
        detail = f"{len(cards)} card(s) in scope"
        if coverage:
            carded = sum(1 for row in coverage if row.get("status") == "carded")
            detail += f" from {carded} of {len(coverage)} eligible section(s)"
        ui.step("stage 4/5 - writing the lightweight reproduction", detail)
        if scope.excluded or scope.relaxed:
            ui.dim(scope.describe())
        feasibility = state.get("feasibility") or {}
        environment = environment_report(deps, check_network=False)
        user = prompts.plan_user_message(
            paper_title=title,
            cards=cards,
            feasibility=json_block(feasibility.get("summary") or feasibility),
            environment=environment,
            time_budget_seconds=deps.settings.runtime.time_budget_seconds,
            plan_round_notes=state.get("plan_carryover") or "",
            lesson_context=deps.lesson_context("execute"),
            excluded=scope.context(),
        )
        try:
            script = _generate_script(deps, prompts.PLAN_SYSTEM, user, label="repro_script")
        except LLMError as exc:
            # Persist the diagnosis: a failed node returns instead of raising, so
            # nothing else writes this into the run directory.
            workspace.log(
                f"stage 4 failed while writing {SCRIPT_NAME}: {type(exc).__name__}: {exc}"
            )
            return {"status": "failed", "error": f"code generation failed: {exc}"}
        path = _write_script(deps, workspace, script, tag="initial")
        if script.notes:
            ui.dim(f"planner notes: {script.notes[:300]}")
        ui.success(f"reproduction script written ({len(script.code.splitlines())} lines)")
        return {
            "status": "running",
            "scope": scope.summary(),
            "exec_plan": script.plan.model_dump(mode="json"),
            "exec_code_path": str(path),
            "exec_round": 1,
            "adjust_round": 0,
            "exec_result": {},
            "verdict": {},
            "verdict_history": [],
        }

    return execute_plan


def make_preflight_node(deps: Deps) -> Callable[[RunState], NodeReturn]:
    """Run the cheap dry-run and decide whether the plan fits the budget."""

    def preflight(state: RunState) -> NodeReturn:
        ui = deps.ui
        workspace = deps.workspace(state)
        script = Path(state.get("exec_code_path") or workspace.dir / "code" / SCRIPT_NAME)
        budget = deps.settings.runtime.time_budget_seconds
        ui.step("stage 4/5 - pre-flight timing check", "measuring the real cost of one step")
        outcome = deps.runner.run(
            script,
            cwd=workspace.subdir("code"),
            timeout=deps.settings.runtime.preflight_timeout,
            label="pre-flight",
            args=["--preflight"],
            env={**_dataset_env(state), "ESSAY_AGENT_MODE": "preflight"},
            sink_factory=deps.sink_factory(),
        )
        estimate = estimate_from_events(
            outcome.event_list,
            outcome.duration,
            fallback_total_steps=_total_steps(state.get("exec_plan") or {}, {}),
        )
        if not outcome.ok:
            ui.warn(f"pre-flight did not finish cleanly ({outcome.describe()})")
        ui.show_estimate(estimate)
        too_slow = needs_adjustment(estimate, budget)
        adjust_round = int(state.get("adjust_round") or 0)
        can_adjust = (
            too_slow
            and deps.settings.runtime.auto_adjust
            and adjust_round < deps.settings.runtime.max_adjust_rounds
        )
        if too_slow:
            ui.warn(
                f"the projected run is longer than the {budget:.0f}s budget"
                + ("; shrinking the experiment" if can_adjust else "; no adjustment budget left")
            )
        workspace.write_json(
            "code/preflight.json",
            {
                "estimate": {
                    "seconds": estimate.estimated_full_seconds,
                    "source": estimate.source,
                    "step_fraction": estimate.step_fraction,
                },
                "budget_seconds": budget,
                "needs_adjustment": too_slow,
                "outcome": _outcome_payload(outcome, deps.runner.env_fixes),
            },
        )
        return {
            "status": "running",
            "preflight": {
                "estimate": {
                    "seconds": estimate.estimated_full_seconds,
                    "source": estimate.source,
                    "fraction": estimate.step_fraction,
                    "wall_seconds": estimate.wall_seconds,
                    "params": estimate.params,
                },
                "describe": estimate.describe(),
                "needs_adjustment": too_slow,
                "can_adjust": can_adjust,
            },
        }

    return preflight


def make_adjust_node(deps: Deps) -> Callable[[RunState], NodeReturn]:
    """Rewrite the experiment so it fits the budget, then re-measure."""

    def adjust(state: RunState) -> NodeReturn:
        ui = deps.ui
        workspace = deps.workspace(state)
        scope = card_scope(state)
        cards = scope.cards
        preflight_payload = state.get("preflight") or {}
        budget = deps.settings.runtime.time_budget_seconds
        estimate = Estimate(
            estimated_full_seconds=(preflight_payload.get("estimate") or {}).get("seconds"),
            source=(preflight_payload.get("estimate") or {}).get("source") or "unknown",
            params=(preflight_payload.get("estimate") or {}).get("params") or {},
            step_fraction=(preflight_payload.get("estimate") or {}).get("fraction"),
            wall_seconds=(preflight_payload.get("estimate") or {}).get("wall_seconds") or 0.0,
        )
        suggestion = suggest_adjustment(
            estimate, budget, state.get("exec_plan", {}).get("params") or {}
        )
        round_no = int(state.get("adjust_round") or 0) + 1
        ui.step(f"stage 4/5 - shrinking the run to fit the budget (round {round_no})")
        for note in suggestion["notes"]:
            ui.dim(note)
        user = prompts.adjust_user_message(
            cards=cards,
            estimate_line=preflight_payload.get("describe", "unknown"),
            budget_seconds=budget,
            suggestions="\n".join(f"- {note}" for note in suggestion["notes"]),
            current_params=json_block(suggestion["params"]),
            adjustment_round=round_no,
            lesson_context=deps.lesson_context("execute"),
            excluded=scope.context(),
        )
        try:
            script = _generate_script(
                deps, prompts.ADJUST_SYSTEM, user, label=f"repro_adjust{round_no}"
            )
        except LLMError as exc:
            ui.warn(f"could not regenerate the script ({exc}); trying the original anyway")
            return {
                "adjust_round": round_no,
                "preflight": {**preflight_payload, "needs_adjustment": False, "can_adjust": False},
            }
        path = _write_script(deps, workspace, script, tag=f"adjust{round_no}")
        if script.notes:
            ui.dim(f"adjustment notes: {script.notes[:300]}")
        ui.success(
            f"scaled by ~{suggestion['factor']:.2f}x "
            f"({len(script.code.splitlines())} lines rewritten)"
        )
        return {
            "status": "running",
            "scope": scope.summary(),
            "exec_plan": script.plan.model_dump(mode="json"),
            "exec_code_path": str(path),
            "adjust_round": round_no,
            "preflight": {**preflight_payload, "needs_adjustment": False, "stale": True},
        }

    return adjust


def make_run_node(deps: Deps) -> Callable[[RunState], NodeReturn]:
    """Run the full reproduction with a live bar and a hard timeout."""

    def run(state: RunState) -> NodeReturn:
        ui = deps.ui
        workspace = deps.workspace(state)
        script = Path(state.get("exec_code_path") or workspace.dir / "code" / SCRIPT_NAME)
        round_no = int(state.get("exec_round") or 1)
        budget = deps.settings.runtime.time_budget_seconds
        timeout = deps.settings.runtime.hard_timeout_seconds
        ui.step(f"stage 4/5 - full reproduction (attempt {round_no})", f"budget {budget:.0f}s")
        outcome = deps.runner.run(
            script,
            cwd=workspace.subdir("code"),
            timeout=timeout,
            label="reproduction",
            args=["--out", str(workspace.subdir("code"))],
            env=_dataset_env(state) or None,
            sink_factory=deps.sink_factory(),
            total_steps=_total_steps(
                state.get("exec_plan") or {}, state.get("exec_plan", {}).get("params") or {}
            ),
        )
        figures = _harvest(workspace, outcome)
        payload = _outcome_payload(outcome, deps.runner.env_fixes)
        payload["figures"] = figures
        workspace.write_json("code/run_outcome.json", payload)
        if outcome.ok:
            ui.success(
                f"run finished in {outcome.duration:.1f}s"
                + (f" with {len(outcome.metrics)} metric(s)" if outcome.metrics else "")
            )
        elif outcome.timed_out:
            ui.error(f"run hit the {timeout:.0f}s hard timeout")
        else:
            ui.error(f"run failed: {outcome.error or outcome.exit_code}")
            for line in _error_lines(outcome.stderr_tail):
                ui.dim(f"  {line[:200]}")
        return {"status": "running", "exec_result": payload}

    return run


def _outcome_payload(
    outcome: RunOutcome, env_fixes: dict[str, str] | None = None
) -> dict[str, Any]:
    return {
        "ok": outcome.ok,
        "exit_code": outcome.exit_code,
        "duration": outcome.duration,
        "timed_out": outcome.timed_out,
        "cancelled": outcome.cancelled,
        "metrics": outcome.metrics,
        "figures": outcome.figures,
        "stdout_tail": outcome.stdout_tail[-4000:],
        "stderr_tail": outcome.stderr_tail[-4000:],
        "error": outcome.error,
        "script": outcome.script,
        "env_fixes": dict(env_fixes or {}),
    }


def preflight_needs_adjustment(state: RunState) -> bool:
    """Routing helper for the preflight -> adjust|run decision."""
    return bool((state.get("preflight") or {}).get("can_adjust"))
