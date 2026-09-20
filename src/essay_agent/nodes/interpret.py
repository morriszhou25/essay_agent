"""Stage 5 - write the report and publish the run."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from essay_agent.errors import LLMError
from essay_agent.nodes.base import (
    Deps,
    NodeReturn,
    card_scope,
    cards_of,
    json_block,
    paper_of,
)
from essay_agent.prompts import interpret as prompts
from essay_agent.schemas.card import cards_to_markdown
from essay_agent.state import RunState


def _history_line(entry: dict[str, Any]) -> str:
    """One verification attempt, plus what it noticed without calling it a defect."""
    line = f"- attempt {entry.get('round')}: {entry.get('verdict')} - {entry.get('rationale')}"
    observations = entry.get("observations") or []
    if observations:
        line += "\n  observations (not defects): " + "; ".join(str(item) for item in observations)
    return line


def _strip_fences(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def make_interpret_node(deps: Deps) -> Callable[[RunState], NodeReturn]:
    """Summarise the evidence, embed the figures, and analyse the paper's impact."""

    def interpret(state: RunState) -> NodeReturn:
        ui = deps.ui
        workspace = deps.workspace(state)
        cards = cards_of(state)
        scope = card_scope(state, cards=cards)
        paper = paper_of(state)
        result = state.get("exec_result") or {}
        ui.step("stage 5/5 - writing the report")

        history = state.get("verdict_history") or []
        verdict_text = "\n".join(_history_line(entry) for entry in history) or (
            "(no verification recorded)"
        )
        language = prompts.detect_language(state.get("query") or "")
        try:
            report = deps.llm.text(
                prompts.SYSTEM,
                prompts.user_message(
                    paper_title=paper.title if paper else (state.get("query") or "unknown paper"),
                    bibliographic=paper.bibliographic_line() if paper else "",
                    cards=scope.cards,
                    excluded=scope.context(),
                    metrics=json_block(result.get("metrics") or {}),
                    figures=list(result.get("figures") or []),
                    verdict_history=verdict_text,
                    plan_summary=state.get("exec_plan", {}).get("approach") or "(none)",
                    run_summary=(
                        f"ok={result.get('ok')} exit={result.get('exit_code')} "
                        f"duration={result.get('duration')} error={result.get('error')}"
                    ),
                    language=language,
                    timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
                    run_id=state.get("run_id", "unknown"),
                    model=deps.llm.model_name("main") if hasattr(deps.llm, "model_name") else "n/a",
                    query=state.get("query") or "",
                ),
                label="report",
            )
        except LLMError as exc:
            ui.error(f"report generation failed: {exc}")
            report = ""
        report = _strip_fences(report)
        if not report:
            report = _fallback_report(state, result)
            ui.warn("wrote a data-only report because the model could not be reached")

        workspace.write_text("result/report.md", report)
        workspace.write_text("result/cards.md", cards_to_markdown(cards))
        workspace.write_json("result/cards.json", [c.model_dump(mode="json") for c in cards])
        workspace.write_json("result/verdicts.json", history)
        if paper:
            workspace.write_json("result/paper.json", paper.model_dump(mode="json"))
        ui.success("report written")
        return {"status": "running", "report_path": str(workspace.dir / "result" / "report.md")}

    return interpret


def _fallback_report(state: RunState, result: dict) -> str:
    paper = state.get("paper") or {}
    metrics = json_block(result.get("metrics") or {})
    return (
        f"# Reproducing: {paper.get('title', state.get('query', 'unknown paper'))}\n\n"
        "## Summary\n\n"
        "The language model was unavailable while writing this report, so only the raw evidence "
        "is recorded here.\n\n"
        f"## Raw metrics\n\n```json\n{metrics}\n```\n\n"
        f"## Run summary\n\n- ok: {result.get('ok')}\n- exit code: {result.get('exit_code')}\n"
        f"- duration: {result.get('duration')}\n- error: {result.get('error')}\n"
    )


def make_finalize_node(deps: Deps) -> Callable[[RunState], NodeReturn]:
    """Publish results, merge lessons into long-term memory and consolidate."""

    def finalize(state: RunState) -> NodeReturn:
        ui = deps.ui
        workspace = deps.workspace(state)
        paths = deps.settings.resolved_paths().ensure()
        title = (state.get("paper") or {}).get("title") or state.get("query") or "paper"
        ui.step("publishing", "moving results out of the staging area")
        report = workspace.publish(paths, title=title)
        ui.success(f"published: {report.describe()}")

        lesson_lines: list[str] = []
        try:
            consolidation = deps.lesson_store.commit(deps.lessons(state), llm=deps.llm)
            for item in consolidation:
                lesson_lines.append(item.describe())
                ui.dim(item.describe())
        except Exception as exc:  # never fail a finished run because of memory upkeep
            ui.warn(f"lesson consolidation failed: {type(exc).__name__}: {exc}")

        if state.get("report_path"):
            ui.info(f"report: {state['report_path']}")
        return {
            "status": "completed",
            "publish": {
                "result_dir": str(report.result_dir) if report.result_dir else None,
                "code_dir": str(report.code_dir) if report.code_dir else None,
                "notes": report.notes,
            },
            "lesson_report": lesson_lines,
        }

    return finalize
