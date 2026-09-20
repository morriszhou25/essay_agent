"""The two bounded feedback loops: replan (cards) and re-execute (results), plus
the pre-flight budget adjustment."""

from __future__ import annotations

from pathlib import Path

from essay_agent.agent import run_task
from essay_agent.console import SilentUI
from tests.fakes import FakeLLM
from tests.pipeline import (
    build_deps,
    crashing_script,
    happy_responses,
    plan_payload,
    result_dirs,
    script_with_estimate,
)

REPORT = "# Reproducing: Test Paper\n\n## Summary\nwritten anyway.\n"

VAGUE_ISSUE = {
    "card_id": "c01",
    "field": "success_criteria",
    "severity": "major",
    "problem": "the criterion cannot be measured from the metrics",
    "suggestion": "compare accuracy against baseline_accuracy with an explicit threshold",
}

TIGHTER_PATCH = {
    "card_id": "c01",
    "field": "success_criteria",
    "value": ["accuracy >= baseline_accuracy + 0.02"],
    "reasoning": "the paper's own numbers give an explicit 2.8 point margin",
}


def _replanning_responses(rounds: int) -> dict:
    responses = happy_responses()
    for round_no in range(1, rounds + 1):
        responses[f"plan_review:{round_no}"] = {
            "ok": False,
            "issues": [VAGUE_ISSUE],
            "summary": "the success criterion is still vague",
        }
        responses[f"card_revision:{round_no}"] = {
            "patches": [TIGHTER_PATCH],
            "rejected_issues": [],
            "overall_reasoning": "accepted: the paper states an explicit margin",
        }
    return responses


def test_replan_loop_runs_three_rounds_then_releases(settings) -> None:
    llm = FakeLLM(_replanning_responses(3), texts={"report": REPORT})
    ui = SilentUI()
    deps = build_deps(settings, llm, ui=ui)

    result = run_task("Test Paper 0", deps)

    assert result.status == "completed", result.message
    state = result.state
    assert state["plan_round"] == 3
    assert "unresolved issues" in state["plan_carryover"]
    assert state["cards"][0]["success_criteria"] == ["accuracy >= baseline_accuracy + 0.02"]
    assert len(ui.find("replan")) == 2  # rounds 1 and 2 ask for a replan; round 3 releases
    assert any("limit reached" in message for message in ui.find("warn"))

    lesson_file = settings.resolved_paths().lesson_dir / "lesson_plan.txt"
    assert lesson_file.is_file()
    assert "success_criteria" in lesson_file.read_text(encoding="utf-8")


def test_replan_stops_early_when_the_verifier_is_happy(settings) -> None:
    llm = FakeLLM(happy_responses(), texts={"report": REPORT})
    ui = SilentUI()
    deps = build_deps(settings, llm, ui=ui)
    result = run_task("Test Paper 0", deps)
    assert result.status == "completed"
    assert result.state["plan_round"] == 1
    assert ui.find("replan") == []
    assert result.state["plan_carryover"] == ""


def test_verifier_unavailable_does_not_block_the_run(settings) -> None:
    from essay_agent.errors import LLMError

    llm = FakeLLM(
        happy_responses(),
        texts={"report": REPORT},
        error_labels={"plan_review:1": LLMError("verifier offline")},
    )
    deps = build_deps(settings, llm)
    result = run_task("Test Paper 0", deps)
    assert result.status == "completed"
    assert result.state["plan_round"] == 1


def test_execution_verifier_triggers_one_re_execution(settings) -> None:
    responses = happy_responses()
    responses["exec_verdict:1"] = {
        "verdict": "unsuccessful",
        "rationale": "the baseline arm never ran",
        "problems": ["baseline_accuracy was written but not measured on the same split"],
        "evidence": ["metrics: accuracy only"],
        "per_card": {"c01": "not supported"},
    }
    responses["exec_verdict:2"] = {
        "verdict": "successful",
        "rationale": "both arms now run on the same split",
        "problems": [],
        "evidence": ["accuracy=0.912", "baseline_accuracy=0.884"],
        "per_card": {"c01": "supported"},
    }
    responses["repro_reexec2_plan"] = responses["repro_script_plan"]
    llm = FakeLLM(responses, texts={"report": REPORT})
    ui = SilentUI()
    deps = build_deps(settings, llm, ui=ui)

    result = run_task("Test Paper 0", deps)

    assert result.status == "completed", result.message
    assert result.state["exec_round"] == 2
    assert len(result.state["verdict_history"]) == 2
    assert len(ui.find("replan")) == 1
    assert ("json", "repro_reexec2_plan", "ExecPlan", "main") in llm.calls
    assert ("text", "repro_reexec2_code1", "text", "main") in llm.calls
    lesson_file = settings.resolved_paths().lesson_dir / "lesson_execute.txt"
    assert lesson_file.is_file()
    assert "baseline" in lesson_file.read_text(encoding="utf-8")


def test_three_failed_executions_are_released_with_the_verdict(settings) -> None:
    responses = happy_responses()
    for round_no in (1, 2, 3):
        responses[f"exec_verdict:{round_no}"] = {
            "verdict": "unsuccessful",
            "rationale": f"attempt {round_no} did not measure the baseline",
            "problems": ["baseline_accuracy missing"],
            "evidence": [],
            "per_card": {"c01": "untested"},
        }
    for round_no in (2, 3):
        responses[f"repro_reexec{round_no}_plan"] = responses["repro_script_plan"]
    llm = FakeLLM(responses, texts={"report": REPORT})
    ui = SilentUI()
    deps = build_deps(settings, llm, ui=ui)

    result = run_task("Test Paper 0", deps)

    assert result.status == "completed"
    assert result.state["exec_round"] == 3
    assert result.state["verdict"]["verdict"] == "unsuccessful"
    assert len(result.state["verdict_history"]) == 3
    assert len(ui.find("replan")) == 2
    assert any("last permitted attempt" in message for message in ui.find("warn"))


def test_crashed_run_is_still_verified_and_reported(settings) -> None:
    responses = happy_responses(script=crashing_script())
    llm = FakeLLM(responses, texts={"report": REPORT})
    deps = build_deps(settings, llm)

    result = run_task("Test Paper 0", deps)

    assert result.status == "completed"
    assert result.state["exec_result"]["ok"] is False
    assert result.state["verdict"]["verdict"] == "successful"  # the scripted verifier
    assert result.published.get("result_dir")


def test_over_budget_plan_is_shrunk_before_the_full_run(settings) -> None:
    responses = happy_responses(script=script_with_estimate(5000.0, steps=5000))
    responses["repro_adjust1_plan"] = plan_payload(
        approach="same experiment with far fewer steps",
        params={"steps": 12},
        metrics=["accuracy"],
        risks=[],
    )
    llm = FakeLLM(
        responses,
        texts={"report": REPORT, "repro_adjust1_code1": script_with_estimate(0.2, steps=12)},
    )
    ui = SilentUI()
    deps = build_deps(settings, llm, ui=ui)

    result = run_task("Test Paper 0", deps)

    assert result.status == "completed", result.message
    assert result.state["adjust_round"] == 1
    assert ("json", "repro_adjust1_plan", "ExecPlan", "main") in llm.calls
    assert ("text", "repro_adjust1_code1", "text", "main") in llm.calls
    assert result.state["preflight"]["needs_adjustment"] is False
    assert any("longer than the" in message for message in ui.find("warn"))
    run_dir = Path(result.state["run_dir"])
    assert "0.2" in (run_dir / "code" / "repro.py").read_text(encoding="utf-8")


def test_adjustment_budget_exhausts_and_releases_anyway(settings) -> None:
    responses = happy_responses(script=script_with_estimate(5000.0, steps=5000))
    texts = {"report": REPORT}
    for round_no in (1, 2):
        responses[f"repro_adjust{round_no}_plan"] = plan_payload(
            approach="still too slow",
            params={"steps": 4000},
            metrics=["accuracy"],
            figures=[],
            risks=[],
        )
        texts[f"repro_adjust{round_no}_code1"] = script_with_estimate(4000.0, steps=4000)
    llm = FakeLLM(responses, texts=texts)
    ui = SilentUI()
    deps = build_deps(settings, llm, ui=ui)

    result = run_task("Test Paper 0", deps)

    assert result.status == "completed", result.message
    assert result.state["adjust_round"] == 2  # max_adjust_rounds in the test settings
    assert result_dirs(settings)
    assert any("no adjustment budget" in message for message in ui.find("warn"))


def test_stage4_llm_failure_names_the_call_and_logs_it(settings) -> None:
    """A broken reply while writing repro.py must not surface as a bare failure."""
    from essay_agent.errors import LLMReplyError

    failure = LLMReplyError(
        "could not parse JSON from reply (truncated)",
        reason="truncated",
        schema="ReproScript",
        label="repro_script",
        reply_chars=8123,
        finish_reason="length",
    )
    llm = FakeLLM(
        happy_responses(),
        texts={"report": REPORT},
        error_labels={"repro_script_plan": failure, "repro_script": failure},
    )
    deps = build_deps(settings, llm, ui=SilentUI())

    result = run_task("Test Paper 0", deps)

    assert result.status == "failed"
    assert "code generation failed" in result.message
    assert "reason=truncated" in result.message
    assert "finish_reason=length" in result.message
    run_log = (Path(result.state["run_dir"]) / "logs" / "run.log").read_text(encoding="utf-8")
    assert "stage 4 failed" in run_log
    assert result_dirs(settings) == []
