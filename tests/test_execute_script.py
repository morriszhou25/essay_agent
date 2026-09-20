"""Stage 4 - the plan is small JSON, the script is plain text, the check must bite."""

from __future__ import annotations

import pytest

from essay_agent.errors import LLMError, LLMReplyError
from essay_agent.nodes.execute import _generate_script, check_code, extract_code
from tests.fakes import FakeLLM
from tests.pipeline import build_deps, plan_payload, script_with_estimate

GOOD = script_with_estimate(0.2)
BROKEN = "def main():\n    return 0\n"  # far too short and no --preflight


def code_reply(script: str, seen: list[tuple[str, str]]) -> object:
    """A scripted text reply that records the exact (system, user) it was asked with."""

    def reply(system: str, user: str) -> str:
        seen.append((system, user))
        return script

    return reply


# ---------------------------------------------------------------- code plumbing
def test_extract_code_strips_prose_and_fences() -> None:
    reply = f"Sure, here it is:\n\n```python\n{GOOD}\n```\nHope that helps!"
    assert check_code(extract_code(reply)) is None


def test_extract_code_accepts_bare_python() -> None:
    assert extract_code(GOOD) == GOOD.strip()


# ------------------------------------------------------------- the split flow
def test_stage4_asks_for_a_plan_then_the_code_as_text(settings) -> None:
    llm = FakeLLM({"repro_script_plan": plan_payload()}, texts={"repro_script_code1": GOOD})
    deps = build_deps(settings, llm)

    script = _generate_script(deps, "CODE SYSTEM", "cards context", label="repro_script")

    assert check_code(script.code) is None
    assert script.params == {"steps": 12, "seed": 0}  # params come from the plan
    assert ("json", "repro_script_plan", "ExecPlan", "main") in llm.calls
    assert ("text", "repro_script_code1", "text", "main") in llm.calls


def test_code_prompt_carries_the_context_the_plan_and_the_output_budget(settings) -> None:
    seen: list[tuple[str, str]] = []
    llm = FakeLLM(
        {"repro_script_plan": plan_payload()}, texts={"repro_script_code1": code_reply(GOOD, seen)}
    )
    deps = build_deps(settings, llm)

    _generate_script(deps, "CODE SYSTEM", "cards context", label="repro_script")

    system, user = seen[0]
    assert system == "CODE SYSTEM"
    assert "cards context" in user
    assert "train a small model on the iris dataset" in user  # the decided plan
    assert "OUTPUT BUDGET" in user
    assert "raw python text" in user


def test_code_call_uses_the_configured_code_max_tokens(settings) -> None:
    settings.llm.code_max_tokens = 4096
    llm = FakeLLM({"repro_script_plan": plan_payload()}, texts={"repro_script_code1": GOOD})
    deps = build_deps(settings, llm)

    _generate_script(deps, "CODE SYSTEM", "cards context", label="repro_script")

    assert llm.max_tokens_seen["repro_script_code1"] == 4096


# ---------------------------------------------------------------- the repairs
def test_a_broken_reply_is_repaired_with_the_static_error(settings) -> None:
    seen: list[tuple[str, str]] = []
    llm = FakeLLM(
        {"repro_script_plan": plan_payload()},
        texts={
            "repro_script_code1": code_reply(BROKEN, seen),
            "repro_script_code2": code_reply(GOOD, seen),
        },
    )
    ui = deps_ui(settings, llm)

    script = _generate_script(ui[0], "CODE SYSTEM", "cards context", label="repro_script")

    assert check_code(script.code) is None
    assert "Mandatory fix" in seen[1][1]
    assert "too short" in seen[1][1]
    assert any("failed a static check" in message for message in ui[1].find("warn"))


def test_a_cut_off_reply_is_told_to_write_a_shorter_file(settings) -> None:
    seen: list[tuple[str, str]] = []
    llm = FakeLLM(
        {"repro_script_plan": plan_payload()},
        texts={
            "repro_script_code1": code_reply(GOOD + "\nprint(1", seen),
            "repro_script_code2": code_reply(GOOD, seen),
        },
    )

    script = _generate_script(
        build_deps(settings, llm), "CODE SYSTEM", "cards context", label="repro_script"
    )

    assert check_code(script.code) is None
    assert "CUT OFF" in seen[1][1]
    assert "shorter" in seen[1][1] or "drop optional checks" in seen[1][1]


# --------------------------------------------------------------- the fallback
def test_two_broken_replies_fall_back_to_one_structured_call(settings) -> None:
    responses = {
        "repro_script_plan": plan_payload(),
        "repro_script": {"plan": plan_payload(), "code": GOOD, "params": {"steps": 12}},
    }
    llm = FakeLLM(
        responses,
        texts={"repro_script_code1": BROKEN, "repro_script_code2": BROKEN},
    )

    script = _generate_script(
        build_deps(settings, llm), "CODE SYSTEM", "cards context", label="repro_script"
    )

    assert check_code(script.code) is None
    assert ("json", "repro_script", "ReproScript", "main") in llm.calls


def test_a_broken_fallback_script_raises_instead_of_running(settings) -> None:
    """The old loop returned broken code silently; that must not happen again."""
    responses = {
        "repro_script_plan": plan_payload(),
        "repro_script": {"plan": plan_payload(), "code": BROKEN, "params": {}},
    }
    llm = FakeLLM(responses, texts={"repro_script_code1": BROKEN, "repro_script_code2": BROKEN})

    with pytest.raises(LLMError, match="failed a static check"):
        _generate_script(
            build_deps(settings, llm), "CODE SYSTEM", "cards context", label="repro_script"
        )


def test_a_failed_plan_falls_back_to_the_structured_call(settings) -> None:
    responses = {"repro_script": {"plan": plan_payload(), "code": GOOD, "params": {}}}
    llm = FakeLLM(
        responses,
        texts={},
        error_labels={"repro_script_plan": LLMReplyError("no plan", reason="empty")},
    )

    script = _generate_script(
        build_deps(settings, llm), "CODE SYSTEM", "cards context", label="repro_script"
    )

    assert check_code(script.code) is None
    assert not any(call[0] == "text" for call in llm.calls)


def deps_ui(settings, llm) -> tuple[object, object]:
    """A Deps plus the UI it was built with, so a test can read its warnings."""
    from essay_agent.console import SilentUI

    ui = SilentUI()
    return build_deps(settings, llm, ui=ui), ui
