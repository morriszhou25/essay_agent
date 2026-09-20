"""A failed run carries its cause to the verifier and to the console.

Measured on run-20260920-062827-3e6074: every execution round died from a native abort that only
ever printed on stderr, while stage 4.1 was handed the stdout tail alone. The verifier therefore
spent three rounds asking for a diagnosis it could not make ("The log tail contains only the step-0
progress event with no error traceback, so the failure cause is not diagnosable").

Stage 4.1 now receives both streams, labelled, and a failed run prints the child's own error line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from essay_agent.console import SilentUI
from essay_agent.nodes.execute import log_tail, make_run_node
from essay_agent.nodes.verify_execute import make_verify_execute_node
from essay_agent.runtime.runner import RunOutcome
from essay_agent.schemas.dialogue import ExecuteVerdict
from tests.conftest import make_settings
from tests.fakes import FakeLLM, FakeRunner
from tests.pipeline import build_deps, card_payload, plan_payload

OMP_ERROR = (
    "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.\n"
    "OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the\n"
    "program."
)
STDOUT_TAIL = (
    '{"event": "progress", "step": 0, "total": 9}\n{"event": "log", "message": "start c04"}'
)
STDOUT_LABEL = "[stdout]"
STDERR_LABEL = "[stderr]"


def crashed(script: Path, *, stderr: str = OMP_ERROR) -> RunOutcome:
    return RunOutcome(
        ok=False,
        exit_code=3,
        duration=3.6,
        metrics={},
        figures=[],
        stdout_tail=STDOUT_TAIL,
        stderr_tail=stderr,
        script=str(script),
        error="script exited with code 3",
    )


def result_payload(outcome: RunOutcome) -> dict[str, Any]:
    """The shape stage 4 leaves in ``state['exec_result']``."""
    return {
        "ok": outcome.ok,
        "exit_code": outcome.exit_code,
        "duration": outcome.duration,
        "timed_out": outcome.timed_out,
        "cancelled": outcome.cancelled,
        "metrics": outcome.metrics,
        "figures": outcome.figures,
        "stdout_tail": outcome.stdout_tail,
        "stderr_tail": outcome.stderr_tail,
        "error": outcome.error,
        "script": outcome.script,
    }


def state_for(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": "run-stderr-test",
        "slug": "stderr",
        "query": "attention is all you need",
        "paper": {"id": "p1", "title": "Attention Is All You Need"},
        "cards": [card_payload("c04")],
        "coverage": [
            {
                "key": "3.1",
                "section": "3.1 Encoder and Decoder Stacks",
                "cards": ["c04"],
                "status": "carded",
                "detail": "",
                "passes": 1,
            }
        ],
        "plan_round": 0,
        "exec_round": 1,
        "exec_plan": plan_payload(cards_covered=["c04"]),
        "exec_result": result,
    }


def test_a_failed_run_prints_the_childs_error_line(tmp_path):
    script = tmp_path / "repro.py"
    script.write_text("raise SystemExit(3)\n", encoding="utf-8")
    ui = SilentUI()
    deps = build_deps(make_settings(tmp_path), FakeLLM(), ui=ui)
    deps.runner = FakeRunner(crashed(script))
    payload = (make_run_node(deps)(state_for({}))).get("exec_result") or {}
    shown = [message for _kind, message in ui.events if "libiomp5md" in message]
    assert shown, [message for _kind, message in ui.events]
    assert "libiomp5md" in payload["stderr_tail"]


def test_the_cause_survives_publication(tmp_path):
    script = tmp_path / "repro.py"
    script.write_text("raise SystemExit(3)\n", encoding="utf-8")
    settings = make_settings(tmp_path)
    deps = build_deps(settings, FakeLLM())
    deps.runner = FakeRunner(crashed(script))
    state = state_for({})
    make_run_node(deps)(state)
    published = deps.workspace(state).publish(settings.resolved_paths(), title="Test Paper")
    written = Path(published.code_dir) / "run_outcome.json"
    assert "libiomp5md" in written.read_text(encoding="utf-8")


def verify_once(result: dict[str, Any], tmp_path: Path) -> tuple[str, dict[str, Any], Path]:
    """Run the real stage-4.1 node: the prompt it sent, the state, and the run workspace."""
    captured: dict[str, str] = {}

    def answer(_system: str, user: str) -> dict[str, Any]:
        captured["user"] = user
        return {
            "verdict": "unsuccessful",
            "rationale": "the script crashed before writing any metric",
            "problems": ["c04: the script crashed at step 0"],
            "evidence": [],
            "per_card": {"c04": "untested"},
        }

    deps = build_deps(make_settings(tmp_path), FakeLLM({ExecuteVerdict: answer}))
    state = state_for(result)
    returned = make_verify_execute_node(deps)(state)
    state.update(returned)
    return captured.get("user") or "", state, deps.workspace(state).dir


def capture_verify(result: dict[str, Any], tmp_path: Path) -> str:
    return verify_once(result, tmp_path)[0]


def test_the_verifier_receives_both_streams(tmp_path):
    prompt = capture_verify(result_payload(crashed(tmp_path / "repro.py")), tmp_path)
    assert "libiomp5md" in prompt
    assert STDOUT_LABEL in prompt and STDERR_LABEL in prompt
    assert "start c04" in prompt


def test_a_clean_run_keeps_the_previous_prompt_shape(tmp_path):
    clean = RunOutcome(
        ok=True,
        exit_code=0,
        duration=2.0,
        metrics={"c04_layers": 6.0},
        figures=["fig_c04.png"],
        stdout_tail=STDOUT_TAIL,
        stderr_tail="",
        script=str(tmp_path / "repro.py"),
    )
    prompt = capture_verify(result_payload(clean), tmp_path)
    assert STDERR_LABEL not in prompt
    assert "start c04" in prompt


def test_log_tail_is_bounded_and_labelled():
    combined = log_tail({"stdout_tail": "x" * 9000, "stderr_tail": "y" * 9000})
    assert len(combined) < 9000
    assert STDOUT_LABEL in combined and STDERR_LABEL in combined
    assert combined.index(STDOUT_LABEL) < combined.index(STDERR_LABEL)
    assert log_tail({"stdout_tail": "abc"}) == "abc"
    assert log_tail({}) == ""


def test_the_verdict_is_still_recorded(tmp_path):
    """Widening the log tail must not change how a verdict is stored."""
    _prompt, state, workspace = verify_once(
        result_payload(crashed(tmp_path / "repro.py")), tmp_path
    )
    verdict = state.get("verdict") or {}
    assert verdict["verdict"] == "unsuccessful"
    stored = json.loads((workspace / "code" / "verdict_round1.json").read_text(encoding="utf-8"))
    assert stored["verdict"] == "unsuccessful"
    assert stored["problems"] == ["c04: the script crashed at step 0"]
    assert stored["unhanded"] == []
