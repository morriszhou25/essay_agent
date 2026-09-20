"""A failed run must carry its *cause* to the verifier and to the console.

Measured on run-20260920-062827-3e6074: every round died from a native abort that only ever printed
on stderr, while stage 4.1 was handed the stdout tail alone. The verifier therefore spent its three
rounds asking for a diagnosis it could not make:

    "The log tail contains only the step-0 progress event with no error traceback, so the failure
     cause is not diagnosable from the run output; the script should surface the exception."

The shipped rule: the verification prompt receives both streams, labelled, and the console shows the
child's last error line when a run fails. The run outcome keeps carrying the raw tails, so the cause
also survives publication into ``repro/``.

    python tests/experiment_stderr_visibility.py
"""

from __future__ import annotations

# ruff: noqa: E402 -- the package is imported after the src/ path shim below.
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(ROOT / "src"), str(ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from essay_agent.console import SilentUI
from essay_agent.nodes.execute import make_run_node
from essay_agent.nodes.verify_execute import make_verify_execute_node
from essay_agent.runtime.runner import RunOutcome
from essay_agent.schemas.dialogue import ExecuteVerdict
from tests.conftest import make_settings
from tests.fakes import FakeLLM
from tests.pipeline import build_deps, card_payload, plan_payload

OMP_ERROR = (
    "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.\n"
    "OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the\n"
    "program. That is dangerous, since it can degrade performance or cause incorrect results."
)
STDOUT_TAIL = (
    '{"event": "log", "message": "total checks=9"}\n'
    '{"event": "progress", "step": 0, "total": 9, "epoch": 0}\n'
    '{"event": "log", "message": "start c04"}'
)
STDERR_LABEL = "[stderr]"
STDOUT_LABEL = "[stdout]"
REAL_RUN = ROOT / ".essay_agent" / "runs" / "run-20260920-062827-3e6074"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


class FakeRunner:
    """Hands back one canned outcome instead of starting a process."""

    def __init__(self, outcome: RunOutcome) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []
        self.env_fixes: dict[str, str] = {}

    def run(self, script: Path, **kwargs: Any) -> RunOutcome:
        self.calls.append({"script": str(script), **kwargs})
        return self.outcome


def crashed_outcome(script: Path, *, stderr: str = OMP_ERROR) -> RunOutcome:
    return RunOutcome(
        ok=False,
        exit_code=3,
        duration=3.6,
        timed_out=False,
        cancelled=False,
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
    cards = [card_payload("c04")]
    return {
        "run_id": "run-experiment-stderr",
        "slug": "stderr",
        "query": "attention is all you need",
        "paper": {"id": "p1", "title": "Attention Is All You Need"},
        "cards": cards,
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


def part1_the_measured_failure() -> None:
    print("\n[1] the measured failure: the cause only ever reached stderr")
    verdict_path = REAL_RUN / "code" / "verdict_round1.json"
    outcome_path = REAL_RUN / "code" / "run_outcome.json"
    if not (verdict_path.is_file() and outcome_path.is_file()):
        print("      (run artifacts are not present in this checkout; skipping the archaeology)")
        return
    payload = json.loads(outcome_path.read_text(encoding="utf-8"))
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    problems = " ".join(verdict.get("problems") or [])
    print(f"      run outcome said: {payload.get('error')!s}")
    print(f"      the verifier said: {problems[:110]}...")
    check(
        "[1a] the run carried the abort on stderr",
        "libiomp5md" in (payload.get("stderr_tail") or ""),
        "stderr_tail is captured by the runner",
    )
    check(
        "[1b] and the verifier asked for a diagnosis it had no input for",
        "diagnosable" in problems or "traceback" in problems,
        "the prompt only ever carried stdout",
    )


def part2_the_console() -> None:
    print("\n[2] the console shows the child's own error line")
    with tempfile.TemporaryDirectory(prefix="ea_stderr_console_") as raw:
        directory = Path(raw)
        script = directory / "repro.py"
        script.write_text("raise SystemExit(3)\n", encoding="utf-8")
        settings = make_settings(directory)
        ui = SilentUI()
        deps = build_deps(settings, FakeLLM(), ui=ui)
        deps.runner = FakeRunner(crashed_outcome(script))
        state = state_for({})
        returned = make_run_node(deps)(state)
        payload = returned.get("exec_result") or {}
        shown = [message for _kind, message in ui.events if "libiomp5md" in message]
        print(f"      console said: {shown[0][:100] if shown else '(nothing about the abort)'}")
        check(
            "[2a] the failure cause is printed, not just the exit code",
            bool(shown),
            f"{len(ui.events)} console event(s)",
        )
        check(
            "[2b] the raw tail is still kept in the outcome payload",
            "libiomp5md" in (payload.get("stderr_tail") or ""),
            "stderr_tail survives into code/run_outcome.json",
        )
        written = deps.workspace(state).dir / "code" / "run_outcome.json"
        check(
            "[2c] and it lands on disk",
            written.is_file() and "libiomp5md" in written.read_text(encoding="utf-8"),
            str(written),
        )
        published = deps.workspace(state).publish(settings.resolved_paths(), title="Test Paper")
        published_file = Path(published.code_dir) / "run_outcome.json"
        check(
            "[2d] and it survives publication into the repro/ tree",
            published_file.is_file() and "libiomp5md" in published_file.read_text(encoding="utf-8"),
            str(published_file),
        )


def run_verify(result: dict[str, Any], directory: Path) -> tuple[dict[str, str], dict[str, Any]]:
    captured: dict[str, str] = {}

    def capture(_system: str, user: str) -> dict[str, Any]:
        captured["user"] = user
        return {
            "verdict": "unsuccessful",
            "rationale": "the script crashed before writing any metric",
            "problems": ["c04: the script crashed at step 0, so the metric was never written"],
            "evidence": [],
            "per_card": {"c04": "untested"},
        }

    llm = FakeLLM({ExecuteVerdict: capture})
    deps = build_deps(make_settings(directory), llm, ui=SilentUI())
    returned = make_verify_execute_node(deps)(state_for(result))
    return captured, returned


def part3_the_verifier_prompt() -> None:
    print("\n[3] the verifier receives the error output")
    with tempfile.TemporaryDirectory(prefix="ea_stderr_verify_") as raw:
        directory = Path(raw)
        outcome = crashed_outcome(directory / "repro.py")
        captured, returned = run_verify(result_payload(outcome), directory)
        prompt = captured.get("user") or ""
        print(
            f"      prompt: {len(prompt):,} chars; stderr first line present: "
            f"{'OMP: Error #15' in prompt}"
        )
        check(
            "[3a] the abort reaches the verification prompt",
            "libiomp5md" in prompt and "OMP: Error #15" in prompt,
            "this is what round 1-3 were missing",
        )
        check(
            "[3b] the two streams are labelled so the verifier knows what it reads",
            STDERR_LABEL in prompt and STDOUT_LABEL in prompt,
            f"{STDOUT_LABEL} / {STDERR_LABEL}",
        )
        check(
            "[3c] the stdout tail is still there",
            "start c04" in prompt,
            "no regression for runs that fail without stderr",
        )
        with tempfile.TemporaryDirectory(prefix="ea_stderr_clean_") as clean:
            clean_dir = Path(clean)
            plain = RunOutcome(
                ok=True,
                exit_code=0,
                duration=2.0,
                metrics={"c04_layers": 6.0},
                figures=["fig_c04.png"],
                stdout_tail=STDOUT_TAIL,
                stderr_tail="",
                script=str(clean_dir / "repro.py"),
            )
            clean_captured, _ = run_verify(result_payload(plain), clean_dir)
            clean_prompt = clean_captured.get("user") or ""
        check(
            "[3d] a clean run keeps the prompt shape it had before",
            STDERR_LABEL not in clean_prompt and "start c04" in clean_prompt,
            "no empty error section",
        )
        check(
            "[3e] the verifier still records its verdict normally",
            (returned.get("verdict") or {}).get("verdict") == "unsuccessful",
            str((returned.get("verdict") or {}).get("verdict")),
        )


def main() -> int:
    print("stage-4 failure visibility: the cause reaches the verifier and the console")
    part1_the_measured_failure()
    part2_the_console()
    part3_the_verifier_prompt()
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
