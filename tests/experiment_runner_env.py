"""Stage-4 scripts must run in a known-good environment, not just an inherited one.

Measured on run-20260920-062827-3e6074 (Attention Is All You Need): all three execution rounds died
in ~3.6s with exit code 3, no metrics and no figures, because of a native abort inside the child:

    OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

That is an environment fault, not a script defect: PyTorch ships one copy of the Intel OpenMP
runtime and another library lazily loads a second copy, so the runtime aborts the process instead of
risking wrong numbers. The very same script, run with ``KMP_DUPLICATE_LIB_OK=TRUE``, finished with
exit code 0 and 7 of its 9 checks in 3.7s.

The shipped rule: every generated script is started in a child environment that carries our
documented workarounds on top of the plumbing defaults, so a reproduction does not depend on how the
user's machine happens to be configured. The flag is an Intel-documented *unsafe* workaround ("may
cause crashes or silently produce incorrect results"), so it is recorded rather than silent: the
runner exposes the workarounds it injects and the run outcome carries them.

    python tests/experiment_runner_env.py
"""

from __future__ import annotations

# ruff: noqa: E402 -- the package is imported after the src/ path shim below.
import json
import os
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
from essay_agent.runtime.runner import ScriptRunner
from tests.conftest import make_settings
from tests.fakes import FakeLLM
from tests.pipeline import build_deps, plan_payload

OMP_ERROR = (
    "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
)
WORKAROUND = "KMP_DUPLICATE_LIB_OK"
REAL_RUN = ROOT / ".essay_agent" / "runs" / "run-20260920-062827-3e6074"

# The child reports the environment it was actually given; it also speaks the progress protocol so
# the node under test sees a well-formed run.
PROBE = """
import argparse, json, os, sys

NAMES = ["KMP_DUPLICATE_LIB_OK", "MPLBACKEND", "PYTHONIOENCODING", "PYTHONUNBUFFERED",
         "PYTHONDONTWRITEBYTECODE", "ESSAY_AGENT_DEVICE", "ESSAY_AGENT_ALLOW_SYNTHETIC_DATA"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()
    print(json.dumps({"event": "log", "message": "environment probed"}), flush=True)
    print(json.dumps({"event": "progress", "step": 1, "total": 1}), flush=True)
    with open(os.path.join(args.out, "env.json"), "w", encoding="utf-8") as handle:
        json.dump({name: os.environ.get(name) for name in NAMES}, handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def write_probe(directory: Path) -> Path:
    script = directory / "env_probe.py"
    script.write_text(PROBE, encoding="utf-8")
    return script


def probe_env(directory: Path, *, env: dict[str, Any] | None = None) -> dict[str, Any]:
    """Start the probe in a real child process and return the environment the child saw."""
    script = write_probe(directory)
    out = directory / "probe_out"
    out.mkdir(parents=True, exist_ok=True)
    runner = ScriptRunner(make_settings(directory).runtime, python_executable=sys.executable)
    # The parent must not be the reason the child sees the flag.
    saved = os.environ.pop(WORKAROUND, None)
    try:
        outcome = runner.run(script, cwd=out, timeout=120, args=["--out", str(out)], env=env)
    finally:
        if saved is not None:
            os.environ[WORKAROUND] = saved
    seen: dict[str, Any] = {}
    if (out / "env.json").is_file():
        seen = json.loads((out / "env.json").read_text(encoding="utf-8"))
    return {"seen": seen, "outcome": outcome, "runner": runner, "out": out, "script": script}


def part1_the_measured_failure() -> None:
    print("\n[1] the measured failure: the child never got past its first check")
    outcome_path = REAL_RUN / "code" / "run_outcome.json"
    if not outcome_path.is_file():
        print("      (run artifacts are not present in this checkout; skipping the archaeology)")
        return
    payload = json.loads(outcome_path.read_text(encoding="utf-8"))
    stderr = payload.get("stderr_tail") or ""
    print(f"      exit={payload.get('exit_code')} metrics={payload.get('metrics')}")
    print(f"      stderr: {stderr.splitlines()[0][:96] if stderr else '(empty)'}")
    check(
        "[1a] the only diagnosis of that run went to stderr",
        OMP_ERROR.split(":")[0] in stderr and "libiomp5md" in stderr,
        "the verifier never saw it",
    )
    check(
        "[1b] the run produced no metrics and no figures",
        payload.get("metrics") == {} and payload.get("figures") == [],
        "nothing to grade",
    )


def part2_the_child_environment() -> None:
    print("\n[2] the child environment is known-good by construction")
    with tempfile.TemporaryDirectory(prefix="ea_runner_env_") as raw:
        directory = Path(raw)
        first = probe_env(directory)
        seen = first["seen"]
        print(
            f"      parent had {WORKAROUND}={os.environ.get(WORKAROUND)!r} (removed for the test)"
        )
        print(f"      child saw {WORKAROUND}={seen.get(WORKAROUND)!r}")
        check(
            "[2a] the child receives the OpenMP workaround without anyone setting it",
            seen.get(WORKAROUND) == "TRUE",
            f"exit={first['outcome'].exit_code}",
        )
        check(
            "[2b] the plumbing defaults are still there",
            seen.get("MPLBACKEND") == "Agg"
            and seen.get("PYTHONIOENCODING") == "utf-8"
            and seen.get("PYTHONUNBUFFERED") == "1",
            f"MPLBACKEND={seen.get('MPLBACKEND')}",
        )
        removed_dir = directory / "removed"
        removed_dir.mkdir()
        second = probe_env(removed_dir, env={WORKAROUND: None})
        check(
            "[2c] an explicit None still removes it (the escape hatch survives)",
            second["seen"].get(WORKAROUND) is None,
            f"child saw {second['seen'].get(WORKAROUND)!r}",
        )
        check(
            "[2d] the workaround changes nothing about the honesty switches",
            seen.get("ESSAY_AGENT_ALLOW_SYNTHETIC_DATA") == "false",
            f"allow_synthetic_data={seen.get('ESSAY_AGENT_ALLOW_SYNTHETIC_DATA')!r}",
        )


def part3_recorded_not_silent() -> None:
    print("\n[3] the workaround is recorded, not silent")
    with tempfile.TemporaryDirectory(prefix="ea_runner_env_node_") as raw:
        directory = Path(raw)
        script = write_probe(directory)
        settings = make_settings(directory)
        ui = SilentUI()
        deps = build_deps(settings, FakeLLM(), ui=ui)
        state: dict[str, Any] = {
            "run_id": "run-experiment-runner-env",
            "slug": "runner-env",
            "exec_round": 1,
            "exec_plan": plan_payload(),
            "exec_code_path": str(script),
        }
        saved = os.environ.pop(WORKAROUND, None)
        try:
            returned = make_run_node(deps)(state)
        finally:
            if saved is not None:
                os.environ[WORKAROUND] = saved
        payload = returned.get("exec_result") or {}
        fixes = payload.get("env_fixes") or {}
        print(f"      recorded fixes: {json.dumps(fixes)}")
        check(
            "[3a] the runner reports which workarounds it injected",
            WORKAROUND in getattr(deps.runner, "env_fixes", {}),
            f"keys={sorted(getattr(deps.runner, 'env_fixes', {}))}",
        )
        check(
            "[3b] the run outcome carries the environment it used",
            isinstance(fixes.get(WORKAROUND), str) and bool(fixes.get(WORKAROUND)),
            f"env_fixes={fixes}",
        )
        written = deps.workspace(state).dir / "code" / "run_outcome.json"
        text = written.read_text(encoding="utf-8") if written.is_file() else ""
        check(
            "[3c] it survives into code/run_outcome.json (and thus the published repro/)",
            WORKAROUND in text,
            str(written),
        )


def main() -> int:
    print("stage-4 child environment: a known-good environment for every generated script")
    part1_the_measured_failure()
    part2_the_child_environment()
    part3_recorded_not_silent()
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
