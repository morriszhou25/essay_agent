"""Every generated script runs in a known-good child environment.

Measured on run-20260920-062827-3e6074 (``Attention Is All You Need``): all three execution rounds
died at step 0 with exit code 3, no metrics and no figures, because a lazily loaded library brought
a second copy of Intel's OpenMP runtime into the child and the runtime aborted the process
(``OMP: Error #15``). The very same script finished with exit code 0 and 7 of its 9 checks once
``KMP_DUPLICATE_LIB_OK`` was set.

``ScriptRunner`` now starts every child with the documented workarounds on top of the plumbing
defaults, and reports what it injected (``env_fixes``) instead of applying it silently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from essay_agent.nodes.execute import make_run_node
from essay_agent.runtime.runner import CHILD_ENV_DEFAULTS, ENV_WORKAROUNDS, ScriptRunner
from tests.conftest import make_settings
from tests.fakes import FakeLLM
from tests.pipeline import build_deps, plan_payload

WORKAROUND = "KMP_DUPLICATE_LIB_OK"

# The child reports the environment it was actually handed.
PROBE = """
import argparse, json, os, sys

NAMES = ["KMP_DUPLICATE_LIB_OK", "MPLBACKEND", "PYTHONIOENCODING", "PYTHONUNBUFFERED",
         "ESSAY_AGENT_DEVICE", "ESSAY_AGENT_ALLOW_SYNTHETIC_DATA"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()
    with open(os.path.join(args.out, "env.json"), "w", encoding="utf-8") as handle:
        json.dump({name: os.environ.get(name) for name in NAMES}, handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""


def run_probe(tmp_path: Path, *, env: dict[str, Any] | None = None) -> dict[str, Any]:
    """Start the probe in a real child process and return the environment it saw."""
    script = tmp_path / "env_probe.py"
    script.write_text(PROBE, encoding="utf-8")
    out = tmp_path / "probe_out"
    out.mkdir(exist_ok=True)
    runner = ScriptRunner(make_settings(tmp_path).runtime, python_executable=sys.executable)
    outcome = runner.run(script, cwd=out, timeout=120, args=["--out", str(out)], env=env)
    assert outcome.ok, outcome.describe()
    return json.loads((out / "env.json").read_text(encoding="utf-8"))


def test_the_workaround_and_the_defaults_reach_the_child(tmp_path, monkeypatch):
    monkeypatch.delenv(WORKAROUND, raising=False)
    seen = run_probe(tmp_path)
    assert seen[WORKAROUND] == "TRUE"
    assert seen["MPLBACKEND"] == "Agg"
    assert seen["PYTHONIOENCODING"] == "utf-8"
    assert seen["PYTHONUNBUFFERED"] == "1"


def test_the_honesty_switches_are_not_touched(tmp_path, monkeypatch):
    monkeypatch.delenv("ESSAY_AGENT_ALLOW_SYNTHETIC_DATA", raising=False)
    seen = run_probe(tmp_path)
    assert seen["ESSAY_AGENT_ALLOW_SYNTHETIC_DATA"] == "false"


def test_our_defaults_win_over_an_inherited_value(tmp_path, monkeypatch):
    monkeypatch.setenv(WORKAROUND, "FALSE")
    assert run_probe(tmp_path)[WORKAROUND] == "TRUE"


def test_a_default_can_be_taken_back_explicitly(tmp_path, monkeypatch):
    monkeypatch.delenv(WORKAROUND, raising=False)
    assert run_probe(tmp_path, env={WORKAROUND: None})[WORKAROUND] is None


def test_env_fixes_names_each_workaround_and_its_reason(tmp_path):
    fixes = ScriptRunner(make_settings(tmp_path).runtime).env_fixes
    assert set(fixes) == set(ENV_WORKAROUNDS)
    assert all(reason.strip() for reason in fixes.values())
    assert set(fixes) <= set(CHILD_ENV_DEFAULTS)
    assert "OMP" in fixes[WORKAROUND]


def test_the_run_outcome_records_the_environment_it_used(tmp_path, monkeypatch):
    monkeypatch.delenv(WORKAROUND, raising=False)
    script = tmp_path / "env_probe.py"
    script.write_text(PROBE, encoding="utf-8")
    deps = build_deps(make_settings(tmp_path), FakeLLM())
    state: dict[str, Any] = {
        "run_id": "run-env-test",
        "slug": "env",
        "exec_round": 1,
        "exec_plan": plan_payload(),
        "exec_code_path": str(script),
    }
    payload = (make_run_node(deps)(state)).get("exec_result") or {}
    assert WORKAROUND in (payload.get("env_fixes") or {})
    written = deps.workspace(state).dir / "code" / "run_outcome.json"
    assert WORKAROUND in written.read_text(encoding="utf-8")
