"""The two dataset-path bugs from run-20260920-042507, stated as checks.

P1 - a stale ``HF_ENDPOINT`` is never taken back.  ``dataset_env()`` only *wrote* the
mirror when the primary host failed; when huggingface.co answered it returned ``{}``, so
an inherited ``HF_ENDPOINT=https://hf-mirror.com`` travelled into ``repro.py`` and broke
real downloads (``FileMetadataError: Distant resource does not seem to be on
huggingface.co``).  The verified route has to be written out completely - a category that
was verified *without* a mirror says so.

P2 - a prose requirement string was read as a Hugging Face repository id, the probe asked
for a repository that cannot exist, and the HTTP 401 that came back was reported as
"requires credentials".  An id that is not a plausible id is an *unresolved reference*,
not a gated dataset.

    python tests/experiment_p1p2_semantics.py
    python tests/experiment_p1p2_semantics.py --live

Not collected by pytest (does not start with ``test_``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from essay_agent.config import RuntimeSettings
from essay_agent.connectivity import (
    DATASET_SOURCE,
    HF_ENDPOINT_VAR,
    PROXY_ENV_VARS,
    Attempt,
    ConnectivityReport,
    Endpoint,
    EndpointReport,
)
from essay_agent.runtime.runner import ScriptRunner
from essay_agent.tools.dataset_probe import classify_target, probe_target

PROXY = "http://127.0.0.1:7897"
MIRROR = "https://hf-mirror.example"
ROOT = Path(__file__).resolve().parents[1]
HF_HOME = ROOT / ".essay_agent" / "experiments" / "hf_home_child"
CHANNELS: dict[str, dict[str, str | None] | None] = {
    "os-proxy": {"http": PROXY, "https": PROXY},
    "direct": {"http": None, "https": None},
}
# Verbatim from cards/probes.json of the run that exposed the bug.
PROSE = [
    "a sparse expert architecture with a tunable number of experts / sparsity level",
    "compute for timing/FLOP measurement",
]
REAL = [
    ("hf:imdb", "hf_dataset", "imdb"),
    ("allenai/c4", "hf_dataset", "allenai/c4"),
    ("https://huggingface.co/datasets/squad", "hf_dataset", "squad"),
]
FAILURES: list[str] = []

# Prints the endpoint the dataset libraries would really talk to, then streams two rows.
# Streaming on purpose: a cached download needs symlinks, which a restricted shell lacks.
CHILD = """
import json, os
from huggingface_hub import constants

out = {
    "endpoint": constants.ENDPOINT,
    "hf_endpoint": os.environ.get("HF_ENDPOINT") or "-",
    "proxies": sorted(k for k in os.environ if k.lower().endswith("_proxy")),
}
try:
    from datasets import load_dataset

    rows = list(load_dataset("nyu-mll/glue", "sst2", split="validation", streaming=True).take(2))
    out["rows"] = len(rows)
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
print(json.dumps(out))
"""


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'ok  ' if condition else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def dataset_report(*results: tuple[str, bool, str, bool]) -> ConnectivityReport:
    """``(name, fallback, channel, ok)`` per dataset endpoint."""
    reports = [
        EndpointReport(
            Endpoint(
                name,
                DATASET_SOURCE,
                f"https://example.test/{name}",
                fallback=fallback,
                env={HF_ENDPOINT_VAR: MIRROR} if fallback else {},
            ),
            [Attempt(channel, ok, "HTTP 200" if ok else "blocked", 0.01)],
        )
        for name, fallback, channel, ok in results
    ]
    return ConnectivityReport(reports=reports, default_channel="os-proxy", channels=CHANNELS)


class _Response:
    def __init__(self, status: int = 200) -> None:
        self.status_code = status


class _Session:
    """A session that answers everything with one status and records the URLs."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.calls: list[str] = []

    def head(self, url: str, **kwargs):
        self.calls.append(url)
        return _Response(self.status)

    def get(self, url: str, **kwargs):
        return self.head(url, **kwargs)


def script_runner() -> ScriptRunner:
    return ScriptRunner(RuntimeSettings(), python_executable=sys.executable, sink_factory=None)


def live_checks() -> None:
    """Probe the real dataset hosts, then let a real child process report its endpoint."""
    print("--- live: the verdict, seen from inside the child process ---")
    from essay_agent import connectivity
    from essay_agent.config import Settings

    endpoints = [
        item for item in connectivity.endpoints_for(Settings()) if item.category == DATASET_SOURCE
    ]
    report = connectivity.run_checks(endpoints, budget_seconds=20.0)
    lines = report.lines()
    print(f"  {lines[0]}")
    for line in lines[1:]:
        if "dataset" in line:
            print(f"  {line}")
    env = report.dataset_env()
    print(f"  dataset_env(): {env}")

    HF_HOME.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [sys.executable, "-c", CHILD],
        # The cache has to live inside the workspace, or a restricted shell cannot write it.
        env={**script_runner()._build_env(env), "HF_HOME": str(HF_HOME)},
        capture_output=True,
        text=True,
        timeout=600,
    )
    payload = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith("{")), None
    )
    if payload is None:
        check("the child reports its endpoint", False, completed.stderr.strip()[-300:])
        return
    seen = json.loads(payload)
    print(f"  child: {seen}")
    primary_ok = report.state(DATASET_SOURCE) == "ok"
    if primary_ok:
        check(
            "the child asks the primary host, not an endpoint nobody probed",
            seen["endpoint"] == "https://huggingface.co" and seen["hf_endpoint"] == "-",
            f"endpoint={seen['endpoint']} HF_ENDPOINT={seen['hf_endpoint']}",
        )
    check(
        "two real rows still stream into the child",
        seen.get("rows") == 2,
        seen.get("error", f"rows={seen.get('rows')} via {seen['endpoint']}"),
    )

    # The other direction, on the same real child: only the mirror answered.
    mirror_only = ConnectivityReport(
        reports=[
            EndpointReport(endpoints[0], [Attempt("os-proxy", False, "blocked", 0.01)]),
            *[
                EndpointReport(item, [Attempt("os-proxy", True, "HTTP 200", 0.01)])
                for item in endpoints
                if item.fallback
            ],
        ],
        default_channel="os-proxy",
        channels={"os-proxy": {"http": proxies, "https": proxies}}
        if (proxies := _proxy_url())
        else {},
    )
    print(f"  mirror-only dataset_env(): {mirror_only.dataset_env()}")
    mirrored = subprocess.run(
        [sys.executable, "-c", CHILD],
        env={**script_runner()._build_env(mirror_only.dataset_env()), "HF_HOME": str(HF_HOME)},
        capture_output=True,
        text=True,
        timeout=600,
    )
    payload = next(
        (line for line in reversed(mirrored.stdout.splitlines()) if line.startswith("{")), None
    )
    if payload is None:
        check("the mirror route reaches the child too", False, mirrored.stderr.strip()[-300:])
        return
    seen = json.loads(payload)
    print(f"  child: {seen}")
    check(
        "the mirror is named when it is what answered",
        seen["endpoint"] == seen["hf_endpoint"] and seen["hf_endpoint"] != "-",
        f"endpoint={seen['endpoint']} HF_ENDPOINT={seen['hf_endpoint']}",
    )
    check(
        "two real rows stream through the mirror as well",
        seen.get("rows") == 2,
        seen.get("error", f"rows={seen.get('rows')} via {seen['endpoint']}"),
    )


def _proxy_url() -> str | None:
    from essay_agent import connectivity

    proxy = connectivity.os_proxy()
    return proxy if not proxy or "://" in proxy else f"http://{proxy}"


def p1_checks() -> None:
    print("--- P1: the verified dataset route reaches the child, mirror included ---")
    healthy = dataset_report(
        ("huggingface", False, "os-proxy", True), ("hf-mirror", True, "os-proxy", True)
    ).dataset_env()
    check(
        "a healthy primary host asks for HF_ENDPOINT to be removed",
        healthy.get(HF_ENDPOINT_VAR, "missing") is None,
        f"dataset_env={healthy}",
    )
    check(
        "the verified proxy still travels with it",
        {healthy.get(name) for name in PROXY_ENV_VARS} == {PROXY},
    )

    os.environ[HF_ENDPOINT_VAR] = "https://hf-mirror.com"
    child = script_runner()._build_env(healthy)
    check(
        "the child process no longer inherits the mirror",
        HF_ENDPOINT_VAR not in child,
        f"HF_ENDPOINT={child.get(HF_ENDPOINT_VAR)!r}",
    )

    mirror_only = dataset_report(
        ("huggingface", False, "os-proxy", False), ("hf-mirror", True, "os-proxy", True)
    ).dataset_env()
    check(
        "a working mirror is still named when the primary host is down",
        mirror_only.get(HF_ENDPOINT_VAR) == MIRROR,
        f"dataset_env={mirror_only}",
    )

    nothing = dataset_report(
        ("huggingface", False, "os-proxy", False), ("hf-mirror", True, "os-proxy", False)
    ).dataset_env()
    check("nothing verified means nothing is rewritten", nothing == {}, f"dataset_env={nothing}")


def p2_checks() -> None:
    print("--- P2: a prose reference is not a Hugging Face repository id ---")
    for prose in PROSE:
        kind, _ = classify_target(prose)
        check(f"classified as unresolved, not as a repo: {prose[:40]!r}", kind == "unknown", kind)

        session = _Session(401)
        result = probe_target(prose, session=session)
        check(
            "a 401 on an impossible id is not 'requires credentials'",
            result.requires_credentials is False and result.status() == "unknown",
            f"status={result.status()!r} detail={result.detail!r}",
        )

        prefixed = probe_target(f"hf:{prose}", session=_Session(401))
        check(
            "the explicit hf: prefix cannot smuggle prose past it either",
            prefixed.requires_credentials is False and prefixed.status() == "unknown",
            f"status={prefixed.status()!r}",
        )

    for target, kind, resolved in REAL:
        got_kind, got_id = classify_target(target)
        check(
            f"{target!r} is still a repo id",
            (got_kind, got_id) == (kind, resolved),
            f"{got_kind}/{got_id}",
        )

    session = _Session(200)
    result = probe_target("hf:imdb", session=session)
    check(
        "a real id is still probed over the API",
        result.reachable is True and any("api/datasets/imdb" in url for url in session.calls),
    )
    gated = probe_target("hf:imdb", session=_Session(403))
    check(
        "a real gated repo still reads as requiring credentials", gated.requires_credentials is True
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="also probe and start a real child")
    args = parser.parse_args()

    p1_checks()
    p2_checks()
    if args.live:
        live_checks()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
