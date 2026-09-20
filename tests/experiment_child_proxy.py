"""Live experiment: the route the probe verified must reach the generated script.

The stage-3 probe decides which channel a run's traffic really uses (on a machine behind a
VPN: the OS proxy) and pins the agent's *own* clients to it. ``repro.py`` is a separate
process, and today it receives ``ScriptRunner``'s environment: the parent's environment
plus ``HF_ENDPOINT``. So the script has to guess the route itself - and its guess depends
on the HTTP library (``requests`` may find the OS registry, ``httpx`` reads only the
environment) and on whatever proxy variables happen to be inherited, dead ones included.

This measures the difference instead of arguing it: the real probe runs, both environments
are built, and a real child process reports which proxies it resolved and whether it could
reach HuggingFace.

    python tests/experiment_child_proxy.py                        # probe the real routes
    python tests/experiment_child_proxy.py --proxy http://127.0.0.1:7897
    python tests/experiment_child_proxy.py --skip-live            # no probe, --proxy only

``legacy`` is what ``dataset_env()`` reports today; ``fixed`` applies ``route_env()`` on
top of a bare runner environment, so a regression shows up as the two columns disagreeing.
Before the fix both columns sent the child to the dead proxy; now the verified route wins.
Not collected by pytest (not ``test_*``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from essay_agent import connectivity
from essay_agent.config import Settings
from essay_agent.connectivity import DATASET_SOURCE, Route, proxies_for
from essay_agent.runtime.progress import NullSink
from essay_agent.runtime.runner import ScriptRunner

HF_URL = "https://huggingface.co/api/datasets?limit=1"
DEAD_PROXY = "http://127.0.0.1:9"
PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
FAILURES: list[str] = []

# The child mimics a generated repro.py: a plain requests call with trust_env left on.
CHILD = """
import json, os
import requests

info = {
    "proxy_vars": {k: v for k, v in os.environ.items() if "proxy" in k.lower()},
    "resolved": requests.utils.getproxies(),
}
try:
    info["http"] = requests.get("__URL__", timeout=8).status_code
except Exception as exc:
    info["error"] = f"{type(exc).__name__}: {str(exc)[:110]}"
print(json.dumps(info))
""".replace("__URL__", HF_URL)


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def route_env(route: Route | None) -> dict[str, str | None]:
    """Proposed: turn a verified route into child-process proxy variables.

    Three cases, because ``Route`` has three meanings: nothing was verified (leave the
    environment alone), ``proxies=None`` ("let the client decide" - the environment *is*
    the verified route, so also leave it alone), and a concrete route (write it out, or
    remove every proxy variable when the route is a plain direct connection).
    """
    if route is None or route.proxies is None:
        return {}
    url = next((value for value in route.proxies.values() if value), None)
    if url is None:
        # A direct route must *remove* an inherited proxy, not merely ignore it.
        return dict.fromkeys(PROXY_VARS)
    env: dict[str, str | None] = dict.fromkeys(PROXY_VARS, url)
    if (bundle := connectivity.ca_bundle()) is not None:
        env["REQUESTS_CA_BUNDLE"] = bundle
        env["CURL_CA_BUNDLE"] = bundle
    return env


def apply_extra(base: dict[str, str], extra: dict[str, str | None]) -> dict[str, str]:
    """Proposed: ``_build_env`` semantics - ``None`` removes the variable."""
    env = dict(base)
    for key, value in extra.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    return env


def run_child(env: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-c", CHILD],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        return {"error": f"child produced no JSON: {completed.stderr.strip()[-200:]}"}
    return json.loads(lines[-1])


def outcome(info: dict[str, Any]) -> str:
    if "http" in info:
        return f"HTTP {info['http']}"
    return str(info.get("error", "unknown"))


def http_proxy_vars(info: dict[str, Any]) -> dict[str, str]:
    """Only the variables ``requests``/``httpx``/``urllib`` actually read.

    ``GIT_HTTP_PROXY`` and friends are inherited too, but no HTTP client consults them,
    so they are reported and then ignored.
    """
    return {
        key: value for key, value in (info.get("proxy_vars") or {}).items() if key in PROXY_VARS
    }


def column(
    title: str,
    runner: ScriptRunner,
    report: connectivity.ConnectivityReport | None,
    route: Route | None,
    inherited: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the child twice - today's environment, and the proposed one."""
    saved = dict(os.environ)
    try:
        os.environ.update(inherited)
        legacy_env = runner._build_env(extra=report.dataset_env() if report else {})
        base = runner._build_env(extra={})
    finally:
        os.environ.clear()
        os.environ.update(saved)
    fixed_env = apply_extra(base, route_env(route))

    print(f"  {title}")
    print(f"    inherited: {inherited or '-'}")
    print(
        f"    verified route: {route.name if route else 'none'} ({route.proxies if route else '-'})"
    )
    legacy = run_child(legacy_env)
    fixed = run_child(fixed_env)
    print(f"    legacy env proxies: {legacy.get('proxy_vars') or '-'}")
    print(f"    legacy resolved    : {legacy.get('resolved')} -> {outcome(legacy)}")
    print(f"    fixed  env proxies : {fixed.get('proxy_vars') or '-'}")
    print(f"    fixed  resolved    : {fixed.get('resolved')} -> {outcome(fixed)}")
    return legacy, fixed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", default=None, help="proxy to fall back on / force")
    parser.add_argument("--skip-live", action="store_true", help="do not run the real probe")
    parser.add_argument("--budget", type=float, default=20.0)
    args = parser.parse_args()

    settings = Settings()
    runner = ScriptRunner(
        settings.runtime,
        python_executable=sys.executable,
        sink_factory=lambda label, total=None: NullSink(),
    )

    report: connectivity.ConnectivityReport | None = None
    route: Route | None = None
    if not args.skip_live:
        endpoints = [
            item for item in connectivity.endpoints_for(settings) if item.category == DATASET_SOURCE
        ]
        print(
            f"[0] live probe of the dataset hosts ({len(endpoints)} endpoint(s), "
            f"budget {args.budget:.0f}s)"
        )
        report = connectivity.run_checks(
            endpoints,
            explicit_proxy=settings.http_proxy,
            budget_seconds=args.budget,
        )
        for line in report.lines():
            print(f"  {line}")
        route = report.route(DATASET_SOURCE)
        print(f"  -> dataset route: {route.name if route else 'none'}")
        print(f"  -> dataset_env(): {report.dataset_env()}")

    if route is None and args.proxy:
        route = Route("forced-proxy", proxies_for(args.proxy))
        print(f"  -> no verified route; forcing {args.proxy}")

    print("\n[1] a script started with the environment it gets today")
    legacy, fixed = column("the parent environment as it is", runner, report, route, {})
    print("\n[2] the same, but a dead proxy is inherited on top")
    legacy_dead, fixed_dead = column(
        "parent environment plus a dead proxy",
        runner,
        report,
        route,
        dict.fromkeys(("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"), DEAD_PROXY),
    )

    print("\nverdict")
    if route is None:
        print("  SKIP  no dataset route was verified (VPN off? pass --proxy)")
        check("nothing is injected when nothing was verified", legacy == fixed)
    elif route.proxies is None:
        # "Let the client decide" *is* the verified route; touching the env would break it.
        check("the environment route is left alone", fixed == legacy)
    else:
        expected = str(next(value for value in route.proxies.values() if value))
        dead_vars = http_proxy_vars(fixed_dead)
        fixed_vars = http_proxy_vars(fixed)
        check(
            "the report and the explicit route agree",
            http_proxy_vars(legacy) == fixed_vars,
            f"report: {http_proxy_vars(legacy)} vs route: {fixed_vars}",
        )
        check(
            "the dead inherited proxy never reaches the script",
            DEAD_PROXY not in set(dead_vars.values()),
            f"fixed env proxies: {dead_vars}",
        )
        check(
            "the verified route reaches the script",
            set(fixed_vars.values()) == {expected},
            f"fixed env proxies: {fixed_vars}",
        )
        check("the script reaches HuggingFace", fixed.get("http") == 200, outcome(fixed))
        check(
            "it still reaches it with a dead proxy inherited",
            fixed_dead.get("http") == 200,
            outcome(fixed_dead),
        )
        if legacy.get("http") != 200:
            print(f"  note  today's environment failed: {outcome(legacy)}")
        if legacy_dead.get("http") != 200:
            print(f"  note  today's environment + dead proxy failed: {outcome(legacy_dead)}")

    print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
