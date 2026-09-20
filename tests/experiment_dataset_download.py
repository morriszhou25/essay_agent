"""Can a generated repro.py really download a dataset? Measured, not assumed.

``experiment_child_proxy.py`` proves the *route* reaches the child. This goes one step
further and runs a real ``datasets.load_dataset()`` in that child, with a fresh HF cache
per scenario so a warm cache cannot hide a broken route, plus the Hub API check for the
paper's own dataset.

    python tests/experiment_dataset_download.py
    python tests/experiment_dataset_download.py --skip-live --proxy http://127.0.0.1:7897

Not collected by pytest (not ``test_*``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from essay_agent import connectivity
from essay_agent.config import Settings
from essay_agent.connectivity import DATASET_SOURCE, Route, proxies_for
from essay_agent.runtime.progress import NullSink
from essay_agent.runtime.runner import ScriptRunner

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".essay_agent" / "experiments" / "download"
DEAD_PROXY = "http://127.0.0.1:9"
# What a machine behind the GFW typically has exported: an endpoint no probe verified.
INHERITED_MIRROR = connectivity.HF_MIRROR_URL
# A public file in a public dataset: the path a download asks the mirror for.
MIRROR_FILE = "/datasets/nyu-mll/glue/resolve/main/README.md"
# Small, canonical, public: nothing gated, nothing huge.
LADDER = [
    ["nyu-mll/glue", "sst2", "validation[:5]"],
    ["glue", "sst2", "validation[:5]"],
    ["stanfordnlp/imdb", None, "test[:5]"],
]
PAPER_DATASETS = ["wmt/wmt14", "wmt14", "wmt/wmt16"]
FAILURES: list[str] = []

CHILD_DOWNLOAD = """
import json, os, time

spec = json.loads(os.environ["EA_LADDER"])
out = {
    "proxies": {
        k: v for k, v in os.environ.items() if k.lower() in ("http_proxy", "https_proxy", "all_proxy")
    },
    "hf_endpoint": os.environ.get("HF_ENDPOINT") or "-",
    "hf_home": os.environ.get("HF_HOME") or "-",
}
from datasets import load_dataset

for repo, config, split in spec:
    started = time.perf_counter()
    try:
        dataset = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
        out.update(
            {
                "repo": repo,
                "config": config,
                "rows": len(dataset),
                "columns": list(dataset.column_names),
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
        break
    except Exception as exc:
        out.setdefault("tried", []).append(f"{repo}: {type(exc).__name__}: {str(exc)[:220]}")
json.dump(out, open(os.environ["EA_OUT"], "w", encoding="utf-8"))
"""

CHILD_API = """
import json, requests

out = {}
for repo in json.loads(__import__("os").environ["EA_REPOS"]):
    try:
        response = requests.get(f"https://huggingface.co/api/datasets/{repo}", timeout=20)
        body = response.json() if response.status_code < 400 else {}
        out[repo] = {
            "status": response.status_code,
            "gated": body.get("gated"),
            "private": body.get("private"),
            "downloads": body.get("downloads"),
        }
    except Exception as exc:
        out[repo] = {"error": f"{type(exc).__name__}: {str(exc)[:140]}"}
print(json.dumps(out))
"""


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def note(label: str, detail: str = "") -> None:
    print(f"  NOTE  {label}{(' - ' + detail) if detail else ''}")


def child_env(
    runner: ScriptRunner,
    report: connectivity.ConnectivityReport | None,
    route: Route | None,
    scenario: str,
    inherited: dict[str, str],
    drop: tuple[str, ...] = (),
) -> dict[str, str]:
    """The environment the runner would start the script with, plus this scenario's cache."""
    saved = dict(os.environ)
    try:
        os.environ.update(inherited)
        base = runner._build_env(extra=report.dataset_env() if report else {})
    finally:
        os.environ.clear()
        os.environ.update(saved)
    if route is not None and report is None:
        base = {**base, **{k: v for k, v in connectivity.route_env(route).items() if v is not None}}
    home = WORK / f"hf_home_{scenario}"
    home.mkdir(parents=True, exist_ok=True)
    for name in drop:
        base.pop(name, None)
    base.update(
        {"HF_HOME": str(home), "EA_LADDER": json.dumps(LADDER), "EA_OUT": str(home / "out.json")}
    )
    return base


def run_child(
    env: dict[str, str], program: str, *, extra: dict[str, str] | None = None
) -> dict[str, Any]:
    child = {**env, **(extra or {})}
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=child,
        capture_output=True,
        text=True,
        timeout=600,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    if lines:
        return json.loads(lines[-1])
    out_file = Path(env.get("EA_OUT", ""))
    if out_file.is_file():
        return json.loads(out_file.read_text(encoding="utf-8"))
    return {"error": f"no result: {completed.stderr.strip()[-400:]}"}


def describe_download(info: dict[str, Any]) -> str:
    if "error" in info:
        return f"FAILED - {info['error']}"
    if "repo" in info:
        return (
            f"{info['repo']}"
            f"{'/' + info['config'] if info.get('config') else ''} -> "
            f"{info['rows']} row(s), {info['seconds']}s"
        )
    return f"all candidates failed: {info.get('tried')}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--skip-live", action="store_true")
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
        print(f"[0] live probe of the dataset hosts ({len(endpoints)} endpoint(s))")
        report = connectivity.run_checks(
            endpoints, explicit_proxy=settings.http_proxy, budget_seconds=20.0
        )
        for line in report.lines():
            print(f"  {line}")
        route = report.route(DATASET_SOURCE)
    if route is None and args.proxy:
        route = Route("forced-proxy", proxies_for(args.proxy))
    print(
        f"  -> dataset route: {route.name if route else 'none'} ({route.proxies if route else '-'})"
    )
    print(f"  -> dataset_env(): {report.dataset_env() if report else route_env_preview(route)}")

    print("\n[1] the runner's environment: an inherited HF_ENDPOINT is taken back")
    as_is = child_env(runner, report, route, "as_is", {"HF_ENDPOINT": INHERITED_MIRROR})
    as_is_run = run_child(as_is, CHILD_DOWNLOAD)
    print(f"  env proxies: {as_is_run.get('proxies', '-')}")
    print(f"  hf_endpoint: {as_is_run.get('hf_endpoint')}")
    print(f"  result     : {describe_download(as_is_run)}")
    if report is None:
        # ``--skip-live`` never builds dataset_env(), so there is nothing to take back.
        note("no probe ran, so dataset_env() was not applied to this child")
    else:
        check(
            "the inherited endpoint is gone and the download still works",
            "repo" in as_is_run and as_is_run.get("hf_endpoint") == "-",
            describe_download(as_is_run),
        )

    print("\n[2] the mirror as the *chosen* route: what it does with a file request")
    status, location = mirror_file_request(INHERITED_MIRROR)
    print(f"  HEAD {INHERITED_MIRROR}{MIRROR_FILE} -> {status} {location}")
    clean = child_env(runner, None, route, "clean", {"HF_ENDPOINT": INHERITED_MIRROR})
    started = time.perf_counter()
    download = run_child(clean, CHILD_DOWNLOAD)
    print(f"  env proxies: {download.get('proxies', '-')}")
    print(f"  hf_endpoint: {download.get('hf_endpoint')} | hf_home: {download.get('hf_home')}")
    print(
        f"  result     : {describe_download(download)} ({time.perf_counter() - started:.1f}s wall)"
    )
    check(
        "the mirror redirects file metadata to huggingface.co - the Hub client refuses that",
        status in {301, 302, 307, 308} and "huggingface.co" in location,
        f"HTTP {status} -> {location or '-'}",
    )
    if "repo" not in download:
        note(
            "consequence: a machine that can only reach the mirror cannot download with "
            "huggingface_hub, however healthy the mirror's listing API looks",
            f"the child went to {download.get('hf_endpoint')}",
        )

    print("\n[3] the same, with a dead proxy inherited on top")
    dead = child_env(
        runner,
        report,
        route,
        "dead",
        dict.fromkeys(("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"), DEAD_PROXY),
        drop=("HF_ENDPOINT",),
    )
    download_dead = run_child(dead, CHILD_DOWNLOAD)
    print(f"  env proxies: {download_dead.get('proxies', '-')}")
    print(f"  result     : {describe_download(download_dead)}")
    check(
        "it still downloads with a dead proxy inherited",
        "repo" in download_dead,
        describe_download(download_dead),
    )

    print("\n[4] the paper's own datasets (Hub API, no token configured)")
    hubs = run_child(clean, CHILD_API, extra={"EA_REPOS": json.dumps(PAPER_DATASETS)})
    for repo, info in hubs.items():
        print(f"  {repo}: {info}")
    gated = [repo for repo, info in hubs.items() if isinstance(info, dict) and info.get("gated")]
    if gated:
        note(
            "these are gated, so a token is required - the route is not the problem",
            ", ".join(gated),
        )
    elif all(isinstance(info, dict) and info.get("status") == 404 for info in hubs.values()):
        note("none of those ids exist on the Hub; the paper names its dataset by hand")

    if not args.skip_live and report is not None:
        print("\n[5] the report the run would have written")
        print(f"  {json.dumps(report.dataset_env(), sort_keys=True)}")

    print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


def route_env_preview(route: Route | None) -> dict[str, str | None]:
    return connectivity.route_env(route)


def mirror_file_request(mirror: str) -> tuple[int | None, str]:
    """Ask the mirror for a real file *without* following the redirect.

    ``huggingface_hub`` reads the commit hash out of this answer and refuses to download
    from anywhere else, so a redirect here - which plain ``requests`` would happily
    follow - is the difference between "the mirror works" and "the download fails".
    """
    import requests

    try:
        response = requests.head(
            mirror.rstrip("/") + MIRROR_FILE,
            timeout=20,
            allow_redirects=False,
            proxies=connectivity.proxies_for(connectivity.os_proxy()),
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:80]}"
    return response.status_code, response.headers.get("location", "")


if __name__ == "__main__":
    raise SystemExit(main())
