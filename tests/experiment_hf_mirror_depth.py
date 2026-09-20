"""Is "HTTP 200 on the listing API" enough to call a Hugging Face mirror usable?

The probe calls ``GET {mirror}/api/datasets?limit=1`` and counts a 200 as "the mirror
works".  A machine that had ``HF_ENDPOINT=https://hf-mirror.com`` set globally still
failed real ``datasets.load_dataset()`` downloads with ``FileMetadataError``, so either
the mirror is broken below the listing endpoint or the failure had another cause.

This asks both hosts the questions a download actually asks and prints the raw answers,
so the mirror check can be deepened only if something really differs.

    python tests/experiment_hf_mirror_depth.py

Live network required (the machine's proxy is picked up normally). Not collected by
pytest (does not start with ``test_``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from essay_agent import connectivity

REPOS = ["nyu-mll/glue", "stanfordnlp/imdb"]
HOSTS = ["https://huggingface.co", "https://hf-mirror.com"]
# What the probe asks today, then the metadata API, then the resolve path a download uses.
PATHS = ["/api/datasets?limit=1", "/api/datasets/{repo}", "/datasets/{repo}/resolve/main/README.md"]
HEADERS = ("x-repo-commit", "x-linked-etag", "etag", "location", "content-type")


def hub_checks(proxies: dict[str, str] | None) -> None:
    """Ask the Hub *client* the way a download does, once per endpoint.

    ``HfApi`` is imported here on purpose: ``constants.ENDPOINT`` is read from
    ``HF_ENDPOINT`` at import time, so the variable has to go first or the "primary"
    arm of this test would silently test the mirror.
    """
    os.environ.pop("HF_ENDPOINT", None)
    for name, value in (proxies or {}).items():
        os.environ[f"{name.upper()}_PROXY"] = value
    from huggingface_hub import HfApi

    print("\n=== the Hub client, per endpoint (no cache, no download) ===")
    for endpoint in ("https://huggingface.co", "https://hf-mirror.com"):
        api = HfApi(endpoint=endpoint)
        print(f"\n--- HfApi(endpoint={endpoint}) ---")
        calls = (
            ("dataset_info", lambda api=api: api.dataset_info("nyu-mll/glue")),
            (
                "list_repo_files",
                lambda api=api: api.list_repo_files("nyu-mll/glue", repo_type="dataset")[:4],
            ),
        )
        for label, call in calls:
            try:
                print(f"  ok   {label}: {str(call())[:110]}")
            except Exception as exc:
                print(f"  ERR  {label}: {type(exc).__name__}: {str(exc)[:160]}")


def ask(
    session: requests.Session, url: str, proxies: dict[str, str] | None, method: str = "GET"
) -> str | None:
    """Print what one URL answers; return a one-word verdict for the summary."""
    try:
        response = session.request(
            method, url, timeout=20, stream=True, allow_redirects=True, proxies=proxies
        )
    except Exception as exc:
        print(f"  ERR  {url}")
        print(f"       {type(exc).__name__}: {str(exc)[:110]}")
        return None
    headers = {key.lower(): value for key, value in response.headers.items()}
    print(f"  {method:4} {response.status_code}  {url}")
    for key in HEADERS:
        if key in headers:
            print(f"         {key}: {headers[key][:90]}")
    detail: str | None = None
    if method == "HEAD":
        # huggingface_hub reads the commit hash out of a HEAD response: no header, no download.
        print(f"         commit headers: {headers.get('x-repo-commit', '-')}")
        response.close()
        return "head"
    try:
        body = response.raw.read(240, decode_content=True)
        text = body.decode("utf-8", "replace").strip()
        kind = "json" if text.startswith(("{", "[")) else ("html" if "<" in text[:20] else "text")
        print(f"         body[{kind}]: {text[:110]!r}")
        detail = kind
    except Exception as exc:
        print(f"         body unreadable: {type(exc).__name__}: {str(exc)[:60]}")
    finally:
        response.close()
    return detail


def redirect_of(session: requests.Session, url: str, proxies: dict[str, str] | None) -> str:
    """The status and ``Location`` of a HEAD that is *not* followed - what hf_hub sees."""
    try:
        response = session.head(url, timeout=20, allow_redirects=False, proxies=proxies)
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc)[:70]}"
    return f"HTTP {response.status_code} -> {response.headers.get('location') or '-'}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", default=None, help="proxy URL; default = the OS proxy")
    args = parser.parse_args()

    # The pipeline's own route: the Windows registry proxy, not whatever a shell exports.
    proxy = args.proxy or connectivity.os_proxy()
    proxies = connectivity.proxies_for(proxy)
    print(f"route: os-proxy {proxy!r} -> proxies={proxies}")
    session = requests.Session()
    session.trust_env = False  # this experiment pins its route, exactly like the clients do
    seen: dict[tuple[str, str], str | None] = {}
    for path in PATHS:
        for repo in REPOS if "{repo}" in path else [""]:
            for host in HOSTS:
                url = host + path.format(repo=repo)
                print(f"\n--- {path} ---")
                seen[(host, path)] = ask(session, url, proxies)
                if path.endswith("README.md"):
                    # How huggingface_hub asks for the metadata that carries the commit hash.
                    ask(session, url, proxies, method="HEAD")

    print("\n=== summary: does the *listing* endpoint predict the *resolve* path? ===")
    for host in HOSTS:
        listing = seen.get((host, PATHS[0]))
        meta = seen.get((host, PATHS[1]))
        resolve = seen.get((host, PATHS[2]))
        print(f"  {host}: listing={listing} metadata={meta} resolve={resolve}")
        if listing == "json" and (meta != "json" or resolve != "text"):
            print(f"    -> the listing endpoint over-reports: {host} answers the listing only")
        print(
            "    resolve HEAD, redirects NOT followed: "
            f"{redirect_of(session, host + PATHS[2].format(repo=REPOS[0]), proxies)}"
        )
    print(
        "\nA 200 listing paired with a non-JSON metadata body is what the probe misses. A\n"
        "cross-host redirect on the resolve path is worse: `requests` follows it, but\n"
        "`huggingface_hub` reads the commit hash out of that first answer and refuses a\n"
        "'distant resource', which is why a mirror-only machine cannot download through it."
    )
    hub_checks(proxies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
