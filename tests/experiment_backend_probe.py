"""Diagnostic experiment: why do the paper backends fail?

Read-only on purpose - it imports the real search/proxy code but changes nothing. It
replays one search request per backend over every channel a real run can take
(``direct`` / the OS proxy / the dead env proxy), and reports DNS, TCP, status, timing
and rate-limit headers separately, so "the backend is down" can be told apart from
"our route is wrong".

    python tests/experiment_backend_probe.py
    python tests/experiment_backend_probe.py --attempts 3 --timeout 60

Not collected by pytest (the filename is not ``test_*``).
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests

from essay_agent.schemas.paper import PaperSearchQuery
from essay_agent.tools.paper_search import arxiv_params, s2_params

OS_PROXY = "http://127.0.0.1:7897"
DEAD_PROXY = "http://127.0.0.1:9"
CHANNELS: dict[str, dict[str, str | None] | None] = {
    "direct": {"http": None, "https": None},
    "os-proxy": {"http": OS_PROXY, "https": OS_PROXY},
    "dead-env-proxy": {"http": DEAD_PROXY, "https": DEAD_PROXY},
}

QUERY = PaperSearchQuery(title="attention is all you need", max_results=8)


def targets() -> list[tuple[str, str, dict[str, Any]]]:
    """(label, url, params) - the exact requests the searcher makes, plus controls."""
    return [
        ("arxiv (https, as used)", "https://export.arxiv.org/api/query", arxiv_params(QUERY, 8)),
        ("arxiv (http)", "http://export.arxiv.org/api/query", arxiv_params(QUERY, 8)),
        (
            "semanticscholar",
            "https://api.semanticscholar.org/graph/v1/paper/search",
            s2_params(QUERY, 8),
        ),
        ("openalex (control)", "https://api.openalex.org/works?per-page=1", {}),
        ("huggingface (control)", "https://huggingface.co/api/datasets?limit=1", {}),
    ]


def resolve(host: str) -> tuple[str, float]:
    started = time.perf_counter()
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        address = sorted({item[4][0] for item in infos})[0]
        return address, time.perf_counter() - started
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc)[:60]}", time.perf_counter() - started


def tcp_connect(host: str, timeout: float) -> str:
    started = time.perf_counter()
    try:
        with socket.create_connection((host, 443), timeout=timeout):
            return f"ok in {time.perf_counter() - started:.2f}s"
    except Exception as exc:
        return f"{type(exc).__name__} in {time.perf_counter() - started:.2f}s"


def probe(url: str, params: dict[str, Any], proxies: dict[str, str | None], timeout: float) -> str:
    session = requests.Session()
    session.trust_env = False  # a real client pinned by the probe behaves this way
    session.proxies = dict(proxies)
    session.headers.update({"User-Agent": "essay-agent/0.1 (diagnostic)"})
    started = time.perf_counter()
    try:
        response = session.get(url, params=params, timeout=timeout, stream=True)
        elapsed = time.perf_counter() - started
        body = next(response.iter_content(2048), b"") or b""
        headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() in {"retry-after", "content-type", "x-ratelimit-limit", "server"}
        }
        return (
            f"HTTP {response.status_code} in {elapsed:.2f}s, {len(body)}B, {headers} "
            f"| {body[:70]!r}"
        )
    except Exception as exc:
        return f"{type(exc).__name__} in {time.perf_counter() - started:.2f}s: {str(exc)[:110]}"
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--attempts", type=int, default=1)
    args = parser.parse_args()

    print("env proxies:", {k: os.environ.get(k) for k in ("HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")})
    print(f"channel under test: os-proxy = {OS_PROXY}, dead = {DEAD_PROXY}\n")

    print("== layer 1: DNS + TCP (no HTTP, so no proxy involved) ==")
    for host in ("export.arxiv.org", "api.semanticscholar.org", "api.openalex.org"):
        address, seconds = resolve(host)
        print(f"  {host:26} dns {address} in {seconds:.2f}s | tcp {tcp_connect(host, 8.0)}")

    print("\n== layer 2: the real requests, per channel ==")
    for label, url, params in targets():
        for channel, proxies in CHANNELS.items():
            for attempt in range(1, args.attempts + 1):
                detail = probe(url, params, proxies, args.timeout)
                print(f"  {label:24} {channel:15} #{attempt}: {detail}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
