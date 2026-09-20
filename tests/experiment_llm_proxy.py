"""Experiment harness for issue #1: pinning the model client to a verified route.

The unit tests (``tests/test_llm_proxy.py``) prove the plumbing offline. This harness
answers the question they cannot: on a machine whose ``HTTP_PROXY`` is dead but whose
OS proxy works, does a *pinned* langchain client actually reach the model API while the
old behaviour does not? The ``legacy_*`` functions below keep a copy of the pre-fix
behaviour so the two columns can be compared in one run.

    python tests/experiment_llm_proxy.py            # offline checks only
    python tests/experiment_llm_proxy.py --live     # + a real call (needs DEEPSEEK_API_KEY)

Not collected by pytest (the filename is not ``test_*``).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx

from essay_agent.config import Settings
from essay_agent.connectivity import (
    MODEL_API,
    ConnectivityProbe,
    Route,
    build_http_client,
    ca_bundle,
    endpoints_for,
)

DEAD_PROXY = "http://127.0.0.1:9"
PROXY = "http://127.0.0.1:7897"
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def proxy_of(client: httpx.Client) -> str | None:
    """The proxy a client really routes through, read back from its transport."""

    def text(value: bytes | str) -> str:
        return value.decode() if isinstance(value, bytes) else value

    for transport in client._mounts.values():
        url = getattr(getattr(transport, "_pool", None), "_proxy_url", None)
        if url is not None:
            return f"{text(url.scheme)}://{text(url.host)}:{url.port}"
    return None


# --------------------------------------------------- the behaviour being replaced
def legacy_build_http_client(route: Route | None) -> None:
    """Before the fix: the model client was never pinned, so ``trust_env`` won."""
    return None


# --------------------------------------------------------------- offline checks
def offline_checks() -> None:
    print("\n[1] route -> client (the real helper, next to the old behaviour)")
    check("nothing verified -> no client (old and new agree)", build_http_client(None) is None)
    check(
        "env route -> no client: the environment was just proven good",
        build_http_client(Route("env-proxy", None)) is None,
    )
    direct = build_http_client(Route("direct", {"http": None, "https": None}))
    check(
        "direct route -> a client that ignores the environment",
        direct is not None and direct.trust_env is False and proxy_of(direct) is None,
    )
    pinned = build_http_client(Route("os-proxy", {"http": PROXY, "https": PROXY}))
    check(
        "os-proxy route -> pinned to that proxy",
        pinned is not None and proxy_of(pinned) == PROXY,
        f"got {proxy_of(pinned) if pinned else None}",
    )
    check(
        "old behaviour: the same route produced no client at all",
        legacy_build_http_client(Route("os-proxy", {"http": PROXY, "https": PROXY})) is None,
    )
    for client in (direct, pinned):
        if client is not None:
            client.close()

    print("\n[2] the TLS bundle a corporate user needs")
    bundle = Path(os.environ.get("TEMP", ".")) / "ea-prototype-ca.pem"
    bundle.write_text("-----BEGIN CERTIFICATE-----\nnot a real cert\n", encoding="utf-8")
    check(
        "REQUESTS_CA_BUNDLE is picked up",
        ca_bundle({"REQUESTS_CA_BUNDLE": str(bundle)}) == str(bundle),
    )
    check(
        "CURL_CA_BUNDLE is the fallback",
        ca_bundle({"CURL_CA_BUNDLE": str(bundle)}) == str(bundle),
    )
    check("nothing set -> None", ca_bundle({}) is None)


# -------------------------------------------------------------------- live check
def live_checks() -> None:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("\n[3] SKIP live checks: DEEPSEEK_API_KEY is not set")
        return

    settings = Settings()
    settings.llm.provider = "deepseek"
    settings.llm.model = "deepseek-chat"
    settings.llm.api_key = key
    print("\n[3] live: a dead env proxy is set, the OS proxy is not")
    probe = ConnectivityProbe(timeout=5.0, budget_seconds=20.0)
    route = probe.run(settings).route(MODEL_API)
    check(
        "the probe did not settle on the dead env proxy",
        route is not None and route.name != "env-proxy",
        f"route={route.name if route else None}",
    )
    if route is None:
        return

    from langchain_deepseek import ChatDeepSeek

    endpoints = len([item for item in endpoints_for(settings) if item.category == MODEL_API])
    print(f"  ({endpoints} model endpoint(s) probed, route {route.name})")

    def call(build: object) -> tuple[bool, str]:
        client = build(route) if callable(build) else None
        kwargs = {"http_client": client} if client is not None else {}
        model = ChatDeepSeek(
            model="deepseek-chat",
            api_key=key,
            temperature=0.0,
            timeout=30.0,
            max_retries=0,
            **kwargs,
        )
        started = time.perf_counter()
        try:
            text = str(model.invoke("Reply with the single word: ok").content).strip()
            return True, f"{text[:40]!r} in {time.perf_counter() - started:.2f}s"
        except Exception as exc:  # the control is *meant* to fail
            return False, f"{type(exc).__name__} in {time.perf_counter() - started:.2f}s"
        finally:
            if client is not None:
                client.close()

    ok, detail = call(build_http_client)
    check("pinned client reaches the model API", ok, detail)
    ok, detail = call(legacy_build_http_client)
    check("old behaviour fails on the same machine", not ok, detail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="also run the network checks")
    args = parser.parse_args()

    # The machine this was written for: env proxy vars point at a closed port.
    os.environ["HTTP_PROXY"] = os.environ["HTTPS_PROXY"] = os.environ["ALL_PROXY"] = DEAD_PROXY
    offline_checks()
    if args.live:
        live_checks()
    else:
        print("\n[3] live checks skipped (pass --live to run them)")
    print(f"\n{'ALL PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
