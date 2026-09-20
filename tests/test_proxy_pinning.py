"""Real HTTP clients use the channel the probe verified, not the environment.

``requests`` only honours proxy environment variables, so a machine whose env holds
a dead ``HTTP_PROXY``/``ALL_PROXY`` - or that reaches the network through a proxy
configured only in the OS settings - would ignore the route ``connectivity.py``
just validated. These tests pin that behaviour down without touching the internet.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

import pytest
import requests

from essay_agent.config import SearchSettings
from essay_agent.connectivity import (
    DATASET_SOURCE,
    PAPER_SOURCE,
    Attempt,
    ConnectivityProbe,
    ConnectivityReport,
    Endpoint,
    EndpointReport,
    Route,
    endpoints_for,
    fetch_with_requests,
    pin_session,
    run_checks,
)
from essay_agent.tools.dataset_probe import DatasetProber
from essay_agent.tools.paper_search import PaperSearcher

CHANNELS: dict[str, dict[str, str | None] | None] = {
    "env-proxy": None,
    "os-proxy": {"http": "http://os-proxy.invalid", "https": "http://os-proxy.invalid"},
}


# --------------------------------------------------------------------- test harness
def unused_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


class _Quiet(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass


class _TargetHandler(_Quiet):
    def do_GET(self) -> None:
        body = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _ProxyHandler(_Quiet):
    """A minimal forward proxy, standing in for the OS proxy the probe validated."""

    seen: ClassVar[list[str]] = []

    def do_GET(self) -> None:
        parts = urllib.parse.urlsplit(self.path)
        self.seen.append(self.path)
        path = parts.path + (f"?{parts.query}" if parts.query else "")
        conn = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=5)
        conn.request("GET", path, headers={"Host": parts.netloc})
        reply = conn.getresponse()
        body = reply.read()
        self.send_response(reply.status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture()
def dead_env_proxy(monkeypatch: pytest.MonkeyPatch) -> str:
    """A broken proxy in the environment, exactly like a stale VPN leftover."""
    dead = f"http://127.0.0.1:{unused_port()}"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, dead)
    # Loopback must not be exempt, or these tests would not exercise the proxy.
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    return dead


@pytest.fixture()
def target_and_proxy() -> Any:
    target = serve(_TargetHandler)
    proxy = serve(_ProxyHandler)
    _ProxyHandler.seen.clear()
    yield (
        f"http://127.0.0.1:{target.server_port}/paper",
        f"http://127.0.0.1:{proxy.server_port}",
    )
    target.shutdown()
    proxy.shutdown()


# ------------------------------------------------------------------- the mechanism
def test_a_dead_env_proxy_breaks_a_bare_session(dead_env_proxy: str, target_and_proxy: Any) -> None:
    """Documents the bug: the probe can say "ok" while the real client cannot connect."""
    url, _ = target_and_proxy
    with pytest.raises(requests.exceptions.ProxyError):
        requests.Session().get(url, timeout=2)


def test_pinning_to_the_verified_proxy_beats_a_dead_env_proxy(
    dead_env_proxy: str, target_and_proxy: Any
) -> None:
    url, proxy = target_and_proxy
    session = pin_session(requests.Session(), Route("os-proxy", {"http": proxy, "https": proxy}))

    response = session.get(url, timeout=2)

    assert response.json() == {"ok": True}
    # the request really travelled through the proxy, not straight to the target
    assert _ProxyHandler.seen == [url]
    assert session.trust_env is False
    assert session.proxies == {"http": proxy, "https": proxy}


def test_pinning_to_a_verified_direct_route_ignores_the_env_proxy(
    dead_env_proxy: str, target_and_proxy: Any
) -> None:
    url, _ = target_and_proxy
    session = pin_session(requests.Session(), Route("direct", {"http": None, "https": None}))

    assert session.get(url, timeout=2).status_code == 200
    assert session.trust_env is False


def test_pinning_keeps_the_env_route_when_that_is_what_was_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://env-proxy.invalid")
    session = pin_session(requests.Session(), Route("env-proxy", None))

    assert session.trust_env is True
    assert session.proxies == {}


def test_no_verified_route_leaves_the_session_untouched(dead_env_proxy: str) -> None:
    session = requests.Session()
    session.proxies = {"http": "http://kept.invalid"}

    assert pin_session(session, None) is session
    assert session.trust_env is True
    assert session.proxies == {"http": "http://kept.invalid"}


def test_pinning_preserves_a_corporate_ca_bundle(
    dead_env_proxy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", r"C:\corp\ca.pem")
    session = pin_session(requests.Session(), Route("direct", {"http": None, "https": None}))

    assert session.verify == r"C:\corp\ca.pem"


def test_a_stale_no_proxy_cannot_unpin_the_route(
    dead_env_proxy: str, monkeypatch: pytest.MonkeyPatch, target_and_proxy: Any
) -> None:
    url, proxy = target_and_proxy
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    session = pin_session(requests.Session(), Route("os-proxy", {"http": proxy, "https": proxy}))

    assert session.get(url, timeout=2).status_code == 200


# ------------------------------------------------------------------------- routing
def _report(*attempts: tuple[Endpoint, list[Attempt]]) -> ConnectivityReport:
    return ConnectivityReport(
        reports=[EndpointReport(endpoint, list(items)) for endpoint, items in attempts],
        default_channel="env-proxy",
        channels=dict(CHANNELS),
    )


def test_route_reports_the_channel_that_answered() -> None:
    report = _report(
        (
            Endpoint("arxiv", PAPER_SOURCE, "https://arxiv.org/x"),
            [
                Attempt("env-proxy", False, "blocked", 0.0),
                Attempt("os-proxy", True, "HTTP 200", 0.0),
            ],
        )
    )

    assert report.route(PAPER_SOURCE) == Route("os-proxy", CHANNELS["os-proxy"])


def test_route_falls_back_to_the_mirror_channel_for_datasets() -> None:
    report = _report(
        (
            Endpoint("huggingface", DATASET_SOURCE, "https://huggingface.co/x"),
            [Attempt("env-proxy", False, "timed out", 0.0)],
        ),
        (
            Endpoint("hf-mirror", DATASET_SOURCE, "https://hf-mirror.com/x", fallback=True),
            [
                Attempt("env-proxy", False, "timed out", 0.0),
                Attempt("os-proxy", True, "HTTP 200", 0.0),
            ],
        ),
    )

    assert report.route(DATASET_SOURCE) == Route("os-proxy", CHANNELS["os-proxy"])


def test_route_is_none_when_nothing_answered() -> None:
    report = _report(
        (
            Endpoint("arxiv", PAPER_SOURCE, "https://arxiv.org/x"),
            [Attempt("env-proxy", False, "blocked", 0.0)],
        )
    )

    assert report.route(PAPER_SOURCE) is None


def test_route_is_none_for_a_channel_we_cannot_reconstruct() -> None:
    report = _report(
        (
            Endpoint("arxiv", PAPER_SOURCE, "https://arxiv.org/x"),
            [Attempt("someones-proxy", True, "HTTP 200", 0.0)],
        )
    )

    assert report.route(PAPER_SOURCE) is None


def test_the_channel_map_is_not_polluted_by_the_environment(
    dead_env_proxy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requests merges env proxies into the dict it is handed; the map must stay clean."""
    seen: list[dict[str, str | None] | None] = []

    def fetch(url: str, proxies: dict[str, str | None] | None, timeout: float):
        seen.append(dict(proxies) if proxies is not None else None)
        return False, "blocked", 0.0

    report = run_checks(
        [Endpoint("arxiv", PAPER_SOURCE, "https://arxiv.org/x")],
        fetch=fetch,
        channels=dict(CHANNELS),
        default_channel="env-proxy",
        timeout=0.01,
    )

    # None twice: the deciding channel gets one retry so a single dropped
    # connection cannot read as "unreachable".
    assert seen == [None, None, CHANNELS["os-proxy"]]
    assert report.channels["os-proxy"] == CHANNELS["os-proxy"]


def test_fetch_with_requests_does_not_mutate_the_caller_s_dict(dead_env_proxy: str) -> None:
    proxies: dict[str, str | None] = {"http": None, "https": None}

    fetch_with_requests("http://127.0.0.1:1", proxies, 0.05)

    assert proxies == {"http": None, "https": None}


def test_the_probe_resolves_one_category_without_probing_the_others(settings: Any) -> None:
    endpoints: list[str] = []

    def fetch(url: str, proxies: dict[str, str | None] | None, timeout: float):
        endpoints.append(url)
        return True, "HTTP 200", 0.0

    probe = ConnectivityProbe(fetch=fetch, timeout=0.01, budget_seconds=1.0)

    assert probe.route(settings, PAPER_SOURCE) is not None
    expected = [item.url for item in endpoints_for(settings) if item.category == PAPER_SOURCE]
    assert endpoints == expected  # only the paper sources were asked about
    assert not any("huggingface" in url for url in endpoints)  # no dataset probe
    assert not any(
        url.endswith("/v1/models") or url.endswith("api.deepseek.com/") for url in endpoints
    )


# ---------------------------------------------------------------- the real clients
def test_the_paper_searcher_pins_its_session(dead_env_proxy: str, target_and_proxy: Any) -> None:
    url, proxy = target_and_proxy
    searcher = PaperSearcher(
        SearchSettings(timeout=2.0),
        route=lambda: Route("os-proxy", {"http": proxy, "https": proxy}),
    )

    assert searcher.session.get(url, timeout=2).status_code == 200
    assert searcher.session.trust_env is False


def test_the_dataset_prober_pins_its_session(dead_env_proxy: str, target_and_proxy: Any) -> None:
    url, proxy = target_and_proxy
    prober = DatasetProber(
        timeout=2.0, route=lambda: Route("os-proxy", {"http": proxy, "https": proxy})
    )

    assert prober.session.get(url, timeout=2).status_code == 200
    assert prober.session.trust_env is False


def test_build_deps_hands_the_clients_the_verified_route(settings: Any) -> None:
    from essay_agent.console import SilentUI
    from essay_agent.nodes.base import build_deps
    from tests.fakes import FakeConnectivity

    pinned = {"http": "http://pinned.invalid", "https": "http://pinned.invalid"}

    class Pinned(FakeConnectivity):
        def provider(self, settings: Any, category: str):
            return lambda: Route("os-proxy", dict(pinned))

    deps = build_deps(settings, ui=SilentUI(), connectivity=Pinned())

    assert deps.searcher.session.proxies == pinned
    assert deps.prober.session.proxies == pinned
