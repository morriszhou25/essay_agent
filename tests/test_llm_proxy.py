"""Issue #1: the *model* client is pinned to the route the probe verified.

The tools already pin themselves (``tests/test_proxy_pinning.py``); these tests cover
the langchain clients, which would otherwise follow a dead ``HTTP_PROXY``.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import httpx
import pytest

from essay_agent.connectivity import MODEL_API, Route, build_http_client, ca_bundle
from essay_agent.llm import build_llm
from essay_agent.nodes.base import build_deps

PROXY = "http://127.0.0.1:7897"
ROUTE_PINNED = Route("os-proxy", {"http": PROXY, "https": PROXY})


def proxy_of(client: httpx.Client) -> str | None:
    """The proxy a client really routes through, read back from its transport."""

    def text(value: bytes | str) -> str:
        return value.decode() if isinstance(value, bytes) else value

    for transport in client._mounts.values():
        url = getattr(getattr(transport, "_pool", None), "_proxy_url", None)
        if url is not None:
            return f"{text(url.scheme)}://{text(url.host)}:{url.port}"
    return None


# ---------------------------------------------------------------- the TLS bundle
def test_the_requests_bundle_wins_over_the_curl_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "corp.pem")
    monkeypatch.setenv("CURL_CA_BUNDLE", "other.pem")
    assert ca_bundle() == "corp.pem"


def test_the_curl_bundle_is_a_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.setenv("CURL_CA_BUNDLE", "corp.pem")
    assert ca_bundle() == "corp.pem"


def test_no_bundle_configured_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
    assert ca_bundle() is None


# ------------------------------------------------------------- route -> client
@pytest.mark.parametrize("route", [None, Route("env-proxy", None)])
def test_routes_we_keep_trusting_get_no_client(route: Route | None) -> None:
    """Nothing verified, or "the environment is fine", must not be second-guessed."""
    assert build_http_client(route) is None


def test_a_direct_route_ignores_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://dead.invalid")
    client = build_http_client(Route("direct", {"http": None, "https": None}))
    assert client is not None
    try:
        assert client.trust_env is False
        assert proxy_of(client) is None
    finally:
        client.close()


def test_a_proxy_route_attaches_that_proxy() -> None:
    client = build_http_client(ROUTE_PINNED)
    assert client is not None
    try:
        assert client.trust_env is False
        assert proxy_of(client) == PROXY
    finally:
        client.close()


def test_the_bundle_is_handed_to_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "corp.pem")
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(httpx, "Client", FakeClient)
    assert build_http_client(ROUTE_PINNED) is not None
    assert captured == {"trust_env": False, "proxy": PROXY, "verify": "corp.pem"}


# ------------------------------------------------------------ langchain client
def test_the_model_gets_the_pinned_client(settings) -> None:
    client = build_llm(settings.llm, route=lambda: ROUTE_PINNED)
    model = client._model("main")
    assert isinstance(model.http_client, httpx.Client)
    assert model.http_client.trust_env is False
    assert proxy_of(model.http_client) == PROXY


def test_without_a_route_the_model_keeps_its_own_client(settings) -> None:
    model = build_llm(settings.llm)._model("main")
    assert model.http_client is None


def test_a_route_of_none_leaves_the_model_alone(settings) -> None:
    model = build_llm(settings.llm, route=lambda: None)._model("main")
    assert model.http_client is None


def test_the_deepseek_branch_is_pinned_too(settings) -> None:
    pytest.importorskip("langchain_deepseek")
    llm_settings = settings.llm.model_copy(
        update={"provider": "deepseek", "model": "deepseek-chat"}
    )
    model = build_llm(llm_settings, route=lambda: ROUTE_PINNED)._model("main")
    assert isinstance(model.http_client, httpx.Client)
    assert model.http_client.trust_env is False


def test_the_anthropic_branch_is_pinned_too(settings, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeChatAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    module = types.ModuleType("langchain_anthropic")
    module.ChatAnthropic = FakeChatAnthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_anthropic", module)
    llm_settings = settings.llm.model_copy(
        update={"provider": "anthropic", "model": "claude-3-5-sonnet-latest"}
    )
    build_llm(llm_settings, route=lambda: ROUTE_PINNED)._model("main")
    assert captured["http_client"].trust_env is False
    assert proxy_of(captured["http_client"]) == PROXY


# ------------------------------------------------------------------- wiring
class _Probe:
    """A ConnectivityProbe stand-in that only records what it was asked for."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def provider(self, settings: Any, category: str) -> Any:
        self.asked.append(category)
        return lambda: None


def test_build_deps_hands_the_model_route_to_the_client(
    settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_build_llm(llm_settings: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("essay_agent.nodes.base.build_llm", fake_build_llm)
    probe = _Probe()
    build_deps(settings, connectivity=probe)  # type: ignore[arg-type]

    assert captured["route"] is not None
    assert callable(captured["route"])
    assert probe.asked[0] == MODEL_API  # the model route is resolved before the tools
