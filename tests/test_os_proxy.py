"""The OS-proxy detection and the explicit `http_proxy` override (issues #2 / #2b)."""

from __future__ import annotations

from typing import Any

import pytest

from essay_agent import connectivity
from essay_agent.connectivity import (
    Endpoint,
    active_channels,
    default_channel_name,
    normalise_proxy,
    os_proxy,
    proxies_for,
    run_checks,
)

SCUTIL = """\
<dictionary> {
  HTTPEnable : 1
  HTTPPort : 8080
  HTTPProxy : 10.0.0.1
  HTTPSEnable : 1
  HTTPSPort : 7897
  HTTPSProxy : 127.0.0.1
  SOCKSEnable : 0
}
"""


def fake_tool(mapping: dict[str, str], *, missing: bool = False):
    """A ``_run_tool`` stand-in that answers per command prefix."""

    def run(command: list[str], timeout: float = 2.0) -> str | None:
        if missing:
            return None
        key = " ".join(command)
        for prefix, value in mapping.items():
            if key.startswith(prefix):
                return value
        return None

    return run


# ------------------------------------------------------------------- parsing
@pytest.mark.parametrize(
    ("host", "port", "expected"),
    [
        ("127.0.0.1", "7897", "http://127.0.0.1:7897"),
        ("127.0.0.1", None, "http://127.0.0.1"),
        ("http://proxy:8080", "9999", "http://proxy:8080"),  # a URL already knows better
    ],
)
def test_normalise_proxy(host: str, port: str | None, expected: str) -> None:
    assert normalise_proxy(host, port) == expected


def test_proxies_for_maps_both_schemes_and_keeps_none_meaningful() -> None:
    assert proxies_for(None) is None
    assert proxies_for("127.0.0.1:9") == {
        "http": "http://127.0.0.1:9",
        "https": "http://127.0.0.1:9",
    }


def test_macos_reads_the_https_proxy_from_scutil(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(connectivity, "_run_tool", fake_tool({"scutil --proxy": SCUTIL}))
    assert connectivity._macos_proxy() == "http://127.0.0.1:7897"


def test_macos_without_a_proxy_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    disabled = SCUTIL.replace("HTTPSEnable : 1", "HTTPSEnable : 0").replace(
        "HTTPEnable : 1", "HTTPEnable : 0"
    )
    monkeypatch.setattr(connectivity, "_run_tool", fake_tool({"scutil --proxy": disabled}))
    assert connectivity._macos_proxy() is None


def test_macos_falls_back_to_an_http_only_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    # A machine behind one proxy box often only enables the plain-HTTP entry; using it for
    # HTTPS too is what a lenient client would do, and the probe still proves reachability.
    http_only = SCUTIL.replace("HTTPSEnable : 1", "HTTPSEnable : 0")
    monkeypatch.setattr(connectivity, "_run_tool", fake_tool({"scutil --proxy": http_only}))
    assert connectivity._macos_proxy() == "http://10.0.0.1:8080"


def test_linux_reads_a_manual_gnome_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        connectivity,
        "_run_tool",
        fake_tool(
            {
                "gsettings get org.gnome.system.proxy mode": "'manual'",
                "gsettings get org.gnome.system.proxy.https host": "'127.0.0.1'",
                "gsettings get org.gnome.system.proxy.https port": "7897",
            }
        ),
    )
    assert connectivity._linux_proxy() == "http://127.0.0.1:7897"


def test_linux_ignores_an_automatic_gnome_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        connectivity,
        "_run_tool",
        fake_tool({"gsettings get org.gnome.system.proxy mode": "'auto'"}),
    )
    assert connectivity._linux_proxy() is None


def test_linux_falls_back_to_kde(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        connectivity,
        "_run_tool",
        fake_tool({"kreadconfig6 --group Proxy --key httpsProxy": "10.0.0.9:3128"}),
    )
    assert connectivity._linux_proxy() == "http://10.0.0.9:3128"


def test_a_missing_tool_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(connectivity, "_run_tool", fake_tool({}, missing=True))
    assert connectivity._macos_proxy() is None
    assert connectivity._linux_proxy() is None


def test_run_tool_survives_a_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    def boom(*_: Any, **__: Any) -> Any:
        raise FileNotFoundError("no such tool")

    monkeypatch.setattr(subprocess, "run", boom)
    assert connectivity._run_tool(["definitely-not-installed"]) is None


def test_os_proxy_never_raises_and_is_cached() -> None:
    value = os_proxy()  # the real platform, whatever it is
    assert os_proxy() == value
    assert value is None or isinstance(value, str)
    assert os_proxy.cache_info().hits >= 1


# ---------------------------------------------------------- explicit override
def test_an_explicit_proxy_wins_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")  # the dead one
    channels = active_channels("http://127.0.0.1:7897")

    assert next(iter(channels)) == "configured-proxy"
    assert channels["configured-proxy"] == {
        "http": "http://127.0.0.1:7897",
        "https": "http://127.0.0.1:7897",
    }
    assert default_channel_name("http://127.0.0.1:7897") == "configured-proxy"


def test_the_report_remembers_the_explicit_proxy() -> None:
    endpoints = [Endpoint("model api", "model_api", "https://model.example/v1/models")]
    report = run_checks(
        endpoints,
        fetch=lambda url, proxies, timeout: (True, "HTTP 200", 0.01),
        timeout=0.01,
        explicit_proxy="http://127.0.0.1:7897",
    )
    assert report.proxy_hint == "http://127.0.0.1:7897"
    assert next(iter(report.channels)) == "configured-proxy"
    assert report.to_dict()["os_proxy"] == "http://127.0.0.1:7897"


def test_the_configured_proxy_reaches_the_settings_and_the_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """`ESSAY_AGENT_HTTP_PROXY` is the escape hatch when detection gets it wrong."""
    from essay_agent.config import load_settings
    from essay_agent.connectivity import MODEL_API, ConnectivityProbe

    monkeypatch.setenv("ESSAY_AGENT_HTTP_PROXY", "http://127.0.0.1:7897")
    settings = load_settings(
        config_file=None, env_file=None, overrides={"paths": {"home": str(tmp_path)}}
    )
    assert settings.http_proxy == "http://127.0.0.1:7897"

    seen: list[Any] = []

    def fetch(url: str, proxies: dict[str, str | None] | None, timeout: float) -> Any:
        seen.append(proxies)
        return True, "HTTP 200", 0.01

    route = ConnectivityProbe(fetch=fetch).route(settings, MODEL_API)
    assert route is not None and route.name == "configured-proxy"
    assert route.proxies == {
        "http": "http://127.0.0.1:7897",
        "https": "http://127.0.0.1:7897",
    }
    assert seen[0] is not None  # the first channel really carried the configured proxy
