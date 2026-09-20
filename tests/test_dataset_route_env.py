"""The route the probe verified has to reach the child process that runs ``repro.py``.

``requests`` reads the environment (and then, on Windows, the registry) while ``httpx``
reads only the environment, so "the probe worked" is not enough on a machine behind a
VPN: the script is started with the verified route written out, and with an inherited
dead proxy *removed* when the verified route is a direct connection.
"""

from __future__ import annotations

import sys

import pytest

from essay_agent.config import RuntimeSettings
from essay_agent.connectivity import (
    DATASET_SOURCE,
    PROXY_ENV_VARS,
    Attempt,
    ConnectivityReport,
    Endpoint,
    EndpointReport,
    Route,
    ca_bundle,
    route_env,
)
from essay_agent.runtime.runner import ScriptRunner

PROXY = "http://127.0.0.1:7897"
DEAD = "http://127.0.0.1:9"
DIRECT_CHANNEL: dict[str, str | None] = {"http": None, "https": None}
MIRROR_ENV = {"HF_ENDPOINT": "https://hf-mirror.example"}
CHANNELS: dict[str, dict[str, str | None] | None] = {
    "env-proxy": None,
    "os-proxy": {"http": PROXY, "https": PROXY},
    "direct": DIRECT_CHANNEL,
}


def target(name: str, *, fallback: bool = False, env: dict[str, str] | None = None) -> Endpoint:
    return Endpoint(
        name, DATASET_SOURCE, f"https://example.test/{name}", fallback=fallback, env=env or {}
    )


def report(*results: tuple[Endpoint, str, bool]) -> ConnectivityReport:
    """One attempt per endpoint, on the channel named in ``results``."""
    reports = [
        EndpointReport(endpoint, [Attempt(channel, ok, "HTTP 200" if ok else "blocked", 0.01)])
        for endpoint, channel, ok in results
    ]
    return ConnectivityReport(reports=reports, default_channel="env-proxy", channels=CHANNELS)


def runner() -> ScriptRunner:
    return ScriptRunner(RuntimeSettings(), python_executable=sys.executable, sink_factory=None)


HF = target("huggingface")
MIRROR = target("hf-mirror", fallback=True, env=MIRROR_ENV)


def test_a_verified_proxy_is_handed_to_the_script() -> None:
    env = report((HF, "os-proxy", True)).dataset_env()

    assert set(env) == {*PROXY_ENV_VARS, "HF_ENDPOINT"}
    assert {env[name] for name in PROXY_ENV_VARS} == {PROXY}
    assert env["HF_ENDPOINT"] is None  # not a proxy: the primary host was verified


def test_a_verified_direct_route_takes_back_an_inherited_proxy() -> None:
    env = report((HF, "direct", True)).dataset_env()

    assert env == {**dict.fromkeys(PROXY_ENV_VARS), "HF_ENDPOINT": None}
    assert all(value is None for value in env.values())


def test_an_environment_route_is_left_alone() -> None:
    # "Let the client decide" is itself a verified route: the environment already says it.
    # Only the endpoint is rewritten, and only to take back an inherited mirror.
    assert report((HF, "env-proxy", True)).dataset_env() == {"HF_ENDPOINT": None}


def test_nothing_verified_means_nothing_reported() -> None:
    built = report((HF, "env-proxy", False), (MIRROR, "env-proxy", False))

    assert built.dataset_env() == {}


def test_the_mirror_is_named_only_when_the_primary_host_failed() -> None:
    fallback = report((HF, "os-proxy", False), (MIRROR, "os-proxy", True)).dataset_env()
    assert fallback["HF_ENDPOINT"] == MIRROR_ENV["HF_ENDPOINT"]
    assert fallback["HTTPS_PROXY"] == PROXY  # the mirror is reached the same way

    healthy = report((HF, "os-proxy", True), (MIRROR, "os-proxy", True)).dataset_env()
    assert healthy["HF_ENDPOINT"] is None  # the inherited mirror must not outrank the probe


def test_an_inherited_hf_endpoint_is_taken_back_when_the_primary_host_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A machine-wide ``HF_ENDPOINT`` is a route nobody probed, so it must not survive."""
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.invalid")

    env = runner()._build_env(report((HF, "os-proxy", True)).dataset_env())

    assert "HF_ENDPOINT" not in env


def test_a_ca_bundle_travels_with_the_route(monkeypatch) -> None:
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "C:/certs/corp.pem")

    env = route_env(Route("os-proxy", {"http": PROXY, "https": PROXY}))

    assert ca_bundle() == "C:/certs/corp.pem"
    assert env["REQUESTS_CA_BUNDLE"] == "C:/certs/corp.pem"
    assert env["CURL_CA_BUNDLE"] == "C:/certs/corp.pem"


def test_a_channel_without_a_proxy_url_means_direct() -> None:
    # ``active_channels`` spells "no proxy" as an empty mapping; that is a direct route.
    assert route_env(Route("direct", {})) == dict.fromkeys(PROXY_ENV_VARS)
    assert route_env(Route("os-proxy", DIRECT_CHANNEL)) == dict.fromkeys(PROXY_ENV_VARS)
    assert route_env(None) == {}


def test_the_child_environment_removes_variables_marked_none(monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", DEAD)
    monkeypatch.setenv("ALL_PROXY", DEAD)
    built = runner()

    env = built._build_env({"HTTPS_PROXY": PROXY, "ALL_PROXY": None})

    assert env["HTTPS_PROXY"] == PROXY  # the verified route wins
    assert env["HTTP_PROXY"] == DEAD  # a variable nobody mentioned survives
    assert "ALL_PROXY" not in env  # None removes instead of blanking
    assert env["ESSAY_AGENT_DEVICE"] == built.settings.device  # the usual additions stay


def test_the_route_survives_the_round_trip_into_the_child(monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", DEAD)
    monkeypatch.setenv("ALL_PROXY", DEAD)

    env = runner()._build_env(report((HF, "os-proxy", True)).dataset_env())

    assert {env[name] for name in PROXY_ENV_VARS} == {PROXY}
