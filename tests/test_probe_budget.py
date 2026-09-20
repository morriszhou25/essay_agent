"""The stage-3 probe must spend its budget fairly, under the budget, and honestly.

Measured on a real (partially blocked) machine: six endpoints x three channels with a
20s budget took 22.8s and never reached ``huggingface``/``hf-mirror``, so the whole
dataset group came back ``unknown`` and the feasibility stage had nothing to judge on.

These tests drive the probe with a virtual clock, so the scenarios cost no real time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from essay_agent import connectivity
from essay_agent.connectivity import (
    BUDGET_CHANNEL,
    DATASET_SOURCE,
    HF_MIRROR_URL,
    MODEL_API,
    PAPER_SOURCE,
    PROXY_ENV_VARS,
    Endpoint,
    active_channels,
    run_checks,
)


class Clock:
    """A virtual monotonic clock: what an attempt costs is what advances it."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def spend(self, seconds: float) -> None:
        self.now += seconds

    def elapsed(self) -> float:
        return self.now - 1000.0


CHANNELS: dict[str, dict[str, str | None] | None] = {
    "env-proxy": None,
    "os-proxy": {"http": "http://os.invalid", "https": "http://os.invalid"},
    "direct": {"http": None, "https": None},
}
CHANNEL_IDS = {id(value): name for name, value in CHANNELS.items() if value is not None}


def targets() -> list[Endpoint]:
    """The shape the real config produces: 1 model, 3 papers, 2 datasets."""
    return [
        Endpoint("model api", MODEL_API, "https://model.example/v1/models", priority=True),
        Endpoint("arxiv", PAPER_SOURCE, "https://arxiv.example/api", priority=True),
        Endpoint("semantic scholar", PAPER_SOURCE, "https://s2.example/api"),
        Endpoint("openalex", PAPER_SOURCE, "https://openalex.example/api"),
        Endpoint("huggingface", DATASET_SOURCE, "https://huggingface.example/api", priority=True),
        Endpoint(
            "hf-mirror",
            DATASET_SOURCE,
            "https://mirror.example/api",
            fallback=True,
            priority=True,
            env={"HF_ENDPOINT": HF_MIRROR_URL},
        ),
    ]


def rules(env_proxy: float, os_proxy: float, direct: float) -> dict[Any, tuple[bool, float]]:
    """A python 2d-ish table: every channel fails unless the OS proxy can help."""
    table: dict[tuple[str, str], tuple[bool, float]] = {}
    for channel, latency in (("env-proxy", env_proxy), ("direct", direct)):
        for endpoint in targets():
            table[(channel, endpoint.url)] = (False, latency)
    for endpoint in targets():
        works = not endpoint.url.startswith("https://huggingface.")  # only the mirror helps
        table[("os-proxy", endpoint.url)] = (True, os_proxy) if works else (False, os_proxy)
    return table


def fetch_for(
    clock: Any, calls: list[tuple[str, str]], table: dict[Any, tuple[bool, float]]
) -> Callable[..., tuple[bool, str, float]]:
    def fetch(url: str, proxies: dict[str, str | None] | None, timeout: float):
        channel = "env-proxy" if proxies is None else CHANNEL_IDS[id(proxies)]
        calls.append((channel, url))
        ok, latency = table[(channel, url)]
        clock.spend(min(latency, timeout))  # an attempt can never outlive its timeout
        return ok, "HTTP 200" if ok else "blocked", latency

    return fetch


# ------------------------------------------------------------------- the ordering
def test_the_deciding_channel_is_settled_for_every_endpoint_first() -> None:
    clock, calls = Clock(), []
    run_checks(
        targets(),
        fetch=fetch_for(clock, calls, rules(0.1, 0.1, 0.1)),
        channels=CHANNELS,
        default_channel="env-proxy",
        timeout=5.0,
        budget_seconds=20.0,
        clock=clock.monotonic,
    )

    round_one = [channel for channel, _ in calls[: len(targets())]]
    assert round_one == ["env-proxy"] * len(targets())


def test_priority_endpoints_are_probed_before_the_detail_ones() -> None:
    clock, calls = Clock(), []
    run_checks(
        targets(),
        fetch=fetch_for(clock, calls, rules(1.5, 1.5, 1.5)),
        channels=CHANNELS,
        default_channel="env-proxy",
        timeout=5.0,
        budget_seconds=6.0,  # exactly the four go/no-go endpoints, nothing more
        clock=clock.monotonic,
    )

    probed = [url for _, url in calls]
    assert probed == [
        "https://model.example/v1/models",
        "https://arxiv.example/api",
        "https://huggingface.example/api",
        "https://mirror.example/api",
    ]  # the go/no-go endpoints, in order
    assert "https://openalex.example/api" not in probed  # reporting detail waits


# ------------------------------------------------------------- the headline fix
def test_a_slow_network_still_reaches_the_dataset_verdict() -> None:
    """20s on a machine where the env proxy is dead and a direct connection hangs."""
    clock, calls = Clock(), []
    report = run_checks(
        targets(),
        fetch=fetch_for(clock, calls, rules(env_proxy=2.0, os_proxy=0.7, direct=5.5)),
        channels=CHANNELS,
        default_channel="env-proxy",
        timeout=5.0,
        budget_seconds=20.0,
        clock=clock.monotonic,
    )

    assert clock.elapsed() <= 20.0
    assert report.state(DATASET_SOURCE) == "degraded"
    assert report.route(DATASET_SOURCE).name == "os-proxy"
    # The mirror answered through the os proxy, so the script is started on that route.
    assert report.dataset_env() == {
        "HF_ENDPOINT": HF_MIRROR_URL,
        **dict.fromkeys(PROXY_ENV_VARS, "http://os.invalid"),
    }
    assert report.state(MODEL_API) in {"ok", "degraded"}


def test_the_probe_cannot_overshoot_its_budget() -> None:
    clock, calls = Clock(), []
    run_checks(
        targets(),
        fetch=fetch_for(clock, calls, rules(env_proxy=2.0, os_proxy=0.7, direct=5.5)),
        channels=CHANNELS,
        default_channel="env-proxy",
        timeout=5.0,
        budget_seconds=3.0,
        clock=clock.monotonic,
    )

    assert clock.elapsed() <= 3.0


def test_a_hanging_host_cannot_starve_the_fallback_host() -> None:
    """The primary dataset host hangs; the mirror must still be reached."""
    table = rules(env_proxy=2.0, os_proxy=0.7, direct=5.5)
    table[("os-proxy", "https://huggingface.example/api")] = (False, 5.0)  # a full hang
    clock, calls = Clock(), []
    report = run_checks(
        targets(),
        fetch=fetch_for(clock, calls, table),
        channels=CHANNELS,
        default_channel="env-proxy",
        timeout=5.0,
        budget_seconds=20.0,
        clock=clock.monotonic,
    )

    assert any(url == "https://mirror.example/api" for _, url in calls)
    assert report.state(DATASET_SOURCE) == "degraded"


# ---------------------------------------------------------------------- retrying
def test_a_dropped_connection_is_retried() -> None:
    clock, calls = Clock(), []
    table = rules(0.1, 0.1, 0.1)
    attempts: list[str] = []

    def flaky(url: str, proxies: dict[str, str | None] | None, timeout: float):
        channel = "env-proxy" if proxies is None else CHANNEL_IDS[id(proxies)]
        attempts.append(channel)
        if len(attempts) == 1:  # the first attempt is dropped
            clock.spend(0.05)
            return False, "ProxyError: connection reset", 0.05
        return fetch_for(clock, calls, table)(url, proxies, timeout)

    run_checks(
        [Endpoint("arxiv", PAPER_SOURCE, "https://arxiv.example/api")],
        fetch=flaky,
        channels=CHANNELS,
        default_channel="env-proxy",
        timeout=5.0,
        budget_seconds=20.0,
        clock=clock.monotonic,
    )

    assert attempts[:2] == ["env-proxy", "env-proxy"]  # the drop was absorbed


def test_a_hang_is_not_retried() -> None:
    clock, calls = Clock(), []
    table = rules(env_proxy=5.0, os_proxy=0.1, direct=0.1)
    report = run_checks(
        [Endpoint("arxiv", PAPER_SOURCE, "https://arxiv.example/api")],
        fetch=fetch_for(clock, calls, table),
        channels=CHANNELS,
        default_channel="env-proxy",
        timeout=5.0,
        budget_seconds=1.0,
        clock=clock.monotonic,
    )

    env_attempts = [a for a in report.reports[0].attempts if a.channel == "env-proxy"]
    assert len(env_attempts) == 1  # a timeout is not a flake


# ---------------------------------------------------------------------- verdicts
def test_an_endpoint_the_budget_cut_short_is_unknown_not_unreachable() -> None:
    clock, calls = Clock(), []
    report = run_checks(
        targets(),
        fetch=fetch_for(clock, calls, rules(0.2, 0.1, 5.0)),
        channels=CHANNELS,
        default_channel="env-proxy",
        timeout=5.0,
        budget_seconds=1.0,
        clock=clock.monotonic,
    )

    assert report.state(DATASET_SOURCE) == "unknown"
    assert report.state(MODEL_API) == "unknown"
    assert all(
        item.attempts[-1].channel == BUDGET_CHANNEL
        for item in report.reports
        if not item.ok and item.endpoint.category == MODEL_API
    )


def test_an_endpoint_that_tried_every_channel_and_failed_is_unreachable() -> None:
    clock, calls = Clock(), []
    everything_fails = {
        (channel, endpoint.url): (False, 0.1) for channel in CHANNELS for endpoint in targets()
    }
    report = run_checks(
        targets(),
        fetch=fetch_for(clock, calls, everything_fails),
        channels=CHANNELS,
        default_channel="env-proxy",
        timeout=5.0,
        budget_seconds=30.0,
        clock=clock.monotonic,
    )

    assert report.state(DATASET_SOURCE) == "unreachable"
    assert report.state(MODEL_API) == "unreachable"


def test_a_healthy_machine_probes_each_endpoint_exactly_once() -> None:
    clock, calls = Clock(), []
    table = {(channel, endpoint.url): (True, 0.1) for channel in CHANNELS for endpoint in targets()}
    report = run_checks(
        targets(),
        fetch=fetch_for(clock, calls, table),
        channels=CHANNELS,
        default_channel="env-proxy",
        timeout=5.0,
        budget_seconds=20.0,
        clock=clock.monotonic,
    )

    assert len(calls) == len(targets())
    assert all(item.ok for item in report.reports)


# ------------------------------------------------------------------- the channels
def test_the_os_proxy_is_offered_before_a_bare_direct_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A machine with a proxy configured is unlikely to reach anything *directly*."""
    monkeypatch.setattr(connectivity, "os_proxy", lambda: "127.0.0.1:7897")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")

    assert list(active_channels()) == ["env-proxy", "os-proxy", "direct"]


def test_a_machine_without_any_proxy_only_gets_the_direct_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(connectivity, "os_proxy", lambda: None)
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)

    assert list(active_channels()) == ["direct"]
