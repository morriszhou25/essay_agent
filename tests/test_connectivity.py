"""Connectivity is reported per purpose and per channel - never as one global flag."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from essay_agent.agent import run_task
from essay_agent.config import DEFAULT_DATASET_MIRROR_URL
from essay_agent.connectivity import (
    DATASET_SOURCE,
    HF_MIRROR_URL,
    MODEL_API,
    PAPER_SOURCE,
    PROXY_ENV_VARS,
    Endpoint,
    endpoints_for,
    run_checks,
)
from essay_agent.console import SilentUI
from tests.fakes import FakeConnectivity, FakeLLM
from tests.pipeline import build_deps, happy_responses

REPORT = "# Reproducing: Test Paper\n\n## Summary\nwritten anyway.\n"

CHANNELS: dict[str, dict[str, str | None] | None] = {
    "env-proxy": None,
    "os-proxy": {"http": "x", "https": "x"},
    "direct": {},
}
MODEL = "https://api.deepseek.com/"
HF = "https://huggingface.co/api/datasets?limit=1"
MIRROR = f"{HF_MIRROR_URL}/api/datasets?limit=1"


def probe_targets() -> list[Endpoint]:
    return [
        Endpoint("model api", MODEL_API, MODEL),
        Endpoint("huggingface", DATASET_SOURCE, HF),
        Endpoint(
            "hf-mirror",
            DATASET_SOURCE,
            MIRROR,
            fallback=True,
            env={"HF_ENDPOINT": DEFAULT_DATASET_MIRROR_URL},
        ),
    ]


def offline_fetch(rules: dict[str, dict[str, tuple[bool, str]]], calls: list[tuple[str, str]]):
    """A fetch that answers according to the channel it was handed."""
    by_proxies = {id(proxies): name for name, proxies in CHANNELS.items()}

    def fetch(url: str, proxies: dict[str, str | None] | None, timeout: float):
        channel = by_proxies[id(proxies)]
        calls.append((channel, url))
        ok, detail = rules.get(channel, {}).get(url, (False, "blocked"))
        return ok, detail, 0.01

    return fetch


def check(rules: dict[str, dict[str, tuple[bool, str]]], **kwargs: Any):
    calls: list[tuple[str, str]] = []
    report = run_checks(
        probe_targets(),
        fetch=offline_fetch(rules, calls),
        channels=CHANNELS,
        default_channel="env-proxy",
        timeout=0.01,
        **kwargs,
    )
    return report, calls


def everything_up(channel: str = "env-proxy") -> dict[str, dict[str, tuple[bool, str]]]:
    return {channel: {endpoint.url: (True, "HTTP 200") for endpoint in probe_targets()}}


def test_a_healthy_machine_is_ok_everywhere_and_needs_no_action() -> None:
    report, calls = check(everything_up())
    assert [report.state(c) for c in (MODEL_API, DATASET_SOURCE)] == ["ok", "ok"]
    assert report.hint() is None
    # Only the endpoint: an inherited mirror must not outrank a verified primary host.
    assert report.dataset_env() == {"HF_ENDPOINT": None}
    assert len(calls) == 3  # one attempt per endpoint: a working channel stops the probe


def test_retrying_the_deciding_channel_absorbs_one_flaky_attempt() -> None:
    attempts = {"count": 0}

    def flaky(url: str, proxies: dict[str, str | None] | None, timeout: float):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return False, "ProxyError: dropped", 0.01
        return True, "HTTP 200", 0.01

    report = run_checks(
        probe_targets(),
        fetch=flaky,
        channels=CHANNELS,
        default_channel="env-proxy",
        timeout=0.01,
    )
    assert report.state(MODEL_API) == "ok"
    assert report.hint() is None


def test_a_dataset_host_that_needs_the_os_proxy_is_degraded_not_missing() -> None:
    rules = everything_up()
    rules["env-proxy"].pop(HF)
    rules["os-proxy"] = {HF: (True, "HTTP 200")}
    report, _ = check(rules)

    assert report.state(DATASET_SOURCE) == "degraded"
    assert report.state(MODEL_API) == "ok"  # the machine is not "unreachable"
    assert "os-proxy" in " ".join(report.lines())
    # The primary host does work, just elsewhere - so the script is told how to get there.
    assert report.dataset_env() == {**dict.fromkeys(PROXY_ENV_VARS, "x"), "HF_ENDPOINT": None}


def test_the_mirror_fallback_names_the_setting_it_needs() -> None:
    rules = everything_up()
    rules["env-proxy"].pop(HF)
    rules["direct"] = {MIRROR: (True, "HTTP 200")}
    report, _ = check(rules)

    assert report.state(DATASET_SOURCE) == "degraded"
    assert report.dataset_env() == {"HF_ENDPOINT": HF_MIRROR_URL}
    assert "HF_ENDPOINT" in (report.hint() or "")


def test_an_offline_machine_says_stop_instead_of_guessing() -> None:
    report, _ = check({})
    assert report.state(MODEL_API) == "unreachable"
    assert report.state(DATASET_SOURCE) == "unreachable"
    assert report.dataset_env() == {}
    assert "stop before" in (report.hint() or "")
    assert "NOT reachable" not in " ".join(report.lines())


def test_a_spent_probe_budget_is_unknown_not_unreachable() -> None:
    report, calls = check(everything_up(), budget_seconds=0.0)
    assert report.state(MODEL_API) == "unknown"
    assert calls == []  # nothing was attempted
    assert "probe budget" in " ".join(report.lines())


def test_endpoints_follow_the_configured_provider_and_backends(settings) -> None:
    configured = settings.model_copy(
        update={
            "llm": settings.llm.model_copy(update={"provider": "deepseek"}),
            "search": settings.search.model_copy(update={"backends": ["arxiv", "semanticscholar"]}),
        }
    )
    endpoints = endpoints_for(configured)
    urls = [endpoint.url for endpoint in endpoints]
    assert any("api.deepseek.com" in url for url in urls)
    assert sum(endpoint.category == PAPER_SOURCE for endpoint in endpoints) == 2
    assert sum(endpoint.category == DATASET_SOURCE for endpoint in endpoints) == 2


def test_the_dataset_mirror_url_is_configurable(settings) -> None:
    configured = settings.model_copy(
        update={
            "search": settings.search.model_copy(
                update={"dataset_mirror_url": "https://mirror.example"}
            )
        }
    )
    mirror = next(endpoint for endpoint in endpoints_for(configured) if endpoint.fallback)
    assert mirror.url == "https://mirror.example/api/datasets?limit=1"
    assert mirror.env == {"HF_ENDPOINT": "https://mirror.example"}


def test_an_empty_mirror_url_disables_the_fallback(settings) -> None:
    configured = settings.model_copy(
        update={"search": settings.search.model_copy(update={"dataset_mirror_url": ""})}
    )
    endpoints = endpoints_for(configured)
    assert sum(endpoint.category == DATASET_SOURCE for endpoint in endpoints) == 1
    assert not any(endpoint.fallback for endpoint in endpoints)


def test_the_reported_dataset_env_follows_the_configured_mirror() -> None:
    mirror_url = "https://mirror.example"
    targets = [
        Endpoint("model api", MODEL_API, MODEL),
        Endpoint("huggingface", DATASET_SOURCE, HF),
        Endpoint(
            "hf-mirror",
            DATASET_SOURCE,
            f"{mirror_url}/api/datasets?limit=1",
            fallback=True,
            env={"HF_ENDPOINT": mirror_url},
        ),
    ]
    rules = {
        "env-proxy": {
            targets[0].url: (True, "HTTP 200"),
            targets[1].url: (False, "blocked"),
            targets[2].url: (True, "HTTP 200"),
        },
        "direct": {targets[2].url: (True, "HTTP 200")},
    }
    report = run_checks(
        targets,
        fetch=offline_fetch(rules, []),
        channels=CHANNELS,
        default_channel="env-proxy",
        timeout=0.01,
    )
    assert report.dataset_env() == {"HF_ENDPOINT": mirror_url}
    assert f"HF_ENDPOINT={mirror_url}" in (report.hint() or "")


def _run_pipeline(settings, connectivity: FakeConnectivity):
    llm = FakeLLM(happy_responses(), texts={"report": REPORT})
    deps = build_deps(settings, llm, ui=SilentUI(), connectivity=connectivity)
    return run_task("Test Paper 0", deps)


def test_the_mirror_reaches_the_repro_script_when_the_dataset_host_is_blocked(
    settings, monkeypatch
) -> None:
    # The machine may already point HF_ENDPOINT somewhere; the agent must override it
    # with the mirror its own probe verified, not inherit whatever was there.
    monkeypatch.setenv("HF_ENDPOINT", "https://decoy.invalid")
    result = _run_pipeline(settings, FakeConnectivity({DATASET_SOURCE: "degraded"}))

    assert result.status == "completed", result.message
    assert result.state["connectivity"]["dataset_env"] == {"HF_ENDPOINT": HF_MIRROR_URL}
    assert result.state["exec_result"]["metrics"]["hf_endpoint_set"] == 1
    assert f"HF_ENDPOINT={HF_MIRROR_URL}" in result.state["exec_result"]["stdout_tail"]
    assert (Path(result.state["run_dir"]) / "logs" / "connectivity.json").is_file()


def test_no_mirror_is_forced_when_the_dataset_host_answers(settings, monkeypatch) -> None:
    # A machine-wide HF_ENDPOINT is a route nobody probed, so it must not reach the script.
    monkeypatch.setenv("HF_ENDPOINT", "https://decoy.invalid")
    result = _run_pipeline(settings, FakeConnectivity())

    assert result.status == "completed", result.message
    assert result.state["connectivity"]["dataset_env"] == {"HF_ENDPOINT": None}
    assert result.state["exec_result"]["metrics"]["hf_endpoint_set"] == 0
    assert '"message": "HF_ENDPOINT="' in result.state["exec_result"]["stdout_tail"]
