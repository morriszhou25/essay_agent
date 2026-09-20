"""Prototype: the connectivity report that should replace today's network probe.

Standalone on purpose - it imports nothing from ``essay_agent``, so it cannot
affect the pipeline. ``run_checks`` is written to be lifted into
``nodes/base.py`` later, with a thin adapter for the pipeline's text format.

What it does differently from ``network_reachable()``:

1. No single host stands in for "the network". Endpoints are grouped by what the
   agent needs them for: the model API, the paper sources, the dataset sources.
2. Each endpoint is tried through the channels real traffic uses (environment
   proxy, OS proxy, direct) instead of a raw socket, so the verdict matches what
   requests/httpx will actually experience.
3. The verdict is per endpoint ("arxiv: ok", "huggingface: blocked") instead of a
   single global ``network: NOT reachable``.
4. The dataset group has a fallback: if huggingface.co is blocked but a mirror
   answers, the group is "degraded" and the report hands over the concrete
   setting to use, instead of declaring that no data can be fetched.

    python scripts/connectivity_report.py             # live check
    python scripts/connectivity_report.py --json      # machine readable
    python scripts/connectivity_report.py --selftest  # offline logic check
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

# proxies is None ("let the client decide"), {} ("no proxy") or {"https": url}.
Fetch = Callable[[str, "dict[str, str | None] | None", float], "tuple[bool, str, float]"]


@dataclass(frozen=True)
class Endpoint:
    name: str
    category: str
    url: str
    fallback: bool = False  # an alternative route for the same need


@dataclass
class Attempt:
    channel: str
    ok: bool
    detail: str
    seconds: float


@dataclass
class EndpointReport:
    endpoint: Endpoint
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return any(attempt.ok for attempt in self.attempts)

    @property
    def channel(self) -> str:
        for attempt in self.attempts:
            if attempt.ok:
                return attempt.channel
        return "-"

    @property
    def why(self) -> str:
        failures = [f"{a.channel}: {a.detail}" for a in self.attempts if not a.ok]
        return "; ".join(failures) or "not attempted"

    def describe(self) -> str:
        if self.ok:
            attempt = next(a for a in self.attempts if a.ok)
            return f"{self.endpoint.name} ok ({attempt.detail} via {attempt.channel})"
        return f"{self.endpoint.name} blocked ({self.why})"


DEFAULT_ENDPOINTS: tuple[Endpoint, ...] = (
    Endpoint("model api", "model_api", "https://api.deepseek.com/"),
    Endpoint(
        "arxiv",
        "paper_source",
        "https://export.arxiv.org/api/query?search_query=all:electron&max_results=1",
    ),
    Endpoint(
        "semantic scholar",
        "paper_source",
        "https://api.semanticscholar.org/graph/v1/paper/search?query=adam&limit=1",
    ),
    Endpoint("openalex", "paper_source", "https://api.openalex.org/works?search=adam&per-page=1"),
    Endpoint("huggingface", "dataset_source", "https://huggingface.co/api/datasets?limit=1"),
    Endpoint(
        "hf-mirror", "dataset_source", "https://hf-mirror.com/api/datasets?limit=1", fallback=True
    ),
)

CATEGORY_LABELS = {
    "model_api": "model api",
    "paper_source": "paper sources",
    "dataset_source": "dataset sources",
}
MIRROR_HINT = "HF_ENDPOINT=https://hf-mirror.com"


# ------------------------------------------------------------------- channels
def windows_proxy() -> str | None:
    """The OS-level proxy real clients inherit on Windows."""
    if os.name != "nt":
        return None
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        )
        enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
        if not enabled:
            return None
        server, _ = winreg.QueryValueEx(key, "ProxyServer")
        return str(server) if server else None
    except Exception:
        return None


def default_channel_name() -> str:
    """Which route a plain ``requests`` call takes here: env vars, else the OS proxy."""
    for name in ("https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        if os.environ.get(name):
            return "env-proxy"
    return "os-proxy" if windows_proxy() else "direct"


def active_channels() -> dict[str, dict[str, str | None] | None]:
    """Channel name -> proxies argument; ``None`` lets the client decide.

    The first entry is the route the pipeline itself would take, so its verdict is
    the one that matters; the rest only exist to explain *why* it failed. Entries
    that would duplicate the first route are not added - two names for the same
    proxy would turn a flaky first attempt into a bogus "degraded" verdict.
    """
    default = default_channel_name()
    channels: dict[str, dict[str, str | None] | None] = {default: None}
    if default != "direct":
        channels["direct"] = {"http": None, "https": None}
    if default != "os-proxy" and (proxy := windows_proxy()) is not None:
        url = proxy if "://" in proxy else f"http://{proxy}"
        channels["os-proxy"] = {"http": url, "https": url}
    return channels


def fetch_with_requests(
    url: str, proxies: dict[str, str | None] | None, timeout: float
) -> tuple[bool, str, float]:
    """Any HTTP status counts as reachable; only transport errors fail."""
    import requests

    started = time.perf_counter()
    try:
        response = requests.get(url, timeout=timeout, proxies=proxies, stream=True)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:90]}", time.perf_counter() - started
    return True, f"HTTP {response.status_code}", time.perf_counter() - started


def probe_endpoint(
    endpoint: Endpoint,
    channels: dict[str, dict[str, str | None] | None],
    *,
    fetch: Fetch,
    timeout: float,
    retries: int = 1,
) -> EndpointReport:
    """Try the channels in order and stop at the first one that answers.

    The channel that decides the verdict gets ``retries`` extra attempts: a local
    proxy that drops one connection must not read as "unreachable".
    """
    report = EndpointReport(endpoint)
    for index, (name, proxies) in enumerate(channels.items()):
        for _ in range(retries + 1 if index == 0 else 1):
            ok, detail, seconds = fetch(endpoint.url, proxies, timeout)
            report.attempts.append(Attempt(name, ok, detail, seconds))
            if ok:
                return report
    return report


# --------------------------------------------------------------------- report
@dataclass
class ConnectivityReport:
    reports: list[EndpointReport]
    default_channel: str = "direct"
    proxy_hint: str | None = None

    def group(self, category: str) -> list[EndpointReport]:
        return [item for item in self.reports if item.endpoint.category == category]

    def state(self, category: str) -> str:
        """``ok`` / ``degraded`` / ``unreachable`` for one category of need."""
        items = self.group(category)
        primary = next((item for item in items if item.ok and not item.endpoint.fallback), None)
        if primary is not None:
            return "ok" if primary.channel == self.default_channel else "degraded"
        return "degraded" if any(item.ok for item in items) else "unreachable"

    def hint(self) -> str | None:
        if self.state("model_api") == "unreachable":
            return "the model API is unreachable: stop before spending a run on planning"
        dataset = self.group("dataset_source")
        primary_ok = any(item.ok and not item.endpoint.fallback for item in dataset)
        mirrors = [item for item in dataset if item.ok and item.endpoint.fallback]
        if not primary_ok and mirrors:
            return (
                f"set {MIRROR_HINT} for dataset downloads "
                f"(mirror '{mirrors[0].endpoint.name}' answered)"
            )
        degraded = [
            label
            for category, label in CATEGORY_LABELS.items()
            if self.state(category) == "degraded"
        ]
        if degraded:
            return (
                f"{', '.join(degraded)} only answer through a non-default channel; "
                f"the default channel ({self.default_channel}) does not reach them"
            )
        return None

    def lines(self) -> list[str]:
        """The replacement for ``network: NOT reachable`` in the environment block."""
        tried = sorted({a.channel for item in self.reports for a in item.attempts})
        out = [
            f"connectivity (default channel: {self.default_channel}; tried: {', '.join(tried)}):"
        ]
        for category, label in CATEGORY_LABELS.items():
            detail = " | ".join(item.describe() for item in self.group(category))
            out.append(f"  - {label}: {self.state(category)} - {detail}")
        if (hint := self.hint()) is not None:
            out.append(f"  - action: {hint}")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_channel": self.default_channel,
            "windows_proxy": self.proxy_hint,
            "categories": {category: self.state(category) for category in CATEGORY_LABELS},
            "hint": self.hint(),
            "endpoints": [
                {"endpoint": asdict(item.endpoint), "attempts": [asdict(a) for a in item.attempts]}
                for item in self.reports
            ],
        }


def run_checks(
    endpoints: Iterable[Endpoint] = DEFAULT_ENDPOINTS,
    *,
    fetch: Fetch = fetch_with_requests,
    timeout: float = 6.0,
    channels: dict[str, dict[str, str | None] | None] | None = None,
    default_channel: str | None = None,
) -> ConnectivityReport:
    """Probe every endpoint once and package the verdicts."""
    active = channels or active_channels()
    reports = [
        probe_endpoint(endpoint, active, fetch=fetch, timeout=timeout) for endpoint in endpoints
    ]
    return ConnectivityReport(
        reports=reports,
        default_channel=default_channel or next(iter(active)),
        proxy_hint=windows_proxy(),
    )


# ------------------------------------------------------------------- selftest
def _offline_fetch(
    rules: dict[str, dict[str, tuple[bool, str]]],
    channels: dict[str, dict[str, str | None] | None],
) -> Fetch:
    """A fetch that reports what each channel would do, keyed by channel name."""
    by_proxies = {id(proxies): name for name, proxies in channels.items()}

    def fetch(url: str, proxies: dict[str, str | None] | None, timeout: float):
        channel = by_proxies[id(proxies)]
        ok, detail = rules.get(channel, {}).get(url, (False, "blocked"))
        return ok, detail, 0.01

    return fetch


def selftest() -> int:
    """Deterministic checks of the decision logic, with no network involved."""
    channels = {"env-proxy": None, "os-proxy": {"http": "x", "https": "x"}, "direct": {}}
    hf = "https://huggingface.co/api/datasets?limit=1"
    mirror = "https://hf-mirror.com/api/datasets?limit=1"
    failures: list[str] = []

    def check(name: str, condition: bool, extra: str = "") -> None:
        mark = "PASS" if condition else "FAIL"
        print(f"  {mark}  {name}" + ("" if condition else f"  -> got: {extra}"))
        if not condition:
            failures.append(name)

    def run(
        rules: dict[str, dict[str, tuple[bool, str]]], default: str = "env-proxy"
    ) -> ConnectivityReport:
        return run_checks(
            fetch=_offline_fetch(rules, channels),
            channels=channels,
            default_channel=default,
            timeout=0.01,
        )

    def everything_up(channel: str) -> dict[str, dict[str, tuple[bool, str]]]:
        return {channel: {endpoint.url: (True, "HTTP 200") for endpoint in DEFAULT_ENDPOINTS}}

    healthy = run(everything_up("env-proxy"))
    check(
        "healthy machine -> all three categories ok",
        all(healthy.state(c) == "ok" for c in CATEGORY_LABELS),
        {c: healthy.state(c) for c in CATEGORY_LABELS},
    )
    check("healthy machine -> no action needed", healthy.hint() is None, healthy.hint())

    # The same machine seen from a shell whose default channel is the OS proxy:
    # nothing to warn about, and definitely nothing to "fix".
    via_os_proxy = run(everything_up("os-proxy"), default="os-proxy")
    check(
        "healthy machine via the OS proxy -> still ok",
        all(via_os_proxy.state(c) == "ok" for c in CATEGORY_LABELS),
        {c: via_os_proxy.state(c) for c in CATEGORY_LABELS},
    )
    check("healthy machine via the OS proxy -> no action needed", via_os_proxy.hint() is None)

    # The bug we are fixing: HF answers only through the OS proxy.
    rules = everything_up("env-proxy")
    rules["env-proxy"].pop(hf)
    rules["os-proxy"] = {hf: (True, "HTTP 200")}
    proxied = run(rules)
    check(
        "huggingface via OS proxy -> degraded, not unreachable",
        proxied.state("dataset_source") == "degraded",
        proxied.state("dataset_source"),
    )
    check(
        "huggingface via OS proxy -> the other categories stay ok",
        proxied.state("model_api") == "ok" and proxied.state("paper_source") == "ok",
    )
    check(
        "huggingface via OS proxy -> the working channel is named",
        "os-proxy" in " ".join(proxied.lines()),
        proxied.lines(),
    )

    rules = everything_up("env-proxy")
    rules["env-proxy"].pop(hf)
    rules["direct"] = {mirror: (True, "HTTP 200")}
    mirrored = run(rules)
    check(
        "mirror fallback -> dataset sources degraded",
        mirrored.state("dataset_source") == "degraded",
        mirrored.state("dataset_source"),
    )
    check(
        "mirror fallback -> hands over HF_ENDPOINT",
        MIRROR_HINT in (mirrored.hint() or ""),
        mirrored.hint(),
    )

    offline = run({})
    check("offline -> model api unreachable", offline.state("model_api") == "unreachable")
    check("offline -> tells the caller to stop", "stop before" in (offline.hint() or ""))
    check(
        "offline -> no false 'reachable' anywhere",
        all(offline.state(c) == "unreachable" for c in CATEGORY_LABELS),
    )

    print(f"\nselftest: {'all checks passed' if not failures else f'{len(failures)} failed'}")
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    timeout = 6.0
    if "--timeout" in argv:
        timeout = float(argv[argv.index("--timeout") + 1])
    report = run_checks(timeout=timeout)
    if "--json" in argv:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print("\n".join(report.lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
