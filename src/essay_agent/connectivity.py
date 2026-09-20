"""What the agent can actually reach, and through which channel.

This replaces a single-host socket probe whose verdict ("network: NOT reachable")
was wrong in two ways: it picked one frequently-blocked host to stand in for the
whole network, and it used a raw socket, which never takes the proxy that the
pipeline's HTTP clients use. A blocked ``huggingface.co`` therefore read as "this
machine is offline" and steered the plan away from datasets that were reachable.

Four rules instead:

1. Endpoints are grouped by what the agent needs them for - the model API, the
   paper sources, the dataset sources.
2. Every endpoint is tried through the channels real traffic uses, so the verdict
   matches what requests/httpx will experience; a raw socket is only used to
   explain a failure, never to conclude one.
3. The verdict is per endpoint ("arxiv: ok", "huggingface: blocked"), never a
   single global flag.
4. The dataset group carries a fallback: if the primary host is unreachable but a
   mirror answers, the group is ``degraded`` and :meth:`ConnectivityReport.dataset_env`
   hands over the environment variable that makes the mirror work - instead of
   declaring that no data can be fetched.
   That method also takes ``HF_ENDPOINT`` *back* when the primary host answered: an
   inherited mirror is a route nobody verified, and it would silently win over the
   verdict the probe just reached.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

from essay_agent.config import DEFAULT_DATASET_MIRROR_URL, Settings

MODEL_API = "model_api"
PAPER_SOURCE = "paper_source"
DATASET_SOURCE = "dataset_source"

CATEGORY_LABELS = {
    MODEL_API: "model api",
    PAPER_SOURCE: "paper sources",
    DATASET_SOURCE: "dataset sources",
}

HF_DATASETS_API = "https://huggingface.co/api/datasets?limit=1"
HF_MIRROR_URL = DEFAULT_DATASET_MIRROR_URL
HF_ENDPOINT_VAR = "HF_ENDPOINT"

# The proxy variables an HTTP client inside the *generated script* actually reads:
# `urllib`/`requests` take either case, `httpx` prefers the upper-case names, and nothing
# consults `GIT_HTTP_PROXY`. A ``None`` value for one of these means "remove it".
PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

_MODEL_APIS = {
    "openai": "https://api.openai.com/v1/models",
    "deepseek": "https://api.deepseek.com/",
    "anthropic": "https://api.anthropic.com/v1/models",
}
_PAPER_APIS = {
    "arxiv": (
        "arxiv",
        "https://export.arxiv.org/api/query?search_query=all:electron&max_results=1",
    ),
    "semanticscholar": (
        "semantic scholar",
        "https://api.semanticscholar.org/graph/v1/paper/search?query=adam&limit=1",
    ),
    "openalex": ("openalex", "https://api.openalex.org/works?search=adam&per-page=1"),
}

# An attempt that never ran because the stage ran out of its probe budget.
BUDGET_CHANNEL = "budget"

# A failure that came back this fast looks like a dropped connection and is worth one
# retry; a *hang* is not a flake, and paying the timeout twice is what ate the budget.
RETRY_MAX_SECONDS = 1.0

# proxies: None ("let the client decide"), {} ("no proxy") or {"https": url}.
Fetch = Callable[[str, "dict[str, str | None] | None", float], "tuple[bool, str, float]"]
# What an HTTP client asks for before its first request: the route to pin itself to.
RouteProvider = Callable[[], "Route | None"]


@dataclass(frozen=True)
class Route:
    """The channel that actually reached one category of endpoints.

    ``proxies=None`` means the route *is* "let the client decide" (the env proxy),
    so pinning it keeps ``trust_env`` on. Every other value is a concrete route and
    is pinned with ``trust_env=False``, which is what stops a stale
    ``HTTP_PROXY``/``ALL_PROXY``/``NO_PROXY`` from hijacking the real clients.
    """

    name: str
    proxies: dict[str, str | None] | None


def pin_session(session: Any, route: Route | None) -> Any:
    """Make ``session`` use exactly ``route``; ``None`` leaves it as it was.

    ``requests`` only consults proxy environment variables, so a machine whose env
    carries a dead ``HTTP_PROXY`` (or whose proxy lives only in the OS settings)
    otherwise ignores the channel the probe just validated.
    """
    if route is None:
        return session
    if route.proxies is None:
        session.trust_env = True
        session.proxies = {}
    else:
        session.trust_env = False
        session.proxies = dict(route.proxies)
        # trust_env=False also drops REQUESTS_CA_BUNDLE/CURL_CA_BUNDLE, which a
        # corporate user needs for TLS; carry it onto the session instead of losing it.
        if getattr(session, "verify", True) is True and (bundle := ca_bundle()) is not None:
            session.verify = bundle
    return session


def ca_bundle(env: dict[str, str] | None = None) -> str | None:
    """The TLS bundle a corporate user set, which ``trust_env=False`` would drop."""
    source = os.environ if env is None else env
    return source.get("REQUESTS_CA_BUNDLE") or source.get("CURL_CA_BUNDLE") or None


def build_http_client(route: Route | None) -> Any:
    """An ``httpx`` client pinned to a verified route; ``None`` keeps the default one.

    The langchain chat models accept an ``http_client`` and hand it straight to the
    provider SDK. Only a *concrete* route gets one: ``route=None`` (nothing was
    verified) and ``route.proxies=None`` ("the environment is the verified route")
    must both leave langchain's own client alone, because pinning there would
    replace a working environment with our guess. Everything else is pinned with
    ``trust_env=False`` so a stale ``HTTP_PROXY``/``NO_PROXY`` cannot hijack a call
    the probe already answered, and the TLS bundle is carried over by hand.
    """
    if route is None or route.proxies is None:
        return None
    import httpx

    proxy = route.proxies.get("https") or route.proxies.get("http")
    kwargs: dict[str, Any] = {"trust_env": False}
    if proxy:
        kwargs["proxy"] = proxy
    if (bundle := ca_bundle()) is not None:
        kwargs["verify"] = bundle
    return httpx.Client(**kwargs)


@dataclass(frozen=True)
class Endpoint:
    """One thing the agent might need to reach."""

    name: str
    category: str
    url: str
    fallback: bool = False  # an alternative route for the same need
    priority: bool = False  # decides a go/no-go verdict: probe it before the detail
    # Environment a client needs in order to actually use this endpoint (the mirror
    # is only reachable when HF_ENDPOINT points at it).
    env: dict[str, str] | None = None


@dataclass
class Attempt:
    channel: str
    ok: bool
    detail: str
    seconds: float


@dataclass
class EndpointReport:
    """Every channel we tried for one endpoint, in order."""

    endpoint: Endpoint
    attempts: list[Attempt] = field(default_factory=list)
    # Set when the budget ran out before this endpoint had tried every channel.
    skipped_for_budget: bool = False

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
    def probed(self) -> bool:
        return any(attempt.channel != BUDGET_CHANNEL for attempt in self.attempts)

    @property
    def unfinished(self) -> bool:
        """The budget stopped us short, so no verdict may be claimed for this endpoint."""
        return self.skipped_for_budget and not self.ok

    @property
    def settled(self) -> bool:
        """No point trying again: it answered, or the budget is already gone."""
        return self.ok or self.skipped_for_budget

    @property
    def why(self) -> str:
        failures = [f"{a.channel}: {a.detail}" for a in self.attempts if not a.ok]
        return "; ".join(failures) or "not attempted"

    def describe(self) -> str:
        if self.ok:
            attempt = next(a for a in self.attempts if a.ok)
            return f"{self.endpoint.name} ok ({attempt.detail} via {attempt.channel})"
        return f"{self.endpoint.name} blocked ({self.why})"


@dataclass
class ConnectivityReport:
    """The per-endpoint verdicts, plus the action the caller should take."""

    reports: list[EndpointReport]
    default_channel: str = "direct"
    proxy_hint: str | None = None
    # channel name -> the proxies argument it stands for, so a verdict can be
    # turned back into something a client can pin itself to.
    channels: dict[str, dict[str, str | None] | None] = field(default_factory=dict)

    def group(self, category: str) -> list[EndpointReport]:
        return [item for item in self.reports if item.endpoint.category == category]

    def route(self, category: str) -> Route | None:
        """The channel that reached ``category`` - ``None`` if none was verified."""
        items = self.group(category)
        winner = next((item for item in items if item.ok and not item.endpoint.fallback), None)
        if winner is None:
            winner = next((item for item in items if item.ok), None)
        if winner is None or winner.channel not in self.channels:
            return None
        return Route(winner.channel, self.channels[winner.channel])

    def state(self, category: str) -> str:
        """``ok`` / ``degraded`` / ``unreachable`` / ``unknown`` for one category."""
        items = self.group(category)
        primary = next((item for item in items if item.ok and not item.endpoint.fallback), None)
        if primary is not None:
            return "ok" if primary.channel == self.default_channel else "degraded"
        if any(item.ok for item in items):
            return "degraded"
        # Only claim "unreachable" for what we really finished probing: an endpoint the
        # budget cut short must read as unknown, never as a confident negative.
        if items and not any(item.unfinished for item in items):
            return "unreachable"
        return "unknown"

    def dataset_env(self) -> dict[str, str | None]:
        """Environment the repro script needs to reach the datasets it uses.

        The script runs in its own process, so the channel verified here has to be
        spelled out rather than assumed: ``requests`` reads the environment (then, on
        Windows, the registry) while ``httpx`` reads only the environment.
        :func:`route_env` writes the route out, and its ``None`` values *remove* a
        variable - taking back a dead inherited ``HTTP_PROXY`` is the whole point of a
        "direct" verdict.

        ``HF_ENDPOINT`` belongs to the route in both directions: it is *written* when
        the mirror is what answered, and *removed* when the primary host answered,
        because an inherited endpoint would send the real download somewhere this run
        never probed. Only the primary host answering is enough to know that.

        Nothing is reported when no dataset host answered at all, so an unverified run
        never rewrites the child's environment on a guess.
        """
        dataset = self.group(DATASET_SOURCE)
        if not any(item.ok for item in dataset):
            return {}
        env = route_env(self.route(DATASET_SOURCE))
        primary = next((item for item in dataset if item.ok and not item.endpoint.fallback), None)
        if primary is not None:
            env[HF_ENDPOINT_VAR] = None
        else:
            for item in dataset:
                if item.ok and item.endpoint.fallback and item.endpoint.env:
                    env.update(dict(item.endpoint.env))
                    break
        return env

    def hint(self) -> str | None:
        if self.state(MODEL_API) == "unknown":
            return "connectivity could not be verified within the probe budget"
        if self.state(MODEL_API) == "unreachable":
            return "the model API is unreachable: stop before spending a run on planning"
        dataset = self.group(DATASET_SOURCE)
        primary_ok = any(item.ok and not item.endpoint.fallback for item in dataset)
        mirrors = [item for item in dataset if item.ok and item.endpoint.fallback]
        if not primary_ok and mirrors:
            setting = ", ".join(f"{k}={v}" for k, v in (mirrors[0].endpoint.env or {}).items())
            return f"set {setting} for dataset downloads (mirror '{mirrors[0].endpoint.name}' answered)"
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
        """The environment-report block the prompts receive."""
        tried = sorted({a.channel for item in self.reports for a in item.attempts})
        header = (
            f"connectivity (default channel: {self.default_channel}; tried: {', '.join(tried)}):"
        )
        out = [header]
        for category, label in CATEGORY_LABELS.items():
            detail = " | ".join(item.describe() for item in self.group(category))
            out.append(f"  - {label}: {self.state(category)} - {detail}")
        if (hint := self.hint()) is not None:
            out.append(f"  - action: {hint}")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_channel": self.default_channel,
            "os_proxy": self.proxy_hint,
            "categories": {category: self.state(category) for category in CATEGORY_LABELS},
            "dataset_env": self.dataset_env(),
            "hint": self.hint(),
            "endpoints": [
                {"endpoint": asdict(item.endpoint), "attempts": [asdict(a) for a in item.attempts]}
                for item in self.reports
            ],
        }


# ------------------------------------------------------------------- channels
def windows_proxy() -> str | None:
    """The OS-level proxy real clients inherit on Windows."""
    if os.name != "nt":  # pragma: no cover - platform specific
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
    except Exception:  # pragma: no cover - registry layout dependent
        return None


def _run_tool(command: list[str], timeout: float = 2.0) -> str | None:
    """Run a small system command; any failure (missing tool, timeout) means ``None``."""
    import subprocess

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _clean(value: str | None) -> str | None:
    """``gsettings`` quotes its output: ``'127.0.0.1'`` -> ``127.0.0.1``."""
    if value is None:
        return None
    return value.strip().strip("'\"").strip() or None


def _macos_proxy() -> str | None:
    """``scutil --proxy`` prints one ``Key : value`` line per setting."""
    output = _run_tool(["scutil", "--proxy"])
    if not output:
        return None
    values: dict[str, str] = {}
    for line in output.splitlines():
        key, _, value = line.partition(":")
        values[key.strip()] = value.strip()
    for scheme in ("HTTPS", "HTTP"):
        if values.get(f"{scheme}Enable", "1") not in {"1", "true"}:
            continue  # the entry is listed but switched off
        host = values.get(f"{scheme}Proxy")
        if host and host not in {"0", "0.0.0.0"}:
            return normalise_proxy(host, values.get(f"{scheme}Port"))
    return None


def _linux_proxy() -> str | None:
    """GNOME first (``gsettings``), then KDE (``kreadconfig``)."""
    mode = _clean(_run_tool(["gsettings", "get", "org.gnome.system.proxy", "mode"]))
    if mode == "manual":
        host = _clean(_run_tool(["gsettings", "get", "org.gnome.system.proxy.https", "host"]))
        port = _clean(_run_tool(["gsettings", "get", "org.gnome.system.proxy.https", "port"]))
        if host:
            return normalise_proxy(host, port)
    for tool in ("kreadconfig6", "kreadconfig5"):
        value = _clean(
            _run_tool([tool, "--group", "Proxy", "--key", "httpsProxy"])
            or _run_tool([tool, "--group", "Proxy", "--key", "httpProxy"])
        )
        if value:
            return value if "://" in value else f"http://{value}"
    return None


@lru_cache(maxsize=1)
def os_proxy() -> str | None:
    """The system-wide proxy a plain HTTP client inherits on *this* machine.

    Windows reads the registry, macOS asks ``scutil``, Linux asks GNOME/KDE. Any failure
    means "no system proxy": the caller then tries a direct connection instead of
    crashing. Cached, because it is consulted several times per run and may spawn a
    subprocess.
    """
    if os.name == "nt":
        return windows_proxy()
    if sys.platform == "darwin":
        return _macos_proxy()
    return _linux_proxy()


def normalise_proxy(host: str, port: str | None = None) -> str:
    """``127.0.0.1`` + ``7897`` -> ``http://127.0.0.1:7897``; a URL is kept as-is."""
    host = (host or "").strip()
    if "://" in host:
        return host
    port = (port or "").strip()
    return f"http://{host}:{port}" if port else f"http://{host}"


def proxies_for(proxy: str | None) -> dict[str, str] | None:
    """The proxies mapping for one proxy URL (``None`` = let the client decide)."""
    if not proxy:
        return None
    url = proxy if "://" in proxy else f"http://{proxy}"
    return {"http": url, "https": url}


def route_env(route: Route | None) -> dict[str, str | None]:
    """Turn a verified route into the environment a child process needs.

    A ``None`` value means "remove this variable", not "set it empty": a direct route is
    only direct once an inherited ``HTTP_PROXY``/``ALL_PROXY`` is gone, and a script that
    keeps a dead one fails no matter how healthy the probe found the network.

    ``Route`` carries three meanings, so there are three answers: ``None`` (nothing was
    verified - change nothing), ``proxies=None`` ("let the client decide" *is* the
    verified route, and the environment already describes it - also change nothing), and
    a concrete mapping (write it out, or clear every proxy variable when the mapping
    holds no URL, which is how the direct channel is spelled).
    """
    if route is None or route.proxies is None:
        return {}
    url = next((value for value in route.proxies.values() if value), None)
    if url is None:
        return dict.fromkeys(PROXY_ENV_VARS)
    env: dict[str, str | None] = {}
    for name in PROXY_ENV_VARS:
        env[name] = url
    if (bundle := ca_bundle()) is not None:
        env["REQUESTS_CA_BUNDLE"] = bundle
        env["CURL_CA_BUNDLE"] = bundle
    return env


def default_channel_name(explicit_proxy: str | None = None) -> str:
    """Which route a plain ``requests`` call takes here: env vars, else the OS proxy."""
    if explicit_proxy:
        return "configured-proxy"
    for name in ("https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        if os.environ.get(name):
            return "env-proxy"
    return "os-proxy" if os_proxy() else "direct"


def active_channels(
    explicit_proxy: str | None = None,
) -> dict[str, dict[str, str | None] | None]:
    """Channel name -> proxies argument; ``None`` lets the client decide.

    The first entry is the route the pipeline itself takes, so its verdict is the
    one that decides the report. The rest only exist to explain *why* a failure
    happened; they are ordered cheapest-first, because a bare direct connection is the
    least likely route on a machine that has a proxy configured at all (and costs a
    full timeout to rule out). A channel that duplicates the first route is not added:
    two names for the same proxy would turn one flaky attempt into a bogus verdict.

    An ``explicit_proxy`` (``ESSAY_AGENT_HTTP_PROXY``) wins over everything else: on a
    machine whose proxy we cannot detect, the user can simply say where it is.
    """
    default = default_channel_name(explicit_proxy)
    channels: dict[str, dict[str, str | None] | None] = {}
    if default == "configured-proxy":
        channels[default] = proxies_for(explicit_proxy)
    else:
        channels[default] = None
    if default != "os-proxy" and (proxy := os_proxy()) is not None:
        channels["os-proxy"] = proxies_for(proxy)
    if default != "direct":
        channels["direct"] = {"http": None, "https": None}
    return channels


def fetch_with_requests(
    url: str, proxies: dict[str, str | None] | None, timeout: float
) -> tuple[bool, str, float]:
    """Any HTTP status counts as reachable; only transport errors fail."""
    import requests

    started = time.perf_counter()
    try:
        # requests merges the environment into the dict it is given, in place. Hand it
        # a copy, or the channel map we later pin clients to inherits the env's
        # (possibly broken) proxy under keys like "all".
        response = requests.get(
            url,
            timeout=timeout,
            proxies=dict(proxies) if proxies is not None else None,
            stream=True,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:90]}", time.perf_counter() - started
    return True, f"HTTP {response.status_code}", time.perf_counter() - started


def endpoints_for(settings: Settings) -> list[Endpoint]:
    """The endpoints this configuration will actually need.

    ``priority`` marks the ones that decide a go/no-go verdict - the model API, the
    first paper backend and both dataset hosts (primary *and* its mirror). Everything
    else is reporting detail, and is the first thing a tight budget may drop.
    """
    endpoints: list[Endpoint] = []
    api = _MODEL_APIS.get(settings.llm.provider) or settings.llm.base_url
    if api:
        endpoints.append(Endpoint("model api", MODEL_API, api.rstrip("/") + "/", priority=True))
    first_paper_backend = True
    for backend in settings.search.backends:
        key = str(backend).lower().replace("-", "").replace("_", "")
        entry = _PAPER_APIS.get(key)
        if entry is not None:
            endpoints.append(
                Endpoint(entry[0], PAPER_SOURCE, entry[1], priority=first_paper_backend)
            )
            first_paper_backend = False
    endpoints.append(Endpoint("huggingface", DATASET_SOURCE, HF_DATASETS_API, priority=True))
    mirror = (settings.search.dataset_mirror_url or "").strip().rstrip("/")
    if mirror:
        endpoints.append(
            Endpoint(
                "hf-mirror",
                DATASET_SOURCE,
                f"{mirror}/api/datasets?limit=1",
                fallback=True,
                priority=True,
                env={HF_ENDPOINT_VAR: mirror},
            )
        )
    return endpoints


# -------------------------------------------------------------------- probing
def probe_channel(
    reports: list[EndpointReport],
    name: str,
    proxies: dict[str, str | None] | None,
    *,
    fetch: Fetch,
    timeout: float,
    deadline: float,
    clock: Callable[[], float],
    attempts: int,
    retry_max_seconds: float,
) -> None:
    """Give every endpoint that is still open one go at this channel.

    Endpoints are visited in priority order and never revisited once settled, so a
    hanging host cannot eat the budget of the endpoints that decide feasibility.
    """
    for report in sorted(reports, key=lambda item: not item.endpoint.priority):
        if report.settled:
            continue
        last_seconds: float | None = None
        for _ in range(attempts):
            # Only a failure that returned quickly looks like a dropped connection; a
            # hang is not a flake, and retrying it doubles the loss for nothing.
            if last_seconds is not None and last_seconds > retry_max_seconds:
                break
            remaining = deadline - clock()
            if remaining <= 0:
                report.skipped_for_budget = True
                report.attempts.append(
                    Attempt(BUDGET_CHANNEL, False, "skipped: probe budget exhausted", 0.0)
                )
                break
            ok, detail, seconds = fetch(report.endpoint.url, proxies, min(timeout, remaining))
            report.attempts.append(Attempt(name, ok, detail, seconds))
            last_seconds = seconds
            if ok:
                break


def run_checks(
    endpoints: Iterable[Endpoint],
    *,
    fetch: Fetch = fetch_with_requests,
    timeout: float = 5.0,
    budget_seconds: float = 20.0,
    channels: dict[str, dict[str, str | None] | None] | None = None,
    default_channel: str | None = None,
    explicit_proxy: str | None = None,
    clock: Callable[[], float] = time.monotonic,
    retries: int = 1,
    retry_max_seconds: float = RETRY_MAX_SECONDS,
) -> ConnectivityReport:
    """Probe every endpoint, channel by channel, under one shared budget.

    Channel-major on purpose: the route real traffic uses is settled for *every*
    endpoint before any fallback channel is explored, so running out of budget costs
    detail rather than the verdict. Every attempt is bounded by what is left of the
    budget, so the probe cannot overshoot it.
    """
    active = channels or active_channels(explicit_proxy)
    deadline = clock() + budget_seconds
    reports = [EndpointReport(endpoint) for endpoint in endpoints]
    for index, (name, proxies) in enumerate(active.items()):
        probe_channel(
            reports,
            name,
            proxies,
            fetch=fetch,
            timeout=timeout,
            deadline=deadline,
            clock=clock,
            attempts=retries + 1 if index == 0 else 1,
            retry_max_seconds=retry_max_seconds,
        )
    return ConnectivityReport(
        reports=reports,
        default_channel=default_channel or next(iter(active)),
        proxy_hint=explicit_proxy or os_proxy(),
        channels=dict(active),
    )


class ConnectivityProbe:
    """The collaborator ``Deps`` carries; tests swap in their own.

    Besides the full :meth:`run` report, it answers :meth:`route` - the channel
    that reaches one category - so the HTTP clients can pin themselves to what was
    actually verified instead of trusting whatever the environment says.
    """

    def __init__(
        self,
        *,
        fetch: Fetch = fetch_with_requests,
        timeout: float = 5.0,
        budget_seconds: float = 20.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.fetch = fetch
        self.timeout = timeout
        self.budget_seconds = budget_seconds
        self.clock = clock
        self._routes: dict[str, Route | None] = {}

    def run(self, settings: Settings) -> ConnectivityReport:
        report = run_checks(
            endpoints_for(settings),
            fetch=self.fetch,
            timeout=self.timeout,
            budget_seconds=self.budget_seconds,
            clock=self.clock,
            explicit_proxy=settings.http_proxy,
        )
        for category in CATEGORY_LABELS:
            self._routes[category] = report.route(category)
        return report

    def route(self, settings: Settings, category: str) -> Route | None:
        """The route to ``category``, probing just that category on first use."""
        if category not in self._routes:
            endpoints = [item for item in endpoints_for(settings) if item.category == category]
            report = run_checks(
                endpoints,
                fetch=self.fetch,
                timeout=self.timeout,
                explicit_proxy=settings.http_proxy,
                budget_seconds=self.budget_seconds,
                clock=self.clock,
            )
            self._routes[category] = report.route(category)
        return self._routes[category]

    def provider(self, settings: Settings, category: str) -> RouteProvider:
        """A zero-arg callable the clients can keep and call before their first request."""
        return lambda: self.route(settings, category)
