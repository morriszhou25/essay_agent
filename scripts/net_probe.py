"""Standalone network diagnostic for essay-agent.

Deliberately imports nothing from ``essay_agent``: this exists to explain what the
in-pipeline probe sees, without touching the pipeline. Run it with the same
interpreter the agent uses:

    python scripts/net_probe.py                      # default target list
    python scripts/net_probe.py hf-mirror.com:443    # extra/specific targets

For every target it separates the layers that ``socket.create_connection``
collapses into one "unreachable" verdict:

    DNS      - can the name be resolved at all, and to what addresses?
    TCP      - does a socket connect to each address, and how long does it take?
    TLS      - does a direct handshake work (raw socket + ssl, never a proxy)?
    HTTPS    - does a real request through the (proxy/TLS) stack succeed?
    PROXY    - which proxy settings are in force for this process?

``--proxy URL`` pins one proxy for the HTTPS column instead of the environment;
``--no-proxy`` forces the direct route. Comparing the two columns is what tells a
blocked host apart from a machine with no network at all.
"""

from __future__ import annotations

import json
import os
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request

DEFAULT_TARGETS = (
    # what the agent's own probe checks
    "huggingface.co:443",
    # what actually worked in the failing run
    "arxiv.org:443",
    "api.deepseek.com:443",
    # dataset hosting, and the usual mirror
    "cdn-lfs.huggingface.co:443",
    "hf-mirror.com:443",
    # a control the pipeline never touches
    "pypi.org:443",
)

PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)

DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def parse_target(text: str) -> tuple[str, int]:
    host, _, port = text.partition(":")
    return host, int(port or 443)


def resolve(host: str, port: int) -> tuple[str, list[str], float]:
    started = time.perf_counter()
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        return f"FAIL {type(exc).__name__}: {exc}", [], time.perf_counter() - started
    addresses = sorted({str(info[4][0]) for info in infos})
    return "ok", addresses, time.perf_counter() - started


def tcp_connect(address: str, port: int, timeout: float) -> tuple[str, float]:
    started = time.perf_counter()
    try:
        with socket.create_connection((address, port), timeout=timeout):
            return "ok", time.perf_counter() - started
    except OSError as exc:
        return f"FAIL {type(exc).__name__}: {exc}", time.perf_counter() - started


def https_get(host: str, timeout: float) -> tuple[str, float]:
    started = time.perf_counter()
    request = urllib.request.Request(
        f"https://{host}/", headers={"User-Agent": "essay-agent-netprobe"}
    )
    try:
        with DIRECT_OPENER.open(request, timeout=timeout) as response:
            return f"ok HTTP {response.status}", time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        return f"ok HTTP {exc.code} (reached the server)", time.perf_counter() - started
    except Exception as exc:
        return f"FAIL {type(exc).__name__}: {exc}", time.perf_counter() - started


def direct_tls(host: str, timeout: float) -> tuple[str, float]:
    """TLS handshake straight to the resolved address: no proxy, no env."""
    started = time.perf_counter()
    context = ssl.create_default_context()
    try:
        with (
            socket.create_connection((host, 443), timeout=timeout) as raw,
            context.wrap_socket(raw, server_hostname=host) as tls,
        ):
            return f"ok {tls.version()}", time.perf_counter() - started
    except Exception as exc:
        return f"FAIL {type(exc).__name__}: {exc}", time.perf_counter() - started


def windows_proxy() -> dict[str, str]:
    """The OS-level proxy the WinHTTP/urllib stack would inherit."""
    if os.name != "nt":
        return {}
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        )
        out: dict[str, str] = {}
        for name in ("ProxyEnable", "ProxyServer", "ProxyOverride", "AutoConfigURL"):
            try:
                value, _ = winreg.QueryValueEx(key, name)
                out[name] = str(value)
            except FileNotFoundError:
                pass
        return out
    except Exception as exc:  # pragma: no cover - platform dependent
        return {"error": f"{type(exc).__name__}: {exc}"}


def proxy_settings() -> dict[str, str]:
    settings = {name: os.environ[name] for name in PROXY_ENV_VARS if name in os.environ}
    settings.update({f"windows:{k}": v for k, v in windows_proxy().items()})
    return settings


def probe(
    target: str, *, tcp_timeout: float, http_timeout: float, proxy: str | None
) -> dict[str, object]:
    host, port = parse_target(target)
    dns_state, addresses, dns_seconds = resolve(host, port)
    row: dict[str, object] = {
        "target": f"{host}:{port}",
        "dns": dns_state,
        "dns_seconds": round(dns_seconds, 3),
        "addresses": addresses,
        "tcp": {},
        "tls": "",
        "https": "",
    }
    tcp: dict[str, str] = {}
    for address in addresses or [host]:
        state, seconds = tcp_connect(address, port, tcp_timeout)
        tcp[address] = f"{state} ({seconds:.2f}s)"
    row["tcp"] = tcp
    if any(value.startswith("ok") for value in tcp.values()):
        state, seconds = direct_tls(host, tcp_timeout)
        row["tls"] = f"{state} ({seconds:.2f}s)"
    else:
        row["tls"] = "skipped: no TCP connection"
    state, seconds = https_get_via(host, http_timeout, proxy)
    row["https"] = f"{state} ({seconds:.2f}s)" + (f" via {proxy}" if proxy else " (direct)")
    return row


def https_get_via(host: str, timeout: float, proxy: str | None) -> tuple[str, float]:
    """An HTTPS GET with an explicit proxy choice (``None`` = direct)."""
    if proxy is None:
        return https_get(host, timeout)
    started = time.perf_counter()
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    )
    request = urllib.request.Request(
        f"https://{host}/", headers={"User-Agent": "essay-agent-netprobe"}
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            return f"ok HTTP {response.status}", time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        return f"ok HTTP {exc.code} (reached the server)", time.perf_counter() - started
    except Exception as exc:
        return f"FAIL {type(exc).__name__}: {exc}", time.perf_counter() - started


def main(argv: list[str]) -> int:
    args = list(argv[1:])
    proxy: str | None = None
    if "--proxy" in args:
        index = args.index("--proxy")
        proxy = args[index + 1]
        del args[index : index + 2]
    if "--no-proxy" in args:
        args.remove("--no-proxy")
        os.environ["NO_PROXY"] = "*"
    targets = args or list(DEFAULT_TARGETS)
    tcp_timeout = float(os.environ.get("NETPROBE_TCP_TIMEOUT", "8"))
    http_timeout = float(os.environ.get("NETPROBE_HTTP_TIMEOUT", "10"))
    print(f"python: {sys.version.split()[0]}  socket default timeout: {socket.getdefaulttimeout()}")
    print(f"tcp timeout: {tcp_timeout}s  http timeout: {http_timeout}s")
    settings = proxy_settings()
    print("proxy settings:", json.dumps(settings, ensure_ascii=False) if settings else "(none)")
    print(f"https column uses: {proxy or 'no proxy (direct)'}")
    print("=" * 100)
    for target in targets:
        row = probe(target, tcp_timeout=tcp_timeout, http_timeout=http_timeout, proxy=proxy)
        verdict = "REACHABLE" if str(row["https"]).startswith("ok") else "NOT REACHABLE"
        print(f"{verdict:>13}  {row['target']}")
        print(f"               dns   : {row['dns']} ({row['dns_seconds']}s) {row['addresses']}")
        for address, state in dict(row["tcp"]).items():
            print(f"               tcp   : {address} -> {state}")
        print(f"               tls   : {row['tls']}")
        print(f"               https : {row['https']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
