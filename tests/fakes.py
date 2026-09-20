"""Offline stand-ins for the LLM, the paper search tool and the dataset prober."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel

from essay_agent.connectivity import (
    Attempt,
    ConnectivityReport,
    EndpointReport,
    Route,
    endpoints_for,
)
from essay_agent.runtime.runner import RunOutcome
from essay_agent.schemas.paper import Candidate, PaperRecord, PaperSection
from essay_agent.tools.dataset_probe import ProbeResult

SAMPLE_REPRO_SCRIPT = """
import argparse, json, os, sys, time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()
    out = args.out
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    total = 12
    if args.preflight:
        time.sleep(0.02)
        print(json.dumps({"event": "estimate", "estimated_full_seconds": 0.4,
                          "params": {"steps": total}}), flush=True)
        return 0
    for step in range(1, total + 1):
        print(json.dumps({"event": "progress", "step": step, "total": total}), flush=True)
        time.sleep(0.002)
    print(json.dumps({"event": "metric", "name": "accuracy", "value": 0.912}), flush=True)
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({"accuracy": 0.912, "baseline_accuracy": 0.884, "delta": 0.028}, handle)
    with open(os.path.join(out, "figures", "fig1.png"), "wb") as handle:
        handle.write(b"\\x89PNG\\r\\n\\x1a\\n")
    print(json.dumps({"event": "done", "metrics": {"accuracy": 0.912}}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""

_CODE_LABEL = re.compile(r"_code\d+$")


class FakeLLM:
    """Scripted LLM.

    ``responses`` maps either a call label or a pydantic schema class to a value
    (or a callable taking ``(system, user)``). Labels win over schema classes.
    """

    def __init__(
        self,
        responses: dict[Any, Any] | None = None,
        *,
        texts: dict[str, Any] | None = None,
        tool_results: dict[str, Any] | None = None,
        interrupt_labels: tuple[str, ...] = (),
        error_labels: dict[str, Exception] | None = None,
        model: str = "fake-model",
    ) -> None:
        self.responses = responses or {}
        self.texts = texts or {}
        self.tool_results = tool_results or {}
        self.interrupt_labels = set(interrupt_labels)
        self.error_labels = dict(error_labels or {})
        self.model = model
        self.calls: list[tuple[str, str | None, str, str]] = []
        self.max_tokens_seen: dict[str | None, int | None] = {}

    # --------------------------------------------------------------- protocol
    def model_name(self, role: str = "main") -> str:
        return self.model

    def text(
        self,
        system: str,
        user: str,
        *,
        role: str = "main",
        label: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(("text", label, "text", role))
        self.max_tokens_seen[label] = max_tokens
        self._maybe_interrupt(label)
        value = None
        if label and _CODE_LABEL.search(label):
            value = self._stage4_code(label)
        if value is None:
            value = self._lookup(self.texts, label, "default")
        return value(system, user) if callable(value) else str(value)

    def json(
        self,
        system: str,
        user: str,
        schema: type[BaseModel],
        *,
        role: str = "main",
        label: str | None = None,
        max_tokens: int | None = None,
    ) -> BaseModel:
        self.calls.append(("json", label, schema.__name__, role))
        self.max_tokens_seen[label] = max_tokens
        self._maybe_interrupt(label)
        value = self._lookup(self.responses, label, schema)
        if callable(value):
            value = value(system, user)
        if isinstance(value, BaseModel):
            return value.model_copy(deep=True)
        return schema.model_validate(value)

    def tool_args(
        self,
        system: str,
        user: str,
        tool: Any,
        schema: type[BaseModel],
        *,
        role: str = "main",
        label: str | None = None,
    ) -> BaseModel:
        name = getattr(tool, "name", "tool")
        self.calls.append(("tool_args", label, name, role))
        self._maybe_interrupt(label)
        # Tool scripts may live in either mapping; ``tool_results`` wins.
        value = self._lookup({**self.responses, **self.tool_results}, label, name)
        if callable(value):
            value = value(system, user)
        return schema.model_validate(value)

    # ---------------------------------------------------------------- helpers
    def _stage4_code(self, label: str) -> Any:
        """Serve the stage-4 script to the plain-text code calls.

        Stage 4 asks for `repro.py` as text (`<label>_code1`), while a happy-run fixture
        scripts the same source once under `repro_script`. A test can override a single
        round by listing that exact label in ``texts``.
        """
        if label in self.texts:
            return self.texts[label]
        if "code" in self.texts:
            return self.texts["code"]
        return (self.responses.get("repro_script") or {}).get("code")

    def _maybe_interrupt(self, label: str | None) -> None:
        if label and label in self.error_labels:
            raise self.error_labels[label]
        if label and label in self.interrupt_labels:
            raise KeyboardInterrupt()

    def _lookup(self, mapping: dict[Any, Any], label: str | None, fallback: Any) -> Any:
        if label and label in mapping:
            return mapping[label]
        if fallback in mapping:
            return mapping[fallback]
        raise KeyError(
            f"FakeLLM has no scripted answer for label={label!r} fallback={fallback!r}; "
            f"known keys: {sorted(str(k) for k in mapping)}"
        )

    def labels(self) -> list[str | None]:
        return [call[1] for call in self.calls]


@dataclass
class FakeSearcher:
    """Duck-typed replacement for :class:`essay_agent.tools.paper_search.PaperSearcher`."""

    candidates: list[Candidate] = field(default_factory=list)
    text: str = "paper text " * 40
    sections: list[PaperSection] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    # Mirrors the real client's report of what the backends did (kept in sync so the
    # fetch node's warning path is exercised by the pipeline tests too).
    backend_notes: list[str] = field(default_factory=list)
    backend_failures: list[str] = field(default_factory=list)
    full_text_available: bool | None = None
    search_error: Exception | None = None
    searches: list[Any] = field(default_factory=list)

    def search(self, request: Any) -> list[Candidate]:
        self.searches.append(request)
        if self.search_error is not None:
            raise self.search_error
        return list(self.candidates)

    def fetch_full_text(
        self, candidate: Candidate
    ) -> tuple[str, list[PaperSection], dict[str, Any]]:
        return self.text, list(self.sections), dict(self.info)

    def build_record(
        self,
        candidate: Candidate,
        *,
        text: str,
        sections: list[PaperSection],
        info: dict[str, Any] | None = None,
    ) -> PaperRecord:
        available = (
            len(text) >= 100 if self.full_text_available is None else self.full_text_available
        )
        return PaperRecord(
            id=candidate.candidate_id,
            title=candidate.title,
            authors=candidate.authors,
            year=candidate.year,
            venue=candidate.venue,
            abstract=candidate.abstract,
            url=candidate.url,
            pdf_url=candidate.pdf_url,
            source=candidate.source,
            full_text_available=available,
            full_text=text,
            sections=list(sections),
            retrieval=dict(info or {}),
        )


class FakeProber:
    """Returns canned probe results, keyed by target (defaults to 'available')."""

    def __init__(self, results: dict[str, ProbeResult] | None = None, default: bool = True) -> None:
        self.results = results or {}
        self.default = default
        self.probed: list[str] = []

    def probe(self, target: str) -> ProbeResult:
        self.probed.append(target)
        if target in self.results:
            return self.results[target]
        return ProbeResult(
            target=target,
            kind="unknown",
            reachable=self.default,
            detail="canned probe result",
        )

    def probe_many(self, targets: list[str]) -> list[ProbeResult]:
        return [self.probe(target) for target in targets if target and str(target).strip()]

    def report(self, targets: list[str]) -> list[dict[str, Any]]:
        return [result.to_dict() for result in self.probe_many(targets)]


class FakeRunner:
    """A ``ScriptRunner`` stand-in: one canned outcome instead of a real child process.

    ``env_fixes`` mirrors the real runner so the nodes that record the child's environment
    keep working, and ``calls`` keeps every invocation (script, args, env) for assertions.
    """

    def __init__(self, outcome: RunOutcome | None = None) -> None:
        self.outcome = outcome or RunOutcome(ok=True, exit_code=0)
        self.calls: list[dict[str, Any]] = []
        self.env_fixes: dict[str, str] = {}

    def run(self, script: Path, **kwargs: Any) -> RunOutcome:
        self.calls.append({"script": Path(script), **kwargs})
        return self.outcome

    def describe_calls(self) -> str:
        return "; ".join(str(call.get("args")) for call in self.calls) or "(no call)"


class FakeConnectivity:
    """Canned connectivity verdicts; everything is reachable unless told otherwise.

    ``states`` maps a category (``model_api`` / ``paper_source`` / ``dataset_source``)
    to ``ok``, ``degraded`` or ``unreachable``. ``degraded`` means the primary host is
    blocked but its fallback answers.
    """

    def __init__(self, states: dict[str, str] | None = None, *, channel: str = "os-proxy") -> None:
        self.states = dict(states or {})
        self.channel = channel
        self.calls = 0

    def run(self, settings: Any) -> ConnectivityReport:
        self.calls += 1
        reports = []
        for endpoint in endpoints_for(settings):
            state = self.states.get(endpoint.category, "ok")
            ok = state == "ok" or (state == "degraded" and endpoint.fallback)
            reports.append(
                EndpointReport(
                    endpoint,
                    [Attempt(self.channel, ok, "HTTP 200" if ok else "blocked", 0.01)],
                )
            )
        return ConnectivityReport(reports=reports, default_channel=self.channel)

    def provider(self, settings: Any, category: str) -> Callable[[], Route | None]:
        """Mirrors ``ConnectivityProbe.provider``; the fake invents no proxy URL."""
        return lambda: Route(self.channel, None)


def make_candidate(index: int = 0, **overrides: Any) -> Candidate:
    payload: dict[str, Any] = {
        "candidate_id": f"arxiv:0000.0000{index}",
        "title": f"Test Paper {index}: A Study",
        "authors": ["Ada Lovelace", "Alan Turing"],
        "year": 2020,
        "venue": "TestConf",
        "abstract": "We study things and report numbers.",
        "url": f"https://arxiv.org/abs/0000.0000{index}",
        "pdf_url": f"https://arxiv.org/pdf/0000.0000{index}",
        "source": "arxiv",
        "score": 100.0 - index,
    }
    payload.update(overrides)
    return Candidate(**payload)


def make_callable(value: Any) -> Callable[..., Any]:
    return lambda *args, **kwargs: value


# ------------------------------------------------------------- scripted network
# Paper-backend fixtures: the shapes the three APIs really returned on 2026-09-20, when
# arXiv stalled once (ReadTimeout), Semantic Scholar answered 429 and OpenAlex's top hit
# pointed at a dead PDF link. They exercise the searcher's retry, PDF validation and
# fallback paths without touching the network.

ARXIV_API = "https://export.arxiv.org/api/query"
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_API = "https://api.openalex.org/works"
DEAD_PDF_URL = "https://langtaosha.org.cn/index.php/lts/preprint/download/10/108"
ARXIV_PDF_URL = "https://arxiv.org/pdf/1706.03762"
ARXIV_PDF_URL_V7 = "https://arxiv.org/pdf/1706.03762v7"
NOT_A_PDF = b"<h1>404 Not Found</h1>\n"

ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All You Need</title>
    <summary>The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks. We propose a new simple network architecture, the
Transformer, based solely on attention mechanisms.</summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <link href="http://arxiv.org/abs/1706.03762v7" rel="alternate" type="text/html"/>
    <link title="pdf" href="https://arxiv.org/pdf/1706.03762v7" rel="related"
          type="application/pdf"/>
  </entry>
</feed>
"""

S2_JSON: dict[str, Any] = {
    "total": 7424,
    "data": [
        {
            "paperId": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
            "title": "Attention is All you Need",
            "abstract": "The dominant sequence transduction models are based on recurrent nets.",
            "year": 2017,
            "venue": "Neural Information Processing Systems",
            "authors": [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}],
            "externalIds": {"ArXiv": "1706.03762", "DOI": "10.48550/arXiv.1706.03762"},
            "openAccessPdf": {"url": ARXIV_PDF_URL, "status": "GREEN"},
            "citationCount": 100000,
            "url": "https://www.semanticscholar.org/paper/204e3073",
        }
    ],
}


def inverted_index(text: str) -> dict[str, list[int]]:
    """OpenAlex ships abstracts as word -> positions; rebuild that shape for a fixture."""
    index: dict[str, list[int]] = {}
    for position, word in enumerate(text.split()):
        index.setdefault(word, []).append(position)
    return index


# The "Attention Is All You Need" duplicate OpenAlex ranks first: year 2025, no venue and
# a PDF link on a mirror that answers 404.
OPENALEX_JSON: dict[str, Any] = {
    "results": [
        {
            "id": "https://openalex.org/W2626778328",
            "title": "Attention Is All You Need",
            "publication_year": 2025,
            "doi": "https://doi.org/10.65215/2q58a426",
            "cited_by_count": 5,
            "authorships": [
                {"author": {"display_name": "Ashish Vaswani"}},
                {"author": {"display_name": "Illia Polosukhin"}},
            ],
            "abstract_inverted_index": inverted_index(
                "The dominant sequence transduction models are based on complex recurrent"
            ),
            "primary_location": {"source": {"display_name": None}},
            "best_oa_location": {"pdf_url": DEAD_PDF_URL},
        },
        {
            "id": "https://openalex.org/W3163652268",
            "title": "Attention Is All You Need In Speech Separation",
            "publication_year": 2021,
            "cited_by_count": 900,
            "authorships": [{"author": {"display_name": "Cem Subakan"}}],
            "primary_location": {"source": {"display_name": "ICASSP"}},
        },
    ]
}


def make_pdf(words: int = 240) -> bytes:
    """A minimal one-page PDF that pypdf really extracts text from (>800 chars)."""
    text = " ".join(f"transformer token {i}" for i in range(words))
    stream = "\n".join(
        ["BT", "/F1 10 Tf", "12 TL", "40 750 Td"]
        + [f"({line}) Tj T*" for line in (text[i : i + 90] for i in range(0, len(text), 90))]
        + ["ET"]
    ).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


class ScriptedResponse:
    """The bit of ``requests.Response`` the searcher touches."""

    def __init__(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        self.content = body
        self.headers = headers or {}

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)

    def iter_content(self, chunk_size: int = 65536) -> Any:
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]


class ScriptedSession:
    """A ``requests.Session`` stand-in: one scripted reply list per URL path.

    The last entry repeats, so a one-entry script means "this always answers that".
    """

    def __init__(self, script: dict[str, list[Any]]) -> None:
        self.script = {key: list(value) for key, value in script.items()}
        self.calls: list[str] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, params: Any = None, timeout: Any = None, stream: bool = False) -> Any:
        key = url.split("?")[0]
        self.calls.append(key)
        entries = self.script.get(key)
        if not entries:
            raise AssertionError(f"unscripted request to {key}")
        item = entries.pop(0) if len(entries) > 1 else entries[0]
        if isinstance(item, type) and issubclass(item, Exception):
            raise item(f"scripted failure for {key}")
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        return None


def ok(body: bytes, content_type: str) -> ScriptedResponse:
    return ScriptedResponse(200, body, {"Content-Type": content_type})


def atom_response() -> ScriptedResponse:
    return ok(ARXIV_ATOM.encode(), "application/atom+xml; charset=utf-8")


def s2_response() -> ScriptedResponse:
    return ok(json.dumps(S2_JSON).encode(), "application/json")


def openalex_response() -> ScriptedResponse:
    return ok(json.dumps(OPENALEX_JSON).encode(), "application/json")


def too_many_requests() -> ScriptedResponse:
    return ScriptedResponse(429, b'{"message": "Too Many Requests."}')


def backend_incident_session() -> ScriptedSession:
    """The live incident: arXiv never answers, S2 is rate-limited, the PDF is gone."""
    return ScriptedSession(
        {
            ARXIV_API: [requests.exceptions.ReadTimeout("read timeout=30.0")],
            S2_API: [too_many_requests()],
            OPENALEX_API: [openalex_response()],
            DEAD_PDF_URL: [ScriptedResponse(404, NOT_A_PDF, {"Content-Type": "text/html"})],
        }
    )


def backend_recovering_session() -> ScriptedSession:
    """The same machine moments later: one stall, two 429s, then everything answers."""
    return ScriptedSession(
        {
            ARXIV_API: [requests.exceptions.ReadTimeout("read timeout=30.0"), atom_response()],
            S2_API: [too_many_requests(), too_many_requests(), s2_response()],
            OPENALEX_API: [openalex_response()],
            # 200 with an HTML body: only a content check can catch this one.
            DEAD_PDF_URL: [ok(NOT_A_PDF, "text/html")],
            ARXIV_PDF_URL: [ok(make_pdf(), "application/pdf")],
            ARXIV_PDF_URL_V7: [ok(make_pdf(), "application/pdf")],
        }
    )
