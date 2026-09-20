"""A flaky backend must cost a retry, never the source.

The 2026-09-20 incident: arXiv stalled once and Semantic Scholar answered 429. A single
attempt per backend lost both, leaving one low-quality OpenAlex record whose PDF link was
dead. These tests replay that network offline; ``tests/experiment_backend_resilience.py``
is the end-to-end replay of the same fixtures.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import requests

from essay_agent.config import SearchSettings
from essay_agent.errors import PaperSearchError
from essay_agent.schemas.paper import Candidate, PaperSearchQuery
from essay_agent.tools import paper_search
from essay_agent.tools.paper_search import PaperSearcher
from tests.fakes import (
    ARXIV_API,
    ARXIV_PDF_URL,
    ARXIV_PDF_URL_V7,
    DEAD_PDF_URL,
    NOT_A_PDF,
    OPENALEX_API,
    ScriptedResponse,
    ScriptedSession,
    backend_incident_session,
    backend_recovering_session,
    make_pdf,
    ok,
    openalex_response,
)

QUERY = PaperSearchQuery(title="attention is all you need", max_results=8)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrying is what is under test, not the waiting."""
    monkeypatch.setattr(paper_search, "RETRY_BACKOFF_SECONDS", 0.0)


def searcher(session: object, cache: Path) -> PaperSearcher:
    return PaperSearcher(SearchSettings(), session=session, cache_dir=cache)  # type: ignore[arg-type]


def dead_pdf_session() -> ScriptedSession:
    return ScriptedSession({DEAD_PDF_URL: [ok(NOT_A_PDF, "text/html")]})


# --------------------------------------------------------------------- retrying
def test_a_stalled_backend_is_retried_and_kept(tmp_path: Path) -> None:
    client = searcher(backend_recovering_session(), tmp_path)
    candidates = client.search(QUERY)

    assert any(c.arxiv_id == "1706.03762" for c in candidates)
    assert any("answered on attempt 2" in note for note in client.backend_notes)
    assert client.backend_failures == []  # nothing was lost, so nothing to report


def test_a_rate_limited_backend_is_retried(tmp_path: Path) -> None:
    client = searcher(backend_recovering_session(), tmp_path)
    candidates = client.search(QUERY)

    assert any(c.source == "semanticscholar" for c in candidates)
    assert any("HTTPError (attempt 1)" in note for note in client.backend_notes)
    assert any("HTTPError (attempt 2)" in note for note in client.backend_notes)


def test_a_timeout_gets_a_bounded_number_of_attempts(tmp_path: Path) -> None:
    session = ScriptedSession(
        {
            ARXIV_API: [requests.exceptions.ReadTimeout("read timeout=30.0")],
            OPENALEX_API: [openalex_response()],
        }
    )
    client = searcher(session, tmp_path)
    client.search(QUERY)

    retries = [note for note in client.backend_notes if "attempt" in note]
    assert len(retries) == paper_search.RETRY_ATTEMPTS - 1


def test_the_authoritative_paper_outranks_the_recycled_record(tmp_path: Path) -> None:
    candidates = searcher(backend_recovering_session(), tmp_path).search(QUERY)

    assert candidates[0].year == 2017
    assert candidates[0].title.lower() == "attention is all you need"


# ------------------------------------------------------------------- reporting
def test_a_lost_backend_is_reported_instead_of_swallowed(tmp_path: Path) -> None:
    client = searcher(backend_incident_session(), tmp_path)
    candidates = client.search(QUERY)

    assert {c.source for c in candidates} == {"openalex"}
    assert len(client.backend_failures) == 2
    assert any("arxiv" in failure for failure in client.backend_failures)
    assert any("semanticscholar" in failure for failure in client.backend_failures)


def test_the_report_is_reset_between_searches(tmp_path: Path) -> None:
    client = searcher(backend_incident_session(), tmp_path)
    client.search(QUERY)
    assert client.backend_failures

    client._session = backend_recovering_session()
    client.search(QUERY)
    assert client.backend_failures == []


def test_search_still_raises_when_every_backend_fails(tmp_path: Path) -> None:
    session = backend_incident_session()
    session.script.pop(OPENALEX_API)  # the one backend that had answered
    with pytest.raises(PaperSearchError):
        searcher(session, tmp_path).search(QUERY)


# ------------------------------------------------------------------ downloads
def test_an_html_error_page_is_never_accepted_as_a_pdf(tmp_path: Path) -> None:
    with pytest.raises(PaperSearchError):
        searcher(dead_pdf_session(), tmp_path).download_pdf(DEAD_PDF_URL)
    assert list(tmp_path.glob("*.pdf")) == []  # the rejected body is not kept


def test_a_poisoned_cache_entry_is_rejected_and_removed(tmp_path: Path) -> None:
    digest = hashlib.sha1(DEAD_PDF_URL.encode()).hexdigest()[:16]
    poisoned = tmp_path / f"{digest}.pdf"
    poisoned.write_bytes(NOT_A_PDF + b" " * 2000)  # big enough to look like a cache hit

    with pytest.raises(PaperSearchError):
        searcher(dead_pdf_session(), tmp_path).download_pdf(DEAD_PDF_URL)
    assert not poisoned.exists()


# ------------------------------------------------------------- the pdf ladder
def test_the_pdf_ladder_falls_back_to_arxiv_by_title(tmp_path: Path) -> None:
    client = searcher(backend_recovering_session(), tmp_path)
    duplicate = next(c for c in client.search(QUERY) if c.year == 2025)
    assert duplicate.pdf_url == DEAD_PDF_URL  # its own link is the dead one

    text, sections, info = client.fetch_full_text(duplicate)

    assert info["pdf_source_url"] == ARXIV_PDF_URL
    assert info["extractor"] == "pypdf"
    assert len(text) >= 800
    assert sections
    assert any(DEAD_PDF_URL in attempt for attempt in info["pdf_attempts"])
    assert "fallback" not in info


def test_the_primary_pdf_is_used_when_it_works(tmp_path: Path) -> None:
    client = searcher(backend_recovering_session(), tmp_path)
    arxiv = next(c for c in client.search(QUERY) if c.source == "arxiv")
    assert arxiv.pdf_url == ARXIV_PDF_URL_V7

    text, _, info = client.fetch_full_text(arxiv)

    assert info["extractor"] == "pypdf"
    assert info["pdf_source_url"] == arxiv.pdf_url  # its own link, not a fallback
    assert info["pdf_attempts"] == []
    assert len(text) >= 800


def test_a_dead_primary_link_falls_back_to_the_arxiv_id(tmp_path: Path) -> None:
    """A versioned arXiv link can rot while the id still resolves."""
    session = ScriptedSession(
        {
            ARXIV_PDF_URL_V7: [ScriptedResponse(404, NOT_A_PDF)],
            ARXIV_PDF_URL: [ok(make_pdf(), "application/pdf")],
        }
    )
    client = searcher(session, tmp_path)
    arxiv = Candidate(
        candidate_id="arxiv:1706.03762",
        title="Attention Is All You Need",
        abstract="a short abstract",
        pdf_url=ARXIV_PDF_URL_V7,
        arxiv_id="1706.03762",
        source="arxiv",
    )

    text, _, info = client.fetch_full_text(arxiv)

    assert info["pdf_source_url"] == ARXIV_PDF_URL
    assert info["extractor"] == "pypdf"
    assert len(text) >= 800


def test_a_dead_primary_link_with_no_fallback_degrades_but_says_why(tmp_path: Path) -> None:
    session = dead_pdf_session()
    session.script[ARXIV_API] = [ok(b"", "application/atom+xml")]  # the title lookup finds nothing
    client = searcher(session, tmp_path)
    orphan = Candidate(
        candidate_id="openalex:W1",
        title="Some Paper Nobody Archived",
        abstract="a short abstract",
        pdf_url=DEAD_PDF_URL,
        source="openalex",
    )

    text, sections, info = client.fetch_full_text(orphan)

    assert text == "a short abstract"
    assert sections == []
    assert info["fallback"] == "abstract only"
    assert info["pdf_error"].startswith("PaperSearchError")
