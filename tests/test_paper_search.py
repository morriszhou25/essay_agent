"""Paper retrieval: parsing, ranking, de-duplication and URL handling (all offline)."""

from __future__ import annotations

import pytest

from essay_agent.errors import PaperSearchError
from essay_agent.schemas.paper import Candidate, PaperSearchQuery, title_overlap
from essay_agent.tools.paper_search import (
    MIN_TITLE_OVERLAP,
    PaperSearcher,
    _first_title_like_line,
    arxiv_params,
    dedupe_candidates,
    metadata_from_html,
    parse_arxiv_atom,
    parse_openalex,
    parse_semanticscholar,
    rank_candidates,
    request_from_url,
    split_into_sections,
)
from tests.pipeline import ICLR_PAGE_1

ADAM_TITLE = "Adam: A Method for Stochastic Optimization"

ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All You Need</title>
    <summary> The dominant sequence transduction models are based on complex recurrent networks.
    </summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <link href="http://arxiv.org/abs/1706.03762v5" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/1706.03762v5" rel="related" type="application/pdf"/>
    <arxiv:doi>10.5555/3295222.3295349</arxiv:doi>
    <arxiv:journal_ref>NeurIPS 2017</arxiv:journal_ref>
  </entry>
</feed>
"""


def test_parse_arxiv_atom() -> None:
    candidates = parse_arxiv_atom(ARXIV_ATOM)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.title == "Attention Is All You Need"
    assert candidate.arxiv_id == "1706.03762"
    assert candidate.year == 2017
    assert candidate.pdf_url == "http://arxiv.org/pdf/1706.03762v5"
    assert candidate.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert candidate.doi == "10.5555/3295222.3295349"
    assert candidate.venue == "NeurIPS 2017"


def test_parse_arxiv_atom_tolerates_garbage() -> None:
    assert parse_arxiv_atom("") == []
    assert parse_arxiv_atom("<not-xml") == []


def test_parse_semanticscholar() -> None:
    payload = {
        "data": [
            {
                "paperId": "abc123",
                "title": "Attention Is All You Need",
                "abstract": "transformers",
                "year": 2017,
                "venue": "NeurIPS",
                "authors": [{"name": "Ashish Vaswani"}],
                "externalIds": {"DOI": "10.5555/3295222.3295349", "ArXiv": "1706.03762"},
                "openAccessPdf": {"url": "https://example.org/paper.pdf"},
                "citationCount": 100000,
                "url": "https://www.semanticscholar.org/paper/abc123",
            }
        ]
    }
    candidates = parse_semanticscholar(payload)
    assert len(candidates) == 1
    assert candidates[0].source == "semanticscholar"
    assert candidates[0].arxiv_id == "1706.03762"
    assert candidates[0].pdf_url == "https://example.org/paper.pdf"
    assert candidates[0].citation_count == 100000


def test_parse_openalex_rebuilds_the_abstract() -> None:
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W123",
                "title": "Attention Is All You Need",
                "publication_year": 2017,
                "doi": "https://doi.org/10.5555/3295222.3295349",
                "authorships": [{"author": {"display_name": "Ashish Vaswani"}}],
                "primary_location": {"source": {"display_name": "NeurIPS"}},
                "best_oa_location": {"pdf_url": "https://example.org/oa.pdf"},
                "cited_by_count": 42,
                "abstract_inverted_index": {"We": [0], "propose": [1], "transformers": [2]},
            }
        ]
    }
    candidates = parse_openalex(payload)
    assert candidates[0].abstract == "We propose transformers"
    assert candidates[0].venue == "NeurIPS"
    assert candidates[0].authors == ["Ashish Vaswani"]


def test_dedupe_merges_across_backends() -> None:
    arxiv = Candidate(
        candidate_id="arxiv:1706.03762",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani"],
        arxiv_id="1706.03762",
        source="arxiv",
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )
    s2 = Candidate(
        candidate_id="abc123",
        title="Attention is all you need.",
        authors=[],
        arxiv_id="1706.03762",
        source="semanticscholar",
        citation_count=100,
        year=2017,
    )
    merged = dedupe_candidates([arxiv, s2])
    assert len(merged) == 1
    assert merged[0].citation_count == 100
    assert merged[0].year == 2017
    assert merged[0].pdf_url == "https://arxiv.org/pdf/1706.03762"


def test_rank_prefers_the_exact_title() -> None:
    request = PaperSearchQuery(title="Attention Is All You Need")
    exact = Candidate(candidate_id="a", title="Attention Is All You Need", source="arxiv")
    near = Candidate(candidate_id="b", title="Attention is not all you need", source="arxiv")
    other = Candidate(candidate_id="c", title="A survey of sorting", source="arxiv")
    ranked = rank_candidates([other, near, exact], request)
    assert ranked[0].candidate_id == "a"
    assert ranked[0].score > ranked[1].score > ranked[2].score


def test_identifier_match_outranks_title_noise() -> None:
    request = PaperSearchQuery(title="attention", arxiv_id="1706.03762")
    wrong = Candidate(candidate_id="w", title="Attention mechanisms in birds", source="arxiv")
    right = Candidate(
        candidate_id="r",
        title="Transformers",
        arxiv_id="1706.03762",
        source="arxiv",
    )
    assert rank_candidates([wrong, right], request)[0].candidate_id == "r"


def test_request_from_url_recognises_identifiers() -> None:
    assert request_from_url("https://arxiv.org/abs/1706.03762").arxiv_id == "1706.03762"
    assert request_from_url("https://arxiv.org/pdf/1706.03762v5.pdf").arxiv_id == "1706.03762"
    assert request_from_url("https://doi.org/10.5555/3295222.3295349").doi == (
        "10.5555/3295222.3295349"
    )
    assert request_from_url("https://example.org/paper.pdf").arxiv_id is None


def test_backend_params() -> None:
    request = PaperSearchQuery(title="Attention Is All You Need", year=2017)
    params = arxiv_params(request, 5)
    assert params["search_query"] == 'ti:"Attention Is All You Need"'
    assert params["max_results"] == "5"
    assert arxiv_params(PaperSearchQuery(arxiv_id="1706.03762"), 5)["id_list"] == "1706.03762"


class _FakeResponse:
    def __init__(self, *, text: str = "", payload=None, status: int = 200) -> None:
        self.text = text
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, routes: dict[str, object]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url: str, params=None, timeout=None):
        self.calls.append(url)
        for key, response in self.routes.items():
            if key in url:
                if isinstance(response, Exception):
                    raise response
                return response
        raise AssertionError(f"unexpected URL: {url}")


def test_search_aggregates_backends(settings) -> None:
    session = _FakeSession(
        {
            "export.arxiv.org": _FakeResponse(text=ARXIV_ATOM),
            "api.semanticscholar.org": _FakeResponse(
                payload={
                    "data": [
                        {
                            "paperId": "x",
                            "title": "A totally different paper",
                            "authors": [{"name": "Someone"}],
                            "year": 2019,
                        }
                    ]
                }
            ),
            "api.openalex.org": _FakeResponse(payload={"results": []}),
        }
    )
    searcher = PaperSearcher(settings.search, session=session)
    candidates = searcher.search(PaperSearchQuery(title="Attention Is All You Need"))
    assert next(c.source for c in candidates) == "arxiv"
    assert len(candidates) == 2
    assert any("api.openalex.org" in url for url in session.calls)


def test_search_survives_a_dead_backend(settings) -> None:
    session = _FakeSession(
        {
            "export.arxiv.org": _FakeResponse(text=ARXIV_ATOM),
            "api.semanticscholar.org": ConnectionError("boom"),
            "api.openalex.org": _FakeResponse(payload={"results": []}),
        }
    )
    searcher = PaperSearcher(settings.search, session=session)
    candidates = searcher.search(PaperSearchQuery(title="Attention Is All You Need"))
    assert len(candidates) == 1


def test_search_raises_when_every_backend_dies(settings) -> None:
    session = _FakeSession(
        {
            "export.arxiv.org": ConnectionError("boom"),
            "api.semanticscholar.org": ConnectionError("boom"),
            "api.openalex.org": ConnectionError("boom"),
        }
    )
    searcher = PaperSearcher(settings.search, session=session)
    with pytest.raises(PaperSearchError):
        searcher.search(PaperSearchQuery(title="Attention Is All You Need"))


def test_search_falls_back_to_a_pasted_pdf_url(settings) -> None:
    session = _FakeSession({"export.arxiv.org": _FakeResponse(payload={"results": []})})
    searcher = PaperSearcher(settings.search, session=session)
    candidates = searcher.search(PaperSearchQuery(url="https://example.org/paper.pdf"))
    assert candidates and candidates[0].source == "user_url"
    assert candidates[0].pdf_url == "https://example.org/paper.pdf"


def test_metadata_from_html() -> None:
    html = """
    <html><head>
      <meta name="citation_title" content="Attention Is All You Need">
      <meta name="citation_author" content="Ashish Vaswani">
      <meta name="citation_author" content="Noam Shazeer">
      <meta name="citation_pdf_url" content="https://example.org/p.pdf">
      <meta name="citation_publication_date" content="2017/06/12">
      <meta name="citation_doi" content="10.5555/3295222.3295349">
    </head></html>
    """
    meta = metadata_from_html(html)
    assert meta["title"] == "Attention Is All You Need"
    assert meta["authors"] == ["Ashish Vaswani", "Noam Shazeer"]
    assert meta["pdf_url"] == "https://example.org/p.pdf"
    assert meta["year"] == 2017


def test_split_into_sections_finds_numbered_headings() -> None:
    text = (
        "1 Introduction\nWe study things.\nMore text here.\n"
        "2 Method\nWe build a model.\nIt has layers.\n"
        "3 Experiments\nWe measure accuracy on CIFAR-10.\n"
        "References\n[1] Someone et al.\n"
    )
    sections = split_into_sections(text)
    names = [section.name for section in sections]
    assert names[:3] == ["1 Introduction", "2 Method", "3 Experiments"]
    assert sections[0].text.startswith("We study things.")
    assert sections[-1].is_reference is True


def test_split_into_sections_falls_back_to_one_block() -> None:
    sections = split_into_sections("just prose with no headings at all")
    assert len(sections) == 1 and sections[0].name == "Full text"


def test_title_extraction_skips_camera_ready_banners() -> None:
    """The ICLR/NeurIPS banner is page furniture, not the title."""
    found = _first_title_like_line(ICLR_PAGE_1, ADAM_TITLE)
    assert found == "ADAM : A M ETHOD FOR STOCHASTIC OPTIMIZATION"
    assert title_overlap(found, ADAM_TITLE) >= MIN_TITLE_OVERLAP


def test_title_extraction_without_an_expected_title_skips_the_banner() -> None:
    assert _first_title_like_line(ICLR_PAGE_1) == "ADAM : A M ETHOD FOR STOCHASTIC OPTIMIZATION"


def test_title_extraction_still_flags_a_different_document() -> None:
    page = (
        "Proceedings of the 40th International Conference on Machine Learning\n"
        "Whales of the North Atlantic: a population survey\n"
        "Jane Roe\n"
        "Institute of Marine Biology\n"
    )
    found = _first_title_like_line(page, ADAM_TITLE)
    assert found == "Whales of the North Atlantic: a population survey"
    assert title_overlap(found, ADAM_TITLE) < MIN_TITLE_OVERLAP


def test_title_extraction_returns_none_without_candidates() -> None:
    assert _first_title_like_line("arXiv:1412.6980v9 [cs.LG] 30 Jan 2017\n1 2 3 4 5 6\n") is None
