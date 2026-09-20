"""Paper retrieval.

Every lookup goes through this module - the model is never allowed to answer
from memory. Three public APIs are used, in this order of preference:

1. **arXiv** - best when we have a title, arXiv id or an arXiv URL.
2. **Semantic Scholar** - best for a DOI or a fuzzy title, and it exposes
   ``openAccessPdf`` links.
3. **OpenAlex** - broadest coverage, used as a safety net.

``parse_*`` and ``rank_candidates`` are pure functions so they can be tested
without network access.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import requests

from essay_agent.config import SearchSettings
from essay_agent.connectivity import RouteProvider, pin_session
from essay_agent.errors import PaperSearchError
from essay_agent.schemas.paper import (
    Candidate,
    PaperRecord,
    PaperSearchQuery,
    PaperSection,
    normalize_title,
    title_overlap,
)

ARXIV_API = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_DOI_API = "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
OPENALEX_API = "https://api.openalex.org/works"
_S2_FIELDS = "title,abstract,year,venue,authors,externalIds,openAccessPdf,citationCount,url"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

# A backend that stalls or rate-limits us must not cost the whole source: one quiet
# retry turns "arXiv timed out once" into a non-event. Only *fast* failures are retried
# (a hang is not a flake), and a retry is never allowed to hide the failure: every
# give-up lands in ``PaperSearcher.backend_failures`` for the caller to report.
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 0.5
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
PDF_MAGIC = b"%PDF-"

_ARXIV_ID_RE = re.compile(r"(?P<id>\d{4}\.\d{4,5})(?P<version>v\d+)?")
_OLD_ARXIV_ID_RE = re.compile(r"(?P<id>[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?P<version>v\d+)?")
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+")


# --------------------------------------------------------------------------- ids
def _clean_arxiv_id(value: str | None) -> str | None:
    if not value:
        return None
    raw = value.strip()
    raw = re.sub(r"^arxiv\s*[:/]\s*", "", raw, flags=re.I)
    match = _ARXIV_ID_RE.search(raw) or _OLD_ARXIV_ID_RE.search(raw)
    return match.group("id") if match else None


def _clean_doi(value: str | None) -> str | None:
    if not value:
        return None
    match = _DOI_RE.search(value.strip())
    return match.group(0).rstrip(".").lower() if match else None


def request_from_url(url: str) -> PaperSearchQuery:
    """Turn a pasted URL into a structured search request."""
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    request = PaperSearchQuery(url=url)
    if "arxiv.org" in host:
        request.arxiv_id = _clean_arxiv_id(parsed.path)
    elif "doi.org" in host:
        request.doi = _clean_doi(parsed.path)
    elif "semanticscholar.org" in host or "openreview.net" in host:
        doi = _clean_doi(url)
        if doi:
            request.doi = doi
    return request


# ---------------------------------------------------------------------- arxiv
def arxiv_params(request: PaperSearchQuery, max_results: int) -> dict[str, str]:
    if request.arxiv_id:
        return {"id_list": _clean_arxiv_id(request.arxiv_id) or request.arxiv_id}
    if request.title:
        query = f'ti:"{request.title.strip()}"'
    else:
        query = f"all:{request.free_text_query() or request.title or ''}"
    return {
        "search_query": query,
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "relevance",
    }


def parse_arxiv_atom(payload: str) -> list[Candidate]:
    """Parse the arXiv Atom response into candidates."""
    if not payload or not payload.strip():
        return []
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return []
    candidates: list[Candidate] = []
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = (entry.findtext(f"{ATOM}id") or "").strip()
        arxiv_id = _clean_arxiv_id(raw_id)
        title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
        if not title:
            continue
        authors = [
            " ".join((author.findtext(f"{ATOM}name") or "").split())
            for author in entry.findall(f"{ATOM}author")
        ]
        authors = [author for author in authors if author]
        abstract = " ".join((entry.findtext(f"{ATOM}summary") or "").split()) or None
        published = (entry.findtext(f"{ATOM}published") or "")[:4]
        year = int(published) if published.isdigit() else None
        pdf_url = None
        abs_url = raw_id or None
        for link in entry.findall(f"{ATOM}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
            elif link.get("rel") == "alternate":
                abs_url = link.get("href")
        doi = _clean_doi(entry.findtext(f"{ARXIV_NS}doi"))
        venue = entry.findtext(f"{ARXIV_NS}journal_ref")
        candidates.append(
            Candidate(
                candidate_id=f"arxiv:{arxiv_id or title[:40]}",
                title=title,
                authors=authors,
                year=year,
                venue=" ".join(venue.split()) if venue else "arXiv",
                abstract=abstract,
                url=abs_url,
                pdf_url=pdf_url,
                source="arxiv",
                doi=doi,
                arxiv_id=arxiv_id,
            )
        )
    return candidates


# ----------------------------------------------------------- semantic scholar
def s2_params(request: PaperSearchQuery, max_results: int) -> dict[str, Any]:
    query = request.title or request.free_text_query()
    params: dict[str, Any] = {"query": query, "limit": max_results, "fields": _S2_FIELDS}
    if request.year:
        params["year"] = str(request.year)
    if request.venue:
        params["venue"] = request.venue
    return params


def parse_semanticscholar(payload: dict[str, Any]) -> list[Candidate]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("data") or []
    candidates: list[Candidate] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("title"):
            continue
        external = row.get("externalIds") or {}
        pdf = (row.get("openAccessPdf") or {}).get("url")
        candidates.append(
            Candidate(
                candidate_id=str(row.get("paperId") or normalize_title(row["title"])[:40]),
                title=" ".join(str(row["title"]).split()),
                authors=[a.get("name", "") for a in (row.get("authors") or []) if a.get("name")],
                year=row.get("year"),
                venue=row.get("venue") or None,
                abstract=row.get("abstract"),
                url=row.get("url"),
                pdf_url=pdf,
                source="semanticscholar",
                doi=_clean_doi(external.get("DOI")),
                arxiv_id=_clean_arxiv_id(external.get("ArXiv")),
                citation_count=row.get("citationCount"),
            )
        )
    return candidates


# ------------------------------------------------------------------- openalex
def openalex_params(
    request: PaperSearchQuery, max_results: int, email: str | None = None
) -> dict[str, Any]:
    params: dict[str, Any] = {"search": request.free_text_query(), "per-page": max_results}
    if email:
        params["mailto"] = email
    if request.year:
        params["filter"] = f"publication_year:{request.year}"
    return params


def abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    positions: list[tuple[int, str]] = []
    for word, where in index.items():
        for position in where or []:
            positions.append((position, word))
    if not positions:
        return None
    positions.sort()
    return " ".join(word for _, word in positions)


def parse_openalex(payload: dict[str, Any]) -> list[Candidate]:
    if not isinstance(payload, dict):
        return []
    candidates: list[Candidate] = []
    for row in payload.get("results") or []:
        if not isinstance(row, dict):
            continue
        title = row.get("title") or row.get("display_name")
        if not title:
            continue
        best = row.get("best_oa_location") or {}
        primary = row.get("primary_location") or {}
        source = (primary.get("source") or {}).get("display_name")
        openalex_id = str(row.get("id") or "").rsplit("/", 1)[-1]
        candidates.append(
            Candidate(
                candidate_id=f"openalex:{openalex_id or normalize_title(title)[:40]}",
                title=" ".join(str(title).split()),
                authors=[
                    (auth.get("author") or {}).get("display_name", "")
                    for auth in (row.get("authorships") or [])
                ],
                year=row.get("publication_year"),
                venue=source,
                abstract=abstract_from_inverted_index(row.get("abstract_inverted_index")),
                url=row.get("id") or (primary.get("landing_page_url")),
                pdf_url=best.get("pdf_url") or (row.get("open_access") or {}).get("oa_url"),
                source="openalex",
                doi=_clean_doi(row.get("doi")),
                citation_count=row.get("cited_by_count"),
            )
        )
    return [candidate for candidate in candidates if any(candidate.authors)]


# --------------------------------------------------------------------- ranking
def _tokens(text: str) -> set[str]:
    return {token for token in normalize_title(text).split() if len(token) > 1}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def score_candidate(candidate: Candidate, request: PaperSearchQuery) -> float:
    """Relevance score in roughly [0, 200]."""
    score = 0.0
    wanted_arxiv = _clean_arxiv_id(request.arxiv_id)
    wanted_doi = _clean_doi(request.doi)
    if wanted_arxiv and candidate.arxiv_id and candidate.arxiv_id == wanted_arxiv:
        score += 90
    if wanted_doi and candidate.doi and candidate.doi == wanted_doi:
        score += 90
    if request.title:
        wanted = normalize_title(request.title)
        if wanted and wanted == candidate.title_key:
            score += 100
        else:
            score += 70 * _jaccard(_tokens(request.title), _tokens(candidate.title))
    elif wanted_arxiv or wanted_doi:
        pass
    else:
        query_text = request.free_text_query()
        if query_text:
            score += 25 * _jaccard(
                _tokens(query_text), _tokens(candidate.title + " " + (candidate.abstract or ""))
            )
    if request.authors:
        surnames = {author.split()[-1].lower() for author in request.authors if author.strip()}
        if surnames:
            hits = sum(
                1
                for author in candidate.authors
                if author.split() and author.split()[-1].lower() in surnames
            )
            score += 20 * (hits / len(surnames))
    if request.year and candidate.year:
        score += 4 if candidate.year == request.year else -6
    if request.venue and candidate.venue:
        score += 10 * _jaccard(_tokens(request.venue), _tokens(candidate.venue))
    if candidate.citation_count:
        score += min(3.0, candidate.citation_count / 400)
    if candidate.pdf_url:
        score += 2
    return round(score, 4)


def _dedupe_key(candidate: Candidate) -> tuple[str, ...]:
    if candidate.doi:
        return ("doi", candidate.doi)
    if candidate.arxiv_id:
        return ("arxiv", candidate.arxiv_id)
    return ("title", candidate.title_key)


def dedupe_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Collapse duplicates across backends, preferring richer records."""
    merged: dict[tuple[str, ...], Candidate] = {}
    for candidate in candidates:
        key = _dedupe_key(candidate)
        current = merged.get(key)
        if current is None:
            merged[key] = candidate
            continue
        current.citation_count = (
            max(current.citation_count or 0, candidate.citation_count or 0) or None
        )
        current.pdf_url = current.pdf_url or candidate.pdf_url
        current.url = current.url or candidate.url
        current.abstract = current.abstract or candidate.abstract
        current.arxiv_id = current.arxiv_id or candidate.arxiv_id
        current.doi = current.doi or candidate.doi
        current.year = current.year or candidate.year
        current.venue = current.venue or candidate.venue
        if not current.authors and candidate.authors:
            current.authors = candidate.authors
        current.extra.setdefault("also_found_in", []).append(candidate.source)
    return list(merged.values())


def rank_candidates(candidates: Iterable[Candidate], request: PaperSearchQuery) -> list[Candidate]:
    scored = []
    for candidate in dedupe_candidates(candidates):
        candidate.score = score_candidate(candidate, request)
        scored.append(candidate)
    scored.sort(key=lambda c: (c.score, c.citation_count or 0), reverse=True)
    return scored


class PaperSearcher:
    """Backend-agnostic paper lookup + full-text retrieval."""

    def __init__(
        self,
        settings: SearchSettings | None = None,
        *,
        session: requests.Session | None = None,
        cache_dir: Path | None = None,
        route: RouteProvider | None = None,
    ) -> None:
        self.settings = settings or SearchSettings()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._session = session
        self._route = route
        self._pinned = False
        # Filled in by :meth:`search`: what went wrong on the way (for the run report).
        self.backend_notes: list[str] = []
        self.backend_failures: list[str] = []

    # ------------------------------------------------------------------ http
    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": self.settings.user_agent, "Accept": "*/*"})
        if not self._pinned:
            # Whatever the probe verified for the paper sources, the environment may
            # disagree (dead HTTP_PROXY, proxy only in the OS settings). Pin to it.
            self._pinned = True
            pin_session(self._session, self._route() if self._route else None)
        return self._session

    def _record_failure(self, label: str, exc: Exception) -> None:
        note = f"{label}: {type(exc).__name__}: {exc}"
        if note not in self.backend_failures:  # the list is deduped: no double reporting
            self.backend_failures.append(note)

    def _transient(self, exc: Exception) -> bool:
        """Worth another go? A stall, a dropped connection, a 429 or a 5xx."""
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return isinstance(exc, requests.HTTPError) and status in RETRY_STATUS

    def _retried(self, label: str, call: Callable[[], Any]) -> Any:
        """Run ``call`` with bounded retries, leaving a trace of what happened."""
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                result = call()
                if attempt > 1:
                    self.backend_notes.append(f"{label}: answered on attempt {attempt}")
                return result
            except Exception as exc:
                if attempt == RETRY_ATTEMPTS or not self._transient(exc):
                    self._record_failure(label, exc)
                    raise
                self.backend_notes.append(f"{label}: {type(exc).__name__} (attempt {attempt})")
                time.sleep(RETRY_BACKOFF_SECONDS)
        raise AssertionError("unreachable")  # pragma: no cover

    def _get_json(
        self, url: str, params: dict[str, Any], *, label: str | None = None
    ) -> dict[str, Any]:
        def call() -> dict[str, Any]:
            response = self.session.get(url, params=params, timeout=self.settings.timeout)
            response.raise_for_status()
            return response.json()

        return self._retried(label or urlparse(url).netloc or url, call)

    def _get_text(
        self, url: str, params: dict[str, Any] | None = None, *, label: str | None = None
    ) -> str:
        def call() -> str:
            response = self.session.get(url, params=params or {}, timeout=self.settings.timeout)
            response.raise_for_status()
            return response.text

        return self._retried(label or urlparse(url).netloc or url, call)

    # --------------------------------------------------------------- backends
    def search_arxiv(self, request: PaperSearchQuery, max_results: int) -> list[Candidate]:
        payload = self._get_text(ARXIV_API, arxiv_params(request, max_results), label="arxiv")
        return parse_arxiv_atom(payload)

    def search_semanticscholar(
        self, request: PaperSearchQuery, max_results: int
    ) -> list[Candidate]:
        doi = _clean_doi(request.doi)
        if doi and not request.title:
            url = SEMANTIC_SCHOLAR_DOI_API.format(doi=quote(doi, safe=""))
            payload = self._get_json(url, {"fields": _S2_FIELDS}, label="semanticscholar")
            return parse_semanticscholar({"data": [payload] if payload else []})
        return parse_semanticscholar(
            self._get_json(
                SEMANTIC_SCHOLAR_API, s2_params(request, max_results), label="semanticscholar"
            )
        )

    def search_openalex(self, request: PaperSearchQuery, max_results: int) -> list[Candidate]:
        params = openalex_params(request, max_results, self.settings.contact_email)
        return parse_openalex(self._get_json(OPENALEX_API, params, label="openalex"))

    # ----------------------------------------------------------------- search
    def search(self, request: PaperSearchQuery) -> list[Candidate]:
        """Query every enabled backend and return ranked, de-duplicated candidates."""
        if request.url and not request.arxiv_id and not request.doi:
            request = _absorbs_url_hints(request)
        effective = request.model_copy(deep=True)
        if effective.max_results is None:
            effective.max_results = None

        max_results = min(effective.max_results or self.settings.max_results, 50)
        candidates: list[Candidate] = []
        errors: list[str] = []
        self.backend_notes = []
        self.backend_failures = []
        backends = [b.lower() for b in self.settings.backends]

        for name, func in (
            ("arxiv", self.search_arxiv),
            ("semanticscholar", self.search_semanticscholar),
            ("openalex", self.search_openalex),
        ):
            if name not in backends:
                continue
            if name != "arxiv" and (not effective.free_text_query() and not effective.doi):
                continue
            try:
                candidates.extend(func(effective, max_results))
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                self._record_failure(name, exc)

        if effective.url and not candidates:
            direct = self.candidate_from_url(effective.url)
            if direct is not None:
                candidates.append(direct)

        if not candidates and errors:
            raise PaperSearchError("every search backend failed: " + " | ".join(errors))
        ranked = rank_candidates(candidates, effective)
        return ranked[:max_results]

    # -------------------------------------------------------------- url input
    def candidate_from_url(self, url: str) -> Candidate | None:
        """Build a candidate from a pasted URL (PDF or landing page)."""
        path = urlparse(url).path.lower()
        if path.endswith(".pdf"):
            return Candidate(
                candidate_id=f"user_url:{hashlib.sha1(url.encode()).hexdigest()[:8]}",
                title=Path(path).stem.replace("_", " ") or url,
                url=url,
                pdf_url=url,
                source="user_url",
            )
        try:
            html = self._get_text(url)
        except Exception:
            return None
        meta = metadata_from_html(html)
        if not meta.get("title"):
            return None
        return Candidate(
            candidate_id=f"user_url:{hashlib.sha1(url.encode()).hexdigest()[:8]}",
            title=meta["title"],
            authors=meta.get("authors", []),
            year=meta.get("year"),
            venue=meta.get("venue"),
            abstract=meta.get("abstract"),
            url=url,
            pdf_url=meta.get("pdf_url"),
            source="user_url",
            doi=_clean_doi(meta.get("doi")),
            arxiv_id=_clean_arxiv_id(meta.get("arxiv_id") or url),
        )

    # ------------------------------------------------------------ full text
    def download_pdf(self, url: str) -> Path:
        """Download (or reuse a cached copy of) a PDF, rejecting anything else.

        A dead or hijacked link often answers ``200`` with an HTML error page, which
        ``pypdf`` would happily "parse" into nothing; the magic bytes are checked
        before the file is trusted, and a rejected download is removed from the cache
        so it can never be picked up again.
        """
        cache_dir = self.cache_dir or Path.cwd() / ".essay_agent" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(url.encode()).hexdigest()[:16]
        target = cache_dir / f"{digest}.pdf"
        if self.settings.cache_pdfs and target.is_file() and target.stat().st_size > 1024:
            path = target
        else:
            response = self.session.get(url, timeout=max(60.0, self.settings.timeout), stream=True)
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        handle.write(chunk)
            path = target
        if not _is_pdf(path):
            path.unlink(missing_ok=True)  # the handle is closed: Windows allows this
            raise PaperSearchError(f"{url} did not return a PDF (an HTML error page?)")
        return target

    def _pdf_urls(self, candidate: Candidate) -> list[str]:
        """The links worth trying for ``candidate``, best first.

        The record's own link can be dead (a mirror that has since 404'd), so the
        paper's arXiv copy is tried next - first by id, then, for records that carry
        no arXiv id at all, by looking the title up on arXiv.
        """
        urls: list[str] = []
        if candidate.pdf_url:
            urls.append(candidate.pdf_url)
        if candidate.arxiv_id:
            urls.append(f"https://arxiv.org/pdf/{candidate.arxiv_id}")
        elif candidate.title:
            urls.extend(self._arxiv_pdfs_by_title(candidate.title))
        unique: list[str] = []
        for url in urls:
            if url and url not in unique:
                unique.append(url)
        return unique

    def _arxiv_pdfs_by_title(self, title: str) -> list[str]:
        """A last resort: the record has no arXiv id, so ask arXiv for the title."""
        try:
            hits = self.search_arxiv(PaperSearchQuery(title=title, max_results=3), 3)
        except Exception as exc:
            self._record_failure("arxiv title lookup", exc)
            return []
        return [f"https://arxiv.org/pdf/{hit.arxiv_id}" for hit in hits if hit.arxiv_id][:1]

    def fetch_full_text(
        self, candidate: Candidate
    ) -> tuple[str, list[PaperSection], dict[str, Any]]:
        """Return ``(text, sections, info)`` for a candidate, using its open PDF when possible."""
        info: dict[str, Any] = {"source": candidate.source, "pdf_url": candidate.pdf_url}
        attempts: list[str] = []
        info["pdf_attempts"] = attempts  # recorded even when one of them works
        for index, pdf_url in enumerate(self._pdf_urls(candidate)):
            try:
                path = self.download_pdf(pdf_url)
                text = extract_pdf_text(path)
                if len(text) >= 800:
                    sections = split_into_sections(text)
                    info.update(
                        {
                            "pdf_path": str(path),
                            "pdf_source_url": pdf_url,
                            "extractor": "pypdf",
                            "chars": len(text),
                        }
                    )
                    info["pdf_title"] = _first_title_like_line(text, candidate.title)
                    return text, sections, info
                attempts.append(f"{pdf_url}: too little text (scanned document?)")
            except Exception as exc:
                attempts.append(f"{pdf_url}: {type(exc).__name__}: {exc}")
                if index == 0:  # keep the primary error where callers already look for it
                    info["pdf_error"] = f"{type(exc).__name__}: {exc}"
        abstract = candidate.abstract or ""
        info["fallback"] = "abstract only"
        return abstract, [], info

    def build_record(
        self,
        candidate: Candidate,
        *,
        text: str,
        sections: list[PaperSection],
        info: dict[str, Any] | None = None,
    ) -> PaperRecord:
        title = candidate.title
        if info and info.get("pdf_title") and len(title) < 25:
            title = str(info["pdf_title"])
        return PaperRecord(
            id=candidate.candidate_id,
            title=title,
            authors=candidate.authors,
            year=candidate.year,
            venue=candidate.venue,
            abstract=candidate.abstract,
            url=candidate.url,
            pdf_url=candidate.pdf_url,
            source=candidate.source,
            doi=candidate.doi,
            arxiv_id=candidate.arxiv_id,
            full_text_available=len(text) >= 2000,
            full_text=text,
            sections=sections,
            retrieval=dict(info or {}),
        )


def _absorbs_url_hints(request: PaperSearchQuery) -> PaperSearchQuery:
    """Fill arxiv/doi fields from a URL so the right backend is chosen."""
    hinted = request_from_url(request.url or "")
    merged = request.model_copy(deep=True)
    merged.arxiv_id = request.arxiv_id or hinted.arxiv_id
    merged.doi = request.doi or hinted.doi
    if not merged.title and (merged.arxiv_id or merged.doi):
        # The identifier is enough; a title guess would only add noise.
        merged.title = None
    return merged


def search_papers(
    request: PaperSearchQuery,
    settings: SearchSettings | None = None,
    *,
    session: requests.Session | None = None,
    cache_dir: Path | None = None,
) -> list[Candidate]:
    """Convenience wrapper around :class:`PaperSearcher`."""
    return PaperSearcher(settings, session=session, cache_dir=cache_dir).search(request)


# ------------------------------------------------------------------- pdf/html
def _is_pdf(path: Path) -> bool:
    """True when the file on disk really starts with the PDF signature."""
    with path.open("rb") as handle:
        return handle.read(len(PDF_MAGIC)) == PDF_MAGIC


def extract_pdf_text(path: Path, *, max_pages: int | None = None) -> str:
    """Extract text from a PDF, keeping page markers for provenance."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for number, page in enumerate(reader.pages, start=1):
        if max_pages and number > max_pages:
            break
        try:
            content = page.extract_text() or ""
        except Exception:
            content = ""
        pages.append(f"\n[[page {number}]]\n{content}")
    return "".join(pages).strip()


def metadata_from_html(html: str) -> dict[str, Any]:
    """Pull Highwire ``citation_*`` metadata (and a title fallback) out of a landing page."""
    meta: dict[str, Any] = {"authors": []}
    for match in re.finditer(
        r"<meta[^>]+name=[\"'](?P<name>[^\"']+)[\"'][^>]*content=[\"'](?P<content>[^\"']*)[\"']",
        html,
        re.I,
    ):
        name = match.group("name").lower().strip()
        content = _unescape(match.group("content"))
        if name == "citation_title" and content:
            meta["title"] = content
        elif name == "citation_author" and content:
            meta["authors"].append(content)
        elif name in {"citation_pdf_url", "citation_pdf"} and content:
            meta["pdf_url"] = content
        elif name == "citation_doi" and content:
            meta["doi"] = content
        elif name == "citation_arxiv_id" and content:
            meta["arxiv_id"] = content
        elif name in {"citation_publication_date", "citation_date"} and content:
            year = re.search(r"(\d{4})", content)
            if year:
                meta["year"] = int(year.group(1))
        elif name in {"citation_journal_title", "citation_conference_title"} and content:
            meta["venue"] = meta.get("venue") or content
        elif name == "description" and content and not meta.get("abstract"):
            meta["abstract"] = content
    if not meta.get("title"):
        title = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
        if title:
            meta["title"] = _unescape(" ".join(title.group(1).split()))
    return meta


def _unescape(text: str) -> str:
    import html as html_module

    return html_module.unescape(text).strip()


MIN_TITLE_OVERLAP = 0.34

# Page-1 furniture that is never the title: camera-ready banners, arXiv stamps,
# journal headers, copyright notices and contact lines.
_TITLE_STOP_RE = re.compile(
    r"^\s*("
    r"published\s+as|preprint|under\s+review|submitted\s+to|accepted\s+(at|to|for)|"
    r"to\s+appear|proceedings\s+of|advances\s+in|journal\s+of|transactions\s+on|"
    r"camera[- ]ready|copyright|©|all\s+rights\s+reserved|this\s+(paper|work|article)|"
    r"arxiv:|doi:|https?://|www\.|volume\s+\d|page\s+\d"
    r")",
    re.IGNORECASE,
)
_NON_TITLE_RE = re.compile(r"^\s*(abstract|keywords|index\s+terms|introduction)\b", re.IGNORECASE)


def _title_candidates(text: str, *, limit: int = 12, window: int = 4000) -> list[str]:
    """Lines near the top of page 1 that could plausibly be the paper title."""
    candidates: list[str] = []
    for raw_line in text[:window].splitlines():
        line = " ".join(raw_line.split())
        if not 12 <= len(line) <= 200:
            continue
        if _TITLE_STOP_RE.match(line) or _NON_TITLE_RE.match(line):
            continue
        if "@" in line:  # author contact line
            continue
        if sum(char.isalpha() for char in line) < 12:  # mostly numbers/symbols
            continue
        candidates.append(line)
        if len(candidates) >= limit:
            break
    return candidates


def _first_title_like_line(text: str, expected_title: str | None = None) -> str | None:
    """Best guess at the document's title from the top of page 1.

    Camera-ready PDFs open with furniture ("Published as a conference paper at
    ICLR 2015", "Preprint. Under review."), so the first non-empty line is rarely
    the title. With an ``expected_title`` the candidate sharing the most tokens
    with it wins; when nothing matches we fall back to the first plausible line
    so the caller can still detect that the document is the wrong one.
    """
    candidates = _title_candidates(text)
    if not candidates:
        return None
    if not expected_title:
        return candidates[0]
    best = max(candidates, key=lambda line: title_overlap(line, expected_title))
    if title_overlap(best, expected_title) >= MIN_TITLE_OVERLAP:
        return best
    return candidates[0]


# ------------------------------------------------------------------- sections
_PAGE_MARKER_RE = re.compile(r"\[\[page (\d+)\]\]")
_NUMBERED_HEADING_RE = re.compile(
    r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(?P<name>[A-Za-z].{2,60})$"
)
_KNOWN_HEADINGS = (
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "methodology",
    "approach",
    "model",
    "experiments",
    "experiment",
    "experimental setup",
    "setup",
    "evaluation",
    "results",
    "result",
    "results and discussion",
    "analysis",
    "ablation study",
    "ablation",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "future work",
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "appendix",
)
_REFERENCE_HEADINGS = ("references", "bibliography", "acknowledgements", "acknowledgments")


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not 3 <= len(stripped) <= 72:
        return False
    if stripped.endswith((".", ",", ";", ":")) and not stripped.endswith(":"):
        return False
    words = stripped.split()
    if len(words) > 12:
        return False
    lowered = stripped.lower().strip("0123456789. ")
    if lowered in _KNOWN_HEADINGS:
        return True
    match = _NUMBERED_HEADING_RE.match(stripped)
    if match:
        name = match.group("name")
        if name[:1].isupper() and sum(ch.isdigit() for ch in name) <= 2:
            return True
    return stripped.isupper() and 1 <= len(words) <= 8 and stripped.isalpha()


def _page_at(offset: int, markers: list[tuple[int, int]], fallback: int = 1) -> int:
    page = fallback
    for position, number in markers:
        if position <= offset:
            page = number
        else:
            break
    return page


def split_into_sections(text: str) -> list[PaperSection]:
    """Heuristic section split used when the paper is too long to send whole.

    The model still performs the authoritative split; this only guarantees that
    chunking happens on plausible boundaries.
    """
    if not text.strip():
        return []
    markers = [(m.start(), int(m.group(1))) for m in _PAGE_MARKER_RE.finditer(text)]
    lines = text.splitlines()
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line) + 1

    headings: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        if _is_heading(line):
            headings.append((index, line.strip()))

    if len(headings) < 2:
        return [
            PaperSection(
                index=0, name="Full text", text=text, page_hint=_page_range(markers, len(text))
            )
        ]

    sections: list[PaperSection] = []
    for position, (line_index, name) in enumerate(headings):
        start = line_index + 1
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        if not body and position + 1 < len(headings):
            continue
        offset = offsets[line_index] if line_index < len(offsets) else 0
        page = _page_at(offset, markers)
        sections.append(
            PaperSection(
                index=len(sections),
                name=name,
                text=body,
                page_hint=f"p.{page}",
                is_reference=any(
                    name.lower().strip("0123456789. ").startswith(r) for r in _REFERENCE_HEADINGS
                ),
            )
        )
    if not sections:
        return [PaperSection(index=0, name="Full text", text=text)]
    return sections


def _page_range(markers: list[tuple[int, int]], length: int) -> str | None:
    if not markers:
        return None
    return f"p.{markers[0][1]}-{_page_at(length, markers)}"


# ----------------------------------------------------------------------- tool
def build_search_tool(searcher: PaperSearcher | None = None):
    """The forced-call search tool handed to the model.

    The body executes a real search, but ``nodes.fetch`` normally consumes only
    the validated arguments and calls :class:`PaperSearcher` itself, which keeps
    the retrieval deterministic and auditable.
    """
    from langchain_core.tools import StructuredTool

    def _run(**kwargs: Any) -> str:
        request = PaperSearchQuery(**kwargs)
        candidate_searcher = searcher or PaperSearcher()
        try:
            results = candidate_searcher.search(request)
        except PaperSearchError as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps([c.model_dump() for c in results], ensure_ascii=False)

    return StructuredTool.from_function(
        func=_run,
        name="search_papers",
        description=(
            "Search bibliographic databases (arXiv, Semantic Scholar, OpenAlex) for a paper. "
            "You MUST call this tool for every paper lookup and you MUST NOT answer from your own "
            "memory or training data. Fill in every field the user's request supports; leave the "
            "rest null. Never invent a title or an identifier the user did not provide."
        ),
        args_schema=PaperSearchQuery,
    )
