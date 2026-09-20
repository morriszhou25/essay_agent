"""Paper-side contracts: the search request, search results and the resolved paper."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_title(title: str) -> str:
    """Return a comparison key for a paper title (case/punctuation insensitive)."""
    return _NON_ALNUM.sub(" ", (title or "").lower()).strip()


def title_tokens(text: str) -> set[str]:
    """Content tokens of a title-like string (single letters and "of"/"a" drop out)."""
    return {token for token in normalize_title(text).split() if len(token) > 2}


def title_overlap(left: str, right: str) -> float:
    """Jaccard similarity of two titles; 1.0 when either side has no content tokens."""
    first, second = title_tokens(left), title_tokens(right)
    if not first or not second:
        return 1.0
    return len(first & second) / len(first | second)


class PaperSearchQuery(BaseModel):
    """Arguments the model must fill in when it calls the paper-search tool.

    Every field is optional, but at least one identifier (``url``, ``arxiv_id``,
    ``doi``) or a ``title`` must be provided. Prefer the most specific identifier
    the user gave you.
    """

    title: str | None = Field(
        default=None,
        description="Full or partial paper title, verbatim from the user. Do not invent or correct it.",
    )
    authors: list[str] | None = Field(
        default=None,
        description="Author surnames (or 'Surname, F.') that appear in the user's request.",
    )
    year: int | None = Field(default=None, description="Publication year, if stated by the user.")
    venue: str | None = Field(
        default=None, description="Conference or journal name, if stated by the user."
    )
    keywords: list[str] | None = Field(
        default=None,
        description="Distinctive topical keywords from the user's request, used only to rank candidates.",
    )
    arxiv_id: str | None = Field(
        default=None,
        description="arXiv identifier such as '1706.03762' or 'arXiv:1706.03762v5'.",
    )
    doi: str | None = Field(default=None, description="DOI, with or without the 'doi:' prefix.")
    url: str | None = Field(
        default=None,
        description="A URL the user pasted: an abstract page, a DOI link or a direct PDF link.",
    )
    max_results: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="How many candidates to retrieve. Defaults to the configured maximum.",
    )

    @field_validator("title", "arxiv_id", "doi", "url", "venue", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def free_text_query(self) -> str:
        """Build a keyword query for backends that only accept free text."""
        parts: list[str] = []
        if self.title:
            parts.append(self.title)
        if self.authors:
            parts.extend(self.authors)
        if self.venue:
            parts.append(self.venue)
        if self.keywords:
            parts.extend(self.keywords)
        return " ".join(part.strip() for part in parts if part and part.strip()).strip()

    def is_empty(self) -> bool:
        return not any([self.title, self.arxiv_id, self.doi, self.url, self.authors, self.keywords])


class Candidate(BaseModel):
    """A single search hit, before the agent commits to a paper."""

    candidate_id: str = Field(description="Stable short id, e.g. 'arxiv:1706.03762'.")
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    source: str = Field(
        default="unknown",
        description="Backend that produced the hit: arxiv | semanticscholar | openalex | user_url.",
    )
    doi: str | None = None
    arxiv_id: str | None = None
    citation_count: int | None = None
    score: float = Field(default=0.0, description="Internal relevance score (higher is better).")
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def title_key(self) -> str:
        return normalize_title(self.title)

    def brief(self) -> str:
        """One-line description used by the command-line disambiguation menu."""
        bits = [self.title]
        if self.authors:
            bits.append(" / ".join(self.authors[:3]) + (" et al." if len(self.authors) > 3 else ""))
        if self.year:
            bits.append(str(self.year))
        if self.venue:
            bits.append(self.venue)
        return " | ".join(bits)


class PaperSection(BaseModel):
    """A chapter/section of the paper, as used by the card-writing step."""

    index: int = Field(description="0-based position in the document.")
    name: str = Field(description="Section heading, e.g. '4.2 Ablation Study'.")
    text: str
    page_hint: str | None = Field(
        default=None, description="Page or page-range hint if it can be recovered."
    )
    is_reference: bool = False

    @property
    def char_count(self) -> int:
        return len(self.text)


class PaperRecord(BaseModel):
    """The resolved paper: bibliographic data plus whatever full text we obtained."""

    id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    source: str = "unknown"
    doi: str | None = None
    arxiv_id: str | None = None
    full_text_available: bool = False
    full_text: str = ""
    sections: list[PaperSection] = Field(default_factory=list)
    retrieval: dict[str, Any] = Field(
        default_factory=dict, description="Provenance notes: backend, http attempts, extractor."
    )

    @property
    def full_text_chars(self) -> int:
        return len(self.full_text)

    def bibliographic_line(self) -> str:
        authors = ", ".join(self.authors[:5]) or "unknown authors"
        tail = f" ({self.year})" if self.year else ""
        venue = f" - {self.venue}" if self.venue else ""
        return f"{self.title}{tail}{venue} [{authors}]"

    def outline(self, max_chars: int = 6000) -> str:
        """A budgeted view of the paper used when the full text is too large."""
        if len(self.full_text) <= max_chars:
            return self.full_text
        head = self.full_text[: int(max_chars * 0.7)]
        tail = self.full_text[-int(max_chars * 0.3) :]
        return f"{head}\n\n[... truncated {len(self.full_text) - max_chars} chars ...]\n\n{tail}"
