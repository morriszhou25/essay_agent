"""Replay of the 2026-09-20 paper-backend incident, against the real searcher.

Network-free and deterministic. ``[1]`` reproduces the incident (arXiv stalls, S2 is
rate-limited, the only PDF link is dead) and shows the *outcome* the user saw: a
single-source candidate list and an abstract-only "full text". ``[2]`` replays the same
machine moments later - one stall, two 429s, then everything answers - and checks that
the searcher now recovers, validates the download and reports what happened.

    python tests/experiment_backend_resilience.py

Not collected by pytest (the filename is not ``test_*``).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from essay_agent.config import SearchSettings
from essay_agent.schemas.paper import Candidate, PaperSearchQuery
from essay_agent.tools import paper_search
from essay_agent.tools.paper_search import PaperSearcher
from tests.fakes import (
    ARXIV_PDF_URL,
    DEAD_PDF_URL,
    backend_incident_session,
    backend_recovering_session,
)

QUERY = PaperSearchQuery(title="attention is all you need", max_results=8)
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def search_with(session: Any, cache: Path) -> tuple[PaperSearcher, list[Candidate]]:
    searcher = PaperSearcher(SearchSettings(), session=session, cache_dir=cache)
    return searcher, searcher.search(QUERY)


def current_behaviour(cache: Path) -> None:
    print("[1] the incident: arXiv stalls, S2 is rate-limited, the PDF link is dead")
    searcher, candidates = search_with(backend_incident_session(), cache)
    print(f"  candidates: {[(c.candidate_id, c.year) for c in candidates]}")
    check(
        "only OpenAlex answered, so the list is single-source",
        {c.source for c in candidates} == {"openalex"},
        f"sources={sorted({c.source for c in candidates})}",
    )
    check("the 2025 duplicate is what the model would pick", candidates[0].year == 2025)
    check(
        "both lost backends are now reported instead of swallowed",
        len(searcher.backend_failures) == 2,
        f"{searcher.backend_failures}",
    )

    text, _, info = searcher.fetch_full_text(candidates[0])
    print(f"  full text: {len(text)} chars, info={info}")
    check("its PDF link is dead (404)", "pdf_error" in info)
    check(
        "with nothing else to try, the run degrades to the abstract",
        info.get("fallback") == "abstract only",
    )


def recovered_behaviour(cache: Path) -> None:
    print("\n[2] the same machine moments later: one stall, two 429s, then it answers")
    searcher, candidates = search_with(backend_recovering_session(), cache)
    print(f"  candidates: {[(c.candidate_id[:28], c.year) for c in candidates]}")
    check("the arXiv hit survives the stall", any(c.arxiv_id == "1706.03762" for c in candidates))
    check(
        "the rate-limited source survives too",
        any(c.source == "semanticscholar" for c in candidates),
    )
    check("the real 2017 paper outranks the 2025 duplicate", candidates[0].year == 2017)
    check(
        "nothing was lost, so nothing is reported as a failure",
        searcher.backend_failures == [],
        f"{searcher.backend_failures}",
    )
    print(f"  retry trace: {searcher.backend_notes}")
    check(
        "the retries are visible in the trace",
        any("attempt 1" in note for note in searcher.backend_notes),
    )

    arxiv = next(c for c in candidates if c.arxiv_id == "1706.03762")
    text, _, info = searcher.fetch_full_text(arxiv)
    print(f"  full text: {len(text)} chars via {info.get('pdf_source_url')}")
    check("the real PDF is used", info.get("extractor") == "pypdf" and len(text) >= 800)

    duplicate = next(c for c in candidates if c.year == 2025)
    text, _, info = searcher.fetch_full_text(duplicate)
    print(f"  dead-link candidate recovered: {len(text)} chars via {info.get('pdf_source_url')}")
    print(f"    attempts: {info.get('pdf_attempts')}")
    check(
        "the HTML body was rejected instead of being parsed as a PDF",
        any(
            DEAD_PDF_URL in item and "PaperSearchError" in item
            for item in info.get("pdf_attempts", [])
        ),
    )
    check(
        "the dead link fell back to arXiv by title",
        info.get("pdf_source_url") == ARXIV_PDF_URL and info.get("extractor") == "pypdf",
        f"used {info.get('pdf_source_url')}",
    )
    check(
        "the rejected download left no poisoned cache entry",
        not [p for p in cache.glob("*.pdf") if p.stat().st_size < 1024],
    )


def main() -> int:
    paper_search.RETRY_BACKOFF_SECONDS = 0.0  # keep the replay instant
    root = Path(__file__).resolve().parents[1] / ".essay_agent" / "experiments" / "backends"
    shutil.rmtree(root, ignore_errors=True)
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    try:
        current_behaviour(cache)
        recovered_behaviour(cache)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
