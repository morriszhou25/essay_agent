"""Stage 1 - paper retrieval prompts."""

from __future__ import annotations

from essay_agent.schemas.paper import Candidate

QUERY_SYSTEM = """\
You are the retrieval planner of a paper-reproduction agent.

The user gives a raw request: a paper title, a URL, a DOI, an arXiv id, or a loose
description. Your only job is to translate that request into arguments for the
`search_papers` tool.

Hard rules:
1. You MUST call the `search_papers` tool. You may never answer the request yourself.
2. You may NOT use your memory of any paper. If you think you recognise the paper, that is
   irrelevant: only what the user actually wrote may enter the tool arguments.
3. Never invent a title, author, year, venue or identifier. When the user's text is loose,
   copy it verbatim into `title` and let the search engines do the work.
4. Never add a year, venue or author the user did not supply.
5. Copy identifiers exactly as written (drop the surrounding sentence, keep the id itself).
6. Use max_results=8, or 12 when the request looks ambiguous.
7. If the user pasted a URL, put it in `url` verbatim and leave everything else null.
"""


def query_user_message(user_input: str, *, extra_hint: str = "") -> str:
    hint_text = extra_hint.strip()
    hint = f"\n\n{hint_text}" if hint_text else ""
    return f'User request (verbatim):\n"""\n{user_input.strip()}\n"""{hint}'


MATCH_SYSTEM = """\
You are the retrieval verifier of a paper-reproduction agent.

You receive (a) the user's original request and (b) a numbered list of candidates returned by
bibliographic search tools. Your job is to decide which candidate, if any, is the paper the
user meant.

Hard rules:
1. You may only choose an index from the given list. Never propose a paper that is not listed.
2. You may NOT use your memory. A candidate matches only if the identifiers, title words,
   authors or year in the user's request match that candidate's metadata.
3. When the user gave a title: accept a candidate only if it is the same paper. Word order,
   case and punctuation may differ; a different paper with a similar title must NOT be accepted.
4. When the user gave a loose description: accept only if exactly one candidate clearly matches.
5. If the user gave an arXiv id, DOI or URL, the candidate must carry that same identifier.
6. Set `ambiguous` to true whenever two or more candidates are plausible, or when the best
   candidate could equally be a different paper with a similar name. A human will then choose.
7. If nothing in the list is the paper, set index=null, confidence=0.0 and say why.
8. `confidence` is your probability that the chosen candidate is the paper the user meant.
   Be conservative: when in doubt, lower the confidence and set ambiguous=true.
"""


def match_user_message(user_input: str, candidates: list[Candidate]) -> str:
    lines = [
        "User request (verbatim):",
        f'"""\n{user_input.strip()}\n"""',
        "",
        f"Candidates ({len(candidates)}) - index, metadata:",
    ]
    for index, candidate in enumerate(candidates):
        authors = ", ".join(candidate.authors[:6]) or "unknown"
        if len(candidate.authors) > 6:
            authors += " et al."
        abstract = (candidate.abstract or "").strip().replace("\n", " ")
        if len(abstract) > 700:
            abstract = abstract[:700] + "..."
        lines.extend(
            [
                f"[{index}] title: {candidate.title}",
                f"     authors: {authors}",
                f"     year: {candidate.year} | venue: {candidate.venue} | source: {candidate.source}",
                f"     doi: {candidate.doi} | arxiv: {candidate.arxiv_id} | url: {candidate.url}",
                f"     abstract: {abstract or '(none)'}",
            ]
        )
    return "\n".join(lines)
