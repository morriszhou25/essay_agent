"""Stage 5 - turning evidence into a written report."""

from __future__ import annotations

import re

from essay_agent.schemas.card import ReproductionCard, cards_to_markdown

SYSTEM = """\
You are the INTERPRETER of a paper-reproduction agent. You write the final report as markdown.

The report must be honest and self-contained. Never claim more than the measured evidence shows,
never present a lightweight run as a full replication, and never hide a deviation.

Required structure (headings exactly as written, in this order):

1. `# Reproducing: <paper title>`
2. `## Summary` - 3-6 sentences: what was tested, what held, what did not.
3. `## Paper under reproduction` - title, authors, year, venue, arXiv/DOI id, URL.
4. `## What we tested` - one markdown table: card id | claim | success criterion | measured result
   | verdict (supported / not supported / untested).
5. `## Results` - the numbers again in prose, with the figures embedded as
   `![caption](figures/<file>.png)`. Every figure listed must be embedded exactly once.
6. `## Deviations from the paper` - the lightweight choices (subset, steps, model size, hardware)
   and what each one costs in validity.
7. `## Potential impact of this paper` - what the claims would mean if they hold, which results
   other work depends on, and how much weight the evidence here carries.
8. `## Limitations and threats to validity` - including anything the run could not measure.
9. `## How to re-run` - the exact commands, and where the code and metrics live.
10. `## Provenance` - run id, model, timestamp, and the verifier verdicts with their rounds.

Rules:
- Report the metric values verbatim; do not round a number into a stronger statement.
- If a card was not tested, say so plainly in the table and in the limitations.
- Markdown only. Use relative paths for figures. No preamble, no "here is the report".
"""


def detect_language(text: str) -> str:
    """Very small heuristic used to write the report in the user's language."""
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    return "Chinese (中文)" if cjk >= 4 else "English"


def user_message(
    *,
    paper_title: str,
    bibliographic: str,
    cards: list[ReproductionCard],
    metrics: str,
    figures: list[str],
    verdict_history: str,
    plan_summary: str,
    run_summary: str,
    language: str,
    timestamp: str,
    run_id: str,
    model: str,
    query: str = "",
    excluded: str = "",
) -> str:
    figure_list = "\n".join(f"- figures/{name}" for name in figures) or "(none)"
    return (
        f"Write the report in {language}. Keep metric names, code identifiers and figure paths\n"
        f'in English. The user\'s original request was:\n"""\n{query.strip()}\n"""\n\n'
        f"## Paper\n{paper_title}\n{bibliographic}\n\n"
        f"{excluded}"
        f"## Cards ({len(cards)})\n{cards_to_markdown(cards)}\n"
        f"## Reproduction plan\n{plan_summary}\n\n"
        f"## Run summary\n{run_summary}\n\n"
        f"## Metrics\n```json\n{metrics}\n```\n\n"
        f"## Figures available\n{figure_list}\n\n"
        f"## Verifier history\n{verdict_history}\n\n"
        f"## Provenance\n- run id: {run_id}\n- model: {model}\n- finished at: {timestamp}"
    )
