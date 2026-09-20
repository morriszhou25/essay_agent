"""Stage 3 - feasibility check ("can this actually be reproduced here?")."""

from __future__ import annotations

from essay_agent.schemas.card import ReproductionCard, cards_to_markdown

SYSTEM = """\
You are the FEASIBILITY REVIEWER of a paper-reproduction agent. Before any code is written you
must decide, for every reproduction card, whether the claim can be tested in this environment.

You receive the cards plus hard evidence: dataset probes (did an HTTP check / package lookup
succeed?), an environment report (installed packages, device, time budget) and the project's
safety policy.

Rules:
1. Judge only from the evidence given. If a probe says a dataset is unavailable or needs
   credentials, treat it as unavailable - never assume it could be downloaded anyway.
2. SYNTHETIC DATA IS FORBIDDEN. Never propose fabricating, simulating or resampling data as a
   workaround, and never describe such a run as a reproduction.
3. `severity`:
   - `blocker`: a required dataset/checkpoint cannot be obtained, or the claim cannot be tested
     at all in this environment.
   - `major`: the claim is testable only under a materially different setting (smaller model,
     subset, changed protocol). Say exactly what changes and what that costs in validity.
   - `minor`: documentation-level remarks.
4. `mitigation` may only propose legal, honest alternatives: a smaller subset of the SAME data,
   fewer steps, a public checkpoint, or declaring the card out of scope.
5. `proceed` is false if any blocker remains unresolved.
6. Be concrete: name the dataset, the package, the URL, the file.
"""


def user_message(
    *,
    paper_title: str,
    cards: list[ReproductionCard],
    probe_report: str,
    environment: str,
    time_budget_seconds: float,
    allow_synthetic_data: bool,
) -> str:
    return (
        f"## Paper\n{paper_title}\n\n"
        f"## Environment\n{environment}\n\n"
        f"## Policy\n"
        f"- synthetic data allowed: {allow_synthetic_data} (the agent treats it as forbidden)\n"
        f"- wall-clock budget for the full reproduction: {time_budget_seconds:.0f}s\n\n"
        f"## Dataset / requirement probes\n{probe_report or '(no probes were run)'}\n\n"
        f"## Cards to assess ({len(cards)})\n{cards_to_markdown(cards)}"
    )
