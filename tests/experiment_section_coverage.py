"""Per-section card coverage - the shipped behaviour, measured (NOT collected by pytest).

Run-20260920-042507 carded 7 of its paper's 26 sections while the CLI still printed
"24 card(s) in scope": one global budget (``MAX_CARDS``) broke out of the per-section loop
the moment it was reached.  Coverage is a property of *each* section, so the planner now
allocates per section - a coverage pass, one retry pass, then the leftover budget on the
richest sections - and writes a ledger that accounts for every eligible section.

Part 1 measures that on the real 28-heading fixture of that run, parts 2-4 check the
failure paths and the outline budget, and part 5 does the arithmetic for the verifier's
output ceiling, which is the next thing that would break.

    python tests/experiment_section_coverage.py
"""

from __future__ import annotations

# ruff: noqa: E402 -- the package is imported after the src/ path shim below.
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(ROOT / "src"), str(ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from essay_agent.config import Settings
from essay_agent.console import SilentUI
from essay_agent.errors import LLMError
from essay_agent.nodes.base import section_outline
from essay_agent.nodes.plan import (
    CARD_BUDGET,
    MAX_CARDS_PER_SECTION,
    make_plan_node,
    section_key,
    unique_keys,
)
from tests.conftest import make_settings
from tests.fakes import FakeLLM
from tests.pipeline import build_deps, card_payload

# The 28 headings the splitter produced for run-20260920-042507, verbatim.  Note the
# duplicated "FFN" (a figure caption) and the small-caps spacing of the PDF.
REAL_SECTION_NAMES = [
    "ABSTRACT",
    "1 I NTRODUCTION",
    "FFN",
    "FFN",
    "2 S PARSE EXPERT MODELS",
    "2.1 I N DEEP LEARNING",
    "2.2 O N MODERN HARDWARE",
    "3 S CALING PROPERTIES OF SPARSE EXPERT MODELS",
    "3.1 U PSTREAM SCALING",
    "3.2 D OWNSTREAM SCALING",
    "3.3 S CALING THE NUMBER , SIZE AND FREQUENCY OF EXPERT LAYERS",
    "4 R OUTING ALGORITHMS",
    "4.1 R OUTING TAXONOMY",
    "4.2 L OAD BALANCING",
    "5 S PARSE EXPERT MODELS ACROSS DOMAINS",
    "5.1 N ATURAL LANGUAGE PROCESSING",
    "5.2 C OMPUTER VISION",
    "5.3 S PEECH RECOGNITION",
    "5.4 M ULTIMODAL AND MULTI -TASK",
    "6 W HEN TO USE A SPARSE VERSUS DENSE MODEL",
    "7 S PARSE MODEL TRAINING IMPROVEMENTS",
    "7.1 I NSTABILITY",
    "7.2 T RANSFER TO NEW DISTRIBUTIONS",
    "7.3 I NFERENCE",
    "8 I NTERPRETABILITY",
    "9 F UTURE DIRECTIONS AND CONCLUSIONS",
    "ACKNOWLEDGEMENTS",
    "REFERENCES",
]
REFERENCE_NAMES = {"ACKNOWLEDGEMENTS", "REFERENCES"}
NOT_A_HEADING = "101 B, T, W, H, k"

# Measured on that run: cards.md 40,198 chars / 24 cards, review_round3.json 20,892 chars
# / 25 issues.  deepseek-chat caps output at 8192 tokens.
ISSUE_CHARS = 835
OUTPUT_TOKEN_CEILING = 8192
CHARS_PER_TOKEN = 4
ISSUE_CAP_PER_ROUND = 12
OLD_OUTLINE_LIMIT = 1200

WORKDIR = ROOT / ".essay_agent" / "experiments" / "coverage"
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """Record one check; a failure is printed immediately, in context."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def rows_in(text: str) -> int:
    """Outline rows ("- ..."), not raw occurrences of "- " inside a coverage suffix."""
    return sum(1 for line in text.splitlines() if line.startswith("- "))


# ------------------------------------------------------------------ fixtures
def numbered_sections(count: int, *, body: int = 0) -> list[dict[str, Any]]:
    return [
        {
            "index": position,
            "name": f"{position + 1} Section {position + 1}",
            "text": f"Section {position + 1} claims that method {position + 1} "
            f"improves accuracy by {position + 1}.0 points.",
            "is_reference": False,
        }
        for position in range(count)
    ]


def real_paper_sections() -> list[dict[str, Any]]:
    """The 28 headings of the run that exposed the coverage gap."""
    return [
        {
            "index": position,
            "name": name,
            "text": "x" * (15 if position == 2 else 3043),
            "is_reference": name in REFERENCE_NAMES,
        }
        for position, name in enumerate(REAL_SECTION_NAMES)
    ]


def label_of(section: dict[str, Any]) -> str:
    return f"cards:{section['name'][:24]}"


def card_stating(statement: str) -> dict[str, Any]:
    payload = card_payload("pending")
    payload["identity"]["section"] = ""
    payload["claim"]["statement"] = statement
    return payload


def one_card_each(sections: list[dict[str, Any]]) -> dict[str, Any]:
    responses: dict[str, Any] = {"section_split": {"sections": sections}}
    for section in sections:
        responses[label_of(section)] = {
            "cards": [card_stating(f"{section['name']} claim holds")],
            "skipped": [],
        }
    return responses


def fresh_settings() -> Settings:
    shutil.rmtree(WORKDIR, ignore_errors=True)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    return make_settings(WORKDIR)


def run_plan(
    sections: list[dict[str, Any]],
    responses: dict[str, Any],
    *,
    errors: dict[str, Exception] | None = None,
    settings: Settings | None = None,
) -> tuple[Any, dict[str, Any], SilentUI]:
    """Drive the real planner node only, with a scripted model."""
    llm = FakeLLM(responses, error_labels=errors)
    ui = SilentUI()
    deps = build_deps(settings or fresh_settings(), llm, ui=ui)
    state: dict[str, Any] = {
        "run_id": "run-experiment-coverage",
        "slug": "coverage",
        "paper": {"id": "p1", "title": "Coverage Fixture"},
        "paper_text": "\n\n".join(f"{s['name']}\n{s['text']}" for s in sections),
    }
    state.update(make_plan_node(deps)(state))
    return deps, state, ui


# --------------------------------------------- part 1: the real paper headings
def part1_real_headings() -> None:
    print(f"\n[1] the real 28 headings: {len(REAL_SECTION_NAMES)} of them, one card each")
    sections = real_paper_sections()
    deps, state, ui = run_plan(sections, one_card_each(sections))
    eligible = list(state["coverage"])
    dropped = [row for row in eligible if row["status"] != "carded"]
    success = [message for kind, message in ui.events if kind == "success"]
    warnings = [message for kind, message in ui.events if kind == "warn"]
    print(f"      sections after merge: {len(state['sections'])}")
    print(f"      cards: {len(state['cards'])}, sections carded: {len(eligible) - len(dropped)}")
    print(f"      coverage line: {success}")
    print(f"      warnings: {warnings}")

    check(
        "[1a] an adjacent duplicate heading is no longer a section of its own",
        len(state["sections"]) == len(REAL_SECTION_NAMES) - 1
        and [s["name"] for s in state["sections"]].count("FFN") == 1,
        f"{len(state['sections'])} sections",
    )
    check(
        "[1b] every eligible section is carded (26 -> 25 after the merge)",
        len(eligible) == 25 and not dropped,
        f"{len(eligible)} sections, {len(dropped)} without a card",
    )
    check(
        "[1c] the references are excluded but stay in the section list",
        all(row["key"] not in {"references", "acknowledgements"} for row in eligible),
        f"keys={sorted(row['key'] for row in eligible)[:6]}...",
    )
    check(
        "[1d] the CLI line states the coverage instead of a bare card count",
        bool(success) and "covering 25 of 25" in success[0],
        success[0] if success else "(no success line)",
    )
    check(
        "[1e] nothing is silently abandoned",
        not any("card budget reached" in message for message in warnings),
        f"{len(warnings)} warning(s)",
    )
    workspace = deps.workspace(state)
    check(
        "[1f] the ledger is written next to the cards",
        workspace.exists("cards", "coverage.json") and workspace.exists("cards", "coverage.md"),
        str(workspace.path("cards", "coverage.md")),
    )


# ------------------------------------------------- part 2: the failure paths
def part2_failure_paths() -> None:
    print("\n[2] the failure paths")
    sections = numbered_sections(4)
    responses = one_card_each(sections)
    responses[label_of(sections[1])] = {"cards": [], "skipped": ["no measurable outcome"]}
    responses[label_of(sections[2])] = {"cards": [], "skipped": []}
    attempts: list[int] = []

    def silent_then_answered(*_ignored: Any) -> dict[str, Any]:
        attempts.append(1)
        if len(attempts) == 1:
            return {"cards": [], "skipped": []}
        return {"cards": [card_stating("recovered on the retry")], "skipped": []}

    responses[label_of(sections[2])] = silent_then_answered
    failed = label_of(sections[3])
    _deps, state, ui = run_plan(
        sections, responses, errors={failed: LLMError("scripted outage")}, settings=fresh_settings()
    )
    rows = {row["key"]: row for row in state["coverage"]}
    print(f"      statuses: { {key: row['status'] for key, row in rows.items()} }")
    print(f"      passes:   { {key: row['passes'] for key, row in rows.items()} }")

    check(
        "[2a] every section has an explicit status (nothing vanishes)",
        len(rows) == 4
        and {row["status"] for row in rows.values()} == {"carded", "no_testable_claim", "failed"},
        f"{len(rows)} rows",
    )
    check(
        "[2b] a section with nothing testable records its reason",
        rows["2"]["status"] == "no_testable_claim"
        and "no measurable outcome" in rows["2"]["detail"],
        rows["2"]["detail"],
    )
    check(
        "[2c] a silent empty answer is retried and then carded",
        rows["3"]["status"] == "carded" and rows["3"]["passes"] == 2,
        f"status={rows['3']['status']} passes={rows['3']['passes']}",
    )
    check(
        "[2d] a hard failure is retried once and then reported",
        rows["4"]["status"] == "failed" and rows["4"]["passes"] == 2,
        rows["4"]["detail"],
    )
    check(
        "[2e] the run still completes and says a section is missing",
        len(state["cards"]) == 2
        and any("produced no card" in message for _kind, message in ui.events if _kind == "warn"),
        f"{len(state['cards'])} cards",
    )


# ----------------------------------------- part 3: identity and the outline view
def part3_identity_and_outline() -> None:
    print("\n[3] section identity and the verifier's outline")
    check(
        "[3a] a numbered heading keeps its number",
        section_key("2.1 I N DEEP LEARNING") == "2.1"
        and section_key("3.3 S CALING THE NUMBER , SIZE") == "3.3",
        section_key("2.1 I N DEEP LEARNING"),
    )
    check(
        "[3b] a page-number-like line is not read as a section number",
        section_key(NOT_A_HEADING) == "101-b-t-w-h-k",
        section_key(NOT_A_HEADING),
    )
    check(
        "[3c] a far-apart duplicate name stays a separate section",
        unique_keys(["Method", "Results", "Method"]) == ["method", "results", "method#2"],
        str(unique_keys(["Method", "Results", "Method"])),
    )
    sections = real_paper_sections()
    _deps, state, _ui = run_plan(sections, one_card_each(sections), settings=fresh_settings())
    outline = section_outline(state)
    old = section_outline(state, OLD_OUTLINE_LIMIT)
    print(
        f"      outline: {len(outline)} chars (default budget), "
        f"{len(old)} chars at the old {OLD_OUTLINE_LIMIT} budget"
    )
    check(
        "[3d] the coverage view fits the default outline budget in full",
        "truncated" not in outline and "cards c01" in outline,
        f"{len(outline)} chars, {rows_in(outline)} rows",
    )
    check(
        "[3e] the old 1200-char budget would have cut it",
        rows_in(old) < rows_in(outline),
        f"{rows_in(old)} of {rows_in(outline)} rows survive",
    )
    big = section_outline({"sections": numbered_sections(60), "coverage": []}, OLD_OUTLINE_LIMIT)
    check(
        "[3f] a 60-section paper still truncates at the old budget",
        rows_in(big) < 60,
        f"{rows_in(big)} of 60 rows survive",
    )


# ------------------------------------------------- part 4: the allocation rules
def part4_allocation_rules() -> None:
    print("\n[4] the allocation rules")
    sections = numbered_sections(1)
    minted: list[int] = []

    def growing_claims(_system: str, user: str) -> dict[str, Any]:
        """Answer with as many fresh claims as the call asks for (the quota is in the prompt)."""
        quota = int(re.search(r"at most (\d+) card", user).group(1))
        cards = [card_stating(f"claim {len(minted) + offset}") for offset in range(quota)]
        minted.extend(range(quota))
        return {"cards": cards, "skipped": []}

    responses = {"section_split": {"sections": sections}, label_of(sections[0]): growing_claims}
    _deps, state, _ui = run_plan(sections, responses, settings=fresh_settings())
    row = state["coverage"][0]
    print(
        f"      per-section cards: {len(row['cards'])} (ceiling {MAX_CARDS_PER_SECTION}), "
        f"budget {CARD_BUDGET}"
    )
    check(
        "[4a] the per-section ceiling, not the budget, caps one section",
        len(row["cards"]) == MAX_CARDS_PER_SECTION < CARD_BUDGET,
        f"{len(row['cards'])} cards with {CARD_BUDGET - 1} of the budget unspent",
    )
    sections = numbered_sections(CARD_BUDGET)
    _deps, state, _ui = run_plan(sections, one_card_each(sections), settings=fresh_settings())
    check(
        "[4b] a paper with as many sections as the budget is still fully carded",
        len(state["cards"]) == CARD_BUDGET
        and all(row["status"] == "carded" for row in state["coverage"]),
        f"{len(state['cards'])} cards for {CARD_BUDGET} sections",
    )
    check(
        "[4c] the leftover budget is not spent on a section with no testable claim",
        all(row["status"] != "no_testable_claim" or not row["cards"] for row in state["coverage"]),
        "no untestable section carries cards",
    )


# ------------------------------------------- part 5: the verifier output budget
def part5_verifier_budget() -> None:
    print("\n[5] the verifier's output budget (unchanged by this fix)")
    ceiling = OUTPUT_TOKEN_CEILING * CHARS_PER_TOKEN
    full = ISSUE_CHARS * 25
    capped = ISSUE_CHARS * ISSUE_CAP_PER_ROUND
    print(f"      one issue per card: {full:,} chars ({full / ceiling:.0%} of the ceiling)")
    print(f"      capped at {ISSUE_CAP_PER_ROUND}:  {capped:,} chars ({capped / ceiling:.0%})")
    check(
        "[5a] one issue per card eats most of the 8192-token output ceiling",
        full / ceiling > 0.6,
        f"{full / ceiling:.0%}",
    )
    check(
        "[5b] an issue cap keeps the review inside the ceiling",
        capped / ceiling < 0.5,
        f"{capped / ceiling:.0%}",
    )


def main() -> int:
    print("section coverage: the shipped behaviour, measured (offline, no network, no key)")
    part1_real_headings()
    part2_failure_paths()
    part3_identity_and_outline()
    part4_allocation_rules()
    part5_verifier_budget()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
