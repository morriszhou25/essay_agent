"""Batched card verification - the shipped behaviour, measured (NOT collected by pytest).

``verify_plan`` used to send the whole card set to one model call.  Measured on
run-20260920-042507: 25 issues came back as 20,892 characters of JSON - 64% of
deepseek-chat's hard 8192-token *output* ceiling for only 24 cards, and the card ceiling has
since moved to 30.  The verifier is now called once per batch of ``VERIFY_BATCH_SIZE`` cards
and the answers are merged (deduplicated per (card, field), capped, with ``ok`` decided on the
uncapped set).

This harness drives the real ``verify_plan`` node with a scripted verifier that answers at the
measured size (835 chars per issue), so the numbers below are the ones the provider will see.

    python tests/experiment_verify_batching.py
"""

from __future__ import annotations

# ruff: noqa: E402 -- the package is imported after the src/ path shim below.
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(ROOT / "src"), str(ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from essay_agent.console import SilentUI
from essay_agent.errors import LLMError
from essay_agent.nodes.plan import (
    VERIFY_BATCH_SIZE,
    VERIFY_ISSUE_LIMIT,
    VERIFY_ISSUE_TOTAL,
    card_batches,
    make_verify_plan_node,
    plan_review_ok,
)
from essay_agent.schemas.dialogue import PlanReview
from tests.conftest import make_settings
from tests.fakes import FakeLLM
from tests.pipeline import build_deps, card_payload

# Measured on run-20260920-042507: review_round3.json 20,892 chars / 25 issues, cards.md
# 40,198 chars / 24 cards, and deepseek-chat caps output at 8192 tokens.
ISSUE_CHARS = 835
CARD_CHARS = 1674
OUTPUT_TOKEN_CEILING = 8192
CHARS_PER_TOKEN = 4
CEILING_CHARS = OUTPUT_TOKEN_CEILING * CHARS_PER_TOKEN

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ------------------------------------------------------------------ scripted model
def cards_for(count: int) -> list[dict[str, Any]]:
    payloads = []
    for position in range(1, count + 1):
        payload = card_payload(f"c{position:02d}")
        payload["identity"]["section"] = f"{position} Section {position}"
        payloads.append(payload)
    return payloads


def batch_cards(user: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"`(c\d{2})`", user)))


def measured_issue(card_id: str, severity: str = "major") -> dict[str, Any]:
    """An issue the size the real verifier produced (835 chars)."""
    filler = "the criterion is not measurable from the metrics this script writes" * 5
    return {
        "card_id": card_id,
        "field": "success_criteria",
        "severity": severity,
        "problem": filler[:360],
        "suggestion": filler[:360],
    }


def one_issue_per_card(_system: str, user: str) -> dict[str, Any]:
    ids = batch_cards(user)
    return {
        "ok": False,
        "issues": [measured_issue(card) for card in ids],
        "summary": "batch review",
    }


def run_verify(
    cards: list[dict[str, Any]],
    responses: dict[Any, Any],
    *,
    errors: dict[str, Exception] | None = None,
) -> tuple[FakeLLM, dict[str, Any], SilentUI, list[int]]:
    sizes: list[int] = []

    def wrap(_system: str, user: str) -> dict[str, Any]:
        sizes.append(len(batch_cards(user)))
        return responses[PlanReview](_system, user)

    scripted = dict(responses)
    scripted[PlanReview] = wrap
    llm = FakeLLM(scripted, error_labels=errors)
    ui = SilentUI()
    deps = build_deps(make_settings(ROOT / ".essay_agent" / "experiments" / "verifier"), llm, ui=ui)
    state: dict[str, Any] = {
        "run_id": "run-experiment-verify",
        "slug": "verify",
        "paper": {"id": "p1", "title": "Batching Fixture"},
        "cards": cards,
        "sections": [
            {"name": f"{index} Section {index}", "text": "x" * 1200, "is_reference": False}
            for index in range(1, 31)
        ],
        "coverage": [
            {
                "key": str(index),
                "section": f"{index} Section {index}",
                "cards": [f"c{index:02d}"],
                "status": "carded",
                "detail": "",
                "passes": 1,
            }
            for index in range(1, 31)
        ],
        "plan_round": 0,
    }
    state.update(make_verify_plan_node(deps)(state))
    return llm, state, ui, sizes


def review_chars(review: dict[str, Any]) -> int:
    return len(json.dumps(review, ensure_ascii=False))


# --------------------------------------------------------------------- parts
def part1_output_budget() -> None:
    print("\n[1] the output budget: one call vs batches")
    single = ISSUE_CHARS * 30
    per_batch = ISSUE_CHARS * VERIFY_BATCH_SIZE
    print(
        f"      one call, 30 cards: {single:,} chars ({single / CEILING_CHARS:.0%} of the ceiling)"
    )
    print(
        f"      {VERIFY_BATCH_SIZE} cards/batch:  {per_batch:,} chars "
        f"({per_batch / CEILING_CHARS:.0%}), {len(card_batches(list(range(30))))} calls"
    )
    check(
        "[1a] one call for 30 cards leaves almost no headroom under the hard ceiling",
        single / CEILING_CHARS > 0.7,
        f"{single / CEILING_CHARS:.0%}",
    )
    check(
        "[1b] a batch stays far below the ceiling",
        per_batch / CEILING_CHARS < 0.25,
        f"{per_batch / CEILING_CHARS:.0%}",
    )
    check(
        "[1c] batching also shrinks the prompt (30 cards per call)",
        CARD_CHARS * 30 > 45_000,
        f"{CARD_CHARS * 30:,} chars -> {CARD_CHARS * VERIFY_BATCH_SIZE:,} per batch",
    )


def part2_real_node() -> None:
    print("\n[2] the real verify_plan node with 30 cards")
    llm, state, _ui, sizes = run_verify(cards_for(30), {PlanReview: one_issue_per_card})
    labels = [call[1] for call in llm.calls if call[1] and "plan_review" in call[1]]
    review = state["plan_review"]
    print(f"      calls: {labels}")
    print(f"      cards per batch: {sizes}")
    print(
        f"      merged review: {len(review['issues'])} issues, {review_chars(review):,} chars "
        f"({review_chars(review) / CEILING_CHARS:.0%} of the ceiling)"
    )
    check(
        "[2a] the card set is split into bounded calls",
        sizes == [VERIFY_BATCH_SIZE] * 3 + [6] and len(labels) == 4,
        f"{len(labels)} calls",
    )
    check(
        "[2b] the merged issue list is capped and says what it dropped",
        len(review["issues"]) == VERIFY_ISSUE_TOTAL and "not carried forward" in review["summary"],
        f"{len(review['issues'])} of 30 issues kept",
    )
    # The merged review is an *input* to the replan call, so it is bounded against the cards
    # it reviews rather than against the output ceiling.
    check(
        "[2c] the merged review stays a minority of the replan prompt",
        review_chars(review) < CARD_CHARS * 30 * 0.4 and review["ok"] is False,
        f"{review_chars(review):,} chars of issues vs {CARD_CHARS * 30:,} chars of cards",
    )
    check(
        "[2d] nothing is lost silently: the dropped count is stated",
        str(30 - VERIFY_ISSUE_TOTAL) in review["summary"],
        review["summary"][-60:],
    )
    check(
        "[2e] the loop sees the issues (replan is still triggered)",
        plan_review_ok(state) is False and state["plan_issues"],
        f"{len(state['plan_issues'])} issue line(s)",
    )


def part3_failure_paths() -> None:
    print("\n[3] the failure paths")
    _llm, state, ui, _sizes = run_verify(
        cards_for(16),
        {PlanReview: one_issue_per_card},
        errors={"plan_review:1:2": LLMError("verifier down")},
    )
    warnings = [message for kind, message in ui.events if kind == "warn"]
    print(f"      warnings: {warnings}")
    check(
        "[3a] one dead batch is reported but does not remove the others",
        state["plan_review"]["issues"] and any("unreviewed" in message for message in warnings),
        f"{len(state['plan_review']['issues'])} issues kept",
    )
    _llm, state, ui, _sizes = run_verify(
        cards_for(16),
        {PlanReview: one_issue_per_card},
        errors={
            "plan_review:1:1": LLMError("verifier down"),
            "plan_review:1:2": LLMError("verifier down"),
        },
    )
    warnings = [message for kind, message in ui.events if kind == "warn"]
    check(
        "[3b] every batch dead releases the cards, exactly like the single-call version",
        plan_review_ok(state) is True
        and state["plan_review"]["issues"] == []
        and any("unavailable" in message for message in warnings),
        warnings[0] if warnings else "(no warning)",
    )


def part4_downstream() -> None:
    print("\n[4] what the replan prompt has to carry")
    kept = VERIFY_ISSUE_TOTAL * ISSUE_CHARS
    uncapped = 30 * ISSUE_CHARS
    print(f"      capped: {kept:,} chars of issues; uncapped: {uncapped:,}")
    check(
        "[4a] the merged issue list is bounded for the replan prompt",
        kept < uncapped * 0.7,
        f"{kept:,} vs {uncapped:,} chars",
    )
    check(
        "[4b] the per-batch ceiling is stated to the model",
        VERIFY_ISSUE_LIMIT * ISSUE_CHARS < CEILING_CHARS * 0.5,
        f"{VERIFY_ISSUE_LIMIT} issues = {VERIFY_ISSUE_LIMIT * ISSUE_CHARS:,} chars",
    )


def main() -> int:
    print("batched card verification: the shipped behaviour, measured (offline, no key)")
    part1_output_budget()
    part2_real_node()
    part3_failure_paths()
    part4_downstream()
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
