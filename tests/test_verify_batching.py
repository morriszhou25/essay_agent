"""The card verifier reviews the cards in bounded batches.

One call over the whole card set returned 20,892 characters of JSON for 24 cards on
run-20260920-042507 - two thirds of the provider's hard output ceiling, and the card
ceiling has since moved to 30.  The verifier is now called once per batch of
``VERIFY_BATCH_SIZE`` cards and the answers are merged.
"""

from __future__ import annotations

import re
from typing import Any

from essay_agent.console import SilentUI
from essay_agent.errors import LLMError
from essay_agent.nodes.plan import (
    VERIFY_BATCH_SIZE,
    VERIFY_ISSUE_LIMIT,
    VERIFY_ISSUE_TOTAL,
    make_verify_plan_node,
    plan_review_ok,
)
from essay_agent.schemas.dialogue import PlanReview
from tests.fakes import FakeLLM
from tests.pipeline import build_deps, card_payload


def cards_for(count: int) -> list[dict[str, Any]]:
    payloads = []
    for position in range(1, count + 1):
        payload = card_payload(f"c{position:02d}")
        payload["identity"]["section"] = f"{position} Section {position}"
        payloads.append(payload)
    return payloads


def issue(card_id: str | None, field: str = "success_criteria", severity: str = "major") -> dict:
    return {
        "card_id": card_id,
        "field": field,
        "severity": severity,
        "problem": "the criterion cannot be measured from the metrics",
        "suggestion": "compare accuracy against baseline_accuracy with a threshold",
    }


def batch_cards(user: str) -> list[str]:
    """The card ids the verifier was actually handed for this call."""
    return list(dict.fromkeys(re.findall(r"`(c\d{2})`", user)))


def one_issue_per_card(_system: str, user: str) -> dict[str, Any]:
    ids = batch_cards(user)
    return {"ok": False, "issues": [issue(card_id) for card_id in ids], "summary": "batch review"}


def run_verify(
    settings: Any,
    cards: list[dict[str, Any]],
    responses: dict[Any, Any],
    *,
    errors: dict[str, Exception] | None = None,
) -> tuple[FakeLLM, dict[str, Any], SilentUI]:
    llm = FakeLLM(responses, error_labels=errors)
    ui = SilentUI()
    deps = build_deps(settings, llm, ui=ui)
    state: dict[str, Any] = {
        "run_id": "run-verify",
        "slug": "verify",
        "paper": {"id": "p1", "title": "Batching Fixture"},
        "cards": cards,
        "sections": [],
        "plan_round": 0,
    }
    result = make_verify_plan_node(deps)(state)
    state.update(result)
    return llm, state, ui


def test_the_cards_are_reviewed_in_batches(settings) -> None:
    cards = cards_for(20)
    llm, state, _ui = run_verify(settings, cards, {PlanReview: one_issue_per_card})

    verify_calls = [call for call in llm.calls if call[1] and "plan_review" in call[1]]
    assert len(verify_calls) == 3
    assert [call[1] for call in verify_calls] == [
        "plan_review:1:1",
        "plan_review:1:2",
        "plan_review:1:3",
    ]
    assert len(state["plan_review"]["issues"]) == 20


def test_each_call_carries_at_most_one_batch(settings) -> None:
    seen: list[int] = []

    def reply(_system: str, user: str) -> dict[str, Any]:
        seen.append(len(batch_cards(user)))
        return {"ok": True, "issues": [], "summary": "clean"}

    run_verify(settings, cards_for(20), {PlanReview: reply})

    assert seen == [VERIFY_BATCH_SIZE, VERIFY_BATCH_SIZE, 4]


def test_a_single_batch_keeps_the_historical_label(settings) -> None:
    llm, _state, _ui = run_verify(settings, cards_for(3), {PlanReview: one_issue_per_card})

    assert [call[1] for call in llm.calls if call[1] and "plan_review" in call[1]] == [
        "plan_review:1"
    ]


def test_a_gap_seen_by_two_batches_is_reported_once(settings) -> None:
    def reply(_system: str, user: str) -> dict[str, Any]:
        return {"ok": False, "issues": [issue(None, "coverage:3.1")], "summary": "gap"}

    _llm, state, _ui = run_verify(settings, cards_for(16), {PlanReview: reply})

    issues = state["plan_review"]["issues"]
    assert len(issues) == 1
    assert issues[0]["field"] == "coverage:3.1"
    assert state["plan_review"]["ok"] is False


def test_two_different_gaps_stay_two_issues(settings) -> None:
    def reply(_system: str, user: str) -> dict[str, Any]:
        first = "coverage:3.1" if "c09" not in batch_cards(user) else "coverage:4.2"
        return {"ok": False, "issues": [issue(None, first)], "summary": "gap"}

    _llm, state, _ui = run_verify(settings, cards_for(10), {PlanReview: reply})

    assert {item["field"] for item in state["plan_review"]["issues"]} == {
        "coverage:3.1",
        "coverage:4.2",
    }


def test_the_merged_list_is_capped_and_says_so(settings) -> None:
    _llm, state, _ui = run_verify(settings, cards_for(30), {PlanReview: one_issue_per_card})

    review = state["plan_review"]
    assert len(review["issues"]) == VERIFY_ISSUE_TOTAL
    assert "not carried forward" in review["summary"]
    assert review["ok"] is False


def test_a_blocker_survives_the_cap(settings) -> None:
    def reply(_system: str, user: str) -> dict[str, Any]:
        ids = batch_cards(user)
        if "c09" in ids:
            return {
                "ok": False,
                "issues": [issue("c09", "claim.statement", "blocker")],
                "summary": "b",
            }
        noisy = [issue("c01", f"field{i}", "minor") for i in range(25)]
        return {"ok": True, "issues": noisy, "summary": "a"}

    _llm, state, ui = run_verify(settings, cards_for(16), {PlanReview: reply})

    review = state["plan_review"]
    # The per-batch trim happens first (25 minors -> VERIFY_ISSUE_LIMIT), the blocker is kept.
    assert len(review["issues"]) == VERIFY_ISSUE_LIMIT + 1
    assert review["ok"] is False
    assert any(item["severity"] == "blocker" for item in review["issues"])
    assert any("past the cap" in message for kind, message in ui.events if kind == "dim")


def test_a_dead_batch_is_retried_then_reported(settings) -> None:
    llm, state, ui = run_verify(
        settings,
        cards_for(16),
        {PlanReview: one_issue_per_card},
        errors={"plan_review:1:2": LLMError("verifier down")},
    )

    labels = [call[1] for call in llm.calls if call[1] and "plan_review" in call[1]]
    assert labels.count("plan_review:1:2") == 2
    assert state["plan_review"]["issues"]
    assert any("unreviewed" in message for kind, message in ui.events if kind == "warn")


def test_a_dead_batch_does_not_block_a_clean_review(settings) -> None:
    def clean(_system: str, user: str) -> dict[str, Any]:
        return {"ok": True, "issues": [], "summary": "clean"}

    _llm, state, ui = run_verify(
        settings,
        cards_for(16),
        {PlanReview: clean},
        errors={"plan_review:1:2": LLMError("verifier down")},
    )

    assert plan_review_ok(state) is True
    assert any("unreviewed" in message for kind, message in ui.events if kind == "warn")


def test_every_batch_failing_releases_the_cards(settings) -> None:
    _llm, state, ui = run_verify(
        settings,
        cards_for(16),
        {PlanReview: one_issue_per_card},
        errors={
            "plan_review:1:1": LLMError("verifier down"),
            "plan_review:1:2": LLMError("verifier down"),
        },
    )

    assert plan_review_ok(state) is True
    assert state["plan_review"]["issues"] == []
    assert "verifier error" in state["plan_review"]["summary"]
    assert any("unavailable" in message for kind, message in ui.events if kind == "warn")
