"""The reproduction-card contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from essay_agent.schemas.card import (
    CardSet,
    ReproductionCard,
    apply_card_patch,
    card_brief,
    card_to_markdown,
    cards_to_markdown,
)


def make_card() -> ReproductionCard:
    return ReproductionCard(
        card_id="c01",
        identity={
            "section": "4.1 Main Results",
            "experiment": "Table 2",
            "page": "p.5",
            "original_text": "Our method improves accuracy by 2.8 points.",
        },
        claim={
            "statement": "The proposed method is more accurate than the baseline.",
            "subject": "the proposed method",
            "relation": "is more accurate than",
            "object": "the baseline",
        },
        assumption=["both models see the same training data"],
        scope={"dataset": ["CIFAR-10"], "setting": "5 seeds", "model": "ResNet-18"},
        expected_outcome="accuracy improves by about 2.8 points",
        success_criteria=["accuracy - baseline_accuracy >= 0.02"],
        format="table",
    )


def test_card_requires_identity_and_claim() -> None:
    with pytest.raises(ValidationError):
        ReproductionCard(card_id="c01", claim={"statement": "x"})


def test_scope_entries_default_to_none() -> None:
    card = ReproductionCard(
        card_id="c01",
        identity={"section": "1", "original_text": "text"},
        claim={"statement": "claim"},
        expected_outcome="something",
    )
    assert card.scope.dataset is None
    assert card.scope.model is None
    assert card.reproducible is True


def test_apply_patch_updates_nested_field() -> None:
    card = make_card()
    patched = apply_card_patch(card, "claim.subject", "the proposed tokenizer")
    assert patched.claim.subject == "the proposed tokenizer"
    assert card.claim.subject == "the proposed method"  # original untouched


def test_apply_patch_updates_list_and_scope() -> None:
    card = make_card()
    assert apply_card_patch(card, "assumption", ["a", "b"]).assumption == ["a", "b"]
    assert apply_card_patch(card, "scope.dataset", None).scope.dataset is None
    assert apply_card_patch(card, "reproducible", False).reproducible is False


def test_apply_patch_rejects_bad_input() -> None:
    card = make_card()
    with pytest.raises(ValueError):
        apply_card_patch(card, "", "x")
    with pytest.raises(ValueError):
        apply_card_patch(card, "nonsense", "x")
    with pytest.raises(ValueError):
        apply_card_patch(card, "identity.section.deep", "x")
    with pytest.raises(ValidationError):
        apply_card_patch(card, "claim", "not an object")


def test_markdown_rendering_mentions_the_essentials() -> None:
    text = card_to_markdown(make_card())
    assert "c01" in text
    assert "4.1 Main Results" in text
    assert "CIFAR-10" in text
    assert "accuracy" in text
    assert "reproduction cards" not in text


def test_cards_to_markdown_handles_empty() -> None:
    assert "no reproduction cards" in cards_to_markdown([])


def test_card_brief_is_bounded() -> None:
    brief = card_brief(make_card(), max_chars=60)
    assert len(brief) <= 60
    assert brief.startswith("[c01]")


def test_card_set_wraps_cards_and_skips() -> None:
    card_set = CardSet(cards=[make_card().model_dump()], skipped=["motivation only"])
    assert len(card_set.cards) == 1
    assert card_set.skipped == ["motivation only"]
