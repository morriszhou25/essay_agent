"""The 复现卡片 (reproduction card) - the single contract every stage reasons about."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CardIdentity(BaseModel):
    """Where the claim lives in the paper. Answers "which sentence are we reproducing?"."""

    section: str = Field(description="Section the claim comes from, e.g. '5.1 Main Results'.")
    experiment: str | None = Field(
        default=None, description="Name/number of the experiment or table/figure it belongs to."
    )
    page: str | None = Field(default=None, description="Page number or page range, if known.")
    original_text: str = Field(
        description="Verbatim quote (1-4 sentences) from the paper that justifies the claim."
    )


class CardClaim(BaseModel):
    """What the paper asserts."""

    statement: str = Field(description="One-sentence restatement of the claim in your own words.")
    subject: str | None = Field(
        default=None, description="Who/what is claimed to be better/stronger/faster."
    )
    relation: str | None = Field(
        default=None,
        description="The relation, e.g. 'is better than', 'reduces', 'is equivalent to'.",
    )
    object: str | None = Field(
        default=None, description="What the subject is compared against / acts on."
    )


class CardScope(BaseModel):
    """The conditions the claim is scoped to. Fill what the paper supports; leave the rest null."""

    dataset: list[str] | None = Field(
        default=None, description="Dataset(s) the experiment is trained/evaluated on."
    )
    setting: str | None = Field(
        default=None,
        description="Experimental setting: task, protocol, hyper-parameters, hardware.",
    )
    time_scope: str | None = Field(
        default=None,
        description="Time scope if the claim is temporal (e.g. 'yearly 2015-2020 data').",
    )
    population: str | None = Field(
        default=None, description="Population/subjects the claim generalises to, if any."
    )
    model: str | None = Field(
        default=None,
        description="Model or system used to produce the result (architecture, size, checkpoint).",
    )


class ReproductionCard(BaseModel):
    """One testable claim extracted from the paper, with everything needed to check it."""

    card_id: str = Field(description="Stable id, e.g. 'c01'.")
    identity: CardIdentity
    claim: CardClaim
    assumption: list[str] = Field(
        default_factory=list,
        description="Conditions under which the claim is justified. If one is violated the claim may not hold.",
    )
    scope: CardScope = Field(default_factory=CardScope)
    expected_outcome: str = Field(
        description="The outcome the paper expects, with the numbers it reports, stated so it can be checked."
    )
    success_criteria: list[str] = Field(
        default_factory=list,
        description="Machine-checkable criteria, e.g. 'accuracy >= 0.90 on the test split'.",
    )
    format: str | None = Field(
        default=None,
        description="How the paper presents this result (table, figure, metric), when the paper states it.",
    )
    reproducible: bool = Field(
        default=True, description="False if the claim cannot be checked from public artifacts."
    )
    needs: list[str] = Field(
        default_factory=list,
        description="Concrete requirements: datasets, checkpoints, compute, third-party APIs.",
    )
    notes: str | None = Field(default=None, description="Anything the verifier should know.")

    def field_map(self) -> dict[str, Any]:
        return self.model_dump()


CARD_FIELDS: tuple[str, ...] = (
    "identity",
    "claim",
    "assumption",
    "scope",
    "expected_outcome",
    "success_criteria",
    "format",
    "reproducible",
    "needs",
    "notes",
)


def apply_card_patch(card: ReproductionCard, field: str, value: Any) -> ReproductionCard:
    """Return a new card with ``field`` (dotted paths allowed) replaced by ``value``.

    The whole card is re-validated, so a malformed patch raises ``ValidationError``
    instead of silently corrupting the card.
    """
    field = (field or "").strip()
    if not field:
        raise ValueError("empty field name")
    root = field.split(".")[0]
    if root not in CARD_FIELDS:
        raise ValueError(f"unknown card field: {field!r}")
    data = card.model_dump()
    cursor = data
    parts = field.split(".")
    for part in parts[:-1]:
        if not isinstance(cursor.get(part), dict):
            raise ValueError(f"cannot patch nested field {field!r}: {part!r} is not an object")
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return ReproductionCard.model_validate(data)


def card_brief(card: ReproductionCard, max_chars: int = 400) -> str:
    """Compact single-card summary for logs and prompts."""
    scope_bits = []
    for label, value in (
        ("dataset", card.scope.dataset),
        ("setting", card.scope.setting),
        ("model", card.scope.model),
        ("population", card.scope.population),
        ("time", card.scope.time_scope),
    ):
        if value:
            scope_bits.append(f"{label}={value}")
    text = (
        f"[{card.card_id}] {card.claim.statement} "
        f"(subject={card.claim.subject!r}, relation={card.claim.relation!r}, object={card.claim.object!r}) "
        f"| section={card.identity.section} | expected={card.expected_outcome} "
        f"| scope: {', '.join(scope_bits) or 'n/a'}"
    )
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def card_to_markdown(card: ReproductionCard) -> str:
    """Human-readable rendering used in reports and lesson files."""
    lines = [
        f"### Card `{card.card_id}` - {card.identity.section}",
        "",
        f"- **Experiment**: {card.identity.experiment or 'n/a'}",
        f"- **Page**: {card.identity.page or 'n/a'}",
        f"- **Original text**: {card.identity.original_text.strip()}",
        f"- **Claim**: {card.claim.statement}",
        f"  - subject/s(x)/object: `{card.claim.subject}` / `{card.claim.relation}` / `{card.claim.object}`",
        f"- **Assumption**: {'; '.join(card.assumption) if card.assumption else 'n/a'}",
        f"- **Scope**: dataset={card.scope.dataset}, setting={card.scope.setting}, "
        f"time_scope={card.scope.time_scope}, population={card.scope.population}, model={card.scope.model}",
        f"- **Expected outcome**: {card.expected_outcome}",
        f"- **Success criteria**: {'; '.join(card.success_criteria) if card.success_criteria else 'n/a'}",
        f"- **Format**: {card.format or 'n/a'}",
        f"- **Reproducible**: {card.reproducible} | needs: {', '.join(card.needs) or 'n/a'}",
    ]
    if card.notes:
        lines.append(f"- **Notes**: {card.notes}")
    lines.append("")
    return "\n".join(lines)


def cards_to_markdown(cards: list[ReproductionCard]) -> str:
    if not cards:
        return "_(no reproduction cards)_\n"
    return "\n".join(card_to_markdown(card) for card in cards)


class CardSet(BaseModel):
    """Wrapper used for structured output when mining one section for cards."""

    cards: list[ReproductionCard] = Field(
        default_factory=list,
        description="One card per distinct, testable claim in the section. Empty if the section claims nothing testable.",
    )
    skipped: list[str] = Field(
        default_factory=list,
        description="Claims you deliberately did not card, with the reason (e.g. 'no measurable outcome').",
    )
