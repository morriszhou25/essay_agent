"""Pydantic contracts shared by the nodes, the tools and the LLM."""

from essay_agent.schemas.card import (
    CardSet,
    ReproductionCard,
    apply_card_patch,
    card_brief,
    card_to_markdown,
    cards_to_markdown,
)
from essay_agent.schemas.dialogue import (
    CardPatch,
    ExecPlan,
    ExecuteVerdict,
    MatchDecision,
    PlanIssue,
    PlanReview,
    PlanRevision,
    RealizationReport,
    ReproScript,
    ScopeCheck,
    SectionedPaper,
    VerdictValue,
)
from essay_agent.schemas.paper import (
    Candidate,
    PaperRecord,
    PaperSearchQuery,
    PaperSection,
)

__all__ = [
    "Candidate",
    "CardPatch",
    "CardSet",
    "ExecPlan",
    "ExecuteVerdict",
    "MatchDecision",
    "PaperRecord",
    "PaperSearchQuery",
    "PaperSection",
    "PlanIssue",
    "PlanReview",
    "PlanRevision",
    "RealizationReport",
    "ReproScript",
    "ReproductionCard",
    "ScopeCheck",
    "SectionedPaper",
    "VerdictValue",
    "apply_card_patch",
    "card_brief",
    "card_to_markdown",
    "cards_to_markdown",
]
