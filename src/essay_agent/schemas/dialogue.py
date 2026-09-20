"""Structured messages exchanged between the main model and the two verifiers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from essay_agent.schemas.paper import PaperSection

VerdictValue = Literal["successful", "pending", "unsuccessful"]
Severity = Literal["blocker", "major", "minor"]


class SectionedPaper(BaseModel):
    """Section split of a paper, produced by the main model before card writing."""

    sections: list[PaperSection] = Field(
        description="Every substantive section in order. Drop pure bibliography/acknowledgement blocks."
    )
    notes: str | None = Field(default=None, description="Anything unusual about the split.")


class MatchDecision(BaseModel):
    """The model's pick among tool-returned candidates. It may only pick from that list."""

    index: int | None = Field(
        default=None,
        description="0-based index into the candidate list, or null if none of them is the paper.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence that the picked candidate is the paper.",
    )
    reason: str = Field(description="Why this candidate matches, quoting the user's request.")
    ambiguous: bool = Field(
        default=False,
        description="True when several candidates are plausible and a human should choose.",
    )


class PlanIssue(BaseModel):
    """A single defect the card verifier found in a reproduction card."""

    card_id: str | None = Field(default=None, description="Card the issue belongs to, if specific.")
    field: str = Field(
        description="Dotted field path, e.g. 'claim.statement', 'scope.dataset', 'success_criteria'."
    )
    severity: Severity = Field(
        description="blocker = unusable, major = will mislead the run, minor = polish."
    )
    problem: str = Field(description="What is wrong, in one or two sentences.")
    suggestion: str = Field(description="What the corrected value should look like.")
    evidence: str | None = Field(
        default=None, description="Quote from the paper that supports the issue."
    )


class PlanReview(BaseModel):
    """The card verifier's verdict over the whole card set."""

    ok: bool = Field(description="True only when there are no blocker/major issues.")
    issues: list[PlanIssue] = Field(default_factory=list)
    summary: str = Field(description="One-paragraph assessment.")
    lessons_used: list[str] = Field(
        default_factory=list, description="Which lessons from earlier runs you actually applied."
    )


class CardPatch(BaseModel):
    """A revision of one card field, produced by the main model after deliberation."""

    card_id: str
    field: str = Field(description="Dotted field path being rewritten, e.g. 'claim.subject'.")
    value: Any = Field(description="The new value, matching the field's type.")
    reasoning: str = Field(
        description="Why this value is correct, referencing the paper. State if you disagree with the verifier."
    )


class PlanRevision(BaseModel):
    """The main model's answer to a PlanReview. It must reason, not just comply."""

    patches: list[CardPatch] = Field(default_factory=list)
    rejected_issues: list[str] = Field(
        default_factory=list,
        description="Issues you deliberately did not apply, with the reason inline.",
    )
    overall_reasoning: str = Field(description="How you decided which issues to accept.")


class ScopeCheck(BaseModel):
    """Feasibility of reproducing one card, given what the tools could actually reach."""

    card_id: str
    feasible: bool
    severity: Severity = Field(
        default="minor",
        description="blocker = cannot reproduce meaningfully, major = degraded, minor = note.",
    )
    findings: list[str] = Field(default_factory=list)
    dataset: str | None = None
    dataset_available: bool | None = None
    requirement: str | None = Field(default=None, description="What exactly is needed.")
    mitigation: str | None = Field(
        default=None,
        description="Legal fallback that keeps the claim testable. Never synthetic data.",
    )


class RealizationReport(BaseModel):
    """Stage-3 output: can this paper be reproduced here and now?"""

    checks: list[ScopeCheck] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    proceed: bool = Field(description="False if any blocker remains unresolved.")
    summary: str


class ExecPlan(BaseModel):
    """The lightweight reproduction plan that drives code generation."""

    approach: str = Field(description="How you will test the cards, in plain language.")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Concrete hyper-parameters (lr, steps, subset size, seeds) chosen for a lightweight run.",
    )
    metrics: list[str] = Field(
        default_factory=list, description="Metric names the script must report."
    )
    success_signal: str = Field(
        description="What the metrics must show for the cards to be supported."
    )
    figures: list[str] = Field(
        default_factory=list,
        description="Figures the script must save, one per card when possible.",
    )
    cards_covered: list[str] = Field(default_factory=list, description="card_ids this plan tests.")
    risks: list[str] = Field(default_factory=list)


class ExecuteVerdict(BaseModel):
    """The execution verifier's judgement, checked against claim/assumption/expected_outcome."""

    verdict: VerdictValue
    rationale: str = Field(description="Why this verdict, citing the metrics that support it.")
    problems: list[str] = Field(
        default_factory=list,
        description=(
            "Serious, actionable defects only: things that change a card's conclusion or "
            "invalidate its evidence, and that the executor can fix. Anything you merely noticed "
            "belongs in `observations`."
        ),
    )
    observations: list[str] = Field(
        default_factory=list,
        description=(
            "What you noticed but is NOT a defect: run-to-run noise, a margin inside the "
            "measurement's resolution, the deliberately small scale of this run, naming or "
            "formatting. Recorded and reported, but nobody is asked to act on it."
        ),
    )
    evidence: list[str] = Field(
        default_factory=list, description="Metric names/values or log lines relied on."
    )
    per_card: dict[str, str] = Field(
        default_factory=dict,
        description="card_id -> short status line (supported / not supported / untested).",
    )


class ReproScript(BaseModel):
    """Everything the execute stage needs: the plan, the code and the shared parameters.

    The generated ``repro.py`` must honour this contract:

    * ``python repro.py --preflight`` finishes fast and prints one line
      ``{"event": "estimate", "estimated_full_seconds": <float>, "params": {...}}``.
    * ``python repro.py --out <dir>`` runs the full experiment, prints
      ``{"event": "progress", "step": i, "total": n}`` lines, and writes
      ``metrics.json`` plus every figure under ``<dir>/figures/``.
    * it never fabricates data; if a dataset is missing it must fail loudly.
    """

    plan: ExecPlan
    code: str = Field(
        description="Complete, runnable python source code. No markdown fences, no placeholders."
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="The parameter values actually used, mirrored in the script so they can be adjusted later.",
    )
    notes: str | None = Field(
        default=None, description="Anything the human should know about the run."
    )
