"""Stage 2 - split the paper into sections and write reproduction cards.

Includes the ``replan`` loop: an LLM verifier reviews the cards, records the defects in
``lesson_plan.txt`` and asks the planner to think again. After the configured number of
rounds the work is released with the open issues carried into the execute stage.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from essay_agent.errors import LLMError
from essay_agent.nodes.base import (
    Deps,
    NodeReturn,
    cards_of,
    dump_cards,
    json_block,
    paper_of,
    section_outline,
    truncate,
)
from essay_agent.prompts import plan as prompts
from essay_agent.schemas.card import CardSet, ReproductionCard, apply_card_patch, cards_to_markdown
from essay_agent.schemas.dialogue import PlanIssue, PlanReview, PlanRevision, SectionedPaper
from essay_agent.schemas.paper import PaperSection
from essay_agent.state import RunState
from essay_agent.tools.paper_search import split_into_sections

MAX_SECTION_CHARS = 14000
MAX_CONTEXT_CHARS = 60000

# Coverage comes first: every eligible section gets a card before any section is
# deepened, so a total budget can no longer starve the sections at the end of the paper.
CARD_BUDGET = 30
CARDS_PER_SECTION = 1
MAX_CARDS_PER_SECTION = 3
MAX_PASSES = 2

# The card verifier is called in batches: one call over the whole card set produced 20,892
# characters of JSON for 24 cards, two thirds of the provider's hard output ceiling.
VERIFY_BATCH_SIZE = 8
VERIFY_ISSUE_LIMIT = 12
VERIFY_ISSUE_TOTAL = 20
VERIFY_ATTEMPTS = 2


def make_plan_node(deps: Deps) -> Callable[[RunState], NodeReturn]:
    """Split the paper and mine one card set per section."""

    def plan(state: RunState) -> NodeReturn:
        ui = deps.ui
        workspace = deps.workspace(state)
        paper = paper_of(state)
        text = state.get("paper_text") or (paper.full_text if paper else "")
        title = paper.title if paper else "unknown paper"
        ui.step("stage 2/5 - splitting the paper", f"{len(text):,} chars")
        if not text.strip():
            return {"status": "failed", "error": "no paper text available for planning"}

        sections = _split_sections(deps, state, title, text)
        workspace.write_json(
            "paper/sections.json",
            [section.model_dump(mode="json") for section in sections],
        )
        ui.dim(f"{len(sections)} section(s): " + ", ".join(s.name[:28] for s in sections[:8]))

        miner = _CardMiner(deps, title=title, lesson_context=deps.lesson_context("plan"))
        miner.collect(sections)
        cards = _renumber(miner.cards)
        rows = miner.rows()
        workspace.write_json("cards/coverage.json", rows)
        workspace.write_text("cards/coverage.md", coverage_markdown(rows))
        if miner.skipped:
            workspace.write_json("cards/skipped.json", miner.skipped)
        if not cards:
            ui.error("the planner found no testable claim in this paper")
            workspace.write_text("cards/cards.md", cards_to_markdown(cards))
            return {
                "status": "failed",
                "error": "no reproduction cards could be written",
                "cards": [],
                "coverage": rows,
            }

        workspace.write_json("cards/cards.json", dump_cards(cards))
        workspace.write_text("cards/cards.md", cards_to_markdown(cards))
        ui.success(coverage_summary(rows, len(cards)))
        pending = [row for row in rows if row["status"] in {"failed", "unanswered"}]
        if pending:
            ui.warn(
                f"{len(pending)} section(s) produced no card: "
                + ", ".join(row["section"][:28] for row in pending[:4])
            )
        ui.show_cards(cards)
        return {
            "status": "running",
            "cards": dump_cards(cards),
            "sections": [section.model_dump(mode="json") for section in sections],
            "coverage": rows,
            "plan_round": 0,
        }

    return plan


def _split_sections(deps: Deps, state: RunState, title: str, text: str) -> list[PaperSection]:
    """Ask the model for an authoritative split, falling back to the heuristic one."""
    heuristic = [PaperSection.model_validate(item) for item in state.get("sections") or []]
    usable_heuristic = [section for section in heuristic if section.text.strip()]
    if len(text) <= MAX_CONTEXT_CHARS:
        try:
            result: SectionedPaper = deps.llm.json(
                prompts.SPLIT_SYSTEM,
                prompts.split_user_message(title, text),
                SectionedPaper,
                label="section_split",
            )
            split = [section for section in result.sections if section.text.strip()]
            if split:
                return _merge_adjacent_duplicates(split)
        except LLMError as exc:
            deps.ui.warn(f"model-based section split failed ({exc}); using text heuristics")
    if len(usable_heuristic) >= 2:
        return _merge_adjacent_duplicates(usable_heuristic)
    fallback = split_into_sections(text) or [PaperSection(index=0, name="Full text", text=text)]
    return _merge_adjacent_duplicates(fallback)


NUMBERED_HEADING = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})(?![\d.])(?=\s|$)")


def section_key(name: str) -> str:
    """Stable identity for a section: its own numbering when the heading has one."""
    match = NUMBERED_HEADING.match(name or "")
    if match:
        return match.group(1)
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-") or "unnamed"


def claim_key(card: ReproductionCard) -> str:
    """Comparison key for a claim, so a repeated statement is not carded twice."""
    return re.sub(r"[^a-z0-9]+", " ", (card.claim.statement or "").lower()).strip()


def unique_keys(names: list[str]) -> list[str]:
    """One key per heading; a name that repeats far apart gets a ``#2`` suffix."""
    counts: dict[str, int] = {}
    keys: list[str] = []
    for name in names:
        base = section_key(name)
        counts[base] = counts.get(base, 0) + 1
        keys.append(base if counts[base] == 1 else f"{base}#{counts[base]}")
    return keys


def _merge_adjacent_duplicates(sections: list[PaperSection]) -> list[PaperSection]:
    """Fold a heading the splitter emitted twice in a row into one section.

    A repeated figure caption ("FFN") otherwise becomes a section of its own, and the
    coverage ledger would count a 15-character fragment as a reproduced section.
    """
    merged: list[PaperSection] = []
    for section in sections:
        if merged and section_key(merged[-1].name) == section_key(section.name):
            if len(section.text) > len(merged[-1].text):
                merged[-1] = section
            continue
        merged.append(section)
    for position, section in enumerate(merged):
        section.index = position
    return merged


def _renumber(cards: list[ReproductionCard]) -> list[ReproductionCard]:
    for position, card in enumerate(cards, start=1):
        card.card_id = f"c{position:02d}"
    return cards


SEVERITY_RANK = {"blocker": 0, "major": 1, "minor": 2}


def card_batches(
    cards: list[ReproductionCard], size: int = VERIFY_BATCH_SIZE
) -> list[list[ReproductionCard]]:
    """Split the cards into verification batches, in order, without losing any."""
    if size <= 0:
        raise ValueError("batch size must be positive")
    return [cards[index : index + size] for index in range(0, len(cards), size)] or [[]]


def dedupe_issues(issues: list[PlanIssue]) -> list[PlanIssue]:
    """One issue per (card, field), keeping the most severe, in first-seen order.

    Every batch judges coverage against the same outline, so the same gap is reported more
    than once; two different gaps differ in the field ("coverage:3.1" vs "coverage:4.2").
    """
    chosen: dict[tuple[str, str], PlanIssue] = {}
    order: list[tuple[str, str]] = []
    for issue in issues:
        key = (issue.card_id or "", (issue.field or "").strip().lower())
        current = chosen.get(key)
        if current is None:
            chosen[key] = issue
            order.append(key)
        elif SEVERITY_RANK[issue.severity] < SEVERITY_RANK[current.severity]:
            chosen[key] = issue
    return [chosen[key] for key in order]


def cap_issues(issues: list[PlanIssue], limit: int) -> tuple[list[PlanIssue], int]:
    """Keep the most severe ``limit`` issues, in their original order; report the overflow."""
    if limit <= 0 or len(issues) <= limit:
        return list(issues), 0
    ranked = sorted(
        range(len(issues)), key=lambda index: (SEVERITY_RANK[issues[index].severity], index)
    )
    kept = sorted(ranked[:limit])
    return [issues[index] for index in kept], len(issues) - limit


def merge_reviews(parts: list[PlanReview], *, total_cap: int = VERIFY_ISSUE_TOTAL) -> PlanReview:
    """One review out of many batches.

    ``ok`` is decided on the *uncapped* merged set, so trimming the list for the prompt can
    never turn a blocker into an approval.
    """
    merged = dedupe_issues([issue for part in parts for issue in part.issues])
    severe = [issue for issue in merged if issue.severity in {"blocker", "major"}]
    kept, dropped = cap_issues(merged, total_cap)
    summary = " | ".join(part.summary.strip() for part in parts if part.summary.strip())
    if dropped:
        summary += f" ({dropped} lower-severity issue(s) not carried forward)"
    lessons: list[str] = []
    for part in parts:
        lessons.extend(lesson for lesson in part.lessons_used if lesson not in lessons)
    return PlanReview(ok=not severe, issues=kept, summary=summary[:2000], lessons_used=lessons)


def _review_cards(
    deps: Deps,
    *,
    cards: list[ReproductionCard],
    paper_title: str,
    outline: str,
    lesson_context: str,
    round_no: int,
) -> tuple[PlanReview | None, list[str]]:
    """Review the cards in batches; ``None`` means no batch could be reviewed at all."""
    batches = card_batches(cards)
    parts: list[PlanReview] = []
    notes: list[str] = []
    for position, batch in enumerate(batches, start=1):
        # One batch keeps the historical label, so single-batch transcripts stay comparable.
        label = (
            f"plan_review:{round_no}" if len(batches) == 1 else f"plan_review:{round_no}:{position}"
        )
        review: PlanReview | None = None
        error: Exception | None = None
        for _attempt in range(VERIFY_ATTEMPTS):
            try:
                review = deps.llm.json(
                    prompts.VERIFY_SYSTEM,
                    prompts.verify_user_message(
                        paper_title=paper_title,
                        cards=batch,
                        section_outline=outline,
                        lesson_context=lesson_context,
                        batch=(position, len(batches)),
                        issue_limit=VERIFY_ISSUE_LIMIT,
                    ),
                    PlanReview,
                    role="verifier",
                    label=label,
                )
                break
            except LLMError as exc:
                error = exc
        if review is None:
            note = f"batch {position}/{len(batches)} could not be reviewed ({error})"
            deps.ui.warn(note)
            notes.append(note)
            continue
        review.issues, dropped = cap_issues(review.issues, VERIFY_ISSUE_LIMIT)
        if dropped:
            deps.ui.dim(f"verifier batch {position}: {dropped} issue(s) past the cap were dropped")
        parts.append(review)
    if not parts:
        return None, notes
    return merge_reviews(parts, total_cap=VERIFY_ISSUE_TOTAL), notes


def coverage_summary(rows: list[dict[str, Any]], cards: int) -> str:
    """One honest line: how many cards, covering how many of the eligible sections."""
    carded = sum(1 for row in rows if row["status"] == "carded")
    untestable = sum(1 for row in rows if row["status"] == "no_testable_claim")
    missing = sum(1 for row in rows if row["status"] in {"failed", "unanswered"})
    text = f"{cards} reproduction card(s) covering {carded} of {len(rows)} eligible section(s)"
    if untestable:
        text += f"; {untestable} declared no testable claim"
    if missing:
        text += f"; {missing} not carded"
    return text


def coverage_markdown(rows: list[dict[str, Any]]) -> str:
    """The coverage ledger as a table, for ``cards/coverage.md``."""
    lines = [
        "| section | key | status | cards | note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        note = (row.get("detail") or "").replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {row['section']} | {row['key']} | {row['status']} | "
            f"{', '.join(row.get('cards') or []) or '-'} | {note[:160]} |"
        )
    return "\n".join(lines) + "\n"


class _CardMiner:
    """Mine cards section by section, with a ledger that accounts for every section.

    The order matters.  The coverage pass cards every eligible section once, the retry
    pass re-asks the sections that failed or answered nothing, and only then is the
    leftover budget spent deepening the sections that did answer.
    """

    def __init__(self, deps: Deps, *, title: str, lesson_context: str) -> None:
        self.deps = deps
        self.title = title
        self.lesson_context = lesson_context
        self.cards: list[ReproductionCard] = []
        self.skipped: list[str] = []
        self.ledger: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ mining
    def _entry(self, key: str, section: PaperSection) -> dict[str, Any]:
        return self.ledger.setdefault(
            key,
            {
                "key": key,
                "section": section.name,
                "cards": [],
                "status": "unanswered",
                "detail": "",
                "passes": 0,
            },
        )

    def mine(self, key: str, section: PaperSection, quota: int, *, counted: bool = True) -> None:
        """One card-mining call; the ledger records what came back, or why nothing did."""
        entry = self._entry(key, section)
        if counted:
            entry["passes"] += 1
        used = [f"c{position:02d}" for position in range(1, len(self.cards) + 1)]
        # Renumbering happens after the passes, so quote the ids those cards will end up with.
        already = [f"c{position + 1:02d}" for position in entry["cards"]]
        try:
            card_set: CardSet = self.deps.llm.json(
                prompts.CARD_SYSTEM,
                prompts.card_user_message(
                    paper_title=self.title,
                    section_name=section.name,
                    section_text=truncate(section.text, MAX_SECTION_CHARS),
                    used_card_ids=used,
                    lesson_context=self.lesson_context,
                    max_cards=quota,
                    already_carded=already,
                ),
                CardSet,
                label=f"cards:{section.name[:24]}",
            )
        except LLMError as exc:
            self.deps.ui.warn(f"card mining failed for {section.name[:40]}: {exc}")
            entry["status"] = "failed"
            entry["detail"] = f"{type(exc).__name__}: {exc}"
            return
        fresh: list[ReproductionCard] = []
        seen = {claim_key(self.cards[position]) for position in entry["cards"]}
        for card in list(card_set.cards)[: max(0, quota)]:
            signature = claim_key(card)
            if signature and signature in seen:
                continue
            seen.add(signature)
            fresh.append(card)
        if fresh:
            for card in fresh:
                card.identity.section = card.identity.section or section.name
            start = len(self.cards)
            self.cards.extend(fresh)
            entry["cards"].extend(range(start, start + len(fresh)))
            entry["status"] = "carded"
            entry["detail"] = ""
            return
        if entry["cards"]:
            # The section already answered; a repeat or a "nothing new" adds nothing.
            return
        reasons = [line.strip() for line in card_set.skipped if line.strip()]
        if reasons:
            entry["status"] = "no_testable_claim"
            entry["detail"] = "; ".join(reasons)
            self.skipped.extend(reasons)
            return
        entry["status"] = "unanswered"
        entry["detail"] = "the model returned no card and no reason"

    # ------------------------------------------------------------- allocations
    def collect(self, sections: list[PaperSection]) -> None:
        """Cover every eligible section, retry the stragglers, then deepen."""
        keys = unique_keys([section.name for section in sections])
        eligible = [
            (key, section)
            for key, section in zip(keys, sections, strict=True)
            if not section.is_reference and section.text.strip()
        ]
        for key, section in eligible:
            self.deps.ui.dim(f"card mining: {section.name[:60]}")
            self.mine(key, section, CARDS_PER_SECTION)
        for _round in range(2, MAX_PASSES + 1):
            pending = [
                (key, section)
                for key, section in eligible
                if self.ledger[key]["status"] in {"failed", "unanswered"}
            ]
            if not pending:
                break
            for key, section in pending:
                self.deps.ui.dim(f"card mining retry: {section.name[:60]}")
                self.mine(key, section, CARDS_PER_SECTION)
        self._deepen(eligible)

    def _deepen(self, eligible: list[tuple[str, PaperSection]]) -> None:
        """Spend whatever is left of the budget on the sections that already answered."""
        remaining = max(0, CARD_BUDGET - len(self.cards))
        for key, section in sorted(eligible, key=lambda pair: -len(pair[1].text)):
            if remaining <= 0:
                break
            entry = self.ledger[key]
            room = min(MAX_CARDS_PER_SECTION - len(entry["cards"]), remaining)
            if entry["status"] != "carded" or room <= 0:
                continue
            before = len(self.cards)
            self.mine(key, section, room, counted=False)
            remaining -= len(self.cards) - before

    # ---------------------------------------------------------------- readers
    def rows(self) -> list[dict[str, Any]]:
        """The ledger with the final card ids, so it is read after ``_renumber``."""
        return [
            {**entry, "cards": [self.cards[position].card_id for position in entry["cards"]]}
            for entry in self.ledger.values()
        ]

    @property
    def coverage_ok(self) -> bool:
        return all(
            entry["status"] in {"carded", "no_testable_claim"} for entry in self.ledger.values()
        )


def make_verify_plan_node(deps: Deps) -> Callable[[RunState], NodeReturn]:
    """The card verifier: find defects, write lessons, decide whether to replan."""

    def verify_plan(state: RunState) -> NodeReturn:
        ui = deps.ui
        workspace = deps.workspace(state)
        cards = cards_of(state)
        round_no = int(state.get("plan_round") or 0) + 1
        max_rounds = deps.settings.runtime.max_plan_rounds
        ui.step(
            f"stage 2.1/5 - card verifier (round {round_no}/{max_rounds})", f"{len(cards)} cards"
        )
        paper = paper_of(state)
        lesson_context = deps.lesson_context("plan")
        review, notes = _review_cards(
            deps,
            cards=cards,
            paper_title=paper.title if paper else "unknown paper",
            outline=section_outline(state),
            lesson_context=lesson_context,
            round_no=round_no,
        )
        if review is None:
            message = "; ".join(notes) or "no batch answered"
            ui.warn(f"card verifier unavailable ({message}); releasing the cards as they are")
            return {
                "plan_round": round_no,
                "plan_review": {"ok": True, "issues": [], "summary": f"verifier error: {message}"},
                "plan_issues": [],
                "plan_carryover": "",
            }
        if notes:
            ui.warn(
                f"card verifier left {len(notes)} batch(es) unreviewed: " + "; ".join(notes[:3])
            )

        workspace.write_json(f"cards/review_round{round_no}.json", review.model_dump(mode="json"))
        issues = [
            f"[{issue.card_id or 'paper'}] {issue.field} ({issue.severity}): {issue.problem} "
            f"-> {issue.suggestion}"
            for issue in review.issues
        ]
        if issues:
            deps.lessons(state).append(
                "plan",
                f"card verifier, round {round_no}: {len(issues)} issue(s)",
                issues,
                tags=("plan", "card-review"),
            )
        severe = [issue for issue in review.issues if issue.severity in {"blocker", "major"}]
        final_round = round_no >= max_rounds
        if not severe:
            ui.success(f"card verifier accepted the cards: {review.summary[:160]}")
            return {
                "plan_round": round_no,
                "plan_review": review.model_dump(mode="json"),
                "plan_issues": issues,
                "plan_carryover": "",
            }
        if final_round:
            ui.warn(
                f"replan limit reached ({max_rounds} rounds) - releasing the cards with "
                f"{len(severe)} open issue(s)"
            )
            carryover = _carryover_text(round_no, issues, review)
            workspace.write_text("cards/plan_carryover.md", carryover)
            return {
                "plan_round": round_no,
                "plan_review": review.model_dump(mode="json"),
                "plan_issues": issues,
                "plan_carryover": carryover,
            }
        ui.replan_notice("plan", round_no, max_rounds, issues)
        return {
            "plan_round": round_no,
            "plan_review": review.model_dump(mode="json"),
            "plan_issues": issues,
        }

    return verify_plan


def _carryover_text(round_no: int, issues: list[str], review: PlanReview) -> str:
    lines = [
        f"The card verifier stopped at round {round_no} with unresolved issues.",
        f"Verifier summary: {review.summary}",
        "",
        "Open issues inherited by the execution stage:",
    ]
    lines.extend(f"- {issue}" for issue in issues)
    lines.append("")
    lines.append(
        "Treat these as known risks: work around them where you can, and state honestly "
        "in the metrics and the report which ones could not be resolved."
    )
    return "\n".join(lines)


def make_revise_plan_node(deps: Deps) -> Callable[[RunState], NodeReturn]:
    """The planner deliberates on the verifier's issues and patches the cards."""

    def revise_plan(state: RunState) -> NodeReturn:
        ui = deps.ui
        workspace = deps.workspace(state)
        cards = cards_of(state)
        review_payload = state.get("plan_review") or {}
        round_no = int(state.get("plan_round") or 1)
        ui.step("stage 2.1/5 - replanning", "the planner is reviewing the verifier's issues")
        staged = deps.lessons(state).read("plan")
        lesson_context = (staged[-2500:] if staged else "") or deps.lesson_context("plan")
        try:
            revision: PlanRevision = deps.llm.json(
                prompts.REVISE_SYSTEM,
                prompts.revise_user_message(
                    cards=cards,
                    review_json=json_block(review_payload),
                    lesson_context=lesson_context,
                ),
                PlanRevision,
                label=f"card_revision:{round_no}",
            )
        except LLMError as exc:
            ui.warn(f"replanning failed ({exc}); keeping the current cards")
            return {"cards": dump_cards(cards)}

        by_id = {card.card_id: card for card in cards}
        applied: list[str] = []
        rejected: list[str] = list(revision.rejected_issues)
        for patch in revision.patches:
            card = by_id.get(patch.card_id)
            if card is None:
                rejected.append(f"{patch.card_id}.{patch.field}: unknown card id")
                continue
            try:
                by_id[patch.card_id] = apply_card_patch(card, patch.field, patch.value)
                applied.append(f"{patch.card_id}.{patch.field}")
            except (ValidationError, ValueError) as exc:
                rejected.append(f"{patch.card_id}.{patch.field}: {exc}")
        updated = [by_id[card.card_id] for card in cards]

        workspace.write_json(
            f"cards/revision_round{round_no}.json",
            {
                "applied": applied,
                "rejected": rejected,
                "reasoning": revision.overall_reasoning,
            },
        )
        workspace.write_json("cards/cards.json", dump_cards(updated))
        workspace.write_text("cards/cards.md", cards_to_markdown(updated))
        if revision.overall_reasoning:
            ui.dim(f"planner reasoning: {revision.overall_reasoning[:400]}")
        if rejected:
            ui.dim(f"rejected {len(rejected)} issue(s): " + "; ".join(rejected[:4]))
        ui.success(f"replanned: {len(applied)} field(s) rewritten")
        ui.show_cards(updated)
        return {"cards": dump_cards(updated)}

    return revise_plan


def plan_review_ok(state: RunState) -> bool:
    """Routing helper used by the graph: should the loop stop here?"""
    review = state.get("plan_review") or {}
    if review.get("ok"):
        return True
    issues = review.get("issues") or []
    severe = [issue for issue in issues if issue.get("severity") in {"blocker", "major"}]
    return not severe


def plan_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
