"""Per-section card coverage: the planner must account for every eligible section.

Regression for run-20260920-042507.  The planner mined cards section by section, but a
global budget (``MAX_CARDS = 24``) broke out of the loop the moment it was reached, so
only 7 of the paper's 26 sections were carded while the CLI still reported "24 card(s)
in scope".  Coverage is a property of each section, so the quota is per section now.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from essay_agent.errors import LLMError
from essay_agent.nodes.base import section_outline
from essay_agent.nodes.plan import (
    CARD_BUDGET,
    MAX_CARDS_PER_SECTION,
    coverage_markdown,
    make_plan_node,
    section_key,
    unique_keys,
)
from tests.fakes import FakeLLM
from tests.pipeline import build_deps, card_payload


def paper_sections(count: int) -> list[dict[str, Any]]:
    """``count`` numbered sections, each with a body."""
    return [
        {
            "index": position,
            "name": f"{position + 1} Section {position + 1}",
            "text": f"Section {position + 1} claims that method {position + 1} improves accuracy.",
            "is_reference": False,
        }
        for position in range(count)
    ]


def label_of(section: dict[str, Any]) -> str:
    """The card-mining label the planner uses for one section."""
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


def sequenced(*answers: dict[str, Any]) -> Callable[..., dict[str, Any]]:
    """A scripted reply that walks a list, repeating the last answer."""
    remaining = list(answers)

    def reply(*_ignored: Any) -> dict[str, Any]:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return reply


def run_plan(
    settings: Any,
    sections: list[dict[str, Any]],
    responses: dict[str, Any],
    *,
    errors: dict[str, Exception] | None = None,
    run_id: str = "run-coverage",
) -> tuple[Any, FakeLLM, dict[str, Any]]:
    """Drive the real planner node only; nothing else in the graph runs."""
    llm = FakeLLM(responses, error_labels=errors)
    deps = build_deps(settings, llm)
    state: dict[str, Any] = {
        "run_id": run_id,
        "slug": "coverage",
        "paper": {"id": "p1", "title": "Coverage Fixture"},
        "paper_text": "\n\n".join(f"{s['name']}\n{s['text']}" for s in sections),
    }
    state.update(make_plan_node(deps)(state))
    return deps, llm, state


def rows_by_key(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["key"]: row for row in state["coverage"]}


def test_every_eligible_section_gets_a_card(settings) -> None:
    sections = paper_sections(5)
    deps, _llm, state = run_plan(settings, sections, one_card_each(sections))

    assert len(state["cards"]) == 5
    assert [card["card_id"] for card in state["cards"]] == ["c01", "c02", "c03", "c04", "c05"]
    assert [row["status"] for row in state["coverage"]] == ["carded"] * 5
    assert deps.workspace(state).exists("cards", "coverage.json")
    assert deps.workspace(state).exists("cards", "coverage.md")


def test_the_budget_no_longer_starves_the_end_of_the_paper(settings) -> None:
    sections = paper_sections(CARD_BUDGET)
    _deps, _llm, state = run_plan(settings, sections, one_card_each(sections))

    assert len(state["cards"]) == CARD_BUDGET
    assert all(row["status"] == "carded" for row in state["coverage"])
    assert rows_by_key(state)[str(CARD_BUDGET)]["cards"] == [f"c{CARD_BUDGET:02d}"]


def test_a_section_with_nothing_testable_is_recorded(settings) -> None:
    sections = paper_sections(2)
    responses = one_card_each(sections)
    responses[label_of(sections[1])] = {"cards": [], "skipped": ["motivation only"]}

    _deps, _llm, state = run_plan(settings, sections, responses)

    rows = rows_by_key(state)
    assert rows["2"]["status"] == "no_testable_claim"
    assert "motivation only" in rows["2"]["detail"]
    assert len(state["cards"]) == 1


def test_a_failing_call_is_retried_once_then_reported(settings) -> None:
    sections = paper_sections(2)
    failed = label_of(sections[1])

    _deps, _llm, state = run_plan(
        settings, sections, one_card_each(sections), errors={failed: LLMError("model refused")}
    )

    rows = rows_by_key(state)
    assert rows["2"]["status"] == "failed"
    assert rows["2"]["passes"] == 2
    assert rows["1"]["status"] == "carded"
    assert len(state["cards"]) == 1


def test_the_retry_pass_recovers_a_transient_failure(settings) -> None:
    sections = paper_sections(1)
    attempts: list[int] = []

    def flaky(*_ignored: Any) -> dict[str, Any]:
        attempts.append(1)
        if len(attempts) == 1:
            raise LLMError("temporary outage")
        return {"cards": [card_stating("the claim holds")], "skipped": []}

    _deps, _llm, state = run_plan(
        settings, sections, {"section_split": {"sections": sections}, label_of(sections[0]): flaky}
    )

    rows = rows_by_key(state)
    assert rows["1"]["status"] == "carded"
    assert rows["1"]["passes"] == 2
    assert len(state["cards"]) == 1


def test_deepening_adds_new_claims_from_the_same_section(settings) -> None:
    sections = paper_sections(1)
    responses = {
        "section_split": {"sections": sections},
        label_of(sections[0]): sequenced(
            {"cards": [card_stating("claim one")], "skipped": []},
            {"cards": [card_stating("claim two")], "skipped": []},
        ),
    }

    _deps, _llm, state = run_plan(settings, sections, responses)

    statements = [card["claim"]["statement"] for card in state["cards"]]
    assert statements == ["claim one", "claim two"]
    assert len(rows_by_key(state)["1"]["cards"]) == 2


def test_a_repeated_claim_is_never_carded_twice(settings) -> None:
    sections = paper_sections(1)
    responses = {
        "section_split": {"sections": sections},
        label_of(sections[0]): sequenced(
            {"cards": [card_stating("claim one")], "skipped": []},
            {"cards": [card_stating("claim one")], "skipped": []},
        ),
    }

    _deps, _llm, state = run_plan(settings, sections, responses)

    assert len(state["cards"]) == 1
    assert rows_by_key(state)["1"]["status"] == "carded"


def test_the_per_section_cap_holds(settings) -> None:
    sections = paper_sections(2)
    responses = {
        "section_split": {"sections": sections},
        **{
            label_of(section): {
                "cards": [card_stating(f"claim {index}") for index in range(5)],
                "skipped": [],
            }
            for section in sections
        },
    }

    _deps, _llm, state = run_plan(settings, sections, responses)

    assert len(state["cards"]) <= CARD_BUDGET
    assert all(len(row["cards"]) <= MAX_CARDS_PER_SECTION for row in state["coverage"])


def test_section_outline_carries_the_coverage(settings) -> None:
    sections = paper_sections(2)
    responses = one_card_each(sections)
    responses[label_of(sections[1])] = {"cards": [], "skipped": ["no measurable outcome"]}

    _deps, _llm, state = run_plan(settings, sections, responses)
    outline = section_outline(state)

    assert "cards c01" in outline
    assert "no testable claim" in outline
    assert "truncated" not in outline


def test_an_adjacent_duplicate_heading_is_one_section(settings) -> None:
    sections = [
        {"index": 0, "name": "1 Model", "text": "Body one.", "is_reference": False},
        {"index": 1, "name": "FFN", "text": "caption fragment", "is_reference": False},
        {"index": 2, "name": "FFN", "text": "FFN body " * 20, "is_reference": False},
        {"index": 3, "name": "References", "text": "Smith et al.", "is_reference": True},
    ]
    responses = one_card_each(sections)
    responses[label_of(sections[1])] = {"cards": [], "skipped": ["integration detail"]}

    _deps, _llm, state = run_plan(settings, sections, responses)

    names = [section["name"] for section in state["sections"]]
    keys = [row["key"] for row in state["coverage"]]
    assert names.count("FFN") == 1
    assert keys == ["1", "ffn"]


def test_section_keys_use_the_paper_numbering() -> None:
    assert section_key("2.1 I N DEEP LEARNING") == "2.1"
    assert section_key("3.3 S CALING THE NUMBER , SIZE AND FREQUENCY") == "3.3"
    assert section_key("ABSTRACT") == "abstract"


def test_a_figure_caption_is_not_read_as_a_section_number() -> None:
    assert section_key("101 B, T, W, H, k") == "101-b-t-w-h-k"
    assert unique_keys(["Method", "Results", "Method"]) == ["method", "results", "method#2"]


def test_coverage_markdown_lists_every_row() -> None:
    rows = [
        {
            "section": "1 A",
            "key": "1",
            "status": "carded",
            "cards": ["c01"],
            "detail": "",
            "passes": 1,
        }
    ]
    text = coverage_markdown(rows)

    assert "| section | key | status | cards | note |" in text
    assert "| 1 A | 1 | carded | c01 |" in text
