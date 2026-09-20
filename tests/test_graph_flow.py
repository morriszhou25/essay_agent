"""End-to-end pipeline behaviour on the happy path, aborts and human choices."""

from __future__ import annotations

from pathlib import Path

from essay_agent.agent import run_task
from essay_agent.console import SilentUI
from essay_agent.schemas.paper import PaperSection
from essay_agent.tools.paper_search import _first_title_like_line
from tests.fakes import FakeLLM, FakeProber, FakeSearcher, make_candidate
from tests.pipeline import ICLR_PAGE_1, PAPER_TEXT, build_deps, happy_responses, result_dirs

REPORT = "# Reproducing: Test Paper\n\n## Summary\nThe claim holds in a lightweight run.\n"
ADAM_TITLE = "Adam: A Method for Stochastic Optimization"


def test_happy_path_completes_and_publishes(settings) -> None:
    llm = FakeLLM(happy_responses(), texts={"report": REPORT})
    ui = SilentUI()
    deps = build_deps(settings, llm, ui=ui)

    result = run_task("Test Paper 0: A Study", deps)

    assert result.status == "completed", result.message
    paths = settings.resolved_paths()
    assert result.published.get("result_dir"), result.published
    report_dir = Path(result.published["result_dir"])
    assert (report_dir / "report.md").read_text(encoding="utf-8").strip() == REPORT.strip()
    assert (report_dir / "metrics.json").is_file()
    assert (report_dir / "figures" / "fig1.png").is_file()
    assert (report_dir / "cards.md").is_file()
    code_dir = Path(result.published["code_dir"])
    assert (code_dir / "repro.py").is_file()

    state = result.state
    assert state["fetch_status"] == "ok"
    assert state["verdict"]["verdict"] == "successful"
    assert state["exec_round"] == 1
    assert state["plan_round"] == 1
    assert len(state["cards"]) == 1
    assert Path(state["run_dir"]).is_dir()  # staging area kept for inspection
    assert not list(paths.lesson_dir.glob("*.txt"))  # nothing was learned yet

    # the retrieval tool was forced, and the match was verified against the list
    assert ("tool_args", "paper_search_args", "search_papers", "main") in llm.calls
    assert ("json", "paper_match", "MatchDecision", "main") in llm.calls
    assert ("json", "plan_review:1", "PlanReview", "verifier") in llm.calls


def test_paper_text_is_written_to_the_workspace(settings) -> None:
    llm = FakeLLM(happy_responses(), texts={"report": REPORT})
    deps = build_deps(settings, llm)
    result = run_task("Test Paper 0: A Study", deps)
    run_dir = Path(result.state["run_dir"])
    assert (run_dir / "paper" / "paper.json").is_file()
    assert (run_dir / "paper" / "full_text.md").read_text(encoding="utf-8") == PAPER_TEXT
    assert (run_dir / "cards" / "cards.json").is_file()
    assert (run_dir / "code" / "repro.py").is_file()


def test_no_candidates_aborts_with_the_required_message(settings) -> None:
    llm = FakeLLM(happy_responses())
    deps = build_deps(settings, llm, searcher=FakeSearcher(candidates=[]))

    result = run_task("A paper that does not exist", deps)

    assert result.status == "aborted"
    assert "未匹配到对应论文" in result.message
    assert result_dirs(settings) == []
    assert not (settings.resolved_paths().result_dir / "report.md").exists()


def test_search_backend_failure_is_reported_as_a_failure(settings) -> None:
    from essay_agent.errors import PaperSearchError

    llm = FakeLLM(happy_responses())
    deps = build_deps(
        settings, llm, searcher=FakeSearcher(search_error=PaperSearchError("all backends down"))
    )
    result = run_task("Test Paper 0: A Study", deps)
    assert result.status == "failed"
    assert "search failed" in result.message
    assert result_dirs(settings) == []


def test_ambiguous_match_asks_the_human(settings) -> None:
    candidates = [make_candidate(0), make_candidate(1), make_candidate(2)]
    responses = happy_responses()
    responses["paper_match"] = {
        "index": None,
        "confidence": 0.2,
        "reason": "three candidates share the title",
        "ambiguous": True,
    }
    response_cards = responses["cards:1 Introduction"]
    llm = FakeLLM(responses, texts={"report": REPORT})
    ui = SilentUI(choices=[1])
    deps = build_deps(
        settings, llm, searcher=FakeSearcher(candidates=candidates, text=PAPER_TEXT), ui=ui
    )

    result = run_task("Test Paper: A Study", deps)

    assert result.status == "completed", result.message
    assert ui.asked and "which paper" in ui.asked[0]
    assert result.state["chosen_candidate"]["candidate_id"] == "arxiv:0000.00001"
    assert response_cards  # keeps the reference alive for clarity


def test_low_confidence_match_also_asks_the_human(settings) -> None:
    responses = happy_responses()
    responses["paper_match"] = {
        "index": 0,
        "confidence": 0.31,
        "reason": "title is similar but not identical",
        "ambiguous": False,
    }
    llm = FakeLLM(responses, texts={"report": REPORT})
    ui = SilentUI(choices=[0])
    deps = build_deps(settings, llm, ui=ui)
    result = run_task("Maybe Test Paper 0", deps)
    assert result.status == "completed"
    assert ui.asked  # the user was consulted


def test_human_declining_the_candidate_list_aborts(settings) -> None:
    responses = happy_responses()
    responses["paper_match"] = {
        "index": None,
        "confidence": 0.0,
        "reason": "unclear",
        "ambiguous": True,
    }
    llm = FakeLLM(responses, texts={"report": REPORT})
    ui = SilentUI(choices=[None])
    deps = build_deps(settings, llm, ui=ui)

    result = run_task("Something vague", deps)

    assert result.status == "aborted"
    assert "未匹配到对应论文" in result.message
    assert result_dirs(settings) == []


def test_abstract_only_paper_asks_before_continuing(settings) -> None:
    searcher = FakeSearcher(
        candidates=[make_candidate(0)],
        text="short abstract",
        full_text_available=False,
    )
    llm = FakeLLM(happy_responses(), texts={"report": REPORT})
    ui = SilentUI(confirmations=[False])
    deps = build_deps(settings, llm, searcher=searcher, ui=ui)
    result = run_task("Test Paper 0", deps)
    assert result.status == "aborted"
    assert "full text unavailable" in result.message
    assert ui.find("notice")  # the user was told why this matters


def test_feasibility_blocker_stops_the_run_when_the_user_declines(settings) -> None:
    blocked = {
        "checks": [
            {
                "card_id": "c01",
                "feasible": False,
                "severity": "blocker",
                "findings": ["CIFAR-10 is unreachable from this network"],
                "dataset": "CIFAR-10",
                "dataset_available": False,
                "mitigation": None,
            }
        ],
        "blockers": ["CIFAR-10 cannot be downloaded"],
        "proceed": False,
        "summary": "the required dataset is not obtainable",
    }
    llm = FakeLLM(happy_responses(feasibility=blocked), texts={"report": REPORT})
    ui = SilentUI(confirmations=[False])
    deps = build_deps(settings, llm, ui=ui)

    result = run_task("Test Paper 0", deps)

    assert result.status == "aborted"
    assert "feasibility" in result.message
    assert ui.find("error")
    assert result_dirs(settings) == []


def test_feasibility_blocker_can_be_overridden_by_the_user(settings) -> None:
    blocked = {
        "checks": [],
        "blockers": ["dataset mirror unclear"],
        "proceed": False,
        "summary": "blocked",
    }
    llm = FakeLLM(happy_responses(feasibility=blocked), texts={"report": REPORT})
    ui = SilentUI(confirmations=[True])
    deps = build_deps(settings, llm, ui=ui)

    result = run_task("Test Paper 0", deps)

    assert result.status == "completed", result.message
    assert result.state["blockers"] == ["dataset mirror unclear"]


def test_probes_are_run_against_the_card_datasets(settings) -> None:
    llm = FakeLLM(happy_responses(), texts={"report": REPORT})
    prober = FakeProber()
    deps = build_deps(settings, llm, prober=prober)
    run_task("Test Paper 0", deps)
    assert "sklearn:iris" in prober.probed


def test_planner_failure_stops_the_pipeline(settings) -> None:
    from essay_agent.errors import LLMError

    llm = FakeLLM(
        happy_responses(),
        texts={"report": REPORT},
        error_labels={
            "cards:1 Introduction": LLMError("model refused"),
            "cards:2 Experiments": LLMError("model refused"),
        },
    )
    deps = build_deps(settings, llm)

    result = run_task("Test Paper 0", deps)

    assert result.status == "failed"
    assert "no reproduction cards" in result.message
    assert result_dirs(settings) == []
    assert PaperSection is not None


def test_camera_ready_header_is_not_a_title_mismatch(settings) -> None:
    """Regression: the ICLR banner used to be read as the title and abort the run."""
    responses = happy_responses()
    responses["paper_search_args"] = {"title": ADAM_TITLE, "max_results": 4}
    llm = FakeLLM(responses, texts={"report": REPORT})
    ui = SilentUI()
    searcher = FakeSearcher(
        candidates=[make_candidate(0, title=ADAM_TITLE)],
        text=PAPER_TEXT,
        info={"pdf_title": _first_title_like_line(ICLR_PAGE_1, ADAM_TITLE)},
    )
    deps = build_deps(settings, llm, searcher=searcher, ui=ui)

    result = run_task(ADAM_TITLE, deps)

    assert result.status == "completed", result.message
    assert not any("does not obviously match" in message for message in ui.find("warn"))
    assert "continue with this document?" not in ui.asked


def test_a_genuinely_wrong_document_is_still_questioned(settings) -> None:
    llm = FakeLLM(happy_responses(), texts={"report": REPORT})
    ui = SilentUI()
    searcher = FakeSearcher(
        candidates=[make_candidate(0)],
        text=PAPER_TEXT,
        info={"pdf_title": "Whales of the North Atlantic: a population survey"},
    )
    deps = build_deps(settings, llm, searcher=searcher, ui=ui)

    result = run_task("Test Paper 0", deps)

    assert result.status == "completed", result.message  # ENTER/default carries on
    assert any("does not obviously match" in message for message in ui.find("warn"))
    assert "continue with this document?" in ui.asked


def test_declining_the_wrong_document_aborts(settings) -> None:
    llm = FakeLLM(happy_responses(), texts={"report": REPORT})
    ui = SilentUI(confirmations=[False])
    searcher = FakeSearcher(
        candidates=[make_candidate(0)],
        text=PAPER_TEXT,
        info={"pdf_title": "Whales of the North Atlantic: a population survey"},
    )
    deps = build_deps(settings, llm, searcher=searcher, ui=ui)

    result = run_task("Test Paper 0", deps)

    assert result.status == "aborted"
    assert "user rejected the downloaded document" in result.message
    assert result_dirs(settings) == []


def test_partial_backend_failure_is_warned_and_reported(settings) -> None:
    """A backend that is down must be visible: warned in the UI and written to disk."""
    import json

    notes = ["arxiv: answered on attempt 2"]
    failures = ["semantic_scholar: HTTP 429"]
    searcher = FakeSearcher(
        candidates=[make_candidate(0)],
        text=PAPER_TEXT,
        backend_notes=list(notes),
        backend_failures=list(failures),
    )
    llm = FakeLLM(happy_responses(), texts={"report": REPORT})
    ui = SilentUI()
    deps = build_deps(settings, llm, searcher=searcher, ui=ui)

    result = run_task("Test Paper 0: A Study", deps)

    assert result.status == "completed", result.message
    warnings = [message for kind, message in ui.events if kind == "warn"]
    assert any("semantic_scholar" in message for message in warnings), warnings
    report_path = Path(result.state["run_dir"]) / "paper" / "search_report.json"
    assert json.loads(report_path.read_text(encoding="utf-8")) == {
        "notes": notes,
        "failures": failures,
    }
    assert result.state["search_report"] == {"notes": notes, "failures": failures}


def test_no_backend_warning_when_every_backend_answered(settings) -> None:
    """The warning must not fire on a clean search, and no report file is written."""
    llm = FakeLLM(happy_responses(), texts={"report": REPORT})
    ui = SilentUI()
    deps = build_deps(settings, llm, ui=ui)

    result = run_task("Test Paper 0: A Study", deps)

    assert result.status == "completed", result.message
    warnings = [message for kind, message in ui.events if kind == "warn"]
    assert not any("search backends" in message for message in warnings), warnings
    assert result.state["search_report"] == {"notes": [], "failures": []}
    assert not (Path(result.state["run_dir"]) / "paper" / "search_report.json").exists()


def test_pdf_fallback_link_is_shown_to_the_user(settings) -> None:
    """When the abstract page's own PDF link fails, say where the text came from."""
    fallback = "https://arxiv.org/pdf/0000.00000v2"
    searcher = FakeSearcher(
        candidates=[make_candidate(0)],
        text=PAPER_TEXT,
        info={"pdf_source_url": fallback},
    )
    llm = FakeLLM(happy_responses(), texts={"report": REPORT})
    ui = SilentUI()
    deps = build_deps(settings, llm, searcher=searcher, ui=ui)

    result = run_task("Test Paper 0: A Study", deps)

    assert result.status == "completed", result.message
    dims = [message for kind, message in ui.events if kind == "dim"]
    assert any("fallback link" in message and fallback in message for message in dims), dims


def test_fetch_node_only_returns_declared_state_keys(settings) -> None:
    """LangGraph drops undeclared keys, so a staged file can survive with a lost state key."""
    from essay_agent.nodes.fetch import make_fetch_node
    from essay_agent.state import RunState

    llm = FakeLLM(happy_responses(), texts={"report": REPORT})
    deps = build_deps(
        settings, llm, searcher=FakeSearcher(candidates=[make_candidate(0)], text=PAPER_TEXT)
    )

    partial = make_fetch_node(deps)({"query": "Test Paper 0: A Study", "run_id": "run-schema"})

    undeclared = sorted(set(partial) - set(RunState.__annotations__))
    assert undeclared == [], f"RunState does not declare these keys: {undeclared}"
