"""Lesson memory: staging, merging and LLM consolidation."""

from __future__ import annotations

from pathlib import Path

from essay_agent.config import MemorySettings
from essay_agent.memory.lesson import (
    LESSON_EXECUTE,
    LESSON_PLAN,
    LessonStore,
    RunLessons,
    phase_file,
)
from tests.fakes import FakeLLM


def test_phase_file_names() -> None:
    assert phase_file("plan") == LESSON_PLAN
    assert phase_file("execute") == LESSON_EXECUTE
    try:
        phase_file("nope")
    except ValueError as exc:
        assert "unknown lesson phase" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_run_lessons_stage_entries(tmp_path: Path) -> None:
    lessons = RunLessons(tmp_path / "lesson", "run-1")
    assert lessons.has_content("plan") is False
    lessons.append("plan", "card verifier found 2 issues", ["a problem", "another problem"])
    assert lessons.has_content("plan") is True
    text = lessons.read("plan")
    assert "run-1" in text
    assert "phase=plan" in text
    assert "- a problem" in text
    assert lessons.has_content("execute") is False


def test_run_lessons_ignores_empty_entries(tmp_path: Path) -> None:
    lessons = RunLessons(tmp_path / "lesson", "run-1")
    lessons.append("plan", "nothing", ["", "   "])
    assert lessons.has_content("plan") is False


def test_commit_merges_staged_entries_into_long_term_memory(tmp_path: Path) -> None:
    store = LessonStore(tmp_path / "lesson", MemorySettings(compress_threshold_chars=100_000))
    staged = RunLessons(tmp_path / "runs" / "run-1" / "lesson", "run-1")
    staged.append("plan", "verifier round 1", ["prefer explicit dataset names"])
    reports = store.commit(staged, llm=None)
    assert reports  # one report per phase
    long_term = store.read("plan")
    assert "merged from run-1" in long_term
    assert "prefer explicit dataset names" in long_term
    assert store.read("execute") == ""


def test_commit_is_skipped_when_nothing_was_staged(tmp_path: Path) -> None:
    store = LessonStore(tmp_path / "lesson", MemorySettings(compress_threshold_chars=100_000))
    staged = RunLessons(tmp_path / "runs" / "run-2" / "lesson", "run-2")
    store.commit(staged, llm=None)
    assert not store.path("plan").exists()


def test_consolidation_is_a_no_op_below_the_threshold(tmp_path: Path) -> None:
    store = LessonStore(tmp_path / "lesson", MemorySettings(compress_threshold_chars=10_000))
    store.path("plan").parent.mkdir(parents=True, exist_ok=True)
    store.path("plan").write_text("- small\n", encoding="utf-8")
    report = store.consolidate("plan", llm=FakeLLM())
    assert report.compressed is False
    assert report.archive is None
    assert "no compression needed" in report.describe()


def test_consolidation_compresses_and_archives(tmp_path: Path) -> None:
    settings = MemorySettings(compress_threshold_chars=400, context_window_chars=100)
    store = LessonStore(tmp_path / "lesson", settings)
    store.path("plan").parent.mkdir(parents=True, exist_ok=True)
    original = "\n".join(
        f"- lesson {index}: a long and winding failure mode" for index in range(60)
    )
    store.path("plan").write_text(original, encoding="utf-8")
    llm = FakeLLM(texts={"default": "- merged rule A\n- merged rule B"})
    report = store.consolidate("plan", llm=llm)
    assert report.compressed is True
    assert report.after_chars < report.before_chars
    assert report.archive is not None and report.archive.is_file()
    assert report.archive.read_text(encoding="utf-8") == original
    assert "consolidated by the LLM" in store.read("plan")
    assert "merged rule A" in store.read("plan")


def test_consolidation_without_an_llm_is_reported(tmp_path: Path) -> None:
    store = LessonStore(tmp_path / "lesson", MemorySettings(compress_threshold_chars=100))
    store.path("plan").parent.mkdir(parents=True, exist_ok=True)
    store.path("plan").write_text("x" * 500, encoding="utf-8")
    report = store.consolidate("plan", llm=None)
    assert report.compressed is False
    assert report.error == "no llm available"


def test_consolidation_survives_an_llm_failure(tmp_path: Path) -> None:
    class Boom(FakeLLM):
        def text(self, *args, **kwargs):
            raise RuntimeError("provider down")

    store = LessonStore(tmp_path / "lesson", MemorySettings(compress_threshold_chars=100))
    store.path("plan").parent.mkdir(parents=True, exist_ok=True)
    store.path("plan").write_text("x" * 500, encoding="utf-8")
    report = store.consolidate("plan", llm=Boom())
    assert report.compressed is False
    assert "provider down" in (report.error or "")
    assert len(store.read("plan")) == 500


def test_context_is_truncated_from_the_front(tmp_path: Path) -> None:
    store = LessonStore(
        tmp_path / "lesson",
        MemorySettings(compress_threshold_chars=10_000, context_window_chars=50),
    )
    store.path("plan").parent.mkdir(parents=True, exist_ok=True)
    store.path("plan").write_text("A" * 100 + "TAIL", encoding="utf-8")
    context = store.context("plan")
    assert context.endswith("TAIL")
    assert context.startswith("...(older lessons elided)...")
    assert len(context) < 120


def test_stats_and_clear(tmp_path: Path) -> None:
    store = LessonStore(tmp_path / "lesson", MemorySettings(compress_threshold_chars=100_000))
    store.path("plan").parent.mkdir(parents=True, exist_ok=True)
    store.path("plan").write_text("## [now] run=x | phase=plan\n**t**\n- a\n", encoding="utf-8")
    stats = store.stats()
    assert stats["plan"]["chars"] > 0
    assert stats["plan"]["blocks"] == 1
    store.clear("plan")
    assert store.read("plan") == ""
    assert list(store.archive_dir().glob("*.txt"))
