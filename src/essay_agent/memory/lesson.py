"""Lesson memory with LLM consolidation.

Two long-lived files live in ``lesson/``:

* ``lesson_plan.txt``    - written by the *card* verifier during ``replan``
* ``lesson_execute.txt`` - written by the *execution* verifier during ``re-execute``

During a run, entries are staged inside the run workspace (so Ctrl+C discards
them). At the end of a successful project, :meth:`LessonStore.commit` merges the
staged entries into the long-lived files and, when a file grows past
``memory.compress_threshold_chars``, asks the LLM to compress it - the original
is archived first so nothing is truly lost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from essay_agent.config import MemorySettings

LESSON_PLAN = "lesson_plan.txt"
LESSON_EXECUTE = "lesson_execute.txt"
_PHASE_FILES = {"plan": LESSON_PLAN, "execute": LESSON_EXECUTE}
PHASES = tuple(_PHASE_FILES)

_CONSOLIDATION_SYSTEM = (
    "You are the long-term memory maintainer for a paper-reproduction agent. "
    "You compress accumulated lessons into a dense, durable rule list. "
    "Never invent lessons. Never drop a rule that is still actionable. "
    "Merge duplicates, generalise repeated instances of the same mistake, and keep concrete "
    "field names and failure modes. Output only markdown bullets, no preamble."
)


def phase_file(phase: str) -> str:
    try:
        return _PHASE_FILES[phase]
    except KeyError:
        raise ValueError(
            f"unknown lesson phase: {phase!r} (expected 'plan' or 'execute')"
        ) from None


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ConsolidationReport:
    """Outcome of one consolidation attempt."""

    phase: str
    before_chars: int
    after_chars: int
    compressed: bool = False
    archive: Path | None = None
    error: str | None = None

    @property
    def saved_chars(self) -> int:
        return max(0, self.before_chars - self.after_chars)

    def describe(self) -> str:
        if self.error:
            return f"{self.phase}: consolidation skipped ({self.error})"
        if not self.compressed:
            return f"{self.phase}: {self.before_chars} chars, no compression needed"
        return f"{self.phase}: {self.before_chars} -> {self.after_chars} chars"


class RunLessons:
    """Staged lesson writes for one run (discarded if the run is cancelled)."""

    def __init__(self, staging_dir: Path, run_id: str) -> None:
        self.dir = Path(staging_dir)
        self.run_id = run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._written: set[str] = set()

    def path(self, phase: str) -> Path:
        return self.dir / phase_file(phase)

    def has_content(self, phase: str) -> bool:
        path = self.path(phase)
        return path.is_file() and bool(path.read_text(encoding="utf-8").strip())

    def read(self, phase: str) -> str:
        path = self.path(phase)
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def phases(self) -> list[str]:
        return [phase for phase in PHASES if self.has_content(phase)]

    def append(
        self,
        phase: str,
        title: str,
        lines: list[str],
        *,
        tags: tuple[str, ...] = (),
        meta: str = "",
    ) -> Path:
        """Append one lesson block and return the file path."""
        path = self.path(phase)
        header_bits = [f"## [{_stamp()}]", f"run={self.run_id}", f"phase={phase}"]
        if tags:
            header_bits.append("tags=" + ",".join(tags))
        header = " | ".join(header_bits)
        body_lines = [f"- {line.strip()}" for line in lines if line and line.strip()]
        if not body_lines:
            return path
        block = f"\n{header}\n**{title.strip()}**\n" + "\n".join(body_lines) + "\n"
        if meta:
            block += f"<!-- {meta} -->\n"
        needs_newline = path.is_file() and path.stat().st_size > 0
        with path.open("a", encoding="utf-8") as handle:
            if needs_newline:
                handle.write("\n")
            handle.write(block)
        self._written.add(phase)
        return path


class LessonStore:
    """Reads and maintains the long-lived lesson files."""

    def __init__(self, lesson_dir: Path, settings: MemorySettings | None = None) -> None:
        self.dir = Path(lesson_dir)
        self.settings = settings or MemorySettings()

    # ------------------------------------------------------------------ paths
    def path(self, phase: str) -> Path:
        return self.dir / phase_file(phase)

    def archive_dir(self) -> Path:
        return self.dir / "archive"

    def read(self, phase: str) -> str:
        path = self.path(phase)
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def stats(self) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for phase in PHASES:
            path = self.path(phase)
            text = self.read(phase)
            out[phase] = {
                "file": str(path),
                "exists": path.is_file(),
                "chars": len(text),
                "blocks": text.count("\n## [") + (1 if text.startswith("## [") else 0),
            }
        return out

    # --------------------------------------------------------------- reading
    def context(self, phase: str, max_chars: int | None = None) -> str:
        """Recent lesson memory, sized for prompt injection."""
        text = self.read(phase).strip()
        if not text:
            return ""
        limit = max_chars or self.settings.context_window_chars
        if len(text) > limit:
            text = "...(older lessons elided)...\n" + text[-limit:]
        return text

    # --------------------------------------------------------------- writing
    def commit(
        self,
        run_lessons: RunLessons,
        *,
        llm=None,
        consolidate: bool = True,
    ) -> list[ConsolidationReport]:
        """Merge staged lessons into long-term memory, then compress if needed."""
        reports: list[ConsolidationReport] = []
        self.dir.mkdir(parents=True, exist_ok=True)
        for phase in run_lessons.phases():
            staged = run_lessons.read(phase).strip()
            if not staged:
                continue
            target = self.path(phase)
            banner = (
                f"\n\n{'=' * 78}\n# merged from {run_lessons.run_id} at {_stamp()}\n{'=' * 78}\n"
            )
            with target.open("a", encoding="utf-8") as handle:
                handle.write(banner)
                handle.write(staged)
                handle.write("\n")
        if consolidate:
            for phase in PHASES:
                reports.append(self.consolidate(phase, llm=llm))
        return reports

    # --------------------------------------------------------- consolidation
    def needs_consolidation(self, phase: str) -> bool:
        return len(self.read(phase)) > self.settings.compress_threshold_chars

    def consolidate(self, phase: str, *, llm=None) -> ConsolidationReport:
        """Compress one lesson file with the LLM when it grows past the threshold."""
        text = self.read(phase)
        before = len(text)
        if before <= self.settings.compress_threshold_chars:
            return ConsolidationReport(phase=phase, before_chars=before, after_chars=before)
        if llm is None:
            return ConsolidationReport(
                phase=phase,
                before_chars=before,
                after_chars=before,
                error="no llm available",
            )

        target_chars = max(400, int(self.settings.compress_threshold_chars * 0.6))
        # Never send the model an unbounded prompt: keep the most recent lessons,
        # which are the ones that reflect the current code base.
        keep = max(self.settings.compress_threshold_chars * 3, target_chars * 4)
        payload = text[-keep:] if len(text) > keep else text
        user = (
            f"Lesson file: {phase_file(phase)}\n"
            f"Current size: {before} characters.\n"
            f"Compress the lessons below to at most {target_chars} characters.\n\n"
            "Requirements:\n"
            "1. Keep every rule that would prevent a future mistake; merge near-duplicates.\n"
            "2. Keep concrete names (field paths, dataset names, failure messages).\n"
            "3. Drop run ids, timestamps and one-off narrative.\n"
            "4. Output markdown bullets grouped under short bold headings.\n\n"
            f"--- BEGIN LESSONS ---\n{payload}\n--- END LESSONS ---"
        )
        try:
            compressed = (llm.text(_CONSOLIDATION_SYSTEM, user) or "").strip()
        except Exception as exc:  # pragma: no cover - depends on provider
            return ConsolidationReport(
                phase=phase,
                before_chars=before,
                after_chars=before,
                error=f"{type(exc).__name__}: {exc}",
            )
        if not compressed:
            return ConsolidationReport(
                phase=phase,
                before_chars=before,
                after_chars=before,
                error="model returned nothing",
            )

        self.archive_dir().mkdir(parents=True, exist_ok=True)
        archive = self.archive_dir() / (
            f"{self.path(phase).stem}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.txt"
        )
        archive.write_text(text, encoding="utf-8")

        header = (
            f"# {phase_file(phase)} - consolidated by the LLM at {_stamp()}\n"
            f"# previous revision archived at {archive.name} ({before} chars)\n\n"
        )
        self.path(phase).write_text(header + compressed + "\n", encoding="utf-8")
        return ConsolidationReport(
            phase=phase,
            before_chars=before,
            after_chars=len(header) + len(compressed) + 1,
            compressed=True,
            archive=archive,
        )

    # ------------------------------------------------------------------ misc
    def clear(self, phase: str | None = None) -> None:
        phases = [phase] if phase else list(PHASES)
        for item in phases:
            path = self.path(item)
            if path.is_file():
                self.archive_dir().mkdir(parents=True, exist_ok=True)
                archive = self.archive_dir() / (
                    f"{path.stem}-cleared-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.txt"
                )
                archive.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
                path.unlink()


_BULLET_RE = re.compile(r"^\s*[-*]\s+", re.MULTILINE)
