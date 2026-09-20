"""Per-run staging directory.

Robustness rule #1: while a task is running, *every* file it produces lives inside
the run workspace. Pressing Ctrl+C purges the whole directory, so an interrupted
task leaves nothing behind. Only a gracefully finished run is published into
``result/``, ``repro/`` and ``lesson/``.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from essay_agent.config import ResolvedPaths

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def make_slug(text: str, *, max_length: int = 60) -> str:
    """Turn a paper title into a filesystem-safe slug."""
    slug = _SAFE.sub("-", (text or "").strip()).strip("-._")
    slug = re.sub(r"-{2,}", "-", slug)
    if not slug:
        slug = "paper"
    return slug[:max_length].strip("-._") or "paper"


def new_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


def _on_rm_error(func, path, exc_info):  # pragma: no cover - Windows read-only files
    import os
    import stat

    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


@dataclass
class PublishReport:
    """What ``publish`` actually moved where."""

    result_dir: Path | None = None
    code_dir: Path | None = None
    lesson_files: list[Path] = field(default_factory=list)
    archived_lessons: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        bits = []
        if self.result_dir:
            bits.append(f"result -> {self.result_dir}")
        if self.code_dir:
            bits.append(f"code -> {self.code_dir}")
        if self.lesson_files:
            bits.append("lessons -> " + ", ".join(p.name for p in self.lesson_files))
        if self.archived_lessons:
            bits.append("archived -> " + ", ".join(p.name for p in self.archived_lessons))
        return "; ".join(bits) if bits else "nothing to publish"


class RunWorkspace:
    """Staging area for a single task."""

    SUBDIRS = ("paper", "cards", "code", "lesson", "logs", "result")

    def __init__(self, root: Path, run_id: str, slug: str = "") -> None:
        self.root = Path(root)
        self.run_id = run_id
        self.slug = slug or "paper"
        self.dir = self.root / run_id
        self._published = False
        self._purged = False

    # ------------------------------------------------------------------ setup
    @classmethod
    def create(cls, root: Path, run_id: str | None = None, slug: str = "") -> RunWorkspace:
        workspace = cls(Path(root), run_id or new_run_id(), slug)
        workspace.mkdirs()
        return workspace

    def mkdirs(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        for name in self.SUBDIRS:
            (self.dir / name).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def subdir(self, name: str) -> Path:
        if name not in self.SUBDIRS:
            raise ValueError(f"unknown workspace subdir: {name!r} (expected one of {self.SUBDIRS})")
        path = self.dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def path(self, *parts: str | Path) -> Path:
        path = self.dir.joinpath(*[Path(p) for p in parts])
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def rel(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.dir.resolve()))
        except ValueError:
            return str(path)

    def exists(self, *parts: str | Path) -> bool:
        return self.dir.joinpath(*[Path(p) for p in parts]).exists()

    # ------------------------------------------------------------------- i/o
    def write_text(self, rel_path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
        path = self.path(rel_path)
        path.write_text(text, encoding=encoding)
        return path

    def append_text(self, rel_path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
        path = self.path(rel_path)
        with path.open("a", encoding=encoding) as handle:
            handle.write(text)
        return path

    def write_json(self, rel_path: str | Path, obj: Any, *, indent: int = 2) -> Path:
        payload = json.dumps(obj, ensure_ascii=False, indent=indent, default=str)
        return self.write_text(rel_path, payload)

    def read_text(self, rel_path: str | Path) -> str:
        return self.path(rel_path).read_text(encoding="utf-8")

    def read_json(self, rel_path: str | Path) -> Any:
        return json.loads(self.read_text(rel_path))

    def list_files(self, rel_path: str | Path = ".", pattern: str = "*") -> list[Path]:
        base = self.dir / rel_path
        if not base.exists():
            return []
        return sorted(p for p in base.glob(pattern) if p.is_file())

    def copy_into(self, source: Path | str, rel_path: str | Path) -> Path:
        source = Path(source)
        target = self.path(rel_path) / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
        return target

    # ------------------------------------------------------------------- logs
    def log_path(self, name: str = "run.log") -> Path:
        return self.subdir("logs") / name

    def log(self, message: str) -> None:
        stamp = datetime.now(UTC).strftime("%H:%M:%S")
        with self.log_path().open("a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] {message}\n")

    def log_llm_exchange(self, label: str, system: str, user: str, reply: str) -> None:
        """Append one prompt/response pair to a per-run transcript (debugging aid)."""
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        block = (
            f"\n{'=' * 78}\n[{stamp}] {label}\n{'-' * 30} SYSTEM {'-' * 30}\n{system}\n"
            f"{'-' * 30} USER {'-' * 32}\n{user}\n{'-' * 30} REPLY {'-' * 31}\n{reply}\n"
        )
        self.append_text("logs/llm_transcript.md", block)

    # --------------------------------------------------------------- lifecycle
    def purge(self) -> bool:
        """Delete the staging directory. Refuses to touch anything outside ``root``."""
        if self._purged:
            return True
        target = self.dir.resolve()
        root = self.root.resolve()
        if root not in target.parents:
            raise RuntimeError(f"refusing to purge {target}: not inside workspace root {root}")
        for attempt in range(3):
            if not target.exists():
                break
            shutil.rmtree(target, ignore_errors=True, onerror=_on_rm_error)
            if not target.exists():
                break
            time.sleep(0.2 * (attempt + 1))
        self._purged = not target.exists()
        return self._purged

    @property
    def published(self) -> bool:
        return self._published

    @property
    def purged(self) -> bool:
        return self._purged

    def stage_result_dir(self) -> Path:
        return self.subdir("result")

    def publish(self, paths: ResolvedPaths, title: str = "") -> PublishReport:
        """Copy the finished run into the long-lived ``result/`` and ``repro/`` trees.

        Lessons are committed separately (see ``memory.lesson.LessonStore.commit``)
        because they need LLM-driven consolidation.
        """
        report = PublishReport()
        slug = make_slug(title or self.slug)
        short_id = self.run_id.rsplit("-", 1)[-1]
        target_name = f"{slug}-{short_id}"

        result_src = self.dir / "result"
        if result_src.exists() and any(result_src.iterdir()):
            report.result_dir = _copy_tree(result_src, Path(paths.result_dir) / target_name)

        code_src = self.dir / "code"
        if code_src.exists() and any(code_src.iterdir()):
            report.code_dir = _copy_tree(code_src, Path(paths.code_dir) / target_name)

        self._published = True
        report.notes.append(f"run dir kept at {self.dir}")
        return report


def _copy_tree(source: Path, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        destination = target / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)
    return target
