"""Staging area behaviour, including the Ctrl+C guarantee."""

from __future__ import annotations

from pathlib import Path

import pytest

from essay_agent.config import load_settings
from essay_agent.workspace import RunWorkspace, make_slug, new_run_id


def _paths(tmp_path: Path):
    return load_settings(
        env_file=None, overrides={"paths": {"home": str(tmp_path)}}
    ).resolved_paths()


def test_create_makes_every_subdir(tmp_path: Path) -> None:
    workspace = RunWorkspace.create(tmp_path / "runs", "run-1")
    assert workspace.dir.is_dir()
    for name in RunWorkspace.SUBDIRS:
        assert (workspace.dir / name).is_dir()


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    workspace = RunWorkspace.create(tmp_path / "runs", "run-1")
    workspace.write_text("cards/cards.md", "hello")
    workspace.write_json("cards/cards.json", {"a": 1})
    assert workspace.read_text("cards/cards.md") == "hello"
    assert workspace.read_json("cards/cards.json") == {"a": 1}
    assert workspace.rel(workspace.dir / "cards" / "cards.md") == str(Path("cards") / "cards.md")


def test_purge_removes_everything(tmp_path: Path) -> None:
    workspace = RunWorkspace.create(tmp_path / "runs", "run-1")
    workspace.write_text("result/report.md", "draft")
    assert workspace.purge() is True
    assert not workspace.dir.exists()


def test_purge_refuses_to_leave_the_workspace_root(tmp_path: Path) -> None:
    workspace = RunWorkspace.create(tmp_path / "runs", "run-1")
    workspace.dir = tmp_path  # simulate a corrupted path
    with pytest.raises(RuntimeError):
        workspace.purge()
    assert tmp_path.exists()


def test_publish_copies_results_and_code(tmp_path: Path) -> None:
    paths = _paths(tmp_path).ensure()
    workspace = RunWorkspace.create(paths.workspace_root, "run-20240101-000000-abcd12")
    workspace.write_text("result/report.md", "# report")
    workspace.write_text("result/figures/fig1.png", "png")
    workspace.write_text("code/repro.py", "print('hi')")
    report = workspace.publish(paths, title="A Study of Things")
    assert report.result_dir is not None and (report.result_dir / "report.md").is_file()
    assert (report.result_dir / "figures" / "fig1.png").is_file()
    assert report.code_dir is not None and (report.code_dir / "repro.py").is_file()
    assert "A-Study-of-Things" in report.result_dir.name
    assert workspace.published is True


def test_publish_ignores_empty_result_tree(tmp_path: Path) -> None:
    paths = _paths(tmp_path).ensure()
    workspace = RunWorkspace.create(paths.workspace_root, "run-2")
    report = workspace.publish(paths, title="Nothing")
    assert report.result_dir is None


def test_make_slug_is_filesystem_safe() -> None:
    assert make_slug("Attention Is All You Need!") == "Attention-Is-All-You-Need"
    assert make_slug("") == "paper"
    assert make_slug("///") == "paper"
    assert len(make_slug("x" * 200)) <= 60


def test_new_run_id_is_unique() -> None:
    assert new_run_id() != new_run_id()
