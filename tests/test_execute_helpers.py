"""Static checks applied to generated reproduction code."""

from __future__ import annotations

from pathlib import Path

from essay_agent.nodes.execute import check_code
from essay_agent.runtime.runner import RunOutcome
from essay_agent.workspace import RunWorkspace


def test_valid_script_passes() -> None:
    code = (
        "import argparse\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--preflight', action='store_true')\n"
        "    return 0\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n"
    )
    assert check_code(code) is None


def test_markdown_fences_are_rejected() -> None:
    assert "markdown fences" in (check_code("```python\nimport os\n```" + "x" * 200) or "")


def test_placeholders_are_rejected() -> None:
    code = "import os\n# TODO: implement the rest\n" + "x = 1\n" * 40
    assert check_code(code) is not None


def test_missing_preflight_flag_is_rejected() -> None:
    code = "import os\n" + "x = 1\n" * 40
    assert "preflight" in (check_code(code) or "")


def test_syntax_errors_are_reported() -> None:
    code = "import os\ndef broken(:\n    pass\n" + "x = 1\n" * 40
    assert "SyntaxError" in (check_code(code) or "")


def test_harvest_copies_metrics_and_figures_into_result(tmp_path: Path) -> None:
    from essay_agent.nodes.execute import _harvest

    workspace = RunWorkspace.create(tmp_path / "runs", "run-1")
    workspace.write_text("code/metrics.json", '{"accuracy": 1}')
    workspace.write_text("code/figures/plot.png", "png-bytes")
    figures = _harvest(workspace, RunOutcome(ok=True, metrics={"accuracy": 1.0}))
    assert figures == ["plot.png"]
    assert workspace.exists("result/figures/plot.png")
    assert workspace.exists("result/metrics.json")
