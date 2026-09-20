"""Shared fixtures. The full pipeline is exercised offline with fakes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from essay_agent.config import Settings, load_settings


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """A hermetic Settings object: everything lives under ``tmp_path``."""
    base: dict[str, Any] = {
        "llm": {"provider": "openai", "model": "test-model", "api_key": "test-key"},
        "paths": {
            "home": str(tmp_path),
            "workspace_root": ".ea/runs",
            "cache_dir": ".ea/cache",
            "history_file": ".ea/history",
            "result_dir": "result",
            "code_dir": "repro",
            "lesson_dir": "lesson",
        },
        "runtime": {
            "time_budget_seconds": 30.0,
            "preflight_timeout": 20.0,
            "hard_timeout_factor": 2.0,
            "max_plan_rounds": 3,
            "max_execute_rounds": 3,
            "max_adjust_rounds": 2,
        },
        "memory": {"compress_threshold_chars": 400, "context_window_chars": 200},
        "search": {"max_results": 4, "user_choice_confidence": 0.6},
    }
    return load_settings(
        config_file=None,
        env_file=None,
        overrides=_deep_merge(base, overrides),
    )


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)
