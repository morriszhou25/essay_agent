"""Configuration precedence and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from essay_agent.config import LLMSettings, Settings, load_settings
from essay_agent.errors import ConfigError


def test_shipped_defaults_are_usable() -> None:
    settings = load_settings(env_file=None)
    assert settings.llm.provider == "openai"
    assert settings.runtime.time_budget_seconds == 600.0
    assert settings.runtime.max_plan_rounds == 3
    assert settings.runtime.allow_synthetic_data is False
    assert settings.search.backends == ["arxiv", "semanticscholar", "openalex"]


def test_yaml_file_overrides_defaults(tmp_path: Path) -> None:
    config = tmp_path / "essay-agent.yaml"
    config.write_text(
        "llm:\n  model: custom-model\nruntime:\n  time_budget_seconds: 42\n", encoding="utf-8"
    )
    settings = load_settings(config, env_file=None)
    assert settings.llm.model == "custom-model"
    assert settings.runtime.time_budget_seconds == 42.0
    assert settings.llm.provider == "openai"  # untouched defaults survive


def test_environment_beats_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESSAY_AGENT_LLM__MODEL", "env-model")
    monkeypatch.setenv("ESSAY_AGENT_RUNTIME__TIME_BUDGET_SECONDS", "99")
    settings = load_settings(env_file=None)
    assert settings.llm.model == "env-model"
    assert settings.runtime.time_budget_seconds == 99.0


def test_missing_config_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "nope.yaml", env_file=None)


def test_resolved_paths_live_under_home(tmp_path: Path) -> None:
    settings = load_settings(env_file=None, overrides={"paths": {"home": str(tmp_path)}})
    paths = settings.resolved_paths().ensure()
    assert paths.result_dir == tmp_path / "result"
    assert paths.workspace_root == tmp_path / ".essay_agent" / "runs"
    assert paths.lesson_dir.is_dir()


def test_api_key_falls_back_to_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert LLMSettings(provider="openai").resolve_api_key() == "from-env"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ConfigError):
        LLMSettings(provider="openai", api_key=None).resolve_api_key()


def test_openai_compatible_needs_base_url() -> None:
    with pytest.raises(ConfigError):
        Settings(llm={"provider": "openai_compatible"})


def test_summary_redacts_the_api_key() -> None:
    settings = Settings(llm={"api_key": "sk-super-secret-value"})
    assert "sk-super-secret-value" not in str(settings.summary())
