"""The default config must ship *inside* the package, or an installed wheel has none."""

from __future__ import annotations

import os
import re
import tomllib
from importlib.resources import files
from pathlib import Path

import yaml

from essay_agent.config import DEFAULTS_RESOURCE, read_default_config

# "sk-" + a long token: the shape of every key we must never commit in plain text.
_SECRET_LIKE = re.compile("sk" + r"-[A-Za-z0-9]{24,}")
_SHIPPED_GLOBS = (
    "*.md",
    "*.toml",
    ".env.example",
    "docs/**/*.md",
    "scripts/**/*.py",
    "src/**/*.py",
)


def test_the_default_config_lives_inside_the_package() -> None:
    resource = files("essay_agent").joinpath(DEFAULTS_RESOURCE)
    assert resource.is_file(), f"missing packaged defaults: {resource}"
    # It must not be the deleted repo-root copy any more.
    assert Path(str(resource)).parent.name == "defaults"


def test_the_packaged_defaults_parse_and_cover_every_section() -> None:
    text = files("essay_agent").joinpath(DEFAULTS_RESOURCE).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert set(data) >= {"llm", "search", "runtime", "memory", "paths"}
    assert read_default_config() == data


def test_pyproject_ships_that_file_in_the_wheel() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    declared = pyproject["tool"]["setuptools"]["package-data"]["essay_agent"]
    assert "defaults/*.yaml" in declared


def test_settings_can_be_built_without_any_config_file(tmp_path, monkeypatch) -> None:
    """Simulates an installed run: no repo checkout, no `.env`, defaults must still load."""
    from essay_agent.config import load_settings

    for name in list(os.environ):
        if name.startswith("ESSAY_AGENT_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)

    settings = load_settings(env_file=None)

    assert settings.llm.provider == "openai"
    assert settings.llm.model == "gpt-4o-mini"
    assert settings.runtime.time_budget_seconds == 600.0
    assert settings.search.connectivity_budget_seconds == 20.0


def test_no_live_api_key_is_committed() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders = [
        str(path.relative_to(root))
        for pattern in _SHIPPED_GLOBS
        for path in root.glob(pattern)
        if path.is_file() and _SECRET_LIKE.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert not offenders, f"looks like a committed API key in: {offenders}"


def test_env_example_is_a_usable_config_without_a_key(tmp_path, monkeypatch) -> None:
    """Copying `.env.example` to `.env` must not carry a key, and must still load."""
    from essay_agent.config import load_settings

    for name in list(os.environ):
        if name.startswith("ESSAY_AGENT_"):
            monkeypatch.delenv(name, raising=False)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)

    settings = load_settings(env_file=root / ".env.example")

    assert settings.llm.provider == "openai"
    assert settings.llm.model == "gpt-4o-mini"
    assert not settings.llm.api_key
