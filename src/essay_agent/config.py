"""Configuration layer (framework stage 0).

Precedence, lowest to highest:
    1. the classes below (built-in defaults)
    2. ``essay_agent/defaults/default.yaml`` shipped *inside* the package
    3. a user config file (``--config`` or ``./essay-agent.yaml``)
    4. environment variables and ``.env``  (``ESSAY_AGENT_LLM__MODEL=...``)
"""

from __future__ import annotations

import os
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from essay_agent.errors import ConfigError

Provider = Literal["openai", "openai_compatible", "deepseek", "anthropic"]

# Where dataset downloads fall back to when the primary host is unreachable.
# An empty ``search.dataset_mirror_url`` disables the fallback.
DEFAULT_DATASET_MIRROR_URL = "https://hf-mirror.com"

DEFAULTS_RESOURCE = "defaults/default.yaml"
_USER_CONFIG_CANDIDATES = ("essay-agent.yaml", "essay-agent.yml", "config.yaml")
_ENV_KEY_FALLBACKS = {
    "openai": ("OPENAI_API_KEY",),
    "openai_compatible": ("OPENAI_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
}


class LLMSettings(BaseModel):
    """Which model the agent talks to, and how."""

    provider: Provider = "openai"
    model: str = "gpt-4o-mini"
    verifier_model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.2
    verifier_temperature: float = 0.0
    max_tokens: int = 8192
    code_max_tokens: int | None = None
    timeout: float = 120.0
    max_retries: int = 3

    @property
    def effective_verifier_model(self) -> str:
        return self.verifier_model or self.model

    def resolve_api_key(self) -> str:
        """Return the API key, falling back to well-known provider env vars."""
        if self.api_key:
            return self.api_key
        for name in _ENV_KEY_FALLBACKS.get(self.provider, ()):
            value = os.environ.get(name)
            if value:
                return value
        if self.provider == "openai_compatible" and self.base_url:
            # Local servers (Ollama, vLLM, ...) usually accept any token.
            return "not-needed"
        wanted = " or ".join(_ENV_KEY_FALLBACKS.get(self.provider, ("ESSAY_AGENT_LLM__API_KEY",)))
        raise ConfigError(
            f"No API key for provider {self.provider!r}. Set ESSAY_AGENT_LLM__API_KEY "
            f"(or {wanted}) in your environment or .env file."
        )


class SearchSettings(BaseModel):
    """Paper retrieval knobs."""

    max_results: int = Field(default=8, ge=1, le=50)
    timeout: float = Field(default=30.0, gt=0)
    backends: list[str] = Field(default_factory=lambda: ["arxiv", "semanticscholar", "openalex"])
    user_choice_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    contact_email: str | None = None
    require_full_text: bool = True
    user_agent: str = "essay-agent/0.1 (paper reproduction research tool)"
    cache_pdfs: bool = True
    # Dataset downloads fall back to this mirror when the primary host is unreachable.
    # Set it to an empty string to disable the fallback (or to your own mirror).
    dataset_mirror_url: str = DEFAULT_DATASET_MIRROR_URL
    # How long the stage-3 reachability probe may take in total. Raise it on a slow or
    # partially blocked network to get a complete verdict instead of a partial one.
    connectivity_budget_seconds: float = Field(default=20.0, gt=0)


class RuntimeSettings(BaseModel):
    """Execution budget, verification loops and the safety policy."""

    time_budget_seconds: float = Field(default=600.0, gt=0)
    preflight_timeout: float = Field(default=300.0, gt=0)
    hard_timeout_factor: float = Field(default=1.5, ge=1.0)
    max_plan_rounds: int = Field(default=3, ge=1)
    max_execute_rounds: int = Field(default=3, ge=1)
    max_adjust_rounds: int = Field(default=3, ge=0)
    auto_adjust: bool = True
    allow_synthetic_data: bool = Field(
        default=False,
        description="Hard policy switch. False forbids the agent from fabricating data to make a run pass.",
    )
    device: Literal["auto", "cpu", "cuda"] = "auto"
    python_executable: str | None = None
    progress_refresh_per_second: float = Field(default=8.0, gt=0)

    @property
    def hard_timeout_seconds(self) -> float:
        return self.time_budget_seconds * self.hard_timeout_factor


class MemorySettings(BaseModel):
    """Lesson-folder behaviour."""

    compress_threshold_chars: int = Field(default=4000, ge=50)
    context_window_chars: int = Field(default=2000, ge=50)


class PathSettings(BaseModel):
    """Where the agent keeps its state. Relative paths are resolved against ``home``."""

    home: Path = Path(".")
    workspace_root: Path = Path(".essay_agent/runs")
    cache_dir: Path = Path(".essay_agent/cache")
    history_file: Path = Path(".essay_agent/history")
    result_dir: Path = Path("result")
    code_dir: Path = Path("repro")
    lesson_dir: Path = Path("lesson")


class ResolvedPaths(BaseModel):
    """Absolute paths, ready to use."""

    home: Path
    workspace_root: Path
    cache_dir: Path
    history_file: Path
    result_dir: Path
    code_dir: Path
    lesson_dir: Path

    def ensure(self) -> ResolvedPaths:
        for path in (
            self.workspace_root,
            self.cache_dir,
            self.result_dir,
            self.code_dir,
            self.lesson_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        return self


class Settings(BaseSettings):
    """Root settings object."""

    model_config = SettingsConfigDict(
        env_prefix="ESSAY_AGENT_",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Precedence: environment/.env  >  YAML passed as init kwargs  >  secrets.

        Pydantic's default puts ``init`` first, which would make the YAML file
        beat the environment - the opposite of what this project documents.
        """
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)

    llm: LLMSettings = Field(default_factory=LLMSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    # Explicit proxy for every HTTP client the agent builds (paper search, dataset
    # probe, the model API). Wins over the environment and over OS detection, so a
    # machine whose proxy we cannot detect only needs this one setting.
    http_proxy: str | None = None
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    verbose: bool = False
    config_file: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.llm.provider == "openai_compatible" and not self.llm.base_url:
            raise ConfigError("llm.provider=openai_compatible requires llm.base_url")
        return self

    def resolved_paths(self, base: Path | None = None) -> ResolvedPaths:
        home = self.paths.home
        if not home.is_absolute():
            home = ((base or Path.cwd()) / home).resolve()

        def _abs(value: Path) -> Path:
            return value if value.is_absolute() else (home / value).resolve()

        return ResolvedPaths(
            home=home,
            workspace_root=_abs(self.paths.workspace_root),
            cache_dir=_abs(self.paths.cache_dir),
            history_file=_abs(self.paths.history_file),
            result_dir=_abs(self.paths.result_dir),
            code_dir=_abs(self.paths.code_dir),
            lesson_dir=_abs(self.paths.lesson_dir),
        )

    def python_executable(self) -> str:
        import sys

        return self.runtime.python_executable or sys.executable

    def summary(self) -> dict[str, Any]:
        """Config dump for ``essay config show`` - the API key is redacted."""
        data = self.model_dump(mode="json")
        key = data.get("llm", {}).get("api_key")
        if key:
            data["llm"]["api_key"] = f"***{str(key)[-4:]}" if len(str(key)) > 4 else "***"
        data["paths"] = {k: str(v) for k, v in self.resolved_paths().model_dump().items()}
        return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return _parse_yaml(path.read_text(encoding="utf-8"), name=str(path))


def _parse_yaml(text: str, *, name: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse config file {name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config file {name} must contain a mapping at the top level")
    return data


def read_default_config() -> dict[str, Any]:
    """The packaged defaults.

    Read through ``importlib.resources`` so it works from an installed wheel and
    from the source tree alike - the file lives *inside* the package, and
    ``pyproject.toml`` ships it via ``[tool.setuptools.package-data]``.
    """
    resource = resource_files("essay_agent").joinpath(DEFAULTS_RESOURCE)
    try:
        text = resource.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):  # pragma: no cover - broken installation
        return {}
    return _parse_yaml(text, name=f"essay_agent/{DEFAULTS_RESOURCE}")


def discover_user_config(start: Path | None = None) -> Path | None:
    base = start or Path.cwd()
    for name in _USER_CONFIG_CANDIDATES:
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def load_settings(
    config_file: Path | str | None = None,
    *,
    env_file: Path | str | None = ".env",
    overrides: dict[str, Any] | None = None,
    use_default_file: bool = True,
) -> Settings:
    """Build a :class:`Settings` object from files, environment and explicit overrides."""
    data: dict[str, Any] = {}
    if use_default_file:
        data = _deep_merge(data, read_default_config())

    explicit = Path(config_file) if config_file else discover_user_config()
    if explicit is not None:
        if not explicit.is_file():
            raise ConfigError(f"config file not found: {explicit}")
        data = _deep_merge(data, _read_yaml(explicit))

    if overrides:
        data = _deep_merge(data, overrides)

    env_path: str | None
    if env_file is None:
        env_path = None
    else:
        env_path = str(Path(env_file)) if Path(env_file).is_file() else None

    settings = Settings(_env_file=env_path, **data)
    if explicit is not None:
        settings.config_file = explicit
    return settings
