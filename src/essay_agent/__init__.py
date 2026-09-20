"""essay-agent - a lightweight, human-in-the-loop paper reproduction agent."""

from __future__ import annotations

__all__ = ["__version__", "run_task"]

__version__ = "0.1.0"


def __getattr__(name: str):
    # Keep import-time side effects (settings parsing, LLM clients) out of
    # `import essay_agent`; `run_task` is loaded lazily for convenience.
    if name == "run_task":
        from essay_agent.agent import run_task

        return run_task
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
