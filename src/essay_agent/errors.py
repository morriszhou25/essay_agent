"""Exception hierarchy for essay-agent."""

from __future__ import annotations


class EssayAgentError(Exception):
    """Base class for every error raised by essay-agent."""


class ConfigError(EssayAgentError):
    """Invalid or missing configuration."""


class LLMError(EssayAgentError):
    """The language model could not be reached or returned unusable output."""


class LLMReplyError(LLMError):
    """The model answered, but the reply could not be used.

    Carries the call identity and the failure classification so a failure can be
    located from the message alone, without a transcript:

    ``reason`` is one of ``empty``, ``truncated``, ``invalid_json`` or
    ``schema_mismatch``; ``finish_reason`` is whatever the provider reported
    (``length`` means the reply was cut off by ``max_tokens``).
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str = "invalid_json",
        schema: str = "",
        role: str = "",
        label: str = "",
        reply_chars: int = 0,
        finish_reason: str | None = None,
        preview: str = "",
        note: str = "",
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.schema = schema
        self.role = role
        self.label = label
        self.reply_chars = reply_chars
        self.finish_reason = finish_reason
        self.preview = preview
        self.note = note

    def where(self) -> str:
        bits = [f"role={self.role or '?'}", f"label={self.label or '?'}"]
        if self.schema:
            bits.append(f"schema={self.schema}")
        bits.append(f"reason={self.reason}")
        bits.append(f"reply_chars={self.reply_chars}")
        if self.finish_reason:
            bits.append(f"finish_reason={self.finish_reason}")
        if self.note:
            bits.append(self.note)
        return " ".join(bits)

    def __str__(self) -> str:
        return f"{super().__str__()} [{self.where()}]"


class ToolNotUsedError(LLMError):
    """The model refused to call a tool it was required to call."""


class PaperSearchError(EssayAgentError):
    """Every configured search backend failed (network / API error)."""


class PaperNotFoundError(EssayAgentError):
    """No candidate paper matched the user's request."""


class PaperUnavailableError(EssayAgentError):
    """A paper was matched but its full text could not be retrieved."""


class TaskAborted(EssayAgentError):
    """The task stopped early in a controlled, user-visible way."""


class TaskCancelled(EssayAgentError):
    """The user pressed Ctrl+C; every artifact of the current run must be purged."""


class ExecutionError(EssayAgentError):
    """Script execution failed or exceeded its budget."""


class BudgetExceeded(ExecutionError):
    """The projected runtime could not be squeezed into the time budget."""
