"""LLM access layer.

Two roles share one client:

``main``
    plans, writes cards, writes code, writes the report.
``verifier``
    reviews the cards and the execution results. It uses a colder temperature
    and, optionally, a different (usually stronger) model.

Every call can be mirrored into the run transcript through ``transcript``.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

from essay_agent.config import LLMSettings
from essay_agent.connectivity import RouteProvider, build_http_client
from essay_agent.errors import ConfigError, LLMError, LLMReplyError, ToolNotUsedError

SchemaT = TypeVar("SchemaT", bound=BaseModel)
Transcript = Callable[[str, str, str, str], None]

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_BRACKETS = {"{": "}", "[": "]"}


@runtime_checkable
class LLM(Protocol):
    """What the nodes depend on. Tests provide their own implementation."""

    def text(
        self,
        system: str,
        user: str,
        *,
        role: str = "main",
        label: str | None = None,
        max_tokens: int | None = None,
    ) -> str: ...

    def json(
        self,
        system: str,
        user: str,
        schema: type[SchemaT],
        *,
        role: str = "main",
        label: str | None = None,
        max_tokens: int | None = None,
    ) -> SchemaT: ...

    def tool_args(
        self,
        system: str,
        user: str,
        tool: Any,
        schema: type[SchemaT],
        *,
        role: str = "main",
        label: str | None = None,
    ) -> SchemaT: ...


def reply_finish_reason(reply: Any) -> str | None:
    """The provider's stop reason, when it reports one."""
    metadata = getattr(reply, "response_metadata", None)
    if not isinstance(metadata, dict):
        return None
    for key in ("finish_reason", "stop_reason", "finish_message"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def preview_reply(text: str, head: int = 200, tail: int = 200) -> str:
    """A head+tail preview: for a cut-off reply the tail is the useful part."""
    text = text or ""
    if len(text) <= head + tail:
        return text
    elided = len(text) - head - tail
    return f"{text[:head]}\n...[ {elided} of {len(text)} chars elided ]...\n{text[-tail:]}"


def looks_cut_off(text: str) -> bool:
    """True when the reply stops inside a string or with brackets still open."""
    in_string = False
    escaped = False
    stack: list[str] = []
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in _BRACKETS:
            stack.append(char)
        elif char in "}]" and (not stack or _BRACKETS[stack.pop()] != char):
            return False  # mismatched pairs: malformed, not merely truncated
    return in_string or bool(stack)


def classify_reply(text: str) -> str:
    """Why a reply is not usable JSON: ``empty`` / ``truncated`` / ``invalid_json``."""
    if not text.strip():
        return "empty"
    return "truncated" if looks_cut_off(text.strip()) else "invalid_json"


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from a model reply."""
    if not text.strip():
        raise LLMReplyError(
            "empty reply where JSON was expected",
            reason="empty",
            reply_chars=len(text),
        )
    fenced = _JSON_FENCE.search(text)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass
        depth = 0
        start = None
        for position, char in enumerate(candidate):
            if char in "{[":
                if depth == 0:
                    start = position
                depth += 1
            elif char in "}]":
                depth -= 1
                if depth == 0 and start is not None:
                    chunk = candidate[start : position + 1]
                    try:
                        return json.loads(chunk)
                    except (json.JSONDecodeError, ValueError):
                        start = None
    reason = classify_reply(text)
    preview = preview_reply(text)
    raise LLMReplyError(
        f"could not parse JSON from reply ({reason}): {preview!r}",
        reason=reason,
        reply_chars=len(text),
        preview=preview,
    )


def reply_to_text(reply: Any) -> str:
    """Normalise a LangChain message into plain text."""
    content = getattr(reply, "content", reply)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content)


class LLMClient:
    """LangChain-backed client with retries and a structured-output fallback."""

    def __init__(
        self,
        settings: LLMSettings,
        *,
        transcript: Transcript | None = None,
        verbose: bool = False,
        route: RouteProvider | None = None,
    ) -> None:
        self.settings = settings
        self.transcript = transcript
        self.verbose = verbose
        self.route = route
        self._models: dict[tuple[str, int | None], Any] = {}
        self.calls = 0
        self.retries = 0

    # ------------------------------------------------------------- model setup
    def model_name(self, role: str = "main") -> str:
        if role == "verifier":
            return self.settings.effective_verifier_model
        return self.settings.model

    def _temperature(self, role: str) -> float:
        return (
            self.settings.verifier_temperature if role == "verifier" else self.settings.temperature
        )

    def _model(self, role: str, max_tokens: int | None = None) -> Any:
        """The chat model for ``role``.

        ``max_tokens`` overrides the configured limit for this call only (the code
        call needs more room than the structured calls). Models are cached per
        (role, limit) pair, so a run still builds at most a handful of clients.
        """
        key = (role, max_tokens)
        if key not in self._models:
            self._models[key] = self._build_model(role, max_tokens)
        return self._models[key]

    def _build_model(self, role: str, max_tokens: int | None = None) -> Any:
        settings = self.settings
        model_name = self.model_name(role)
        temperature = self._temperature(role)
        limit = max_tokens or settings.max_tokens
        common: dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "timeout": settings.timeout,
            "max_retries": 0,  # we own the retry policy
        }
        # Pin the provider SDK to the route the probe verified: langchain's own client
        # would otherwise follow a dead env proxy (it even disables httpx's own
        # auto-detection when it injects its socket options, which is why it logs a
        # one-off "injected a custom httpx transport" warning for users who have a
        # proxy configured - expected, and exactly what this argument answers).
        if self.route is not None and (client := build_http_client(self.route())) is not None:
            common["http_client"] = client
        provider = settings.provider
        try:
            if provider in {"openai", "openai_compatible"}:
                from langchain_openai import ChatOpenAI

                kwargs = dict(common)
                kwargs["api_key"] = settings.resolve_api_key()
                if settings.base_url:
                    kwargs["base_url"] = settings.base_url
                if limit:
                    kwargs["max_tokens"] = limit
                return ChatOpenAI(**kwargs)
            if provider == "deepseek":
                from langchain_deepseek import ChatDeepSeek

                kwargs = dict(common)
                kwargs["api_key"] = settings.resolve_api_key()
                if limit:
                    kwargs["max_tokens"] = limit
                if settings.base_url:
                    kwargs["api_base"] = settings.base_url
                return ChatDeepSeek(**kwargs)
            if provider == "anthropic":
                from langchain_anthropic import ChatAnthropic

                kwargs = dict(common)
                kwargs["api_key"] = settings.resolve_api_key()
                if limit:
                    kwargs["max_tokens"] = limit
                return ChatAnthropic(**kwargs)
        except ImportError as exc:
            raise ConfigError(
                f"provider {provider!r} needs an extra package that is not installed: {exc}. "
                "Install it with: pip install 'essay-agent[deepseek]' or 'essay-agent[anthropic]'"
            ) from exc
        raise ConfigError(f"unknown llm provider: {provider!r}")

    # ------------------------------------------------------------------ public
    def text(
        self,
        system: str,
        user: str,
        *,
        role: str = "main",
        label: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = self._messages(system, user)
        reply = self._invoke(
            self._model(role, max_tokens), messages, role=role, label=label or "text"
        )
        return reply_to_text(reply)

    def json(
        self,
        system: str,
        user: str,
        schema: type[SchemaT],
        *,
        role: str = "main",
        label: str | None = None,
        max_tokens: int | None = None,
    ) -> SchemaT:
        model = self._model(role, max_tokens)
        messages = self._messages(system, user)
        tag = label or schema.__name__
        structured_error: Exception | None = None
        if hasattr(model, "with_structured_output"):
            try:
                structured = model.with_structured_output(schema)
                reply = self._invoke(structured, messages, role=role, label=tag)
                if isinstance(reply, schema):
                    return reply
                if isinstance(reply, dict):
                    return schema.model_validate(reply)
                if isinstance(reply, BaseModel):
                    return schema.model_validate(reply.model_dump())
            except Exception as exc:
                # Either the provider has no structured output, or the structured
                # call itself broke (truncated tool arguments, ...). Keep the
                # reason instead of discarding it: it is often the real diagnosis.
                structured_error = exc
        return self._json_via_text(
            model, system, user, schema, role=role, label=tag, structured_error=structured_error
        )

    def tool_args(
        self,
        system: str,
        user: str,
        tool: Any,
        schema: type[SchemaT],
        *,
        role: str = "main",
        label: str | None = None,
    ) -> SchemaT:
        """Call a tool with forced tool-choice and return its validated arguments."""
        model = self._model(role)
        messages = self._messages(system, user)
        tag = label or f"tool:{getattr(tool, 'name', 'tool')}"
        attempts: list[dict[str, Any]] = [{"tool_choice": "required"}]
        name = getattr(tool, "name", None)
        if name:
            attempts.append({"tool_choice": name})
        attempts.append({})

        name = getattr(tool, "name", "tool")
        last_error: Exception | None = None
        saw_call_error = False  # a provider/validation failure, not "no tool call"
        problems: list[str] = []
        for kwargs in attempts:
            try:
                bound = model.bind_tools([tool], **kwargs)
                reply = self._invoke(bound, messages, role=role, label=tag)
                calls = getattr(reply, "tool_calls", None) or []
                if calls:
                    raw_args = calls[0].get("args", {}) if isinstance(calls[0], dict) else {}
                    if isinstance(raw_args, str):
                        raw_args = json.loads(raw_args or "{}")
                    return schema.model_validate(raw_args)
                last_error = ToolNotUsedError(f"model returned no tool call for {name!r}")
            except Exception as exc:
                saw_call_error = True
                last_error = exc
            problems.append(f"{type(last_error).__name__}: {str(last_error)[:120]}")
        # Last resort: ask for the same schema as plain structured output.
        try:
            return self.json(system, user, schema, role=role, label=tag)
        except Exception as exc:
            problems.append(f"structured fallback failed: {type(exc).__name__}: {str(exc)[:120]}")
        detail = "; ".join(problems)
        if saw_call_error:
            # Do not blame the model for a broken connection or a schema mismatch.
            raise LLMError(f"calling tool {name!r} failed: {detail}") from last_error
        raise ToolNotUsedError(
            f"the model would not call {name!r} and no structured fallback worked: {detail}"
        ) from last_error

    # ----------------------------------------------------------------- helpers
    def _messages(self, system: str, user: str) -> list[Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        return [SystemMessage(content=system), HumanMessage(content=user)]

    def _invoke(self, runner: Any, messages: list[Any], *, role: str, label: str) -> Any:
        attempts = max(1, self.settings.max_retries)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                self.calls += 1
                reply = runner.invoke(messages)
                self._record(label, messages, reply, role=role)
                return reply
            except Exception as exc:  # provider/network/rate-limit errors
                last_error = exc
                self.retries += 1
                if attempt + 1 >= attempts:
                    break
                time.sleep(min(8.0, 0.75 * (2**attempt)))
        raise LLMError(
            f"LLM call failed after {attempts} attempt(s) [{role}:{label}]: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def _record(self, label: str, messages: list[Any], reply: Any, *, role: str) -> None:
        if not self.transcript:
            return
        try:
            system = reply_to_text(messages[0]) if messages else ""
            user = reply_to_text(messages[1]) if len(messages) > 1 else ""
            self.transcript(f"{role}:{label}", system, user, reply_to_text(reply))
        except Exception:
            pass

    def _json_via_text(
        self,
        model: Any,
        system: str,
        user: str,
        schema: type[SchemaT],
        *,
        role: str,
        label: str,
        structured_error: Exception | None = None,
    ) -> SchemaT:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
        instruction = (
            f"{user}\n\n"
            "---\nReturn ONLY a single JSON object that validates against this JSON Schema. "
            "No prose, no markdown fence.\n"
            f"```json\n{schema_json}\n```"
        )
        messages = self._messages(system, instruction)
        text = ""
        finish_reason: str | None = None
        for attempt in (1, 2):
            try:
                reply = self._invoke(model, messages, role=role, label=f"{label}:json")
                if (reason := reply_finish_reason(reply)) is not None:
                    finish_reason = reason
                text = reply_to_text(reply)
                return schema.model_validate(extract_json(text))
            except LLMReplyError as exc:
                failure = exc
            except ValidationError as exc:
                failure = LLMReplyError(
                    f"reply does not match {schema.__name__}: {str(exc)[:300]}",
                    reason="schema_mismatch",
                    reply_chars=len(text),
                    preview=preview_reply(text),
                )
            except LLMError as exc:
                failure = LLMReplyError(f"the model call failed: {exc}", reason="call_failed")
            if attempt == 2:
                # Single place that labels the failure, so the caller can locate it.
                failure.schema = schema.__name__
                failure.role = role
                failure.label = label
                failure.finish_reason = failure.finish_reason or finish_reason
                if structured_error is not None:
                    failure.note = (
                        f"structured path failed too: {type(structured_error).__name__}: "
                        f"{str(structured_error)[:160]}"
                    )
                raise failure from (structured_error or failure.__cause__)
            messages = self._messages(
                system,
                f"{instruction}\n\nYour previous answer was rejected: {failure}\n"
                "Fix it and answer again.",
            )


def build_llm(
    settings: LLMSettings,
    *,
    transcript: Transcript | None = None,
    verbose: bool = False,
    route: RouteProvider | None = None,
) -> LLMClient:
    return LLMClient(settings, transcript=transcript, verbose=verbose, route=route)
