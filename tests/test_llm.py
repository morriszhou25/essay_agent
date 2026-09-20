"""A failed LLM call must say which call failed and why."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from essay_agent.errors import LLMError, LLMReplyError, ToolNotUsedError
from essay_agent.llm import LLMClient, classify_reply, extract_json

TRUNCATED = '{"plan": {"approach": "pure numerics over arbitrary gradient sequ'


class Tiny(BaseModel):
    plan: dict[str, Any]


class _Tool:
    name = "paper_search"


class _Reply:
    def __init__(self, content: str, finish_reason: str | None = None) -> None:
        self.content = content
        self.response_metadata = {"finish_reason": finish_reason} if finish_reason else {}


class _Model:
    """Offline stand-in for a LangChain chat model (no structured output)."""

    def __init__(self, replies: list[Any]) -> None:
        self._replies = list(replies)
        self.calls = 0

    def invoke(self, messages: list[Any]) -> Any:
        self.calls += 1
        reply = self._replies[min(self.calls - 1, len(self._replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> _Model:
        return self


class _StructuredFails(_Model):
    def with_structured_output(self, schema: type[BaseModel]) -> Any:
        raise RuntimeError("provider rejected the schema")


def client_for(settings, model: _Model) -> LLMClient:
    """A client whose provider is the offline stand-in, with retries kept short."""
    client = LLMClient(settings.llm.model_copy(update={"max_retries": 1, "timeout": 1.0}))
    client._build_model = lambda role, max_tokens=None: model
    return client


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ('{"a": 1', "truncated"),
        ('{"a": "unterminated', "truncated"),
        ('{"a": [1, 2}', "invalid_json"),
        ('{"a": 1,} trailing prose', "invalid_json"),
    ],
)
def test_parse_failures_are_classified(reply: str, reason: str) -> None:
    assert classify_reply(reply) == reason


def test_truncated_reply_keeps_the_tail_and_the_provider_stop_reason(settings) -> None:
    model = _Model([_Reply(TRUNCATED, finish_reason="length")])
    with pytest.raises(LLMReplyError) as caught:
        client_for(settings, model).json("system", "user", Tiny, label="repro_script")

    error = caught.value
    assert error.reason == "truncated"
    assert error.finish_reason == "length"
    assert error.reply_chars == len(TRUNCATED)
    assert error.preview.endswith(TRUNCATED[-40:])  # the tail shows where it stopped
    assert "label=repro_script" in str(error)
    assert "schema=Tiny" in str(error)
    assert model.calls == 2  # one retry, then a diagnosed failure


def test_schema_mismatch_is_reported_as_such(settings) -> None:
    model = _Model([_Reply('{"plan": 3}')])
    with pytest.raises(LLMReplyError) as caught:
        client_for(settings, model).json("system", "user", Tiny, label="repro_script")
    assert caught.value.reason == "schema_mismatch"


def test_structured_path_failure_is_kept_as_diagnosis(settings) -> None:
    model = _StructuredFails([_Reply("not json at all")])
    with pytest.raises(LLMReplyError) as caught:
        client_for(settings, model).json("system", "user", Tiny, label="repro_script")
    assert "RuntimeError" in caught.value.note
    assert "structured path failed too" in str(caught.value)


def test_extract_json_preview_keeps_head_and_tail() -> None:
    reply = '{"plan": "' + "x" * 600
    with pytest.raises(LLMReplyError) as caught:
        extract_json(reply)
    assert caught.value.preview.startswith('{"plan": "')
    assert caught.value.preview.endswith("x" * 50)
    assert "chars elided" in caught.value.preview


def test_a_broken_tool_call_is_not_blamed_on_the_model(settings) -> None:
    model = _Model([ConnectionError("temporary failure in name resolution")])
    with pytest.raises(LLMError) as caught:
        client_for(settings, model).tool_args("system", "user", _Tool(), Tiny, label="search")
    assert not isinstance(caught.value, ToolNotUsedError)
    assert "ConnectionError" in str(caught.value)
    assert "calling tool 'paper_search' failed" in str(caught.value)


def test_no_tool_call_is_still_reported_as_such(settings) -> None:
    model = _Model([_Reply("I will not use a tool.")])
    with pytest.raises(ToolNotUsedError) as caught:
        client_for(settings, model).tool_args("system", "user", _Tool(), Tiny, label="search")
    assert "would not call 'paper_search'" in str(caught.value)
