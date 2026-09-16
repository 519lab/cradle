from __future__ import annotations

from cradle.cache.records import Principal
from cradle.config import Settings
from cradle.gateway.models import ChatMessage, ChatRequest
from cradle.normalize import canonicalize, is_cacheable, l1_key


def _p() -> Principal:
    return Principal(tenant_id="t1", user_id="u1", key_id="k")


def _req(**kwargs) -> ChatRequest:
    base = {
        "model": "gpt-4o-mini",
        "messages": [ChatMessage(role="user", content="hello world")],
    }
    base.update(kwargs)
    return ChatRequest.model_validate(base)


def test_response_format_and_seed_do_not_collide() -> None:
    s = Settings()
    a = canonicalize(_req(response_format={"type": "json_object"}), _p(), s)
    b = canonicalize(_req(), _p(), s)
    c = canonicalize(_req(seed=1), _p(), s)
    d = canonicalize(_req(seed=2), _p(), s)
    assert l1_key(a) != l1_key(b)
    assert l1_key(c) != l1_key(d)


def test_tools_empty_and_null_collide() -> None:
    s = Settings()
    a = canonicalize(_req(tools=[]), _p(), s)
    b = canonicalize(_req(tools=None), _p(), s)
    assert l1_key(a) == l1_key(b)
    assert a.has_tools is False


def test_tool_choice_and_extras() -> None:
    s = Settings()
    a = canonicalize(_req(tool_choice="auto"), _p(), s)
    b = canonicalize(_req(), _p(), s)
    assert l1_key(a) != l1_key(b)
    extra = ChatRequest.model_validate(
        {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello world"}], "foo": 1}
    )
    extra2 = ChatRequest.model_validate(
        {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello world"}], "foo": 2}
    )
    assert l1_key(canonicalize(extra, _p(), s)) != l1_key(canonicalize(extra2, _p(), s))


def test_tool_calls_in_messages_hash() -> None:
    s = Settings()
    m1 = ChatRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "1", "function": {"name": "a", "arguments": "{}"}}],
                }
            ],
        }
    )
    m2 = ChatRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "2", "function": {"name": "a", "arguments": "{}"}}],
                }
            ],
        }
    )
    assert l1_key(canonicalize(m1, _p(), s)) != l1_key(canonicalize(m2, _p(), s))


def test_developer_role_allowed() -> None:
    s = Settings()
    req = ChatRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {"role": "developer", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
        }
    )
    c = canonicalize(req, _p(), s)
    assert c.system_prompt_version != "none"


def test_stream_tools_uncacheable() -> None:
    s = Settings()
    req = _req(stream=True, tools=[{"type": "function", "function": {"name": "x"}}])
    c = canonicalize(req, _p(), s)
    assert is_cacheable(c, req, s) is False


def test_stream_logprobs_uncacheable() -> None:
    s = Settings()
    req = _req(stream=True, logprobs=True)
    c = canonicalize(req, _p(), s)
    assert is_cacheable(c, req, s) is False


def test_whitespace_and_nfkc() -> None:
    s = Settings()
    a = canonicalize(_req(messages=[ChatMessage(role="user", content="hello   world")]), _p(), s)
    b = canonicalize(_req(messages=[ChatMessage(role="user", content="hello world")]), _p(), s)
    assert l1_key(a) == l1_key(b)
