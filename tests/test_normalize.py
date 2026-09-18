from __future__ import annotations

from cradle.cache.records import Principal
from cradle.config import Settings
from cradle.gateway.models import ChatMessage, ChatRequest
from cradle.normalize import canonicalize, is_cacheable, l1_key, l2_eligible


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


# --- #40: system prompt is scored for identity but excluded from embed_text ---

_BIG_SYS = "You are a precise assistant. " * 40  # a large fixed frame


def test_embed_text_excludes_system_and_developer_turns() -> None:
    s = Settings()
    c = canonicalize(
        _req(messages=[
            ChatMessage(role="system", content=_BIG_SYS),
            ChatMessage(role="user", content="capital of France?"),
        ]),
        _p(), s,
    )
    # The user turn is embedded; the system prompt is not.
    assert "capital of France" in c.embed_text
    assert "precise assistant" not in c.embed_text
    assert c.embed_text.startswith("user:")


def test_same_system_prompt_different_user_diverge_in_embed_text_but_share_identity() -> None:
    """#40: two different user tasks under one shared system prompt must not be
    matched by embed_text, yet must still share system-prompt scoping."""
    s = Settings()
    a = canonicalize(
        _req(messages=[
            ChatMessage(role="system", content=_BIG_SYS),
            ChatMessage(role="user", content="generate a concise title for this content"),
        ]),
        _p(), s,
    )
    b = canonicalize(
        _req(messages=[
            ChatMessage(role="system", content=_BIG_SYS),
            ChatMessage(role="user", content="explain airplanes like I am a goldfish"),
        ]),
        _p(), s,
    )
    # embed_text now reflects only the (very different) user turns...
    assert a.embed_text != b.embed_text
    assert _BIG_SYS not in a.embed_text and _BIG_SYS not in b.embed_text
    # ...but the shared system prompt still binds their cache identity (so a
    # different system prompt would still separate them — correctness kept).
    assert a.system_prompt_version == b.system_prompt_version
    assert a.system_prompt_version != "none"


def test_system_only_request_is_l2_ineligible() -> None:
    """A request with only a system prompt yields empty embed_text and must not
    be embedded (an empty vector matches anything)."""
    s = Settings()
    req = _req(messages=[ChatMessage(role="system", content=_BIG_SYS)])
    c = canonicalize(req, _p(), s)
    assert c.embed_text.strip() == ""
    assert l2_eligible(c, req, s) is False


def test_different_system_prompt_still_separates_identity() -> None:
    """Excluding the system prompt from embed_text must NOT let two requests with
    different system prompts share a cache entry — system_prompt_version guards it."""
    s = Settings()
    a = canonicalize(
        _req(messages=[
            ChatMessage(role="system", content="You are a pirate."),
            ChatMessage(role="user", content="hello"),
        ]),
        _p(), s,
    )
    b = canonicalize(
        _req(messages=[
            ChatMessage(role="system", content="You are a lawyer."),
            ChatMessage(role="user", content="hello"),
        ]),
        _p(), s,
    )
    assert a.embed_text == b.embed_text  # same user turn
    assert a.system_prompt_version != b.system_prompt_version  # but different identity
    assert l1_key(a) != l1_key(b)
