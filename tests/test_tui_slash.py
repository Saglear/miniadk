"""Tests for the Ink TUI runner's slash-command dispatch.

The bridge spawns no real subprocess — we exercise ``_dispatch_slash``
directly. Goal: prove that slash commands (a) never reach the runtime,
(b) mutate ``runtime.messages`` correctly for /reset/undo, (c) emit the
expected wire events.
"""

from __future__ import annotations

import pytest

from miniadk import Agent, ScriptedModel
from miniadk.adapters.tui_ink.runner import _dispatch_slash
from miniadk.core.messages import Message
from miniadk.core.runtime import Runtime


def _runtime_with_messages(*messages: Message) -> Runtime:
    agent = Agent(name="test", instructions="be helpful", tools=[])
    runtime = Runtime(agent=agent, model=ScriptedModel([]))
    runtime.messages[:] = list(messages)
    return runtime


async def _collect(text: str, runtime: Runtime, agent: Agent) -> list:
    events = []
    async for event in _dispatch_slash(text, runtime, agent, ScriptedModel([]), None, None):
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_dispatch_slash_clear_emits_clear_event():
    agent = Agent(name="t", instructions="", tools=[])
    runtime = _runtime_with_messages(
        Message("system", "sys"),
        Message("user", "hello"),
        Message("assistant", "hi"),
    )
    events = await _collect("/clear", runtime, agent)
    assert events[0].type == "clear"
    # /clear should NOT touch conversation history.
    assert len(runtime.messages) == 3


@pytest.mark.asyncio
async def test_dispatch_slash_reset_drops_history_and_keeps_system():
    agent = Agent(name="t", instructions="be helpful", tools=[])
    runtime = _runtime_with_messages(
        Message("system", "sys"),
        Message("user", "a"),
        Message("assistant", "b"),
        Message("user", "c"),
    )
    events = await _collect("/reset", runtime, agent)

    types = [e.type for e in events]
    assert "clear" in types  # TUI is told to clear scrollback
    # Only the system message survives.
    assert [m.role for m in runtime.messages] == ["system"]
    assert runtime.messages[0].content == "sys"


@pytest.mark.asyncio
async def test_dispatch_slash_new_is_alias_for_reset():
    agent = Agent(name="t", instructions="be helpful", tools=[])
    runtime = _runtime_with_messages(
        Message("system", "sys"),
        Message("user", "a"),
        Message("assistant", "b"),
    )
    await _collect("/new", runtime, agent)
    assert [m.role for m in runtime.messages] == ["system"]


@pytest.mark.asyncio
async def test_dispatch_slash_undo_removes_last_turn():
    agent = Agent(name="t", instructions="", tools=[])
    runtime = _runtime_with_messages(
        Message("system", "sys"),
        Message("user", "a"),
        Message("assistant", "b"),
        Message("user", "c"),
        Message("assistant", "d"),
    )
    await _collect("/undo", runtime, agent)
    # Last user + its response are gone.
    assert [m.role for m in runtime.messages] == ["system", "user", "assistant"]


@pytest.mark.asyncio
async def test_dispatch_slash_undo_on_empty_yields_notice():
    agent = Agent(name="t", instructions="", tools=[])
    runtime = _runtime_with_messages(Message("system", "sys"))
    events = await _collect("/undo", runtime, agent)
    assert events[0].type == "notice"
    assert "nothing" in events[0].data["text"].lower()


@pytest.mark.asyncio
async def test_dispatch_slash_status_reports_counts():
    agent = Agent(name="t", instructions="", tools=[])
    runtime = _runtime_with_messages(
        Message("system", "sys"),
        Message("user", "hi"),
    )
    events = await _collect("/status", runtime, agent)
    types = [e.type for e in events]
    assert all(t == "notice" for t in types)
    body = "\n".join(e.data["text"] for e in events)
    assert "agent: t" in body
    assert "messages:" in body


@pytest.mark.asyncio
async def test_dispatch_slash_help_emits_notices():
    agent = Agent(name="t", instructions="", tools=[])
    runtime = _runtime_with_messages(Message("system", "sys"))
    events = await _collect("/help", runtime, agent)
    assert all(e.type == "notice" for e in events)
    body = "\n".join(e.data["text"] for e in events)
    assert "/reset" in body
    assert "/exit" in body


@pytest.mark.asyncio
async def test_dispatch_slash_unknown_command_emits_notice_only():
    agent = Agent(name="t", instructions="", tools=[])
    runtime = _runtime_with_messages(Message("system", "sys"))
    events = await _collect("/totally-fake", runtime, agent)
    assert len(events) == 1
    assert events[0].type == "notice"
    assert "unknown" in events[0].data["text"]


@pytest.mark.asyncio
async def test_dispatch_slash_exit_emits_quit():
    agent = Agent(name="t", instructions="", tools=[])
    runtime = _runtime_with_messages(Message("system", "sys"))
    events = await _collect("/exit", runtime, agent)
    assert events[0].type == "quit"
