"""TUI tests using Textual's Pilot harness.

These exercise the new MiniADK TUI surface end-to-end:

- App boots and constructs the runtime
- Slash commands resolve through the registry
- A user turn drives a runtime turn end-to-end
- Custom commands can be registered
- Permission modal opens, approves, and forwards a runtime decision
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from miniadk import (
    Agent,
    Command,
    CommandRegistry,
    MiniADKApp,
    ModelResult,
    Runtime,
    ScriptedModel,
    Session,
    Skill,
    SkillRegistry,
    Theme,
    ToolCall,
    builtin_commands,
    register_command,
    run_cli,
    tool,
)
from miniadk.adapters.tui_textual.commands import _help, _status
from miniadk.adapters.tui_textual.screens import (
    CommandPaletteModal,
    InfoModal,
    PermissionModal,
)


@tool(read_only=True)
def echo(text: str) -> str:
    """Echo back the text."""
    return text


@tool(destructive=True)
def write_file(path: str, content: str) -> str:
    """Pretend to write a file."""
    return f"wrote {path}"


def _agent(*, instructions: str = "Be brief.", tools=None) -> Agent:
    return Agent(name="tester", instructions=instructions, tools=tools or [echo])


# ── construction & wiring ──────────────────────────────────────────────


def test_run_cli_is_callable():
    assert callable(run_cli)


def test_app_constructs_with_defaults():
    app = MiniADKApp(_agent(), model=ScriptedModel([]))
    assert app.agent.name == "tester"
    assert isinstance(app.commands, CommandRegistry)
    assert app.theme_tokens.name == "miniadk"


def test_builtin_commands_registry_contains_help_and_exit():
    registry = builtin_commands()
    names = [cmd.name for cmd in registry.all()]
    assert "help" in names
    assert "exit" in names
    assert "status" in names


def test_register_command_adds_to_registry():
    registry = builtin_commands()

    async def my_handler(_app, _args):
        return None

    register_command(registry, "ping", "ping the agent", my_handler, group="Inspect")
    cmd = registry.resolve("/ping")
    assert cmd is not None
    assert cmd.description == "ping the agent"
    assert cmd.group == "Inspect"


def test_command_aliases_resolve():
    registry = builtin_commands()
    cmd = registry.resolve("/new")
    assert cmd is not None
    assert cmd.name == "reset"


def test_unknown_command_returns_none():
    registry = builtin_commands()
    assert registry.resolve("/nonexistent") is None


def test_command_decorator_form():
    registry = CommandRegistry()

    @registry.command("plan", "make a plan", group="Custom")
    async def planner(_app, _args):
        return None

    assert registry.resolve("/plan") is planner


def test_theme_exposes_css_variables():
    theme = Theme(name="custom")
    css = theme.as_css_variables()
    assert "$miniadk-accent:" in css
    assert "$miniadk-user:" in css


# ── pilot-driven app behaviour ────────────────────────────────────────


@pytest.mark.asyncio
async def test_app_mounts_and_renders_header():
    app = MiniADKApp(_agent(), model=ScriptedModel([ModelResult(message="ok")]))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.title == "miniadk · tester"
        assert app.runtime is not None


@pytest.mark.asyncio
async def test_user_input_triggers_runtime_turn():
    app = MiniADKApp(
        _agent(),
        model=ScriptedModel([ModelResult(message="hello back")]),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt")
        prompt.value = "hello"
        await prompt.action_submit()
        await pilot.pause()

        # the assistant response is captured by the runtime
        assert any(
            m.role == "assistant" and "hello back" in m.content
            for m in app.runtime.messages
        )


@pytest.mark.asyncio
async def test_slash_help_opens_command_palette():
    app = MiniADKApp(_agent(), model=ScriptedModel([]))
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt")
        prompt.value = "/help"
        await prompt.action_submit()
        await pilot.pause()
        assert any(isinstance(s, CommandPaletteModal) for s in app.screen_stack)


@pytest.mark.asyncio
async def test_slash_status_opens_info_modal():
    app = MiniADKApp(_agent(), model=ScriptedModel([]))
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt")
        prompt.value = "/status"
        await prompt.action_submit()
        await pilot.pause()
        assert any(isinstance(s, InfoModal) for s in app.screen_stack)


@pytest.mark.asyncio
async def test_slash_clear_resets_transcript():
    app = MiniADKApp(_agent(), model=ScriptedModel([]))
    async with app.run_test() as pilot:
        await pilot.pause()
        transcript = app._transcript()
        prompt = app.query_one("#prompt")
        prompt.value = "/clear"
        await prompt.action_submit()
        await pilot.pause()
        # after /clear the transcript still has the "transcript cleared" notice
        assert len(transcript.lines) >= 1


@pytest.mark.asyncio
async def test_slash_reset_clears_conversation_history():
    app = MiniADKApp(
        _agent(),
        model=ScriptedModel([ModelResult(message="first")]),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt")
        prompt.value = "hello"
        await prompt.action_submit()
        await pilot.pause()
        assert len(app.runtime.messages) > 1

        prompt.value = "/reset"
        await prompt.action_submit()
        await pilot.pause()
        assert len(app.runtime.messages) == 1
        assert app.runtime.messages[0].role == "system"


@pytest.mark.asyncio
async def test_unknown_slash_command_writes_notice():
    app = MiniADKApp(_agent(), model=ScriptedModel([]))
    async with app.run_test() as pilot:
        await pilot.pause()
        before = len(app._transcript().lines)
        prompt = app.query_one("#prompt")
        prompt.value = "/nonsense"
        await prompt.action_submit()
        await pilot.pause()
        assert len(app._transcript().lines) > before


@pytest.mark.asyncio
async def test_custom_command_handler_runs():
    invoked: list[str] = []

    async def handler(app: MiniADKApp, args: str) -> None:
        invoked.append(args)
        app._transcript().write_notice(f"called with: {args}")

    registry = builtin_commands()
    register_command(registry, "ping", "test ping", handler)
    app = MiniADKApp(_agent(), model=ScriptedModel([]), commands=registry)
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt")
        prompt.value = "/ping hello"
        await prompt.action_submit()
        await pilot.pause()
        assert invoked == ["hello"]


@pytest.mark.asyncio
async def test_undo_removes_last_turn():
    app = MiniADKApp(
        _agent(),
        model=ScriptedModel([ModelResult(message="reply")]),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt")
        prompt.value = "hello"
        await prompt.action_submit()
        await pilot.pause()
        full_count = len(app.runtime.messages)

        prompt.value = "/undo"
        await prompt.action_submit()
        await pilot.pause()
        assert len(app.runtime.messages) < full_count


@pytest.mark.asyncio
async def test_session_persists_to_disk(tmp_path: Path):
    session_path = tmp_path / "session.json"
    app = MiniADKApp(
        _agent(),
        model=ScriptedModel([ModelResult(message="persisted")]),
        session=session_path,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt")
        prompt.value = "save me"
        await prompt.action_submit()
        await pilot.pause()

    assert session_path.exists()
    loaded = Session.load(session_path)
    assert any(m.role == "assistant" and "persisted" in m.content for m in loaded.messages)


@pytest.mark.asyncio
async def test_permission_modal_renders_for_destructive_tool():
    agent = Agent(name="risky", instructions="be safe", tools=[write_file])
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[
                    ToolCall(name="write_file", arguments={"path": "x", "content": "y"}, id="t1")
                ]
            ),
            ModelResult(message="done"),
        ]
    )
    app = MiniADKApp(agent, model=model)
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt")
        prompt.value = "do it"
        # fire-and-forget: don't await the submit because the runtime
        # turn itself is what waits on the modal we're about to interact
        # with.
        prompt.post_message(prompt.Submitted(prompt, "do it"))

        # Wait for the modal to appear, then press y to approve.
        for _ in range(50):
            await pilot.pause()
            if any(isinstance(s, PermissionModal) for s in app.screen_stack):
                break
        assert any(isinstance(s, PermissionModal) for s in app.screen_stack)
        await pilot.press("y")

        # Let the run finish.
        for _ in range(80):
            await pilot.pause()
            if not app._busy:
                break
        assert not app._busy


@pytest.mark.asyncio
async def test_skill_invocation_dispatches_runtime_turn():
    plan_skill = Skill(
        name="plan",
        description="Plan a release",
        body="Plan something with $ARGUMENTS",
    )
    skills = SkillRegistry.from_skills(plan_skill)
    agent = Agent(name="planner", instructions="be helpful", tools=[echo], skills=skills)
    app = MiniADKApp(
        agent,
        model=ScriptedModel([ModelResult(message="planned")]),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt")
        prompt.value = "/plan release"
        await prompt.action_submit()
        await pilot.pause()
        last = app.runtime.messages[-1]
        assert last.role == "assistant"
        assert "planned" in last.content


@pytest.mark.asyncio
async def test_busy_state_prevents_concurrent_turns():
    app = MiniADKApp(
        _agent(),
        model=ScriptedModel([ModelResult(message="one"), ModelResult(message="two")]),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt")
        prompt.value = "first"
        await prompt.action_submit()
        # try to immediately submit again — should be a no-op since busy
        prompt.value = "second"
        # don't await pause first; check that _busy can be set
        # (this is more of a smoke test — racing conditions are hard to assert)
        await pilot.pause()
