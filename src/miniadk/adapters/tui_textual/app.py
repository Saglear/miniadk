"""The MiniADK Textual app.

Layout (full-screen):

    ┌─────────────────────────────────────────────────────────┐
    │ Header — agent · model · cwd                            │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ Transcript (RichLog) — every committed turn / tool      │
    │                                                         │
    ├─────────────────────────────────────────────────────────┤
    │ Activity — transient streaming line (thinking / tool …) │
    ├─────────────────────────────────────────────────────────┤
    │ Input — single-line prompt                              │
    ├─────────────────────────────────────────────────────────┤
    │ Footer — keybinding hints                               │
    └─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Input

from ..._guards import bind_guards
from ...core.agent import Agent, resolve_composition
from ...core.events import Event
from ...core.messages import Message
from ...core.middleware import Guard, Middleware, PermissionRequest
from ...core.model import Model
from ...core.policy import RunPolicy
from ...core.runtime import Runtime
from ...core.session import Session
from ...core.tools import Tool
from ...run import merge_tools
from ...sessions import CompactSpec, compact as compact_session, sessions
from ...skills import parse_slash_command, resolve_agent, tools_for_skill
from .commands import CommandRegistry, builtin_commands
from .screens import CommandPaletteModal, InfoModal, PermissionModal
from .theme import Theme
from .widgets import (
    ActivityLine,
    StatusBar,
    Transcript,
    render_status,
    render_table,
)


class MiniADKApp(App[None]):
    """Full-screen TUI surface for an ADK agent."""

    CSS_PATH = "theme.tcss"
    BINDINGS = [
        Binding("ctrl+c,ctrl+q", "quit", "Quit", show=True, priority=True),
        Binding("ctrl+l", "clear_transcript", "Clear", show=True),
        Binding("ctrl+p", "open_palette", "Commands", show=True),
        Binding("ctrl+u", "undo_turn", "Undo", show=True),
    ]

    def __init__(
        self,
        agent: Any,
        *,
        model: Model | None = None,
        middleware: list[Middleware] | None = None,
        policy: RunPolicy | None = None,
        session: Session | str | Path | bool | None = None,
        tools: list[Tool] | None = None,
        max_steps: int | None = None,
        compact: CompactSpec = None,
        compact_keep: int = 10,
        commands: CommandRegistry | None = None,
        theme: Theme | None = None,
    ) -> None:
        super().__init__()
        self._raw_agent = agent
        normalized_agent, normalized_middleware, normalized_policy = resolve_composition(
            agent,
            middleware=list(middleware) if middleware else None,
            policy=policy,
        )
        self._raw_middleware = normalized_middleware
        self.agent = normalized_agent
        self.policy = normalized_policy
        self._model = model
        self._tools = tools
        self._max_steps = max_steps
        self.compact = compact
        self.compact_keep = compact_keep
        self.commands = commands or builtin_commands()
        self.theme_tokens = theme or Theme()
        self._session_arg = session

        self.runtime: Runtime | None = None
        self.session_path: Path | None = None
        self._turn = 0
        self._busy = False
        self._stream_buffer: list[str] = []
        self._tool_buffers: dict[int, dict[str, Any]] = {}

    # ── lifecycle ─────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Transcript()
        yield ActivityLine()
        with Horizontal(id="input-bar"):
            yield Input(placeholder="ask me anything · /help for commands", id="prompt")
        yield StatusBar(
            agent_name=self.agent.name,
            model_label=self._describe_model(),
            cwd=os.getcwd(),
        )

    async def on_mount(self) -> None:
        self.title = f"miniadk · {self.agent.name}"
        self.sub_title = self._describe_model()

        self.agent = await resolve_agent(self.agent)
        active_tools = merge_tools(self.agent.tools, self._tools)
        if active_tools is not None:
            self.agent = self.agent.copy(tools=active_tools)

        if self._model is None:
            from ...models.factory import model as default_model
            self._model = default_model()

        self._active_session, self.session_path = _load_session(self._session_arg, self.agent)
        if self._active_session is not None:
            await compact_session(self._active_session, model=self._model, spec=self.compact)

        runtime_middleware = bind_guards(self._raw_middleware or [], ask_user=self._ask_user)
        if not _has_guard(runtime_middleware or []):
            runtime_middleware = list(runtime_middleware or [])
            runtime_middleware.append(Guard("ask", ask_user=self._ask_user))

        self.runtime = Runtime(
            agent=self.agent,
            model=self._model,
            middleware=runtime_middleware,
            policy=self.policy,
            session=self._active_session,
            max_steps=self._max_steps,
        )

        self._transcript().write_intro(
            agent_name=self.agent.name,
            model_label=self._describe_model(),
            cwd=os.getcwd(),
        )
        self.query_one("#prompt", Input).focus()

    # ── public API surface (called from commands.py) ──────────────────
    async def show_help(self) -> None:
        await self.push_screen(CommandPaletteModal(self.commands))

    def show_status(self) -> None:
        runtime = self.runtime
        assert runtime is not None
        stats = runtime.session.stats
        skills = self.agent.skills.all() if self.agent.skills is not None else []
        rows = [
            ("agent", self.agent.name),
            ("model", self._describe_model()),
            ("cwd", os.getcwd()),
            ("messages", str(stats.messages)),
            ("tool calls", str(stats.tool_calls)),
            ("chars", str(stats.chars)),
            ("tools", str(len(self.agent.tools))),
            ("skills", str(len(skills))),
        ]
        self.run_worker(self.push_screen(InfoModal("Session status", render_status(rows))))

    def show_tools(self) -> None:
        rows = [
            (tool.name, _tool_tags(tool), tool.description or "—")
            for tool in self.agent.tools
        ]
        renderable = render_table(["name", "tags", "description"], rows)
        self.run_worker(self.push_screen(InfoModal("Tools", renderable)))

    def show_skills(self) -> None:
        if self.agent.skills is None:
            self.run_worker(self.push_screen(InfoModal("Skills", "no skills loaded")))
            return
        rows = []
        for skill in self.agent.skills.all():
            mode = "user" if skill.user_invocable else "model"
            rows.append((f"/{skill.name}", mode, skill.description or skill.name))
        renderable = render_table(["command", "mode", "description"], rows or [("—", "—", "no skills")])
        self.run_worker(self.push_screen(InfoModal("Skills", renderable)))

    def clear_transcript(self) -> None:
        self._transcript().clear()
        self._transcript().write_notice("transcript cleared")

    def reset_conversation(self) -> None:
        runtime = self.runtime
        assert runtime is not None
        runtime.messages[:] = [Message("system", self.agent.instructions)]
        self._turn = 0
        self._transcript().write_notice("conversation reset")
        self._save_session()

    def undo_last_turn(self) -> None:
        runtime = self.runtime
        assert runtime is not None
        removed = _undo_last_turn(runtime.messages)
        if removed == 0:
            self._transcript().write_notice("nothing to undo")
            return
        self._turn = max(0, self._turn - 1)
        self._transcript().write_notice(f"removed {removed} messages")
        self._save_session()

    async def retry_last_turn(self) -> None:
        runtime = self.runtime
        assert runtime is not None
        retry_input = _pop_last_user_turn(runtime.messages)
        if retry_input is None:
            self._transcript().write_notice("nothing to retry")
            return
        self._turn = max(0, self._turn - 1)
        self.run_worker(self._dispatch_user_turn(retry_input, reuse_turn=False), exclusive=True)

    async def compact_now(self) -> None:
        runtime = self.runtime
        assert runtime is not None
        summary = await runtime.session.summarize(model=self._model, keep=self.compact_keep)
        if summary:
            self._transcript().write_assistant(summary)
        else:
            self._transcript().write_notice("nothing to compact")

    # ── input handling ────────────────────────────────────────────────
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        prompt = self.query_one("#prompt", Input)
        prompt.value = ""
        if not text or self._busy:
            return

        if text.startswith("/"):
            await self._dispatch_slash(text)
            return

        # Run the runtime turn in a worker so the message pump stays free
        # to handle modal interactions (permission prompts) while the run
        # is in flight.
        self.run_worker(self._dispatch_user_turn(text), exclusive=True)

    async def _dispatch_slash(self, raw: str) -> None:
        parsed = parse_slash_command(raw)
        if parsed is None:
            return
        name, args = parsed
        command = self.commands.resolve(name)
        if command is not None:
            await command.invoke(self, args)
            return
        if self.agent.skills is not None:
            skill = self.agent.skills.get(name)
            if skill is not None:
                if not skill.user_invocable:
                    self._transcript().write_notice(f"skill /{name} is model-only")
                    return
                rendered = skill.render(args).text
                self.run_worker(
                    self._dispatch_user_turn(
                        rendered,
                        displayed_input=raw,
                        tools=tools_for_skill(self.agent, skill.name),
                    ),
                    exclusive=True,
                )
                return
        self._transcript().write_notice(f"unknown command: /{name}")

    async def _dispatch_user_turn(
        self,
        user_input: str,
        *,
        displayed_input: str | None = None,
        reuse_turn: bool = False,
        tools: Iterable[Tool] | None = None,
    ) -> None:
        if self._busy:
            return
        runtime = self.runtime
        assert runtime is not None

        if not reuse_turn:
            self._turn += 1
        self._transcript().write_user(displayed_input or user_input, turn=self._turn)
        self._busy = True
        self._activity().set_activity("working")

        try:
            await self._stream_runtime(runtime, user_input, tools=tools)
            await compact_session(runtime.session, model=self._model, spec=self.compact)
        except RuntimeError as error:
            self._transcript().write_error(str(error))
        finally:
            self._busy = False
            self._activity().set_activity(None)
            self._save_session()

    async def _stream_runtime(
        self,
        runtime: Runtime,
        user_input: str,
        *,
        tools: Iterable[Tool] | None = None,
    ) -> None:
        self._stream_buffer.clear()
        self._tool_buffers.clear()
        async for event in runtime.run(user_input, tools=tools):
            self._handle_event(event)

    # ── event → widget bridge ─────────────────────────────────────────
    def _handle_event(self, event: Event) -> None:
        kind = event.type
        if kind == "thinking_delta":
            self._activity().set_activity("thinking")
            return
        if kind == "tool_call_delta":
            self._on_tool_call_delta(event)
            return
        if kind == "tool_call":
            self._activity().set_activity(None)
            self._tool_buffers.pop(event.data.get("index", 0), None)
            self._transcript().write_tool_call(event.data["name"], event.data.get("arguments", {}))
            return
        if kind == "tool_progress":
            self._activity().set_activity(f"{event.data['tool']} · {event.data['message']}")
            return
        if kind == "tool_result":
            self._activity().set_activity(None)
            self._transcript().write_tool_result(
                event.data.get("name") or "result",
                str(event.data.get("text") or event.data["result"]),
            )
            return
        if kind == "tool_denied":
            self._transcript().write_denied(event.data["message"])
            return
        if kind == "tool_invalid" or kind == "tool_error" or kind == "error":
            self._transcript().write_error(event.data.get("message", str(event.data)))
            return
        if kind == "message_delta":
            self._stream_buffer.append(event.data["text"])
            preview = "".join(self._stream_buffer)[-80:]
            self._activity().set_activity(f"writing  {preview}")
            return
        if kind == "message":
            self._activity().set_activity(None)
            text = event.data["text"]
            streamed = event.data.get("streamed", False)
            if streamed and self._stream_buffer:
                text = "".join(self._stream_buffer)
            self._stream_buffer.clear()
            if text:
                self._transcript().write_assistant(text)

    def _on_tool_call_delta(self, event: Event) -> None:
        index = int(event.data.get("index", 0))
        buffer = self._tool_buffers.setdefault(index, {"name": "", "args": ""})
        name = event.data.get("name")
        if name:
            buffer["name"] = str(name)
        arguments = event.data.get("arguments")
        if arguments is not None:
            buffer["args"] = (buffer["args"] + str(arguments)).replace("\n", " ")
        label = buffer["name"] or f"#{index}"
        self._activity().set_activity(f"preparing {label} {buffer['args'][-60:]}")

    # ── permission ────────────────────────────────────────────────────
    async def _ask_user(self, request: PermissionRequest) -> bool:
        result = await self.push_screen_wait(PermissionModal(request))
        return bool(result)

    # ── action handlers (keybindings) ─────────────────────────────────
    def action_clear_transcript(self) -> None:
        self.clear_transcript()

    async def action_open_palette(self) -> None:
        await self.show_help()

    def action_undo_turn(self) -> None:
        self.undo_last_turn()

    async def action_quit(self) -> None:
        self.exit()

    # ── helpers ───────────────────────────────────────────────────────
    @property
    def transcript(self) -> Transcript:
        """Public accessor for the scrollback widget — for custom commands."""
        return self.query_one(Transcript)

    @property
    def activity(self) -> ActivityLine:
        """Public accessor for the transient streaming line."""
        return self.query_one(ActivityLine)

    def _transcript(self) -> Transcript:
        return self.transcript

    def _activity(self) -> ActivityLine:
        return self.activity

    def _describe_model(self) -> str:
        if self._model is None:
            return "—"
        explicit = getattr(self._model, "model", None)
        return str(explicit) if explicit else self._model.__class__.__name__

    def _save_session(self) -> None:
        if self.session_path is not None and self.runtime is not None:
            self.runtime.session.save(self.session_path)


# ── module-level helpers ────────────────────────────────────────────────


def _has_guard(middleware: Iterable[Middleware]) -> bool:
    return any(isinstance(item, Guard) for item in middleware)


def _load_session(
    session: Session | str | Path | bool | None,
    agent: Agent,
) -> tuple[Session | None, Path | None]:
    if session is None:
        return None, None
    if session is True:
        path = sessions(".miniadk/sessions").path(agent.name)
        if path.exists():
            return Session.load(path), path
        return Session(), path
    if session is False:
        return None, None
    if isinstance(session, Session):
        return session, None
    path = Path(session)
    if path.exists():
        return Session.load(path), path
    return Session(), path


def _undo_last_turn(messages: list[Message]) -> int:
    if len(messages) <= 1:
        return 0
    last_user = _last_user_index(messages)
    if last_user is None:
        return 0
    removed = len(messages) - last_user
    del messages[last_user:]
    return removed


def _pop_last_user_turn(messages: list[Message]) -> str | None:
    if len(messages) <= 1:
        return None
    last_user = _last_user_index(messages)
    if last_user is None:
        return None
    user_input = messages[last_user].content
    del messages[last_user:]
    return user_input


def _last_user_index(messages: list[Message]) -> int | None:
    for index in range(len(messages) - 1, 0, -1):
        if messages[index].role == "user":
            return index
    return None


def _tool_tags(tool: Tool) -> str:
    tags = []
    for name, label in (("is_read_only", "read"), ("is_destructive", "destructive"), ("is_concurrency_safe", "safe")):
        try:
            if getattr(tool, name)():
                tags.append(label)
        except Exception:
            pass
    return ", ".join(tags) or "—"


def _recent_commits(limit: int = 3) -> list[str]:
    import subprocess
    try:
        output = subprocess.check_output(
            ["git", "log", "--oneline", f"-n{limit}"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=1,
        ).strip()
    except Exception:
        return []
    return [line for line in output.splitlines() if line.strip()]


__all__ = ["MiniADKApp", "run_cli"]
