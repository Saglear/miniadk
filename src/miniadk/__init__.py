"""MiniADK public surface.

Design contract (see ``goal/GOAL.MD``):

* ``import miniadk`` MUST NOT pull in any TUI rendering layer (Textual,
  Ink). TUI is opt-in and resolved lazily through ``__getattr__``.
* Core ADK names (Agent, Runtime, Tool, Model, Session…) are imported
  eagerly because they're the building blocks every consumer needs.
* Adapter / preset / TUI names remain accessible via
  ``from miniadk import run_cli`` etc., but the actual import work
  defers until the attribute is touched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# ── eager: zero-dependency core ─────────────────────────────────────────
from .core import (
    Agent,
    AskBefore,
    AskBeforeMiddleware,
    DefaultRunPolicy,
    Event,
    Guard,
    Message,
    Model,
    ModelResult,
    ModelStreamEvent,
    PermissionDecision,
    RunDecision,
    RunHook,
    RunPolicy,
    RunState,
    Runtime,
    Session,
    SessionStats,
    ScriptedModel,
    StopReason,
    StreamingModel,
    Tool,
    ToolCall,
    ToolCallDelta,
    ToolValidation,
    as_tool,
    ask_before,
    canonical_tool_name,
    filter_tools,
    normalize_tool_name,
    tool,
    tool_matches_name,
)
from .env import load_env, load_env_upwards
from .models import AnthropicModel, OpenAIModel, model
from .run import arun, run
from .sessions import Compact, SessionStore, compact, sessions

# ── eager-but-cheap: pure-Python compositions over the core ─────────────
# These modules import nothing heavier than .core, so loading them at
# package import keeps the lazy story simple. They're listed eagerly
# (rather than via __getattr__) because Python resolves a ``from
# miniadk import agentic`` statement by attribute lookup AFTER the
# submodule import side-effect — which would otherwise return the
# ``miniadk.agentic`` MODULE instead of the ``agentic`` function and
# shadow our ``__getattr__`` hook.
from .agentic import (
    AGENTIC_INSTRUCTIONS,
    Agentic,
    AgenticPolicy,
    CHAT_INSTRUCTIONS,
    TodoStore,
    agentic,
    make_todo_read,
    make_todo_tool,
    with_agentic_instructions,
    with_chat_instructions,
)

# ── lazy: anything that may pull a heavy / optional dep ────────────────
# Each entry maps `attribute -> (submodule, real_name)`. We resolve on
# first access, then cache on the module so repeat lookups are cheap.

_LAZY: dict[str, tuple[str, str]] = {
    # Adapters: TUI, web, WS, JSON streaming.
    "Command": (".adapters.tui", "Command"),
    "CommandRegistry": (".adapters.tui", "CommandRegistry"),
    "MiniADKApp": (".adapters.tui", "MiniADKApp"),
    "Theme": (".adapters.tui", "Theme"),
    "builtin_commands": (".adapters.tui", "builtin_commands"),
    "register_command": (".adapters.tui", "register_command"),
    "run_cli": (".adapters.tui", "run_cli"),
    "astream_json": (".adapters.json", "astream_json"),
    "event_dict": (".adapters.json", "event_dict"),
    "jsonl": (".adapters.json", "jsonl"),
    "web_html": (".adapters.web", "web_html"),
    "ws_chat": (".adapters.web", "ws_chat"),
    "ws_json": (".adapters.ws", "ws_json"),
    # MCP (heavier — pulls subprocess, json-rpc client).
    "MCPError": (".mcp", "MCPError"),
    "MCPHub": (".mcp", "MCPHub"),
    "MCPInfo": (".mcp", "MCPInfo"),
    "MCPNotice": (".mcp", "MCPNotice"),
    "MCPPrompt": (".mcp", "MCPPrompt"),
    "MCPPromptMessage": (".mcp", "MCPPromptMessage"),
    "MCPPromptResult": (".mcp", "MCPPromptResult"),
    "MCPResource": (".mcp", "MCPResource"),
    "MCPServer": (".mcp", "MCPServer"),
    "MCPToolError": (".mcp", "MCPToolError"),
    # Skills.
    "Skill": (".skills", "Skill"),
    "SkillInvocation": (".skills", "SkillInvocation"),
    "SkillProblem": (".skills", "SkillProblem"),
    "SkillRegistry": (".skills", "SkillRegistry"),
    "parse_slash_command": (".skills", "parse_slash_command"),
    "resolve_agent": (".skills", "resolve_agent"),
    "skill": (".skills", "skill"),
    "tools_for_skill": (".skills", "tools_for_skill"),
    # Standard tools (heavy enough to keep lazy — they pull subprocess,
    # network, filesystem helpers).
    "FetchResult": (".stdtools", "FetchResult"),
    "ShellResult": (".stdtools", "ShellResult"),
    "Spawn": (".stdtools", "Spawn"),
    "SpawnResult": (".stdtools", "SpawnResult"),
    "Work": (".stdtools", "Work"),
    "WorkResult": (".stdtools", "WorkResult"),
    "glob_files": (".stdtools", "glob_files"),
    "list_files": (".stdtools", "list_files"),
    "make_copy_file": (".stdtools", "make_copy_file"),
    "make_delete_file": (".stdtools", "make_delete_file"),
    "make_edit_file": (".stdtools", "make_edit_file"),
    "make_edit_files": (".stdtools", "make_edit_files"),
    "make_fetch_url": (".stdtools", "make_fetch_url"),
    "make_glob_files": (".stdtools", "make_glob_files"),
    "make_list_files": (".stdtools", "make_list_files"),
    "make_move_file": (".stdtools", "make_move_file"),
    "make_read_file": (".stdtools", "make_read_file"),
    "make_search_text": (".stdtools", "make_search_text"),
    "make_shell": (".stdtools", "make_shell"),
    "make_spawn": (".stdtools", "make_spawn"),
    "make_tools": (".stdtools", "make_tools"),
    "make_work": (".stdtools", "make_work"),
    "make_write_file": (".stdtools", "make_write_file"),
    "search_text": (".stdtools", "search_text"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'miniadk' has no attribute {name!r}")
    from importlib import import_module

    module = import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value  # cache for next lookup
    return value


if TYPE_CHECKING:
    # Keep static type-checkers happy by re-exporting the lazy names as
    # if they were imported eagerly. This block is invisible at runtime.
    from .adapters.json import astream_json, event_dict, jsonl
    from .adapters.tui import (
        Command,
        CommandRegistry,
        MiniADKApp,
        Theme,
        builtin_commands,
        register_command,
        run_cli,
    )
    from .adapters.web import web_html, ws_chat
    from .adapters.ws import ws_json
    from .mcp import (
        MCPError,
        MCPHub,
        MCPInfo,
        MCPNotice,
        MCPPrompt,
        MCPPromptMessage,
        MCPPromptResult,
        MCPResource,
        MCPServer,
        MCPToolError,
    )
    from .skills import (
        Skill,
        SkillInvocation,
        SkillProblem,
        SkillRegistry,
        parse_slash_command,
        resolve_agent,
        skill,
        tools_for_skill,
    )
    from .stdtools import (
        FetchResult,
        ShellResult,
        Spawn,
        SpawnResult,
        Work,
        WorkResult,
        glob_files,
        list_files,
        make_copy_file,
        make_delete_file,
        make_edit_file,
        make_edit_files,
        make_fetch_url,
        make_glob_files,
        make_list_files,
        make_move_file,
        make_read_file,
        make_search_text,
        make_shell,
        make_spawn,
        make_tools,
        make_work,
        make_write_file,
        search_text,
    )


# ── public surface ─────────────────────────────────────────────────────
# A short, opinionated top-level set so newcomers can read it. Anything
# not listed here is still importable (lazy ``__getattr__`` keeps every
# symbol reachable as ``miniadk.NAME``), but ``dir(miniadk)`` and
# ``from miniadk import *`` only advertise the curated 30. For the
# rest, use the module path directly: ``from miniadk.stdtools import
# make_read_file``, ``from miniadk.adapters.tui_textual import Theme``,
# etc.
__all__ = sorted(
    [
        # ── building blocks ──
        "Agent",
        "Tool",
        "Runtime",
        "Session",
        "Model",
        "Message",
        "Event",
        "RunPolicy",
        "RunDecision",
        # ── tool helpers ──
        "tool",
        "as_tool",
        "make_tools",
        # ── run shortcuts ──
        "run",
        "arun",
        "run_cli",
        # ── models ──
        "model",
        "AnthropicModel",
        "OpenAIModel",
        # ── presets / composition ──
        "agentic",
        "Agentic",
        # ── skills / MCP entry points ──
        "skill",
        "MCPServer",
        # ── sessions ──
        "sessions",
        "compact",
        # ── env helpers ──
        "load_env",
        "load_env_upwards",
    ],
    key=str.lower,
)


def __dir__() -> list[str]:
    return list(__all__) + ["__version__"]
