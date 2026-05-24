"""Wire the Ink TUI to an ADK ``Runtime``.

This module is the *integration* layer — it owns the things the bridge
doesn't care about:

* unwrapping ``Agentic`` agents,
* loading and saving sessions,
* binding ``Guard.ask_user`` to the bridge's permission round-trip,
* turning each user submission into a runtime turn.

Keeping this glue separate from the bridge means the bridge stays a
small, testable JSON-line shim.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any

from ..._guards import bind_guards
from ...core.agent import Agent, resolve_composition
from ...core.events import Event
from ...core.middleware import Guard, Middleware, PermissionRequest
from ...core.model import Model
from ...core.policy import RunPolicy
from ...core.runtime import Runtime
from ...core.session import Session
from ...core.tools import Tool
from ...run import merge_tools
from ...sessions import CompactSpec, compact as compact_session, sessions
from ...skills import resolve_agent
from .bridge import IntroPayload, TUIBridge, find_tui_command


async def run_ink_cli(
    agent: Any,
    *,
    model: Model | None = None,
    middleware: list[Middleware] | None = None,
    policy: RunPolicy | None = None,
    session: Session | str | Path | bool | None = None,
    tools: list[Tool] | None = None,
    max_steps: int | None = None,
    compact: CompactSpec = None,
) -> None:
    """Run ``agent`` through the Ink TUI.

    Async equivalent of the ``run_cli`` entry point. The synchronous
    version simply wraps this in ``asyncio.run``.
    """

    # Resolve the agent / tools / model.
    raw_middleware_input = list(middleware) if middleware else None
    bare_agent, raw_middleware, resolved_policy = resolve_composition(
        agent, middleware=raw_middleware_input, policy=policy,
    )
    bare_agent = await resolve_agent(bare_agent)
    active_tools = merge_tools(bare_agent.tools, tools)
    if active_tools is not None:
        bare_agent = bare_agent.copy(tools=active_tools)

    if model is None:
        from ...models.factory import model as default_model
        model = default_model()

    # Load session (if any).
    active_session, session_path = _load_session(session, bare_agent)
    if active_session is not None:
        await compact_session(active_session, model=model, spec=compact)

    # Build the bridge — but defer constructing the runtime until we
    # know which permission asker to use.
    bridge_holder: dict[str, TUIBridge] = {}

    async def ask_user(request: PermissionRequest) -> bool:
        return await bridge_holder["bridge"].ask_permission(request)

    runtime_middleware = bind_guards(raw_middleware or [], ask_user=ask_user)
    if not _has_guard(runtime_middleware or []):
        runtime_middleware = list(runtime_middleware or [])
        runtime_middleware.append(Guard("ask", ask_user=ask_user))

    runtime = Runtime(
        agent=bare_agent,
        model=model,
        middleware=runtime_middleware,
        policy=resolved_policy,
        session=active_session,
        max_steps=max_steps,
    )

    # Mode tracker — the bridge will call this whenever the user
    # presses Shift+Tab. We inject a one-shot system note on the next
    # turn so the model is aware of the new constraint.
    mode_state = {"current": "default", "pending_note": None}

    def on_mode_changed(mode: str) -> None:
        mode_state["current"] = mode
        if mode == "plan":
            mode_state["pending_note"] = (
                "[plan mode] You're now in PLAN mode. Do not call any "
                "tools that write, edit, delete, or execute. Outline the "
                "plan in plain text and ask for confirmation before "
                "switching back."
            )
        elif mode == "accept_edits":
            mode_state["pending_note"] = (
                "[accept-edits mode] File-edit tool calls are now "
                "auto-approved. Continue running write/edit/patch tools "
                "without asking; still pause for shell or other "
                "destructive operations."
            )
        else:
            mode_state["pending_note"] = (
                "[default mode] Permission prompts are restored. Ask "
                "before any destructive action."
            )

    async def turn_runner(text: str) -> AsyncIterator[Event]:
        # Slash commands intercepted here — never reach the LLM. This
        # also avoids the "/clear gets sent as a user message" trap.
        if text.startswith("/"):
            async for event in _dispatch_slash(text, runtime, bare_agent, model, compact, session_path):
                yield event
            return

        note = mode_state.pop("pending_note", None) if isinstance(mode_state, dict) else None
        if isinstance(mode_state, dict) and "pending_note" not in mode_state:
            mode_state["pending_note"] = None

        if note:
            try:
                from ...core.messages import Message
                runtime.messages.append(Message("system", note))
            except Exception:
                pass

        async for event in runtime.run(text):
            yield event
        if active_session is not None:
            await compact_session(active_session, model=model, spec=compact)
        if session_path is not None:
            runtime.session.save(session_path)

    intro = IntroPayload(
        agent=bare_agent.name,
        model=_describe_model(model),
        cwd=os.getcwd(),
        tool_count=len(bare_agent.tools),
    )

    cwd_root = Path.cwd()

    def file_lister(prefix: str, limit: int) -> list[str]:
        return _list_files(cwd_root, prefix, limit)

    def turn_accountant() -> tuple[int | None, int]:
        # 1 token ≈ 4 chars (rough English / mixed average; good enough
        # for a status-bar gauge — providers report exact usage but we
        # don't have a unified path to them yet).
        try:
            chars = runtime.session.stats.chars
        except Exception:
            return None, 3000
        tokens = max(0, chars // 4)
        return tokens, 3000

    pending_notices: list[str] = []
    if active_session is not None and session_path is not None and session_path.exists():
        non_system = sum(1 for m in active_session.messages if m.role != "system")
        chars = active_session.stats.chars
        if non_system > 0:
            tokens_est = chars // 4
            label = f"resumed session · {non_system} messages · ~{tokens_est} tokens · /reset to drop"
            pending_notices.append(label)
            # No second "large session" warning — the token count above
            # is enough; doubling up nags the user and breaks immersion.

    bridge = TUIBridge(
        intro=intro,
        turn_runner=turn_runner,
        permission_asker=ask_user,
        file_lister=file_lister,
        turn_accountant=turn_accountant,
        on_mode_changed=on_mode_changed,
        pending_notices=pending_notices,
    )
    bridge_holder["bridge"] = bridge

    argv = find_tui_command()
    if argv is None:
        raise RuntimeError(
            "miniadk-tui not found. Install Bun (https://bun.sh) or set "
            "MINIADK_TUI_BIN to a built binary."
        )

    try:
        await bridge.run(argv)
    finally:
        if session_path is not None and runtime.session is not None:
            runtime.session.save(session_path)


def run_ink_cli_sync(*args: Any, **kwargs: Any) -> None:
    asyncio.run(run_ink_cli(*args, **kwargs))


# ── helpers ────────────────────────────────────────────────────────────


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


def _describe_model(model: Model) -> str:
    explicit = getattr(model, "model", None)
    return str(explicit) if explicit else model.__class__.__name__


# How deep to walk when serving @-completions. Bigger trees just take
# longer to scan; 800 entries is enough to feel snappy in a typical
# project. Hidden files / dirs starting with "." are skipped.
_FILE_LIST_BUDGET = 800


def _list_files(root: Path, prefix: str, limit: int) -> list[str]:
    """Best-effort fast file listing for the @-completion popup.

    The first segment of ``prefix`` may include a directory; everything
    after the trailing ``/`` is the partial name to match. We do a
    bounded BFS from ``root`` and return the matches.
    """
    prefix = prefix.lstrip("@").strip()
    base, partial = _split_prefix(prefix)
    base_path = (root / base) if base else root
    if not base_path.exists():
        return []

    needle = partial.lower()
    results: list[str] = []
    seen = 0

    if base_path.is_file():
        return [str(base_path.relative_to(root))]

    # Single directory listing if the user has typed a trailing slash —
    # cheap and predictable.
    if base and (prefix.endswith("/") or partial == ""):
        try:
            entries = sorted(base_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            display = str(entry.relative_to(root)) + ("/" if entry.is_dir() else "")
            results.append(display)
            if len(results) >= limit:
                break
        return results

    # Otherwise BFS from the base, scoring matches by where the needle
    # appears (prefix > substring).
    stack = [base_path]
    matches_prefix: list[str] = []
    matches_substring: list[str] = []
    while stack and seen < _FILE_LIST_BUDGET:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            seen += 1
            if seen > _FILE_LIST_BUDGET:
                break
            if entry.name.startswith("."):
                continue
            name_lower = entry.name.lower()
            display = str(entry.relative_to(root)) + ("/" if entry.is_dir() else "")
            if needle and name_lower.startswith(needle):
                matches_prefix.append(display)
            elif needle and needle in name_lower:
                matches_substring.append(display)
            if entry.is_dir() and entry.name not in {"node_modules", "__pycache__", "dist", "build", ".venv", ".git"}:
                stack.append(entry)
            if len(matches_prefix) >= limit:
                break
        if len(matches_prefix) >= limit:
            break
    results = matches_prefix[:limit]
    if len(results) < limit:
        results.extend(matches_substring[: limit - len(results)])
    return results


def _split_prefix(prefix: str) -> tuple[str, str]:
    """``src/mini`` → ``("src", "mini")``; ``src/`` → ``("src", "")``."""
    if "/" not in prefix:
        return "", prefix
    base, partial = prefix.rsplit("/", 1)
    return base, partial


# ── slash-command dispatcher ──────────────────────────────────────────


async def _dispatch_slash(
    text: str,
    runtime: Runtime,
    agent: Agent,
    model: Model,
    compact: CompactSpec,
    session_path: Path | None,
) -> AsyncIterator[Event]:
    """Handle a slash command without going to the LLM.

    Yields ``Event`` objects that the bridge passes through to the TUI:

    * ``notice`` — any informational text (status, help, errors).
    * ``clear`` — wipe the visible transcript (purely cosmetic).
    * ``quit`` — exit the TUI.

    Conversation-history mutations (``/reset``, ``/undo``, ``/retry``)
    happen on ``runtime.messages`` directly. ``/retry`` re-runs the
    last user turn through the runtime.
    """
    from ...core.messages import Message

    parts = text[1:].strip().split(maxsplit=1)
    if not parts:
        yield _notice("type a command, e.g. /help")
        return
    name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if name in {"exit", "quit"}:
        yield Event(type="quit", data={})
        return

    if name in {"clear"}:
        yield Event(type="clear", data={})
        yield _notice("transcript cleared (history kept — use /reset to drop history)")
        return

    if name in {"reset", "new"}:
        kept = sum(1 for m in runtime.messages if m.role != "system")
        runtime.messages[:] = [m for m in runtime.messages if m.role == "system"]
        if not any(m.role == "system" for m in runtime.messages) and agent.instructions:
            runtime.messages.insert(0, Message("system", agent.instructions))
        if session_path is not None:
            runtime.session.save(session_path)
        yield Event(type="clear", data={})
        yield _notice(f"reset · removed {kept} messages, kept system prompt")
        return

    if name == "undo":
        removed = _undo_last_turn(runtime.messages)
        if removed == 0:
            yield _notice("nothing to undo")
        else:
            yield _notice(f"removed last {removed} messages")
        if session_path is not None:
            runtime.session.save(session_path)
        return

    if name == "retry":
        last = _pop_last_user_turn(runtime.messages)
        if last is None:
            yield _notice("nothing to retry")
            return
        yield _notice(f"retrying: {last[:80]}")
        async for event in runtime.run(last):
            yield event
        if session_path is not None:
            runtime.session.save(session_path)
        return

    if name == "compact":
        if model is None:
            yield _notice("no model configured for /compact")
            return
        try:
            summary = await runtime.session.summarize(model=model, keep=10)
        except Exception as error:
            yield _notice(f"compact failed: {error}")
            return
        if summary:
            yield _notice("compacted older turns")
        else:
            yield _notice("nothing to compact")
        if session_path is not None:
            runtime.session.save(session_path)
        return

    if name == "status":
        stats = runtime.session.stats
        rows = [
            f"agent: {agent.name}",
            f"model: {_describe_model(model)}",
            f"messages: {stats.messages}",
            f"tool calls: {stats.tool_calls}",
            f"chars: {stats.chars}  (~{stats.chars // 4} tokens)",
            f"tools: {len(agent.tools)}",
        ]
        for row in rows:
            yield _notice(row)
        return

    if name == "tools":
        if not agent.tools:
            yield _notice("no tools registered")
            return
        for tool in agent.tools:
            tags = []
            for attr, label in (("is_read_only", "read"),
                                 ("is_destructive", "destructive"),
                                 ("is_concurrency_safe", "safe")):
                try:
                    if getattr(tool, attr)():
                        tags.append(label)
                except Exception:
                    pass
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            desc = (tool.description or "").splitlines()[0] if tool.description else "—"
            yield _notice(f"{tool.name}{tag_str}: {desc}")
        return

    if name == "skills":
        skills = agent.skills.all() if agent.skills is not None else []
        if not skills:
            yield _notice("no skills loaded")
            return
        for skill in skills:
            mode = "user" if skill.user_invocable else "model"
            yield _notice(f"/{skill.name} [{mode}]: {skill.description or skill.name}")
        return

    if name == "help":
        for line in _HELP_TEXT.splitlines():
            yield _notice(line)
        return

    # Unknown / not yet wired up.
    yield _notice(f"unknown command: /{name} (type /help)")


_HELP_TEXT = """\
commands ─
  /help                show this help
  /status              session info (messages, tokens, tools)
  /tools               list tools
  /skills              list loaded skills
  /clear               clear transcript only (history kept)
  /reset, /new         drop conversation history
  /undo                remove last turn
  /retry               rerun the last user turn
  /compact             summarise older turns
  /exit, /quit         leave
keys ─
  shift+tab            cycle permission mode (default → accept_edits → plan)
  ctrl+r               expand / fold last tool result
  ctrl+l               clear screen (TUI only)
  ctrl+c (twice)       exit
  esc                  cancel running turn"""


def _notice(text: str) -> Event:
    return Event(type="notice", data={"text": text})


def _undo_last_turn(messages: list) -> int:
    if len(messages) <= 1:
        return 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            removed = len(messages) - i
            del messages[i:]
            return removed
    return 0


def _pop_last_user_turn(messages: list) -> str | None:
    if len(messages) <= 1:
        return None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            text = messages[i].content
            del messages[i:]
            return text
    return None


__all__ = ["run_ink_cli", "run_ink_cli_sync"]
