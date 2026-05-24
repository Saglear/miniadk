"""Top-level ``run_cli`` dispatcher for the TUI adapter.

Lives outside ``app.py`` so importing the dispatcher does NOT pull in
Textual. The Textual ``App`` subclass is only loaded when the user
explicitly asks for the Textual backend (or the Ink path can't run).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ...core.agent import Agent
    from ...core.middleware import Middleware
    from ...core.model import Model
    from ...core.policy import RunPolicy
    from ...core.session import Session
    from ...core.tools import Tool
    from ...sessions import CompactSpec
    from ..tui_textual.commands import CommandRegistry
    from ..tui_textual.theme import Theme


def run_cli(
    agent: "Agent | Any",
    *,
    model: "Model | None" = None,
    middleware: "list[Middleware] | None" = None,
    policy: "RunPolicy | None" = None,
    session: "Session | str | Path | bool | None" = None,
    tools: "list[Tool] | None" = None,
    max_steps: int | None = None,
    compact: "CompactSpec" = None,
    compact_keep: int = 10,
    commands: "CommandRegistry | None" = None,
    theme: "Theme | None" = None,
    backend: str = "auto",
) -> None:
    """Launch the default TUI for ``agent``.

    ``backend`` selects the renderer:

    * ``"auto"`` (default) — prefer the Ink TUI if a binary or local
      Bun + source tree is available, otherwise fall back to the
      Textual TUI.
    * ``"ink"`` — force the Ink TUI; raises if not available.
    * ``"textual"`` — force the Textual TUI (requires the
      ``[tui-textual]`` extra to be installed).

    Custom CLIs should consume :class:`miniadk.adapters.tui_textual.MiniADKApp`
    or :class:`miniadk.adapters.tui_ink.TUIBridge` directly when they
    need to override layout, CSS, or React components.
    """

    if backend in {"auto", "ink"}:
        from ..tui_ink.bridge import find_tui_command

        if backend == "ink" or find_tui_command() is not None:
            from ..tui_ink.runner import run_ink_cli_sync

            run_ink_cli_sync(
                agent,
                model=model,
                middleware=middleware,
                policy=policy,
                session=session,
                tools=tools,
                max_steps=max_steps,
                compact=compact,
            )
            return
        if backend == "ink":
            raise RuntimeError(
                "Ink TUI binary not found. Run `miniadk-tui-fetch` after "
                "install, or set MINIADK_TUI_BIN to a built binary."
            )

    # Textual path. Defer the import — `textual` is an optional extra.
    try:
        from ..tui_textual.app import MiniADKApp
    except ImportError as exc:  # pragma: no cover - exercised in headless installs
        raise RuntimeError(
            "Textual is not installed. Either install the optional extra "
            "`pip install miniadk[tui-textual]`, or fetch the Ink TUI "
            "binary with `miniadk-tui-fetch`."
        ) from exc

    MiniADKApp(
        agent,
        model=model,
        middleware=middleware,
        policy=policy,
        session=session,
        tools=tools,
        max_steps=max_steps,
        compact=compact,
        compact_keep=compact_keep,
        commands=commands,
        theme=theme,
    ).run()
