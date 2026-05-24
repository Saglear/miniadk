"""Compatibility shim for the pre-split ``miniadk.adapters.tui`` path.

The TUI adapter was split into two backend-specific packages:

* :mod:`miniadk.adapters.tui_ink` — default Ink (TS subprocess) TUI.
* :mod:`miniadk.adapters.tui_textual` — legacy Textual TUI.

This module re-exports the public names from both, lazily, so existing
imports like ``from miniadk.adapters.tui import MiniADKApp`` keep
working. New code should import from the backend-specific package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


_LAZY: dict[str, tuple[str, str]] = {
    # Dispatcher (no backend deps).
    "run_cli": (".cli_dispatch", "run_cli"),
    # Ink backend.
    "TUIBridge": ("..tui_ink.bridge", "TUIBridge"),
    "IntroPayload": ("..tui_ink.bridge", "IntroPayload"),
    "find_tui_command": ("..tui_ink.bridge", "find_tui_command"),
    "run_ink_cli": ("..tui_ink.runner", "run_ink_cli"),
    "run_ink_cli_sync": ("..tui_ink.runner", "run_ink_cli_sync"),
    # Textual backend (only loaded when actually accessed).
    "MiniADKApp": ("..tui_textual.app", "MiniADKApp"),
    "Command": ("..tui_textual.commands", "Command"),
    "CommandRegistry": ("..tui_textual.commands", "CommandRegistry"),
    "register_command": ("..tui_textual.commands", "register_command"),
    "builtin_commands": ("..tui_textual.commands", "builtin_commands"),
    "Theme": ("..tui_textual.theme", "Theme"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'miniadk.adapters.tui' has no attribute {name!r}")
    from importlib import import_module

    module = import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:
    from .cli_dispatch import run_cli
    from ..tui_ink.bridge import IntroPayload, TUIBridge, find_tui_command
    from ..tui_ink.runner import run_ink_cli, run_ink_cli_sync
    from ..tui_textual.app import MiniADKApp
    from ..tui_textual.commands import (
        Command,
        CommandRegistry,
        builtin_commands,
        register_command,
    )
    from ..tui_textual.theme import Theme


__all__ = sorted(set(_LAZY))
