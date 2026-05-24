"""Adapter sub-packages for MiniADK.

Lazy-load policy: importing ``miniadk.adapters`` should NOT pull in
Textual / Ink rendering deps. Each adapter (``json``, ``tui``, ``web``,
``ws``) is a real submodule; access top-level names through normal
import paths (``from miniadk.adapters.json import jsonl``) or through
the lazy attributes here for back-compat.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


_LAZY: dict[str, tuple[str, str]] = {
    "astream_json": (".json", "astream_json"),
    "astream_runtime": (".json", "astream_runtime"),
    "event_dict": (".json", "event_dict"),
    "jsonl": (".json", "jsonl"),
    "Command": (".tui", "Command"),
    "CommandRegistry": (".tui", "CommandRegistry"),
    "MiniADKApp": (".tui", "MiniADKApp"),
    "Theme": (".tui", "Theme"),
    "builtin_commands": (".tui", "builtin_commands"),
    "register_command": (".tui", "register_command"),
    "run_cli": (".tui", "run_cli"),
    "web_html": (".web", "web_html"),
    "ws_chat": (".web", "ws_chat"),
    "ws_json": (".ws", "ws_json"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'miniadk.adapters' has no attribute {name!r}")
    from importlib import import_module

    module = import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:
    from .json import astream_json, astream_runtime, event_dict, jsonl
    from .tui import (
        Command,
        CommandRegistry,
        MiniADKApp,
        Theme,
        builtin_commands,
        register_command,
        run_cli,
    )
    from .web import web_html, ws_chat
    from .ws import ws_json


__all__ = sorted(set(_LAZY))
