"""Ink-based TUI adapter (default).

Spawns the ``miniadk-tui`` Bun-compiled binary as a subprocess, with
the ADK side talking JSON over stdio. The runner glues the bridge to a
:class:`miniadk.core.runtime.Runtime` instance.

Public surface (lazy-loaded so users without the binary still pay
nothing on ``import miniadk.adapters.tui_ink``):

- :class:`TUIBridge`
- :func:`find_tui_command`
- :class:`IntroPayload`
- :func:`run_ink_cli` / :func:`run_ink_cli_sync`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

_LAZY: dict[str, tuple[str, str]] = {
    "TUIBridge": (".bridge", "TUIBridge"),
    "IntroPayload": (".bridge", "IntroPayload"),
    "find_tui_command": (".bridge", "find_tui_command"),
    "run_ink_cli": (".runner", "run_ink_cli"),
    "run_ink_cli_sync": (".runner", "run_ink_cli_sync"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'miniadk.adapters.tui_ink' has no attribute {name!r}")
    from importlib import import_module

    module = import_module(target[0], __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:
    from .bridge import IntroPayload, TUIBridge, find_tui_command
    from .runner import run_ink_cli, run_ink_cli_sync


__all__ = sorted(set(_LAZY))
