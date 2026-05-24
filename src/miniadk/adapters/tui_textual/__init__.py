"""Textual-based TUI adapter (legacy fallback).

Full-screen Python TUI built on `Textual <https://textual.textualize.io>`_.
Requires the optional ``[tui-textual]`` extra:

.. code-block:: bash

    pip install miniadk[tui-textual]

Use cases:

- Platforms where the Bun-compiled Ink binary can't run.
- Headless CI environments that already speak Textual.

Public surface (lazy so importing this package without ``textual``
installed doesn't crash — only attribute access does):

- :class:`MiniADKApp`
- :class:`Theme`
- :class:`Command`, :func:`register_command`, :func:`builtin_commands`,
  :class:`CommandRegistry`
"""

from __future__ import annotations

from typing import TYPE_CHECKING


_LAZY: dict[str, tuple[str, str]] = {
    "MiniADKApp": (".app", "MiniADKApp"),
    "Command": (".commands", "Command"),
    "CommandRegistry": (".commands", "CommandRegistry"),
    "register_command": (".commands", "register_command"),
    "builtin_commands": (".commands", "builtin_commands"),
    "Theme": (".theme", "Theme"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(
            f"module 'miniadk.adapters.tui_textual' has no attribute {name!r}"
        )
    from importlib import import_module

    try:
        module = import_module(target[0], __name__)
    except ImportError as exc:  # pragma: no cover — exercised only if textual missing
        raise ImportError(
            "Textual TUI requires the 'tui-textual' extra. "
            "Install with: pip install miniadk[tui-textual]"
        ) from exc
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:
    from .app import MiniADKApp
    from .commands import Command, CommandRegistry, builtin_commands, register_command
    from .theme import Theme


__all__ = sorted(set(_LAZY))
