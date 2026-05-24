"""Modal screens layered on top of the chat surface."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rich.markdown import Markdown
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, Static

from ...core.middleware import PermissionRequest

if TYPE_CHECKING:
    from .commands import CommandRegistry


class PermissionModal(ModalScreen[bool]):
    """Yes/no decision for a guarded tool call."""

    BINDINGS = [
        Binding("y,Y,enter", "approve", "Allow", show=True),
        Binding("n,N,escape", "deny", "Deny", show=True),
    ]

    def __init__(self, request: PermissionRequest):
        super().__init__()
        self._request = request

    def compose(self):
        request = self._request
        try:
            args_text = json.dumps(request.arguments, ensure_ascii=False, indent=2)
        except Exception:
            args_text = str(request.arguments)

        with VerticalScroll(id="dialog"):
            yield Label(f"⚠  Allow {request.tool.name}?", classes="title")
            yield Static(request.reason, classes="reason")
            yield Static(f"args:\n{args_text}", classes="args")
            yield Static("press y to allow, n to deny", classes="hint")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


class CommandPaletteModal(ModalScreen[str | None]):
    """Searchable list of slash commands."""

    BINDINGS = [Binding("escape", "cancel", "Close", show=True)]

    def __init__(self, registry: "CommandRegistry"):
        super().__init__()
        self._registry = registry

    def compose(self):
        groups: dict[str, list[tuple[str, str]]] = {}
        for cmd in self._registry.all():
            row = (f"/{cmd.name}", cmd.description)
            groups.setdefault(cmd.group, []).append(row)
            for alias in cmd.aliases:
                groups[cmd.group].append((f"/{alias}", f"alias for /{cmd.name}"))

        with VerticalScroll(id="palette"):
            yield Label("Command palette", classes="title")
            order = ["Inspect", "Capabilities", "Session", "Custom"]
            extra = [g for g in groups if g not in order]
            for group in [*order, *extra]:
                rows = groups.get(group)
                if not rows:
                    continue
                yield Label(group, classes="group")
                for left, right in rows:
                    yield Static(f"  {left:<14} {right}", classes="row")

    def action_cancel(self) -> None:
        self.dismiss(None)


class InfoModal(ModalScreen[None]):
    """Generic info modal — used for /status, /tools, /skills."""

    BINDINGS = [Binding("escape,enter,q", "close", "Close", show=True)]

    def __init__(self, title: str, content):
        super().__init__()
        self._title = title
        self._content = content

    def compose(self):
        with VerticalScroll(id="info"):
            yield Label(self._title, classes="title")
            if isinstance(self._content, str):
                yield Static(Markdown(self._content))
            else:
                yield Static(self._content)

    def action_close(self) -> None:
        self.dismiss(None)


__all__ = ["PermissionModal", "CommandPaletteModal", "InfoModal"]
