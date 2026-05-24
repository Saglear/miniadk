"""Slash-command registry.

Custom CLIs add commands by either:

1. Calling :func:`register_command` on a :class:`CommandRegistry` instance,
2. Decorating a function with ``@registry.command(...)``.

Each command receives ``(app, args)`` where ``app`` is the running
:class:`MiniADKApp` and ``args`` is the trailing string after the slash
command. Handlers may be synchronous or asynchronous.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import MiniADKApp


CommandHandler = Callable[["MiniADKApp", str], Awaitable[None] | None]


@dataclass(slots=True)
class Command:
    name: str
    description: str
    handler: CommandHandler
    aliases: tuple[str, ...] = ()
    group: str = "Custom"

    def matches(self, candidate: str) -> bool:
        normalised = candidate.lstrip("/").lower()
        return normalised == self.name or normalised in self.aliases

    async def invoke(self, app: "MiniADKApp", args: str) -> None:
        result = self.handler(app, args)
        if inspect.isawaitable(result):
            await result


@dataclass(slots=True)
class CommandRegistry:
    items: dict[str, Command] = field(default_factory=dict)

    def register(self, command: Command) -> Command:
        self.items[command.name] = command
        return command

    def remove(self, name: str) -> None:
        self.items.pop(name.lstrip("/").lower(), None)

    def resolve(self, candidate: str) -> Command | None:
        normalised = candidate.lstrip("/").lower().split()[0] if candidate.strip() else ""
        for command in self.items.values():
            if command.matches(normalised):
                return command
        return None

    def all(self) -> list[Command]:
        return list(self.items.values())

    def names(self) -> list[str]:
        names: list[str] = []
        for command in self.items.values():
            names.append(f"/{command.name}")
            names.extend(f"/{alias}" for alias in command.aliases)
        return names

    def command(
        self,
        name: str,
        description: str,
        *,
        aliases: Iterable[str] = (),
        group: str = "Custom",
    ) -> Callable[[CommandHandler], Command]:
        def decorator(handler: CommandHandler) -> Command:
            command = Command(
                name=name.lstrip("/").lower(),
                description=description,
                handler=handler,
                aliases=tuple(alias.lstrip("/").lower() for alias in aliases),
                group=group,
            )
            self.register(command)
            return command

        return decorator


def register_command(
    registry: CommandRegistry,
    name: str,
    description: str,
    handler: CommandHandler,
    *,
    aliases: Iterable[str] = (),
    group: str = "Custom",
) -> Command:
    return registry.register(
        Command(
            name=name.lstrip("/").lower(),
            description=description,
            handler=handler,
            aliases=tuple(alias.lstrip("/").lower() for alias in aliases),
            group=group,
        )
    )


# ── built-in commands ───────────────────────────────────────────────────


def builtin_commands() -> CommandRegistry:
    registry = CommandRegistry()
    register_command(registry, "help", "show command palette", _help, group="Inspect")
    register_command(registry, "status", "show session and model state", _status, group="Inspect")
    register_command(registry, "tools", "list tools available to the model", _tools, group="Inspect")
    register_command(registry, "skills", "list loaded skills", _skills, group="Inspect")
    register_command(registry, "clear", "clear the transcript log", _clear, group="Session")
    register_command(
        registry,
        "reset",
        "clear conversation history",
        _reset,
        aliases=("new",),
        group="Session",
    )
    register_command(registry, "undo", "remove the last user turn", _undo, group="Session")
    register_command(registry, "retry", "rerun the last user turn", _retry, group="Session")
    register_command(registry, "compact", "summarise older turns", _compact, group="Session")
    register_command(registry, "exit", "leave the session", _exit, aliases=("quit",), group="Session")
    return registry


async def _help(app: "MiniADKApp", _args: str) -> None:
    await app.show_help()


async def _status(app: "MiniADKApp", _args: str) -> None:
    app.show_status()


async def _tools(app: "MiniADKApp", _args: str) -> None:
    app.show_tools()


async def _skills(app: "MiniADKApp", _args: str) -> None:
    app.show_skills()


async def _clear(app: "MiniADKApp", _args: str) -> None:
    app.clear_transcript()


async def _reset(app: "MiniADKApp", _args: str) -> None:
    app.reset_conversation()


async def _undo(app: "MiniADKApp", _args: str) -> None:
    app.undo_last_turn()


async def _retry(app: "MiniADKApp", _args: str) -> None:
    await app.retry_last_turn()


async def _compact(app: "MiniADKApp", _args: str) -> None:
    await app.compact_now()


async def _exit(app: "MiniADKApp", _args: str) -> None:
    app.exit()


__all__ = [
    "Command",
    "CommandHandler",
    "CommandRegistry",
    "builtin_commands",
    "register_command",
]
