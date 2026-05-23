from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from ..core.agent import Agent

InputFunc = Callable[[str], str]
OutputFunc = Callable[[str], None]


class CLIInput:
    def __init__(
        self,
        *,
        prompt: str,
        commands: list[str],
        history_path: str | Path | None = None,
    ):
        self.prompt = prompt
        self.commands = commands
        self.history_path = Path(history_path or ".miniadk/cli-history")
        self._session = None

    def __call__(self, prompt: str | None = None) -> str:
        session = self._prompt_session()
        return session.prompt(prompt or self.prompt)

    def ask(self, prompt: str) -> str:
        from prompt_toolkit import prompt as prompt_once

        return prompt_once(prompt)

    def _prompt_session(self):
        if self._session is not None:
            return self._session

        from prompt_toolkit import PromptSession
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.enums import EditingMode
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.shortcuts import CompleteStyle

        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _(event):
            event.current_buffer.validate_and_handle()

        @bindings.add("c-j")
        def _(event):
            event.current_buffer.insert_text("\n")

        self._session = PromptSession(
            completer=WordCompleter(self.commands, ignore_case=True, sentence=True),
            complete_style=CompleteStyle.MULTI_COLUMN,
            auto_suggest=AutoSuggestFromHistory(),
            history=FileHistory(str(self.history_path)),
            key_bindings=bindings,
            multiline=True,
            editing_mode=EditingMode.EMACS,
            complete_while_typing=True,
            enable_history_search=True,
        )
        return self._session


def command_names(agent: Agent) -> list[str]:
    commands = [
        "/help",
        "/status",
        "/usage",
        "/theme",
        "/tools",
        "/skills",
        "/todos",
        "/clear",
        "/new",
        "/reset",
        "/retry",
        "/undo",
        "/compact",
        "/exit",
        "/quit",
    ]
    if agent.skills is not None:
        commands.extend(
            f"/{skill.name}" for skill in agent.skills.all() if skill.user_invocable
        )
    return sorted(set(commands), key=str.lower)


def should_use_prompt_toolkit(
    input_func: InputFunc,
    output_func: OutputFunc,
) -> bool:
    if input_func is not input or output_func is not print:
        return False
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False
    try:
        import prompt_toolkit  # noqa: F401
    except Exception:
        return False
    return True
