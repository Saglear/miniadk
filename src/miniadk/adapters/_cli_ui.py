from __future__ import annotations

import shutil
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Literal

from ..core.agent import Agent
from ..core.events import Event
from ..core.middleware import PermissionRequest


OutputMode = Literal["auto", "plain", "pretty"]


@dataclass(frozen=True, slots=True)
class CLITheme:
    name: str = "miniadk"
    accent: str = "\033[38;5;81m"
    muted: str = "\033[38;5;244m"
    user: str = "\033[38;5;111m"
    assistant: str = "\033[38;5;120m"
    tool: str = "\033[38;5;215m"
    error: str = "\033[38;5;203m"
    reset: str = "\033[0m"


@dataclass(frozen=True, slots=True)
class CLIStatus:
    agent_name: str
    tool_count: int
    skill_count: int

    @classmethod
    def from_agent(cls, agent: Agent) -> "CLIStatus":
        skills = agent.skills.all() if agent.skills is not None else []
        return cls(
            agent_name=agent.name,
            tool_count=len(agent.tools),
            skill_count=len(skills),
        )


class CLIRenderer:
    """Small rendering boundary for the terminal adapter.

    The runtime emits semantic events. This class turns those events into text.
    Keeping that boundary explicit lets products swap the CLI look without
    changing agent, model, or tool code.
    """

    def __init__(
        self,
        output_func,
        *,
        mode: OutputMode = "auto",
        theme: CLITheme | None = None,
        width: int | None = None,
    ):
        self.output_func = output_func
        self.theme = theme or CLITheme()
        self.width = width or shutil.get_terminal_size((88, 24)).columns
        self.pretty = self._should_use_pretty(mode)
        self._streaming = False

    def intro(self, status: CLIStatus) -> None:
        if not self.pretty:
            return

        details = [f"{status.tool_count} tools"]
        if status.skill_count:
            details.append(f"{status.skill_count} skills")
        subtitle = " • ".join(details)
        title = f" {self.theme.name} :: {status.agent_name} "

        inner = max(30, min(self.width - 4, 80))
        top = "╭" + "─" * min(len(title), inner)
        if len(title) < inner:
            top += "─" * (inner - len(title))
        top += "╮"

        self._write(self._color(top, self.theme.accent))
        self._write(self._line(title[:inner], inner, self.theme.accent))
        self._write(self._line(subtitle[:inner], inner, self.theme.muted))
        self._write(self._line("type /exit or /quit to leave", inner, self.theme.muted))
        self._write(self._color("╰" + "─" * inner + "╯", self.theme.accent))

    def prompt(self, text: str) -> str:
        if not self.pretty:
            return text
        label = self._prompt_label(text)
        return (
            f"{self.theme.accent}{label}{self.theme.reset}"
            f"{self.theme.muted} › {self.theme.reset}"
        )

    def skill_not_invocable(self, name: str) -> None:
        self.notice(f"skill /{name} is model-only")

    def notice(self, text: str) -> None:
        if not self.pretty:
            self._write(text)
            return
        self._write(self._color(f"· {text}", self.theme.muted))

    def clear(self) -> None:
        if self.pretty:
            self._write("\033[2J\033[H")
        else:
            self._write("")

    def section(self, title: str, subtitle: str | None = None) -> None:
        if not self.pretty:
            self._write(title)
            if subtitle:
                self._write(subtitle)
            return

        width = max(30, min(self.width - 4, 92))
        title_line = self._truncate(title, width - 2)
        self._write(self._color("┌" + "─" * width + "┐", self.theme.accent))
        self._write(self._color(f"│ {title_line.ljust(width - 1)}│", self.theme.accent))
        if subtitle:
            subtitle_line = self._truncate(subtitle, width - 2)
            self._write(self._color(f"│ {subtitle_line.ljust(width - 1)}│", self.theme.muted))
        self._write(self._color("└" + "─" * width + "┘", self.theme.accent))

    def rows(self, rows: list[tuple[str, str]], *, heading: str | None = None) -> None:
        if heading:
            self.section(heading)
        if not rows:
            self.notice("none")
            return
        if not self.pretty:
            for left, right in rows:
                self._write(f"{left}: {right}")
            return
        label_width = min(max(len(left) for left, _ in rows), 20)
        for left, right in rows:
            value = self._truncate(right, max(20, self.width - label_width - 6))
            self._write(
                self._color(f"{left.ljust(label_width)} ", self.theme.tool)
                + self._color(f" {value}", self.theme.muted)
            )

    def bullets(self, items: list[str], *, heading: str | None = None) -> None:
        if heading:
            self.section(heading)
        if not items:
            self.notice("none")
            return
        if not self.pretty:
            for item in items:
                self._write(f"- {item}")
            return
        for item in items:
            self._write(self._color("• ", self.theme.tool) + item)

    def permission_prompt(self, request: PermissionRequest) -> str:
        if not self.pretty:
            return f"Allow {request.tool.name} ({request.reason})? [y/N] "
        tool = self._color(request.tool.name, self.theme.tool)
        reason = self._color(request.reason, self.theme.muted)
        return f"Allow {tool} {reason}? [y/N] "

    def event(self, event: Event) -> None:
        if event.type == "message":
            self.assistant(event.data["text"], streamed=event.data.get("streamed", False))
        elif event.type == "message_delta":
            self.assistant_delta(event.data["text"])
        else:
            self._finish_stream()
            if event.type == "permission_request":
                self.permission_request(event)
            elif event.type == "thinking_delta":
                self.thinking_delta(event)
            elif event.type == "tool_call_delta":
                self.tool_call_delta(event)
            elif event.type == "tool_call":
                self.tool_call(event)
            elif event.type == "tool_progress":
                self.tool_progress(event)
            elif event.type == "tool_result":
                self.tool_result(event)
            elif event.type == "tool_denied":
                self.denied(event.data["message"])
            elif event.type in {"tool_invalid", "tool_error"}:
                self.error(event.data["message"])
            elif event.type == "error":
                self.error(event.data["message"])

    def assistant(self, text: str, *, streamed: bool = False) -> None:
        if streamed and self._streaming:
            self._finish_stream()
            return
        if not self.pretty:
            self._write(text)
            return
        self._block("assistant", text, self.theme.assistant)

    def assistant_delta(self, text: str) -> None:
        if not self._streaming:
            self._streaming = True
            if self.pretty:
                self._raw(self._color("assistant", self.theme.assistant) + "\n")
                self._raw(self._color("│ ", self.theme.assistant))
        self._raw(text)

    def _finish_stream(self) -> None:
        if self._streaming:
            self._raw("\n")
            self._streaming = False

    def permission_request(self, event: Event) -> None:
        if not self.pretty:
            return
        name = event.data["tool"]
        reason = event.data["reason"]
        self._write(
            self._color("◇ permission ", self.theme.tool)
            + self._color(name, self.theme.tool)
            + self._color(f" · {reason}", self.theme.muted)
        )

    def thinking_delta(self, event: Event) -> None:
        if not self.pretty:
            return
        self._write(
            self._color("◇ thinking ", self.theme.muted)
            + self._color(event.data["text"], self.theme.muted)
        )

    def tool_call_delta(self, event: Event) -> None:
        if not self.pretty:
            return
        parts = []
        name = event.data.get("name")
        if name:
            parts.append(str(name))
        arguments = event.data.get("arguments")
        if arguments:
            parts.append(str(arguments).replace("\n", "\\n"))
        if not parts:
            parts.append(f"#{event.data.get('index', 0)}")
        summary = self._clip(" ".join(parts), limit=160)
        self._write(
            self._color("◇ tool draft ", self.theme.tool)
            + self._color(summary, self.theme.muted)
        )

    def tool_call(self, event: Event) -> None:
        name = event.data["name"]
        if not self.pretty:
            self._write(f"tool: {name}")
            return
        summary = self._format_arguments(event.data.get("arguments", {}))
        suffix = f" {summary}" if summary else ""
        self._write(
            self._color("◇ tool ", self.theme.tool)
            + self._color(name, self.theme.tool)
            + self._color(suffix, self.theme.muted)
        )

    def tool_progress(self, event: Event) -> None:
        message = event.data["message"]
        if not self.pretty:
            self._write(message)
            return
        name = event.data["tool"]
        detail = self._format_arguments(event.data.get("data", {}))
        suffix = f" {detail}" if detail else ""
        self._write(
            self._color("◇ progress ", self.theme.tool)
            + self._color(name, self.theme.tool)
            + self._color(f" · {message}{suffix}", self.theme.muted)
        )

    def tool_result(self, event: Event) -> None:
        result = str(event.data["result"])
        if not self.pretty:
            self._write(result)
            return

        if not result:
            self._write(self._color("  done", self.theme.muted))
            return

        clipped = self._clip(result, limit=900)
        self._block("result", clipped, self.theme.muted)

    def denied(self, message: str) -> None:
        if not self.pretty:
            self._write(message)
            return
        self._write(self._color(f"× {message}", self.theme.error))

    def error(self, message: str) -> None:
        if not self.pretty:
            self._write(f"error: {message}")
            return
        self._write(self._color(f"× error: {message}", self.theme.error))

    def _raw(self, text: str) -> None:
        if self.output_func is print:
            sys.stdout.write(text)
            sys.stdout.flush()
        else:
            self.output_func(text)

    def _block(self, label: str, text: str, color: str) -> None:
        width = max(30, min(self.width - 6, 92))
        label_text = self._color(label, color)
        self._write(f"{label_text}")
        for paragraph in str(text).splitlines() or [""]:
            wrapped = textwrap.wrap(
                paragraph,
                width=width,
                replace_whitespace=False,
                drop_whitespace=False,
            ) or [""]
            for line in wrapped:
                self._write(self._color("│ ", color) + line)

    def _line(self, text: str, width: int, color: str) -> str:
        return self._color(f"│ {text.ljust(width)} │", color)

    def _format_arguments(self, arguments: dict[str, Any]) -> str:
        if not arguments:
            return ""
        parts = []
        for key, value in list(arguments.items())[:3]:
            text = str(value).replace("\n", "\\n")
            if len(text) > 48:
                text = f"{text[:45]}..."
            parts.append(f"{key}={text}")
        if len(arguments) > 3:
            parts.append("...")
        return " ".join(parts)

    def _prompt_label(self, text: str) -> str:
        label = text.strip()
        if label in {"", ">"}:
            return self.theme.name
        return label.rstrip(">").rstrip()

    def _clip(self, text: str, *, limit: int) -> str:
        if len(text) <= limit:
            return text
        return f"{text[:limit].rstrip()}\n... ({len(text) - limit} more chars)"

    def _truncate(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        if limit <= 1:
            return text[:limit]
        return text[: max(1, limit - 1)].rstrip() + "…"

    def _color(self, text: str, color: str) -> str:
        return f"{color}{text}{self.theme.reset}"

    def _write(self, text: str) -> None:
        self.output_func(text)

    def _should_use_pretty(self, mode: OutputMode) -> bool:
        if mode == "pretty":
            return True
        if mode == "plain":
            return False
        return self.output_func is print and sys.stdout.isatty()
