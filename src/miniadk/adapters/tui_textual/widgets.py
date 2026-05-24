"""TUI widgets specific to MiniADK."""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Label, RichLog, Static


# ── activity line with braille spinner ─────────────────────────────────
_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


class ActivityLine(Static):
    """Single transient line for streaming progress.

    Updated in place via :meth:`set_activity`. When active the leading
    glyph cycles through braille spinner frames at 80 ms — the same
    cadence opencode and codewhale use.
    """

    def __init__(self):
        super().__init__("", id="activity")
        self._frame = 0
        self._text = ""
        self._timer = None

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.08, self._tick, pause=True)

    def set_activity(self, text: str | None) -> None:
        if text is None:
            self._text = ""
            self.update("")
            self.set_class(False, "visible")
            if self._timer is not None:
                self._timer.pause()
            return
        self._text = text
        self.set_class(True, "visible")
        if self._timer is not None:
            self._timer.resume()
        self._render_frame()

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(_SPINNER_FRAMES)
        self._render_frame()

    def _render_frame(self) -> None:
        if not self._text:
            return
        glyph = _SPINNER_FRAMES[self._frame]
        line = Text()
        line.append(f"{glyph} ", style="cyan")
        line.append(self._text, style="dim")
        self.update(line)


# ── status bar at the bottom of the screen ────────────────────────────


class StatusBar(Widget):
    """Bottom bar — agent · model · cwd. Single line, terminal-coloured."""

    def __init__(self, *, agent_name: str, model_label: str, cwd: str):
        super().__init__(id="status-bar")
        self._agent_name = agent_name
        self._model_label = model_label
        self._cwd = cwd

    def compose(self) -> ComposeResult:
        cwd_short = self._cwd
        if len(cwd_short) > 28:
            cwd_short = "…" + cwd_short[-27:]
        right = f"{self._model_label} · {cwd_short}"
        with Horizontal():
            yield Label(f"miniadk · {self._agent_name}", classes="status-left")
            yield Label(right, classes="status-right")


# ── transcript ────────────────────────────────────────────────────────


class Transcript(RichLog):
    """Scrollback of every committed turn / tool call / result."""

    def __init__(self):
        super().__init__(
            id="transcript",
            wrap=True,
            highlight=False,
            markup=False,
            auto_scroll=True,
        )

    # ── high-level helpers ────────────────────────────────────────────

    def write_intro(self, agent_name: str, model_label: str, cwd: str) -> None:
        cwd_short = cwd
        if len(cwd_short) > 48:
            cwd_short = "…" + cwd_short[-47:]
        head = Text()
        head.append("miniadk ", style="bold cyan")
        head.append(f"· {agent_name}", style="cyan")
        sub = Text()
        sub.append(f"  {model_label} · {cwd_short}", style="dim")
        hint = Text("  /help · /status · ctrl+c to exit", style="dim")
        self.write(Group(head, sub, hint))
        self.write("")

    def write_user(self, text: str, *, turn: int) -> None:
        # Opencode-style: a coloured left bar marks user input. No header
        # label — the bar is the signal.
        for line in text.splitlines() or [""]:
            row = Text()
            row.append("▌ ", style="cyan")
            row.append(line)
            self.write(row)
        self.write("")

    def write_assistant(self, markdown_text: str) -> None:
        # No "assistant" label — the rendered markdown speaks for itself.
        self.write(Markdown(markdown_text, code_theme="ansi_dark"))
        self.write("")

    def write_tool_call(self, name: str, arguments: dict) -> None:
        head = Text.assemble(
            ("● ", "yellow"),
            (name, "yellow"),
        )
        if arguments:
            head.append("  ")
            head.append(_compact_args(arguments), style="dim")
        self.write(head)

    def write_tool_result(self, name: str, result: str) -> None:
        if not result:
            self.write(Text("  ✓ done", style="green"))
            self.write("")
            return
        line_count = result.count("\n") + (1 if result else 0)
        char_count = len(result)
        head = Text.assemble(
            ("  ↳ ", "dim"),
            (f"{line_count} lines · {char_count} chars", "dim"),
        )
        body_text = (
            result
            if char_count <= 1500
            else f"{result[:1500].rstrip()}\n... ({char_count - 1500} more chars)"
        )
        self.write(head)
        for line in body_text.splitlines()[:20]:  # cap visible lines
            row = Text()
            row.append("  ", style="dim")
            row.append(line)
            self.write(row)
        if len(body_text.splitlines()) > 20:
            self.write(Text(f"  … ({len(body_text.splitlines()) - 20} more lines)", style="dim"))
        self.write("")

    def write_tool_progress(self, name: str, message: str) -> None:
        line = Text.assemble(
            ("  ", ""),
            (name, "yellow"),
            (" · ", "dim"),
            (message, "dim"),
        )
        self.write(line)

    def write_notice(self, text: str) -> None:
        self.write(Text(f"· {text}", style="dim"))

    def write_error(self, message: str) -> None:
        self.write(Text(f"× {message}", style="bold red"))

    def write_denied(self, message: str) -> None:
        self.write(Text(f"× {message}", style="red"))

    def write_rule(self) -> None:
        self.write(Rule(style="dim"))


# ── helpers used by modals ────────────────────────────────────────────


def render_status(rows: list[tuple[str, str]]) -> RenderableType:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="yellow")
    table.add_column()
    for left, right in rows:
        table.add_row(left, right)
    return table


def render_table(headers: list[str], rows: list[tuple[str, ...]]) -> RenderableType:
    table = Table(
        show_header=True,
        header_style="bold cyan",
        expand=True,
        border_style="dim",
    )
    for header in headers:
        table.add_column(header)
    for row in rows:
        table.add_row(*row)
    return table


def render_code(text: str, language: str = "text") -> Syntax:
    return Syntax(text, language, theme="ansi_dark", word_wrap=True, background_color="default")


def _compact_args(arguments: dict) -> str:
    parts = []
    for key, value in list(arguments.items())[:3]:
        text = str(value).replace("\n", "\\n")
        if len(text) > 48:
            text = f"{text[:45]}..."
        parts.append(f"{key}={text}")
    if len(arguments) > 3:
        parts.append("...")
    return " ".join(parts)


__all__ = [
    "ActivityLine",
    "StatusBar",
    "Transcript",
    "render_code",
    "render_status",
    "render_table",
]
