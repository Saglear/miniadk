"""Theme tokens for the MiniADK TUI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Theme:
    """Named tokens used across the TUI.

    Custom CLIs can pass a ``Theme`` instance to :class:`MiniADKApp` or
    :func:`run_cli` to recolour the interface. Tokens map onto Textual
    CSS classes via :meth:`as_css_variables`.
    """

    name: str = "miniadk"
    accent: str = "#5fafff"
    accent_strong: str = "#87afff"
    user: str = "#87afff"
    assistant: str = "#87d787"
    tool: str = "#ffaf5f"
    success: str = "#87d787"
    error: str = "#ff5f5f"
    muted: str = "#8a8a8a"
    faint: str = "#5f5f5f"
    surface: str = "#1c1c1c"
    surface_alt: str = "#262626"
    panel: str = "#121212"

    def as_css_variables(self) -> str:
        return "\n".join(
            f"$miniadk-{name}: {value};"
            for name, value in {
                "accent": self.accent,
                "accent-strong": self.accent_strong,
                "user": self.user,
                "assistant": self.assistant,
                "tool": self.tool,
                "success": self.success,
                "error": self.error,
                "muted": self.muted,
                "faint": self.faint,
                "surface": self.surface,
                "surface-alt": self.surface_alt,
                "panel": self.panel,
            }.items()
        )


__all__ = ["Theme"]
