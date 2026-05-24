"""06 — A minimum CLI.

``run_cli`` launches the default terminal UI. On first use it'll
download the Ink binary into ``~/.cache/miniadk/tui/`` (a one-time
~70 MB fetch). Set ``MINIADK_TUI_NO_FETCH=1`` to skip and use the
Textual fallback instead — install with ``pip install
miniadk[tui-textual]``.

Run:

    uv run python examples/06_run_cli.py
"""

from miniadk import Agent, load_env_upwards, run_cli, tool

load_env_upwards()


@tool
def fortune() -> str:
    """Return a tiny random saying."""
    import random

    sayings = [
        "What you seek is seeking you.",
        "When in doubt, sleep on it.",
        "The best time to plant a tree was twenty years ago.",
    ]
    return random.choice(sayings)


run_cli(
    Agent(
        name="oracle",
        instructions="You are a friendly oracle. Use the fortune tool when asked.",
        tools=[fortune],
    ),
)
