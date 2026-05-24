"""02 — Adding a tool.

A tool is just a Python function with a docstring. Decorate it with
``@tool`` and pass it to the agent. The model will call it on its own
when the question warrants it.

Run:

    uv run python examples/02_with_a_tool.py
"""

from datetime import datetime, UTC

from miniadk import Agent, load_env_upwards, run, tool

load_env_upwards()


@tool
def now_utc() -> str:
    """Return the current UTC time as ISO-8601."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@tool
def add(a: int, b: int) -> int:
    """Return ``a + b``."""
    return a + b


agent = Agent(
    name="time-and-math",
    instructions="Use the tools when they help. Be terse.",
    tools=[now_utc, add],
)

print(run(agent, "What's the current UTC time, and what is 17 + 25?"))
