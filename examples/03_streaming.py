"""03 — Streaming events from the runtime.

``run()`` and ``arun()`` are convenience wrappers that drain the
runtime and return the final answer. For richer UIs (or just to see
what the model is doing) you iterate over events directly.

Each Event has a ``type`` string (``message_delta``, ``tool_call``,
``tool_result``, ``message``, …) and a ``data`` dict. We render a few
of them here.

Run:

    uv run python examples/03_streaming.py
"""

import asyncio

from miniadk import Agent, Runtime, load_env_upwards, model, tool

load_env_upwards()


@tool
def square(n: int) -> int:
    """Square ``n``."""
    return n * n


async def main() -> None:
    agent = Agent(
        name="streamer",
        instructions="Use the tool. Then explain the answer in one sentence.",
        tools=[square],
    )
    rt = Runtime(agent=agent, model=model())

    async for event in rt.run("What is 13 squared?"):
        if event.type == "message_delta":
            # Stream tokens as they arrive.
            print(event.data["text"], end="", flush=True)
        elif event.type == "tool_call":
            print(f"\n[tool] {event.data['name']}({event.data['arguments']})")
        elif event.type == "tool_result":
            print(f"[result] {event.data['text']}")
        elif event.type == "message":
            print()  # final newline after stream

asyncio.run(main())
