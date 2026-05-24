"""08 — A custom run policy.

A ``RunPolicy`` decides what happens after each model + tool call.
The default policy stops as soon as the model answers without calling
a tool. Here we write a tiny variant that caps the loop at three tool
rounds — useful when you want to bound cost in CI.

This is the ADK's main extension point for "agent loop shape". If you
catch yourself wanting a Plan-and-Execute / ReAct / reflection
preset, write it as a RunPolicy and you're done.

Run:

    uv run python examples/08_custom_policy.py
"""

from miniadk import Agent, RunDecision, load_env_upwards, run, tool

load_env_upwards()


@tool
def echo(text: str) -> str:
    """Return ``text`` unchanged. Just a tool to drive the loop."""
    return text


class MaxToolRoundsPolicy:
    """Stop after ``limit`` tool rounds, even if the model keeps asking."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.tool_rounds = 0

    async def after_model(self, state):
        result = state.result
        # Default behaviour: stop if the model didn't request a tool.
        if result is not None and result.message is not None and not result.tool_calls:
            return RunDecision.stop(result.message)
        return RunDecision()

    async def after_tools(self, state):
        self.tool_rounds += 1
        if self.tool_rounds >= self.limit:
            return RunDecision.stop(
                f"hit max-tool-rounds cap of {self.limit}",
                reason="policy",
            )
        return RunDecision()


agent = Agent(
    name="bounded",
    instructions="Use the echo tool a few times, then summarise.",
    tools=[echo],
    policy=MaxToolRoundsPolicy(limit=3),
)

print(run(agent, "Echo 'a', 'b', 'c', 'd' in turn, then list what you echoed."))
