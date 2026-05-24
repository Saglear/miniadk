"""01 — Hello, agent.

The smallest possible MiniADK program. Build an Agent, give it
instructions, and ask it something. The default model is read from
your environment (ANTHROPIC_API_KEY, OPENAI_API_KEY, …).

Run:

    uv run python examples/01_hello_agent.py
"""

from miniadk import Agent, load_env_upwards, run

load_env_upwards()  # picks up a nearby .env, no-op if none

agent = Agent(
    name="hello",
    instructions="You greet the user in the language they used.",
)

print(run(agent, "你好，请用中文打个招呼"))
