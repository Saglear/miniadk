"""04 — Sessions: persisting a conversation.

Pass ``session=<name>`` to ``run`` and the message history is stored
on disk under ``.miniadk/sessions/<name>.json``. The next call to
``run`` with the same session resumes where you left off.

``compact=...`` controls automatic summarisation when the session
grows large; pass ``compact={"max_messages": 50}`` to roll up older
turns into a single summary.

Run twice — the second run sees what you asked first.

    uv run python examples/04_session_and_compaction.py
"""

from miniadk import Agent, load_env_upwards, run

load_env_upwards()

agent = Agent(
    name="memo",
    instructions="You remember what the user told you in earlier turns.",
)

# Two turns sharing one session. In real code you'd interleave them
# with user input; here we stack them so a single run shows the effect.
print(">>", run(agent, "Remember: my favourite colour is teal.", session="memo-demo"))
print(">>", run(agent, "What did I say my favourite colour was?", session="memo-demo"))
