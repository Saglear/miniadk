"""10 — Slash-skills: dispatch /commands to canned playbooks.

A "skill" is a prompt + a set of tools the user can launch with
``/name``. Skills live as Markdown files or in code; the registry
matches the user's first token to a skill, expands the body, and
hands it to the model with the matching toolset.

The TUI's autocomplete picks them up automatically — type ``/`` and
they appear next to the built-in commands.

Run:

    uv run python examples/10_skills_router.py
"""

from miniadk import Agent, load_env_upwards, make_tools, run_cli, skill
from miniadk.skills import SkillRegistry

load_env_upwards()

review = skill(
    "review",
    body=(
        "Read $path and summarise the design choices. "
        "Mention anything that smells off."
    ),
    tools=["read_file"],
    args=["path"],
)

changelog = skill(
    "changelog",
    body=(
        "Read CHANGELOG.md and tell me what changed since the entry titled $since."
    ),
    tools=["read_file"],
    args=["since"],
)

run_cli(
    Agent(
        name="reviewer",
        instructions="You explain code and history. Use tools sparingly.",
        tools=make_tools(write=False, shell=False, web=False),
        skills=SkillRegistry.from_skills(review, changelog),
    ),
)
