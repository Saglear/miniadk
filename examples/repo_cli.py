from __future__ import annotations

from pathlib import Path

from miniadk import (
    Agent,
    AgenticPolicy,
    CLITheme,
    SkillRegistry,
    TodoStore,
    load_env_upwards,
    make_todo_tool,
    model,
    run_cli,
    with_agentic_instructions,
)
from miniadk.stdtools import (
    make_edit_file,
    make_glob_files,
    make_list_files,
    make_read_file,
    make_search_text,
    make_shell,
    make_write_file,
)


def main() -> None:
    load_env_upwards(start=Path(__file__).parent)

    workspace = Path.cwd()
    skills_dir = Path(__file__).with_name("repo_cli_skills")
    todo_store = TodoStore()

    agent = Agent(
        name="repo-cli",
        instructions=with_agentic_instructions(
            "You are a local repository assistant. "
            "Read before writing, inspect context before editing, and search when needed. "
            "If the user enters /skill name, treat it as a request to use that skill."
        ),
        tools=[
            make_todo_tool(todo_store),
            make_read_file(root=workspace),
            make_write_file(root=workspace),
            make_edit_file(root=workspace),
            make_list_files(root=workspace),
            make_glob_files(root=workspace),
            make_search_text(root=workspace),
            make_shell(cwd=workspace),
        ],
        skills=SkillRegistry.from_paths(skills_dir),
    )

    run_cli(
        agent,
        model=model(),
        prompt="repo > ",
        output_mode="pretty",
        theme=CLITheme(name="repo"),
        policy=AgenticPolicy(todo_store=todo_store),
    )


if __name__ == "__main__":
    main()
