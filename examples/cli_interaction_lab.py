"""Maintained playground for the latest MiniADK CLI interaction."""

from __future__ import annotations

from pathlib import Path

from miniadk import (
    Agent,
    CLITheme,
    SkillRegistry,
    load_env_upwards,
    model,
    run_cli,
)
from miniadk.stdtools import make_read_file, make_search_text, make_shell


def build_agent(workspace: Path) -> Agent:
    return Agent(
        name="cli-lab",
        instructions="Help the user try MiniADK's latest CLI interaction.",
        tools=[
            make_read_file(root=workspace),
            make_search_text(root=workspace),
            make_shell(cwd=workspace),
        ],
        skills=SkillRegistry.from_paths(Path(__file__).with_name("repo_cli_skills")),
    )


def main() -> None:
    load_env_upwards(start=Path(__file__).parent)
    run_cli(
        build_agent(Path.cwd()),
        model=model(),
        prompt="mini > ",
        output_mode="pretty",
        theme=CLITheme(name="miniadk"),
    )


if __name__ == "__main__":
    main()
