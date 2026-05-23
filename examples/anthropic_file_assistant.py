from pathlib import Path
import os

from miniadk import Agent, AnthropicModel, load_env_upwards, run_cli, CLITheme
from miniadk.stdtools import make_list_files, make_read_file, make_shell, make_write_file


def build_agent(workspace: Path) -> Agent:
    return Agent(
        name="file-assistant",
        instructions=(
            "You are a careful local file assistant. "
            "Read before writing. Explain changes briefly. "
            "Use shell only when it is clearly useful."
        ),
        tools=[
            make_read_file(root=workspace),
            make_write_file(root=workspace),
            make_list_files(root=workspace),
            make_shell(cwd=workspace),
        ],
    )


def main() -> None:
    load_env_upwards(start=Path(__file__).parent)

    agent = build_agent(Path.cwd())
    model = AnthropicModel(model=os.getenv("ANTHROPIC_MODEL"))
    run_cli(agent, model=model, output_mode="pretty", theme=CLITheme(name="repo"))


if __name__ == "__main__":
    main()
