from __future__ import annotations

from pathlib import Path

from miniadk import Agent, run_cli
from miniadk.presets import coder


def build(root: str | Path = "."):
    reviewer = Agent("reviewer", "Review code changes and point out risks.")
    return coder(
        root,
        agents=[reviewer],
        work=True,
        skills=Path(__file__).with_name("repo_cli_skills"),
    )


def main() -> None:
    run_cli(build("."), session=True)


if __name__ == "__main__":
    main()
