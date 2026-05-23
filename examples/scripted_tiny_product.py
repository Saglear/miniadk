from miniadk import Agent, ModelResult, ScriptedModel, run_cli, tool


@tool
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


def main() -> None:
    agent = Agent(
        name="calculator",
        instructions="You are a tiny calculator product.",
        tools=[add],
    )
    model = ScriptedModel([ModelResult(message="This scripted demo is alive.")])
    run_cli(agent, model=model)


if __name__ == "__main__":
    main()

