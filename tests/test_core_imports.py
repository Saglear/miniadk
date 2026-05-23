from miniadk.core import (
    Agent,
    Event,
    Guard,
    Message,
    ModelResult,
    RunDecision,
    RunState,
    Runtime,
    Session,
    SessionStats,
    ScriptedModel,
    ToolCall,
    ToolValidation,
    arun,
    as_tool,
    run,
    tool,
)


def test_core_exports_atomic_agent_building_blocks():
    @tool
    def greet(name: str) -> str:
        """Greet a person."""
        return f"hello {name}"

    agent = Agent(name="core", instructions="Answer.", tools=[greet])
    runtime = Runtime(agent=agent, model=ScriptedModel([ModelResult(message="ok")]))

    assert runtime.messages == [Message("system", "Answer.")]
    assert Event("message", {"text": "ok"}).type == "message"
    assert Guard("trust").mode == "trust"
    assert Session(runtime.messages).messages[0].role == "system"
    assert SessionStats(messages=1).messages == 1
    assert ToolCall(name="greet", arguments={"name": "Ada"}).name == "greet"
    assert ToolValidation.allow().ok is True
    assert as_tool(agent, model=ScriptedModel([ModelResult(message="ok")])).name == "core"
    assert callable(arun)
    assert callable(run)
    assert RunDecision.stop("done").action == "stop"
    assert RunState(1, 1, runtime.messages, [greet]).step == 1


def test_agent_copy_can_extend_tools_without_mutating_original():
    @tool
    def read() -> str:
        """Read."""
        return "read"

    @tool
    def write() -> str:
        """Write."""
        return "write"

    base = Agent(
        name="base",
        instructions="Answer.",
        tools=[read],
        skills={"review": "skill"},
        mcp={"server": "local"},
    )

    built = base.copy(name="coder", extra=[write])

    assert built.name == "coder"
    assert built.instructions == "Answer."
    assert [item.name for item in built.tools] == ["read", "write"]
    assert [item.name for item in base.tools] == ["read"]
    assert built.skills is base.skills
    assert built.mcp is base.mcp


def test_agent_copy_can_replace_fields_and_clear_integrations():
    @tool
    def read() -> str:
        """Read."""
        return "read"

    base = Agent(
        name="base",
        instructions="Answer.",
        tools=[read],
        skills={"review": "skill"},
        mcp={"server": "local"},
    )

    built = base.copy(
        instructions="Review.",
        tools=[],
        skills=None,
        mcp=None,
    )

    assert built.name == "base"
    assert built.instructions == "Review."
    assert built.tools == []
    assert built.skills is None
    assert built.mcp is None
