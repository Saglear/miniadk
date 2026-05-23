import asyncio
from importlib import import_module

import pytest

from miniadk import (
    Agent,
    Guard,
    Message,
    ModelResult,
    ScriptedModel,
    Session,
    Skill,
    SkillRegistry,
    TodoStore,
    ToolCall,
    agentic,
    arun,
    Compact,
    run,
    tool,
)
from miniadk.presets import coder


run_module = import_module("miniadk.run")


async def test_arun_returns_final_message_and_updates_session():
    session = Session()
    agent = Agent(name="helper", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="hello")])

    answer = await arun(agent, "hi", model=model, session=session)

    assert answer == "hello"
    assert session.messages == [
        Message("system", "Answer."),
        Message("user", "hi"),
        Message("assistant", "hello"),
    ]


async def test_arun_persists_session_to_path(tmp_path):
    path = tmp_path / "session.json"
    agent = Agent(name="helper", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="hello")])

    answer = await arun(agent, "hi", model=model, session=path)

    assert answer == "hello"
    assert Session.load(path).messages == [
        Message("system", "Answer."),
        Message("user", "hi"),
        Message("assistant", "hello"),
    ]


async def test_arun_resumes_session_from_path(tmp_path):
    path = tmp_path / "session.json"
    Session(
        [
            Message("system", "Answer."),
            Message("user", "old"),
            Message("assistant", "old answer"),
        ]
    ).save(path)
    agent = Agent(name="helper", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="new answer")])

    answer = await arun(agent, "again", model=model, session=path)

    assert answer == "new answer"
    messages, _ = model.calls[0]
    assert [message.content for message in messages] == [
        "Answer.",
        "old",
        "old answer",
        "again",
    ]
    assert Session.load(path).messages[-1].content == "new answer"


async def test_arun_can_auto_compact_session_after_run():
    session = Session()
    agent = Agent(name="helper", instructions="Answer.")
    model = ScriptedModel(
        [
            ModelResult(message="hello"),
            ModelResult(message="Conversation summary."),
        ]
    )

    answer = await arun(
        agent,
        "hi",
        model=model,
        session=session,
        compact=Compact(chars=1, keep=1),
    )

    assert answer == "hello"
    assert session.messages == [
        Message("system", "Answer."),
        Message("system", "Conversation summary."),
        Message("assistant", "hello"),
    ]
    assert "user: hi" in model.calls[-1][0][1].content


def test_run_returns_final_message_for_sync_scripts():
    agent = Agent(name="helper", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="hello")])

    assert run(agent, "hi", model=model) == "hello"


async def test_arun_can_use_default_model_helper(monkeypatch):
    built = ScriptedModel([ModelResult(message="hello")])
    monkeypatch.setattr(run_module, "default_model", lambda: built)

    answer = await arun(Agent(name="helper", instructions="Answer."), "hi")

    assert answer == "hello"
    assert built.calls


async def test_arun_resolves_agent_skills_by_default():
    registry = SkillRegistry(
        [Skill(name="review", description="Review.", body="Review $ARGUMENTS")]
    )
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[
                    ToolCall(
                        name="skill",
                        arguments={"skill": "review", "args": "app.py"},
                    )
                ]
            ),
            ModelResult(message="done"),
        ]
    )

    answer = await arun(
        Agent(name="helper", instructions="Use skills.", skills=registry),
        "review app.py",
        model=model,
    )

    assert answer == "done"
    messages, tools = model.calls[0]
    assert [tool.name for tool in tools] == ["skill"]
    assert "Available skills:" in messages[0].content


async def test_arun_can_skip_agent_resolution():
    registry = SkillRegistry(
        [Skill(name="review", description="Review.", body="Review $ARGUMENTS")]
    )
    model = ScriptedModel([ModelResult(message="plain")])

    answer = await arun(
        Agent(name="helper", instructions="Use skills.", skills=registry),
        "review app.py",
        model=model,
        resolve=False,
    )

    assert answer == "plain"
    messages, tools = model.calls[0]
    assert tools == []
    assert "Available skills:" not in messages[0].content


async def test_arun_accepts_agentic_preset_wrapper_policy():
    store = TodoStore()
    kit = agentic(Agent(name="helper", instructions="Use todos."), todos=store)
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[
                    ToolCall(
                        name="todo_write",
                        arguments={
                            "todos": [
                                {"content": "Work", "status": "in_progress"},
                            ]
                        },
                    )
                ]
            ),
            ModelResult(message="too early"),
            ModelResult(
                tool_calls=[
                    ToolCall(
                        name="todo_write",
                        arguments={
                            "todos": [
                                {"content": "Work", "status": "completed"},
                            ]
                        },
                    )
                ]
            ),
            ModelResult(message="done"),
        ]
    )

    answer = await arun(kit, "work", model=model)

    assert answer == "done"
    assert store.items == [{"content": "Work", "status": "completed"}]


async def test_arun_uses_agentic_preset_middleware(tmp_path):
    target = tmp_path / "notes.txt"
    kit = coder(tmp_path, shell=False, chat=False)
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[
                    ToolCall(
                        name="write_file",
                        arguments={"path": "notes.txt", "content": "hello"},
                    )
                ]
            ),
        ]
    )

    answer = await arun(kit, "write notes", model=model)

    assert answer == ""
    assert not target.exists()


async def test_arun_can_bind_guard_to_permission_callback():
    prompts = []

    @tool(destructive=True)
    def write_file(path: str) -> str:
        """Write a file."""
        return f"wrote {path}"

    agent = Agent(name="helper", instructions="Use tools.", tools=[write_file])
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[ToolCall(name="write_file", arguments={"path": "a.py"})]
            ),
            ModelResult(message="done"),
        ]
    )

    answer = await arun(
        agent,
        "write",
        model=model,
        middleware=[Guard("ask")],
        ask_user=lambda request: prompts.append((request.tool.name, request.reason)) or True,
    )

    assert answer == "done"
    assert prompts == [("write_file", "destructive tool use")]


async def test_arun_does_not_mutate_shared_guard_when_binding_permission_callback():
    prompts = []

    @tool(destructive=True)
    def write_file(path: str) -> str:
        """Write a file."""
        return f"wrote {path}"

    agent = Agent(name="helper", instructions="Use tools.", tools=[write_file])
    guard = Guard("ask")

    async def run_once(label: str):
        model = ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="write_file", arguments={"path": f"{label}.py"})
                    ]
                ),
                ModelResult(message="done"),
            ]
        )
        return await arun(
            agent,
            "write",
            model=model,
            middleware=[guard],
            ask_user=lambda request: prompts.append((label, request.arguments["path"])) or True,
        )

    assert await run_once("first") == "done"
    assert await run_once("second") == "done"
    assert guard.ask_user is None
    assert prompts == [("first", "first.py"), ("second", "second.py")]


async def test_arun_can_add_user_middleware_after_preset_middleware():
    seen = []

    class Audit:
        async def before_model_call(self, state):
            seen.append(("model", state.step))

    kit = agentic(
        Agent(name="helper", instructions="Answer."),
        middleware=[Audit()],
    )
    model = ScriptedModel([ModelResult(message="hello")])

    answer = await arun(kit, "hi", model=model)

    assert answer == "hello"
    assert seen == [("model", 1)]


def test_run_can_use_default_model_helper(monkeypatch):
    built = ScriptedModel([ModelResult(message="hello")])
    monkeypatch.setattr(run_module, "default_model", lambda: built)

    answer = run(Agent(name="helper", instructions="Answer."), "hi")

    assert answer == "hello"
    assert built.calls


def test_run_resolves_agent_skills_by_default():
    registry = SkillRegistry(
        [Skill(name="review", description="Review.", body="Review $ARGUMENTS")]
    )
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[
                    ToolCall(
                        name="skill",
                        arguments={"skill": "review", "args": "app.py"},
                    )
                ]
            ),
            ModelResult(message="done"),
        ]
    )

    answer = run(
        Agent(name="helper", instructions="Use skills.", skills=registry),
        "review app.py",
        model=model,
    )

    assert answer == "done"
    messages, tools = model.calls[0]
    assert [tool.name for tool in tools] == ["skill"]
    assert "Available skills:" in messages[0].content


def test_run_accepts_agentic_preset_wrapper_policy():
    store = TodoStore()
    kit = agentic(Agent(name="helper", instructions="Use todos."), todos=store)
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[
                    ToolCall(
                        name="todo_write",
                        arguments={
                            "todos": [
                                {"content": "Work", "status": "completed"},
                            ]
                        },
                    )
                ]
            ),
            ModelResult(message="done"),
        ]
    )

    answer = run(kit, "work", model=model)

    assert answer == "done"
    assert store.items == [{"content": "Work", "status": "completed"}]


def test_run_can_bind_guard_to_permission_callback():
    prompts = []

    @tool(destructive=True)
    def write_file(path: str) -> str:
        """Write a file."""
        return f"wrote {path}"

    agent = Agent(name="helper", instructions="Use tools.", tools=[write_file])
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[ToolCall(name="write_file", arguments={"path": "a.py"})]
            ),
            ModelResult(message="done"),
        ]
    )

    answer = run(
        agent,
        "write",
        model=model,
        middleware=[Guard("ask")],
        ask_user=lambda request: prompts.append(request.tool.name) or True,
    )

    assert answer == "done"
    assert prompts == ["write_file"]


def test_run_can_use_default_session_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = Agent(name="helper", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="hello")])

    answer = run(agent, "hi", model=model, session=True)

    assert answer == "hello"
    loaded = Session.load(tmp_path / ".miniadk" / "sessions" / "helper.json")
    assert loaded.messages[-1].content == "hello"


async def test_arun_accepts_temporary_tools():
    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    model = ScriptedModel(
        [
            ModelResult(tool_calls=[ToolCall(name="ping", arguments={})]),
            ModelResult(message="done"),
        ]
    )
    agent = Agent(name="helper", instructions="Use tools.")

    answer = await arun(agent, "ping", model=model, tools=[ping])

    assert answer == "done"
    assert model.calls[0][1] == [ping]


async def test_arun_merges_resolved_agent_tools_with_temporary_tools():
    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    registry = SkillRegistry(
        [Skill(name="review", description="Review.", body="Review $ARGUMENTS")]
    )
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[
                    ToolCall(
                        name="skill",
                        arguments={"skill": "review", "args": "app.py"},
                    )
                ]
            ),
            ModelResult(tool_calls=[ToolCall(name="ping", arguments={})]),
            ModelResult(message="done"),
        ]
    )

    answer = await arun(
        Agent(name="helper", instructions="Use skills.", skills=registry),
        "review",
        model=model,
        tools=[ping],
    )

    assert answer == "done"
    assert [tool.name for tool in model.calls[0][1]] == ["skill", "ping"]


async def test_arun_temporary_tools_override_agent_tools_by_name():
    @tool
    def child_check(path: str) -> str:
        """Check a path."""
        return f"child:{path}"

    @tool
    def extra_check(path: str) -> str:
        """Check a path."""
        return f"extra:{path}"

    child_check.name = "check"
    extra_check.name = "check"
    model = ScriptedModel(
        [
            ModelResult(tool_calls=[ToolCall(name="check", arguments={"path": "app.py"})]),
            ModelResult(message="done"),
        ]
    )

    answer = await arun(
        Agent(name="helper", instructions="Use tools.", tools=[child_check]),
        "check",
        model=model,
        tools=[extra_check],
    )

    assert answer == "done"
    assert [tool.name for tool in model.calls[0][1]] == ["check"]
    assert model.calls[1][0][-1].content == "extra:app.py"


async def test_arun_cancels_running_tool_and_saves_session(tmp_path):
    started = asyncio.Event()
    cancelled = asyncio.Event()
    path = tmp_path / "session.json"

    @tool
    async def wait(progress) -> str:
        """Wait until cancelled."""
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "done"

    agent = Agent(name="helper", instructions="Use tools.", tools=[wait])
    model = ScriptedModel(
        [ModelResult(tool_calls=[ToolCall(name="wait", arguments={})])]
    )
    task = asyncio.create_task(arun(agent, "wait", model=model, session=path))

    await asyncio.wait_for(started.wait(), timeout=0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(cancelled.wait(), timeout=0.2)
    assert Session.load(path).messages[-1].tool_calls[0].name == "wait"


def test_run_accepts_temporary_tools():
    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    model = ScriptedModel(
        [
            ModelResult(tool_calls=[ToolCall(name="ping", arguments={})]),
            ModelResult(message="done"),
        ]
    )
    agent = Agent(name="helper", instructions="Use tools.")

    assert run(agent, "ping", model=model, tools=[ping]) == "done"
    assert model.calls[0][1] == [ping]
