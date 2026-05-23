import json
import asyncio
from dataclasses import dataclass

import miniadk.adapters.json as json_adapter
from miniadk.adapters.json import astream_runtime
from miniadk import (
    Agent,
    Compact,
    Guard,
    Message,
    ModelResult,
    ModelStreamEvent,
    ScriptedModel,
    Session,
    Skill,
    SkillRegistry,
    ToolCall,
    ToolCallDelta,
    astream_json,
    event_dict,
    jsonl,
    tool,
    Runtime,
)
from miniadk.core import Event


@dataclass(slots=True)
class Box:
    value: str


class StreamingModel:
    async def complete(self, messages, tools):
        raise AssertionError("stream should be used")

    async def stream(self, messages, tools):
        yield ModelStreamEvent(thinking="checking")
        yield ModelStreamEvent(
            tool_call=ToolCallDelta(index=0, id="call_1", name="greet", arguments='{"')
        )
        yield ModelStreamEvent(delta="hi")
        yield ModelStreamEvent(result=ModelResult(message="hi"))


async def test_astream_json_yields_runtime_events_as_dicts():
    @tool
    def echo(text: str) -> str:
        """Echo text."""
        return text

    agent = Agent(name="api", instructions="Answer.", tools=[echo])
    model = ScriptedModel(
        [
            ModelResult(tool_calls=[ToolCall(name="echo", arguments={"text": "ok"})]),
            ModelResult(message="done"),
        ]
    )

    events = [
        event
        async for event in astream_json(agent, "hello", model=model, resolve=False)
    ]

    assert events == [
        {"type": "tool_call", "data": {"name": "echo", "arguments": {"text": "ok"}}},
        {
            "type": "tool_result",
            "data": {"name": "echo", "result": "ok", "text": "ok"},
        },
        {"type": "message", "data": {"text": "done"}},
    ]


async def test_astream_json_merges_resolved_agent_tools_with_extra_tools():
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

    events = [
        event
        async for event in astream_json(
            Agent(name="api", instructions="Use skills.", skills=registry),
            "review",
            model=model,
            tools=[ping],
        )
    ]

    assert [tool.name for tool in model.calls[0][1]] == ["skill", "ping"]
    assert [event["type"] for event in events] == [
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "message",
    ]


async def test_astream_json_serializes_tool_error_events():
    @tool
    def fail() -> str:
        """Fail."""
        raise ValueError("bad tool")

    agent = Agent(name="api", instructions="Use tools.", tools=[fail])
    model = ScriptedModel(
        [ModelResult(tool_calls=[ToolCall(name="fail", arguments={})])]
    )

    events = [
        event
        async for event in astream_json(agent, "fail", model=model, resolve=False)
    ]

    assert events == [
        {"type": "tool_call", "data": {"name": "fail", "arguments": {}}},
        {
            "type": "tool_error",
            "data": {
                "tool": "fail",
                "message": "ValueError: bad tool",
            },
        },
    ]


async def test_astream_json_includes_thinking_delta_events():
    events = [
        event
        async for event in astream_json(
            Agent(name="api", instructions="Answer."),
            "hi",
            model=StreamingModel(),
            resolve=False,
        )
    ]

    assert events == [
        {"type": "thinking_delta", "data": {"text": "checking"}},
        {
            "type": "tool_call_delta",
            "data": {
                "index": 0,
                "id": "call_1",
                "name": "greet",
                "arguments": '{"',
            },
        },
        {"type": "message_delta", "data": {"text": "hi"}},
        {"type": "message", "data": {"text": "hi", "streamed": True}},
    ]


async def test_astream_json_can_include_lifecycle_events():
    agent = Agent(name="api", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="hello")])

    events = [
        event
        async for event in astream_json(
            agent,
            "hi",
            model=model,
            lifecycle=True,
            resolve=False,
        )
    ]

    assert events == [
        {"type": "run_start", "data": {"agent": "api", "input": "hi"}},
        {"type": "message", "data": {"text": "hello"}},
        {
            "type": "run_end",
            "data": {
                "agent": "api",
                "status": "completed",
                "reason": "completed",
                "messages": 3,
            },
        },
    ]


async def test_astream_json_can_include_trace_metadata():
    @tool
    def echo(text: str) -> str:
        """Echo text."""
        return text

    agent = Agent(name="api", instructions="Answer.", tools=[echo])
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[
                    ToolCall(id="call_1", name="echo", arguments={"text": "ok"})
                ]
            ),
            ModelResult(message="done"),
        ]
    )

    events = [
        event
        async for event in astream_json(
            agent,
            "hi",
            model=model,
            lifecycle=True,
            trace=True,
            resolve=False,
        )
    ]

    run_id = events[0]["data"]["run_id"]
    assert run_id
    assert [event["type"] for event in events] == [
        "run_start",
        "tool_call",
        "tool_result",
        "message",
        "run_end",
    ]
    assert all(event["data"]["run_id"] == run_id for event in events)
    assert events[1]["data"]["step"] == 1
    assert events[1]["data"]["tool_call_id"] == "call_1"
    assert events[2]["data"]["tool_call_id"] == "call_1"
    assert events[3]["data"]["step"] == 2


async def test_astream_json_includes_tool_progress_events():
    @tool
    async def build(path: str, progress) -> str:
        """Build a path."""
        await progress("started", step=1)
        return f"built {path}"

    agent = Agent(name="api", instructions="Use tools.", tools=[build])
    model = ScriptedModel(
        [
            ModelResult(tool_calls=[ToolCall(name="build", arguments={"path": "app.py"})]),
            ModelResult(message="done"),
        ]
    )

    events = [
        event
        async for event in astream_json(agent, "build", model=model, resolve=False)
    ]

    assert events == [
        {"type": "tool_call", "data": {"name": "build", "arguments": {"path": "app.py"}}},
        {
            "type": "tool_progress",
            "data": {
                "tool": "build",
                "message": "started",
                "data": {"step": 1},
            },
        },
        {
            "type": "tool_result",
            "data": {"name": "build", "result": "built app.py", "text": "built app.py"},
        },
        {"type": "message", "data": {"text": "done"}},
    ]


async def test_astream_json_can_bind_guard_to_permission_callback():
    prompts = []

    @tool(destructive=True)
    def write_file(path: str) -> str:
        """Write a file."""
        return f"wrote {path}"

    agent = Agent(name="api", instructions="Use tools.", tools=[write_file])
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[ToolCall(name="write_file", arguments={"path": "a.py"})]
            ),
            ModelResult(message="done"),
        ]
    )

    events = [
        event
        async for event in astream_json(
            agent,
            "write",
            model=model,
            middleware=[Guard("ask")],
            ask_user=lambda request: prompts.append((request.tool.name, request.reason)) or True,
            resolve=False,
        )
    ]

    assert prompts == [("write_file", "destructive tool use")]
    assert [event["type"] for event in events] == [
        "permission_request",
        "tool_call",
        "tool_result",
        "message",
    ]
    assert events[2]["data"]["text"] == "wrote a.py"


async def test_astream_json_can_route_ask_before_to_permission_callback():
    prompts = []

    @tool(destructive=True)
    def write_file(path: str) -> str:
        """Write a file."""
        return f"wrote {path}"

    agent = Agent(name="api", instructions="Use tools.", tools=[write_file])
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[ToolCall(name="write_file", arguments={"path": "a.py"})]
            ),
            ModelResult(message="done"),
        ]
    )

    events = [
        event
        async for event in astream_json(
            agent,
            "write",
            model=model,
            middleware=[Guard("ask")],
            ask_user=lambda request: prompts.append(request.arguments["path"]) or True,
            resolve=False,
        )
    ]

    assert prompts == ["a.py"]
    assert [event["type"] for event in events] == [
        "permission_request",
        "tool_call",
        "tool_result",
        "message",
    ]


async def test_astream_json_does_not_mutate_shared_guard_when_binding_permission_callback():
    prompts = []

    @tool(destructive=True)
    def write_file(path: str) -> str:
        """Write a file."""
        return f"wrote {path}"

    agent = Agent(name="api", instructions="Use tools.", tools=[write_file])
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
        return [
            event
            async for event in astream_json(
                agent,
                "write",
                model=model,
                middleware=[guard],
                ask_user=lambda request: prompts.append((label, request.arguments["path"])) or True,
                resolve=False,
            )
        ]

    first = await run_once("first")
    second = await run_once("second")

    assert guard.ask_user is None
    assert prompts == [("first", "first.py"), ("second", "second.py")]
    assert first[2]["data"]["text"] == "wrote first.py"
    assert second[2]["data"]["text"] == "wrote second.py"


def test_jsonl_returns_json_lines_for_sync_adapters():
    agent = Agent(name="api", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="hello")])

    lines = list(jsonl(agent, "hi", model=model, resolve=False))

    assert [json.loads(line) for line in lines] == [
        {"type": "message", "data": {"text": "hello"}},
    ]


def test_jsonl_can_bind_guard_to_permission_callback():
    prompts = []

    @tool(destructive=True)
    def write_file(path: str) -> str:
        """Write a file."""
        return f"wrote {path}"

    agent = Agent(name="api", instructions="Use tools.", tools=[write_file])
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[ToolCall(name="write_file", arguments={"path": "a.py"})]
            ),
            ModelResult(message="done"),
        ]
    )

    lines = list(
        jsonl(
            agent,
            "write",
            model=model,
            middleware=[Guard("ask")],
            ask_user=lambda request: prompts.append(request.tool.name) or True,
            resolve=False,
        )
    )

    assert prompts == ["write_file"]
    assert [json.loads(line)["type"] for line in lines] == [
        "permission_request",
        "tool_call",
        "tool_result",
        "message",
    ]


async def test_astream_json_can_use_default_model_helper(monkeypatch):
    built = ScriptedModel([ModelResult(message="hello")])
    monkeypatch.setattr(json_adapter, "default_model", lambda: built)

    events = [
        event
        async for event in astream_json(
            Agent(name="api", instructions="Answer."),
            "hi",
            resolve=False,
        )
    ]

    assert events == [{"type": "message", "data": {"text": "hello"}}]
    assert built.calls


def test_jsonl_can_use_default_model_helper(monkeypatch):
    built = ScriptedModel([ModelResult(message="hello")])
    monkeypatch.setattr(json_adapter, "default_model", lambda: built)

    lines = list(jsonl(Agent(name="api", instructions="Answer."), "hi", resolve=False))

    assert [json.loads(line) for line in lines] == [
        {"type": "message", "data": {"text": "hello"}},
    ]
    assert built.calls


async def test_astream_json_updates_supplied_session():
    session = Session()
    agent = Agent(name="api", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="hello")])

    events = [
        event
        async for event in astream_json(
            agent,
            "hi",
            model=model,
            session=session,
            resolve=False,
        )
    ]

    assert events == [{"type": "message", "data": {"text": "hello"}}]
    assert session.messages == [
        Message("system", "Answer."),
        Message("user", "hi"),
        Message("assistant", "hello"),
    ]


async def test_astream_json_persists_session_to_path(tmp_path):
    path = tmp_path / "session.json"
    agent = Agent(name="api", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="hello")])

    events = [
        event
        async for event in astream_json(
            agent,
            "hi",
            model=model,
            session=path,
            resolve=False,
        )
    ]

    assert events == [{"type": "message", "data": {"text": "hello"}}]
    assert Session.load(path).messages == [
        Message("system", "Answer."),
        Message("user", "hi"),
        Message("assistant", "hello"),
    ]


async def test_astream_json_resumes_session_from_path(tmp_path):
    path = tmp_path / "session.json"
    Session(
        [
            Message("system", "Answer."),
            Message("user", "old"),
            Message("assistant", "old answer"),
        ]
    ).save(path)
    agent = Agent(name="api", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="new answer")])

    events = [
        event
        async for event in astream_json(
            agent,
            "again",
            model=model,
            session=path,
            resolve=False,
        )
    ]

    assert events == [{"type": "message", "data": {"text": "new answer"}}]
    messages, _ = model.calls[0]
    assert [message.content for message in messages] == [
        "Answer.",
        "old",
        "old answer",
        "again",
    ]
    assert Session.load(path).messages[-1].content == "new answer"


async def test_astream_json_can_use_default_session_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = Agent(name="api", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="hello")])

    events = [
        event
        async for event in astream_json(
            agent,
            "hi",
            model=model,
            session=True,
            resolve=False,
        )
    ]

    assert events == [{"type": "message", "data": {"text": "hello"}}]
    loaded = Session.load(tmp_path / ".miniadk" / "sessions" / "api.json")
    assert loaded.messages[-1].content == "hello"


async def test_astream_json_can_auto_compact_session():
    session = Session()
    agent = Agent(name="api", instructions="Answer.")
    model = ScriptedModel(
        [
            ModelResult(message="hello"),
            ModelResult(message="Conversation summary."),
        ]
    )

    events = [
        event
        async for event in astream_json(
            agent,
            "hi",
            model=model,
            session=session,
            compact=Compact(chars=1, keep=1),
            resolve=False,
        )
    ]

    assert events == [{"type": "message", "data": {"text": "hello"}}]
    assert session.messages == [
        Message("system", "Answer."),
        Message("system", "Conversation summary."),
        Message("assistant", "hello"),
    ]


async def test_astream_runtime_streams_existing_runtime_as_json():
    runtime = Runtime(
        agent=Agent(name="api", instructions="Answer."),
        model=ScriptedModel([ModelResult(message="hello")]),
    )

    events = [
        event
        async for event in astream_runtime(runtime, "hi", lifecycle=True)
    ]

    assert events == [
        {"type": "run_start", "data": {"agent": "api", "input": "hi"}},
        {"type": "message", "data": {"text": "hello"}},
        {
            "type": "run_end",
            "data": {
                "agent": "api",
                "status": "completed",
                "reason": "completed",
                "messages": 3,
            },
        },
    ]


async def test_astream_json_cancels_active_run_when_stream_is_closed():
    cancelled = asyncio.Event()

    @tool
    async def work(progress) -> str:
        """Work until cancelled."""
        await progress("started")
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "done"

    agent = Agent(name="api", instructions="Use tools.", tools=[work])
    model = ScriptedModel(
        [ModelResult(tool_calls=[ToolCall(name="work", arguments={})])]
    )
    stream = astream_json(agent, "work", model=model, resolve=False)

    assert await stream.__anext__() == {
        "type": "tool_call",
        "data": {"name": "work", "arguments": {}},
    }
    assert await stream.__anext__() == {
        "type": "tool_progress",
        "data": {"tool": "work", "message": "started"},
    }

    await stream.aclose()

    await asyncio.wait_for(cancelled.wait(), timeout=0.2)


def test_event_dict_converts_non_json_values():
    assert event_dict(Event("tool_result", {"result": Box("ok")})) == {
        "type": "tool_result",
        "data": {"result": {"value": "ok"}},
    }
