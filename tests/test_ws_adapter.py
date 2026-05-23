import json
import asyncio

import pytest

import miniadk.adapters.json as json_adapter
from miniadk import (
    Agent,
    Compact,
    Guard,
    Message,
    ModelResult,
    ScriptedModel,
    Session,
    Skill,
    SkillRegistry,
    ToolCall,
    tool,
    ws_json,
)


class JsonSocket:
    def __init__(self):
        self.events = []

    async def send_json(self, event):
        self.events.append(event)


class TextSocket:
    def __init__(self):
        self.messages = []

    async def send(self, text):
        self.messages.append(text)


class FailingJsonSocket:
    def __init__(self, fail_at: int):
        self.fail_at = fail_at
        self.events = []

    async def send_json(self, event):
        self.events.append(event)
        if len(self.events) == self.fail_at:
            raise RuntimeError("websocket closed")


class CancellingJsonSocket:
    def __init__(self, cancel_at: int):
        self.cancel_at = cancel_at
        self.events = []

    async def send_json(self, event):
        self.events.append(event)
        if len(self.events) == self.cancel_at:
            raise asyncio.CancelledError


async def test_ws_json_sends_runtime_events_to_send_json():
    @tool
    def echo(text: str) -> str:
        """Echo text."""
        return text

    agent = Agent(name="ws", instructions="Answer.", tools=[echo])
    model = ScriptedModel(
        [
            ModelResult(tool_calls=[ToolCall(name="echo", arguments={"text": "ok"})]),
            ModelResult(message="done"),
        ]
    )
    ws = JsonSocket()

    count = await ws_json(ws, agent, "hello", model=model, resolve=False)

    assert count == 3
    assert ws.events == [
        {"type": "tool_call", "data": {"name": "echo", "arguments": {"text": "ok"}}},
        {
            "type": "tool_result",
            "data": {"name": "echo", "result": "ok", "text": "ok"},
        },
        {"type": "message", "data": {"text": "done"}},
    ]


async def test_ws_json_merges_resolved_agent_tools_with_extra_tools():
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
    ws = JsonSocket()

    await ws_json(
        ws,
        Agent(name="ws", instructions="Use skills.", skills=registry),
        "review",
        model=model,
        tools=[ping],
    )

    assert [tool.name for tool in model.calls[0][1]] == ["skill", "ping"]
    assert [event["type"] for event in ws.events] == [
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "message",
    ]


async def test_ws_json_can_include_lifecycle_events():
    agent = Agent(name="ws", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="hello")])
    ws = JsonSocket()

    count = await ws_json(
        ws,
        agent,
        "hi",
        model=model,
        lifecycle=True,
        resolve=False,
    )

    assert count == 3
    assert ws.events == [
        {"type": "run_start", "data": {"agent": "ws", "input": "hi"}},
        {"type": "message", "data": {"text": "hello"}},
        {
            "type": "run_end",
            "data": {
                "agent": "ws",
                "status": "completed",
                "reason": "completed",
                "messages": 3,
            },
        },
    ]


async def test_ws_json_sends_tool_progress_events():
    @tool
    async def build(path: str, progress) -> str:
        """Build a path."""
        await progress("started", step=1)
        return f"built {path}"

    agent = Agent(name="ws", instructions="Use tools.", tools=[build])
    model = ScriptedModel(
        [
            ModelResult(tool_calls=[ToolCall(name="build", arguments={"path": "app.py"})]),
            ModelResult(message="done"),
        ]
    )
    ws = JsonSocket()

    count = await ws_json(ws, agent, "build", model=model, resolve=False)

    assert count == 4
    assert ws.events == [
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


async def test_ws_json_can_send_json_lines_to_plain_send():
    agent = Agent(name="ws", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="你好")])
    ws = TextSocket()

    count = await ws_json(ws, agent, "hi", model=model, resolve=False)

    assert count == 1
    assert [json.loads(message) for message in ws.messages] == [
        {"type": "message", "data": {"text": "你好"}},
    ]


async def test_ws_json_can_use_default_model_helper(monkeypatch):
    built = ScriptedModel([ModelResult(message="hello")])
    monkeypatch.setattr(json_adapter, "default_model", lambda: built)
    ws = JsonSocket()

    count = await ws_json(
        ws,
        Agent(name="ws", instructions="Answer."),
        "hi",
        resolve=False,
    )

    assert count == 1
    assert ws.events == [{"type": "message", "data": {"text": "hello"}}]
    assert built.calls


async def test_ws_json_can_bind_guard_to_permission_callback():
    prompts = []

    @tool(destructive=True)
    def write_file(path: str) -> str:
        """Write a file."""
        return f"wrote {path}"

    agent = Agent(name="ws", instructions="Use tools.", tools=[write_file])
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[ToolCall(name="write_file", arguments={"path": "a.py"})]
            ),
            ModelResult(message="done"),
        ]
    )
    ws = JsonSocket()

    count = await ws_json(
        ws,
        agent,
        "write",
        model=model,
        middleware=[Guard("ask")],
        ask_user=lambda request: prompts.append((request.tool.name, request.reason)) or True,
        resolve=False,
    )

    assert count == 4
    assert prompts == [("write_file", "destructive tool use")]
    assert [event["type"] for event in ws.events] == [
        "permission_request",
        "tool_call",
        "tool_result",
        "message",
    ]
    assert ws.events[2]["data"]["text"] == "wrote a.py"


async def test_ws_json_does_not_mutate_shared_guard_when_binding_permission_callback():
    prompts = []

    @tool(destructive=True)
    def write_file(path: str) -> str:
        """Write a file."""
        return f"wrote {path}"

    agent = Agent(name="ws", instructions="Use tools.", tools=[write_file])
    guard = Guard("ask")

    async def run_once(label: str):
        ws = JsonSocket()
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
        await ws_json(
            ws,
            agent,
            "write",
            model=model,
            middleware=[guard],
            ask_user=lambda request: prompts.append((label, request.arguments["path"])) or True,
            resolve=False,
        )
        return ws.events

    first = await run_once("first")
    second = await run_once("second")

    assert guard.ask_user is None
    assert prompts == [("first", "first.py"), ("second", "second.py")]
    assert first[2]["data"]["text"] == "wrote first.py"
    assert second[2]["data"]["text"] == "wrote second.py"


async def test_ws_json_updates_supplied_session():
    session = Session()
    agent = Agent(name="ws", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="hello")])
    ws = JsonSocket()

    await ws_json(
        ws,
        agent,
        "hi",
        model=model,
        session=session,
        resolve=False,
    )

    assert session.messages == [
        Message("system", "Answer."),
        Message("user", "hi"),
        Message("assistant", "hello"),
    ]


async def test_ws_json_persists_session_to_path(tmp_path):
    path = tmp_path / "session.json"
    agent = Agent(name="ws", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="hello")])
    ws = JsonSocket()

    count = await ws_json(
        ws,
        agent,
        "hi",
        model=model,
        session=path,
        resolve=False,
    )

    assert count == 1
    assert Session.load(path).messages == [
        Message("system", "Answer."),
        Message("user", "hi"),
        Message("assistant", "hello"),
    ]


async def test_ws_json_resumes_session_from_path(tmp_path):
    path = tmp_path / "session.json"
    Session(
        [
            Message("system", "Answer."),
            Message("user", "old"),
            Message("assistant", "old answer"),
        ]
    ).save(path)
    agent = Agent(name="ws", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="new answer")])
    ws = JsonSocket()

    await ws_json(
        ws,
        agent,
        "again",
        model=model,
        session=path,
        resolve=False,
    )

    messages, _ = model.calls[0]
    assert [message.content for message in messages] == [
        "Answer.",
        "old",
        "old answer",
        "again",
    ]
    assert Session.load(path).messages[-1].content == "new answer"


async def test_ws_json_can_use_default_session_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = Agent(name="ws", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="hello")])
    ws = JsonSocket()

    await ws_json(
        ws,
        agent,
        "hi",
        model=model,
        session=True,
        resolve=False,
    )

    loaded = Session.load(tmp_path / ".miniadk" / "sessions" / "ws.json")
    assert loaded.messages[-1].content == "hello"


async def test_ws_json_can_auto_compact_session():
    session = Session()
    agent = Agent(name="ws", instructions="Answer.")
    model = ScriptedModel(
        [
            ModelResult(message="hello"),
            ModelResult(message="Conversation summary."),
        ]
    )
    ws = JsonSocket()

    count = await ws_json(
        ws,
        agent,
        "hi",
        model=model,
        session=session,
        compact=Compact(chars=1, keep=1),
        resolve=False,
    )

    assert count == 1
    assert ws.events == [{"type": "message", "data": {"text": "hello"}}]
    assert session.messages == [
        Message("system", "Answer."),
        Message("system", "Conversation summary."),
        Message("assistant", "hello"),
    ]


async def test_ws_json_requires_send_method():
    agent = Agent(name="ws", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="hello")])

    with pytest.raises(TypeError, match="send_json or send"):
        await ws_json(object(), agent, "hi", model=model, resolve=False)


async def test_ws_json_cancels_active_run_when_send_fails():
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

    agent = Agent(name="ws", instructions="Use tools.", tools=[work])
    model = ScriptedModel(
        [ModelResult(tool_calls=[ToolCall(name="work", arguments={})])]
    )
    ws = FailingJsonSocket(fail_at=2)

    with pytest.raises(RuntimeError, match="websocket closed"):
        await ws_json(ws, agent, "work", model=model, resolve=False)

    assert [event["type"] for event in ws.events] == ["tool_call", "tool_progress"]
    await asyncio.wait_for(cancelled.wait(), timeout=0.2)


async def test_ws_json_cancels_active_run_when_handler_is_cancelled():
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

    agent = Agent(name="ws", instructions="Use tools.", tools=[work])
    model = ScriptedModel(
        [ModelResult(tool_calls=[ToolCall(name="work", arguments={})])]
    )
    ws = CancellingJsonSocket(cancel_at=2)

    with pytest.raises(asyncio.CancelledError):
        await ws_json(ws, agent, "work", model=model, resolve=False)

    assert [event["type"] for event in ws.events] == ["tool_call", "tool_progress"]
    await asyncio.wait_for(cancelled.wait(), timeout=0.2)
