import json

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
    ToolCall,
    web_html,
    tool,
    ws_chat,
)


class ChatSocket:
    def __init__(self, incoming):
        self.incoming = list(incoming)
        self.sent = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def receive_json(self):
        if not self.incoming:
            return None
        item = self.incoming.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def send_json(self, event):
        self.sent.append(event)


class RecvSocket:
    def __init__(self, incoming):
        self.incoming = list(incoming)
        self.messages = []

    async def recv(self):
        if not self.incoming:
            return None
        return self.incoming.pop(0)

    async def send(self, text):
        self.messages.append(text)


def test_web_html_returns_a_working_chat_page():
    html = web_html(title="Lab <Agent>", ws_path="/agent/ws")

    assert "<!doctype html>" in html
    assert "Lab &lt;Agent&gt;" in html
    assert "new WebSocket" in html
    assert "/agent/ws" in html
    assert "socket.send(JSON.stringify({ text }))" in html
    assert "event.type === \"permission_prompt\"" in html
    assert "socket.send(JSON.stringify({ allow }))" in html
    assert "event.type === \"tool_call\"" in html
    assert "event.type === \"tool_error\"" in html
    assert "event.type === \"run_end\"" in html


async def test_ws_chat_accepts_json_turns_and_updates_session():
    session = Session()
    ws = ChatSocket([{"text": "hi"}, None])
    agent = Agent(name="web", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="hello")])

    await ws_chat(
        ws,
        agent,
        model=model,
        session=session,
        resolve=False,
    )

    assert ws.accepted is True
    assert ws.sent == [{"type": "message", "data": {"text": "hello"}}]
    assert session.messages == [
        Message("system", "Answer."),
        Message("user", "hi"),
        Message("assistant", "hello"),
    ]


async def test_ws_chat_can_ask_for_permission_over_websocket():
    calls = []
    ws = ChatSocket([{"text": "write"}, {"allow": True}, None])

    @tool(destructive=True)
    def write_file(path: str) -> str:
        """Write a file."""
        calls.append(path)
        return f"wrote {path}"

    agent = Agent(name="web", instructions="Use tools.", tools=[write_file])
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[ToolCall(name="write_file", arguments={"path": "a.py"})]
            ),
            ModelResult(message="done"),
        ]
    )

    await ws_chat(
        ws,
        agent,
        model=model,
        middleware=[Guard("ask")],
        resolve=False,
    )

    assert calls == ["a.py"]
    assert [event["type"] for event in ws.sent] == [
        "permission_prompt",
        "permission_request",
        "tool_call",
        "tool_result",
        "message",
    ]
    assert ws.sent[0]["data"] == {
        "tool": "write_file",
        "arguments": {"path": "a.py"},
        "reason": "destructive tool use",
    }


async def test_ws_chat_can_deny_permission_over_websocket():
    calls = []
    ws = ChatSocket([{"text": "write"}, {"allow": "false"}, None])

    @tool(destructive=True)
    def write_file(path: str) -> str:
        """Write a file."""
        calls.append(path)
        return f"wrote {path}"

    agent = Agent(name="web", instructions="Use tools.", tools=[write_file])
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[ToolCall(name="write_file", arguments={"path": "a.py"})]
            )
        ]
    )

    await ws_chat(
        ws,
        agent,
        model=model,
        middleware=[Guard("ask")],
        resolve=False,
    )

    assert calls == []
    assert [event["type"] for event in ws.sent] == [
        "permission_prompt",
        "permission_request",
        "tool_denied",
    ]
    assert "denied" in ws.sent[-1]["data"]["message"].lower()


async def test_ws_chat_persists_session_path_across_turns(tmp_path):
    path = tmp_path / "session.json"
    ws = ChatSocket([{"text": "first"}, {"text": "second"}, None])
    agent = Agent(name="web", instructions="Answer.")
    model = ScriptedModel(
        [
            ModelResult(message="one"),
            ModelResult(message="two"),
        ]
    )

    await ws_chat(
        ws,
        agent,
        model=model,
        session=path,
        resolve=False,
    )

    assert [message.content for message in Session.load(path).messages] == [
        "Answer.",
        "first",
        "one",
        "second",
        "two",
    ]
    assert [message.content for message in model.calls[1][0]] == [
        "Answer.",
        "first",
        "one",
        "second",
    ]


async def test_ws_chat_can_auto_compact_session_across_turns():
    session = Session()
    ws = ChatSocket([{"text": "first"}, {"text": "second"}, None])
    agent = Agent(name="web", instructions="Answer.")
    model = ScriptedModel(
        [
            ModelResult(message="one"),
            ModelResult(message="Summary after first."),
            ModelResult(message="Summary before second."),
            ModelResult(message="two"),
            ModelResult(message="Summary after second."),
        ]
    )

    await ws_chat(
        ws,
        agent,
        model=model,
        session=session,
        compact=Compact(chars=1, keep=1),
        resolve=False,
    )

    assert ws.sent == [
        {"type": "message", "data": {"text": "one"}},
        {"type": "message", "data": {"text": "two"}},
    ]
    assert [message.content for message in model.calls[3][0]] == [
        "Answer.",
        "Summary before second.",
        "one",
        "second",
    ]
    assert session.messages == [
        Message("system", "Answer."),
        Message("system", "Summary after second."),
        Message("assistant", "two"),
    ]


async def test_ws_chat_can_include_lifecycle_events():
    ws = ChatSocket([{"text": "hi"}, None])
    agent = Agent(name="web", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="hello")])

    await ws_chat(ws, agent, model=model, lifecycle=True, resolve=False)

    assert ws.sent == [
        {"type": "run_start", "data": {"agent": "web", "input": "hi"}},
        {"type": "message", "data": {"text": "hello"}},
        {
            "type": "run_end",
            "data": {
                "agent": "web",
                "status": "completed",
                "reason": "completed",
                "messages": 3,
            },
        },
    ]


async def test_ws_chat_reads_json_strings_from_recv():
    ws = RecvSocket([json.dumps({"text": "hi"}), None])
    agent = Agent(name="web", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="hello")])

    await ws_chat(ws, agent, model=model, resolve=False)

    assert [json.loads(message) for message in ws.messages] == [
        {"type": "message", "data": {"text": "hello"}},
    ]


async def test_ws_chat_can_use_default_model_helper(monkeypatch):
    built = ScriptedModel([ModelResult(message="hello")])
    monkeypatch.setattr(json_adapter, "default_model", lambda: built)
    ws = ChatSocket([{"text": "hi"}, None])

    await ws_chat(ws, Agent(name="web", instructions="Answer."), resolve=False)

    assert ws.sent == [{"type": "message", "data": {"text": "hello"}}]
    assert built.calls


async def test_ws_chat_requires_receive_method():
    agent = Agent(name="web", instructions="Answer.")
    model = ScriptedModel([ModelResult(message="hello")])

    with pytest.raises(TypeError, match="receive_json, receive_text, or recv"):
        await ws_chat(object(), agent, model=model, resolve=False)
