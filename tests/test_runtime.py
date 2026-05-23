import asyncio

import pytest

from miniadk import (
    Agent,
    Guard,
    ModelResult,
    ModelStreamEvent,
    PermissionDecision,
    RunDecision,
    Runtime,
    ScriptedModel,
    Session,
    ToolCall,
    ToolCallDelta,
    as_tool,
    tool,
)
from miniadk.core import arun as core_arun, run as core_run
from miniadk.stdtools import make_edit_file, make_search_text


class StreamingScriptedModel:
    def __init__(self, updates):
        self.updates = list(updates)

    async def complete(self, messages, tools):
        raise AssertionError("streaming model should use stream")

    async def stream(self, messages, tools):
        for update in self.updates:
            yield update


class StepStreamingScriptedModel:
    def __init__(self, steps):
        self.steps = [list(step) for step in steps]

    async def complete(self, messages, tools):
        raise AssertionError("streaming model should use stream")

    async def stream(self, messages, tools):
        updates = self.steps.pop(0)
        for update in updates:
            yield update


async def collect(runtime: Runtime, text: str):
    return [event async for event in runtime.run(text)]


async def test_runtime_answers_when_model_returns_message():
    agent = Agent(
        name="simple",
        instructions="Answer plainly.",
        tools=[],
    )
    runtime = Runtime(agent=agent, model=ScriptedModel([ModelResult(message="hello")]))

    events = await collect(runtime, "hi")

    assert [event.type for event in events] == ["message"]
    assert events[0].data == {"text": "hello"}
    assert [message.role for message in runtime.messages] == [
        "system",
        "user",
        "assistant",
    ]


async def test_core_arun_accepts_temporary_tools():
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
    agent = Agent(name="simple", instructions="Use tools.")

    answer = await core_arun(agent, "ping", model=model, tools=[ping])

    assert answer == "done"
    assert model.calls[0][1] == [ping]


async def test_as_tool_inherits_current_permission_callback():
    prompts = []

    @tool(destructive=True)
    def write_file(path: str) -> str:
        """Write a file."""
        return f"wrote {path}"

    child = Agent(name="child", instructions="Use tools.", tools=[write_file])
    child_tool = as_tool(
        child,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="write_file", arguments={"path": "a.py"})
                    ]
                ),
                ModelResult(message="child done"),
            ]
        ),
        middleware=[Guard("ask")],
    )
    parent = Agent(name="parent", instructions="Delegate.", tools=[child_tool])
    runtime = Runtime(
        agent=parent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="child", arguments={"prompt": "write"})
                    ]
                ),
                ModelResult(message="done"),
            ]
        ),
        middleware=[
            Guard(
                "ask",
                ask_user=lambda request: prompts.append(
                    (request.tool.name, request.arguments["path"])
                )
                or True,
            )
        ],
    )

    events = await collect(runtime, "delegate")

    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "message",
    ]
    assert prompts == [("write_file", "a.py")]


def test_core_run_accepts_temporary_tools():
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
    agent = Agent(name="simple", instructions="Use tools.")

    assert core_run(agent, "ping", model=model, tools=[ping]) == "done"
    assert model.calls[0][1] == [ping]


async def test_runtime_can_emit_lifecycle_events():
    agent = Agent(name="simple", instructions="Answer plainly.", tools=[])
    runtime = Runtime(agent=agent, model=ScriptedModel([ModelResult(message="hello")]))

    events = [
        event
        async for event in runtime.run("hi", lifecycle=True)
    ]

    assert [event.type for event in events] == [
        "run_start",
        "message",
        "run_end",
    ]
    assert events[0].data == {"agent": "simple", "input": "hi"}
    assert events[-1].data == {
        "agent": "simple",
        "status": "completed",
        "reason": "completed",
        "messages": 3,
    }


async def test_runtime_trace_adds_run_and_step_metadata_without_changing_default():
    @tool
    async def build(progress) -> str:
        """Build something."""
        await progress("started")
        return "built"

    agent = Agent(name="traced", instructions="Use tools.", tools=[build])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(id="call_1", name="build", arguments={}),
                    ]
                ),
                ModelResult(message="done"),
            ]
        ),
    )

    events = [
        event
        async for event in runtime.run("build", lifecycle=True, trace=True)
    ]

    run_id = events[0].data["run_id"]
    assert run_id
    assert all(event.data.get("run_id") == run_id for event in events)
    assert "step" not in events[0].data
    assert events[1].data["step"] == 1
    assert events[1].data["tool_call_id"] == "call_1"
    assert events[2].data["step"] == 1
    assert events[3].data["step"] == 1
    assert events[3].data["tool_call_id"] == "call_1"
    assert events[4].data["step"] == 2
    assert events[5].data["step"] == 2


async def test_runtime_state_exposes_run_id_to_hooks_and_policy():
    seen = []

    class Audit:
        async def before_model_call(self, state):
            seen.append(("before", state.run_id))

        async def after_model_call(self, state):
            seen.append(("after", state.run_id))

    class StopPolicy:
        async def after_model(self, state):
            seen.append(("policy", state.run_id))
            return RunDecision.stop(state.result.message)

        async def after_tools(self, state):
            return RunDecision()

    agent = Agent(name="traced", instructions="Answer.", tools=[])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel([ModelResult(message="hello")]),
        middleware=[Audit()],
        policy=StopPolicy(),
    )

    events = [
        event
        async for event in runtime.run("hi", lifecycle=True, trace=True)
    ]

    run_id = events[0].data["run_id"]
    assert run_id
    assert seen == [
        ("before", run_id),
        ("after", run_id),
        ("policy", run_id),
    ]


async def test_runtime_lifecycle_marks_errors():
    class IncompleteStreamingModel:
        async def stream(self, messages, tools):
            yield ModelStreamEvent(delta="hel")

    agent = Agent(name="simple", instructions="Answer plainly.", tools=[])
    runtime = Runtime(agent=agent, model=IncompleteStreamingModel())

    events = [
        event
        async for event in runtime.run("hi", lifecycle=True)
    ]

    assert [event.type for event in events] == [
        "run_start",
        "message_delta",
        "error",
        "run_end",
    ]
    assert events[-1].data["status"] == "error"
    assert events[-1].data["reason"] == "stream_missing_result"


async def test_runtime_lifecycle_marks_denials():
    class Deny:
        async def before_tool_call(self, tool, arguments):
            return PermissionDecision("deny", "no")

    @tool
    def echo(text: str) -> str:
        """Echo text."""
        return text

    agent = Agent(name="guarded", instructions="Use tools.", tools=[echo])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[ToolCall(name="echo", arguments={"text": "ok"})]
                )
            ]
        ),
        middleware=[Deny()],
    )

    events = [
        event
        async for event in runtime.run("echo", lifecycle=True)
    ]

    assert [event.type for event in events] == [
        "run_start",
        "tool_denied",
        "run_end",
    ]
    assert events[-1].data["status"] == "stopped"
    assert events[-1].data["reason"] == "tool_denied"


async def test_runtime_lifecycle_marks_tool_errors():
    @tool
    def fail() -> str:
        """Fail."""
        raise ValueError("bad tool")

    agent = Agent(name="broken", instructions="Use tools.", tools=[fail])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [ModelResult(tool_calls=[ToolCall(name="fail", arguments={})])]
        ),
    )

    events = [
        event
        async for event in runtime.run("fail", lifecycle=True)
    ]

    assert [event.type for event in events] == [
        "run_start",
        "tool_call",
        "tool_error",
        "run_end",
    ]
    assert events[-1].data["status"] == "error"
    assert events[-1].data["reason"] == "tool_error"


async def test_runtime_lifecycle_marks_policy_stop():
    class StopPolicy:
        async def after_model(self, state):
            return RunDecision.stop("blocked", reason="policy_stop")

        async def after_tools(self, state):
            return RunDecision()

    agent = Agent(name="simple", instructions="Answer plainly.", tools=[])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel([ModelResult(message="hello")]),
        policy=StopPolicy(),
    )

    events = [
        event
        async for event in runtime.run("hi", lifecycle=True)
    ]

    assert [event.type for event in events] == [
        "run_start",
        "message",
        "run_end",
    ]
    assert events[-1].data["status"] == "stopped"
    assert events[-1].data["reason"] == "policy_stop"


async def test_runtime_ask_returns_final_message_and_keeps_history():
    agent = Agent(name="simple", instructions="Answer plainly.", tools=[])
    runtime = Runtime(agent=agent, model=ScriptedModel([ModelResult(message="hello")]))

    answer = await runtime.ask("hi")

    assert answer == "hello"
    assert [message.role for message in runtime.messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert runtime.messages[-1].content == "hello"


async def test_runtime_streams_message_delta_then_final_message():
    agent = Agent(name="simple", instructions="Answer plainly.", tools=[])
    runtime = Runtime(
        agent=agent,
        model=StreamingScriptedModel(
            [
                ModelStreamEvent(delta="hel"),
                ModelStreamEvent(delta="lo"),
                ModelStreamEvent(result=ModelResult(message="hello")),
            ]
        ),
    )

    events = await collect(runtime, "hi")

    assert [event.type for event in events] == [
        "message_delta",
        "message_delta",
        "message",
    ]
    assert [event.data["text"] for event in events] == ["hel", "lo", "hello"]
    assert events[-1].data["streamed"] is True
    assert runtime.messages[-1].content == "hello"


async def test_runtime_can_cancel_streaming_run():
    class CancellableStreamingModel:
        async def stream(self, messages, tools):
            yield ModelStreamEvent(delta="hel")
            runtime.cancel("user_stop")
            yield ModelStreamEvent(delta="late")

    agent = Agent(name="simple", instructions="Answer plainly.", tools=[])
    runtime = Runtime(agent=agent, model=CancellableStreamingModel())

    events = [
        event
        async for event in runtime.run("hi", lifecycle=True)
    ]

    assert [event.type for event in events] == [
        "run_start",
        "message_delta",
        "cancelled",
        "run_end",
    ]
    assert events[2].data == {"reason": "user_stop"}
    assert events[-1].data["status"] == "stopped"
    assert events[-1].data["reason"] == "user_stop"
    assert runtime.messages[-1].role == "assistant"
    assert runtime.messages[-1].content == "hel"


async def test_runtime_streams_thinking_delta_events():
    agent = Agent(name="simple", instructions="Answer plainly.", tools=[])
    runtime = Runtime(
        agent=agent,
        model=StreamingScriptedModel(
            [
                ModelStreamEvent(thinking="checking"),
                ModelStreamEvent(delta="hi"),
                ModelStreamEvent(result=ModelResult(message="hi")),
            ]
        ),
    )

    events = await collect(runtime, "hi")

    assert [event.type for event in events] == [
        "thinking_delta",
        "message_delta",
        "message",
    ]
    assert events[0].data == {"text": "checking"}
    assert events[1].data == {"text": "hi"}


async def test_runtime_streams_tool_call_delta_events():
    @tool
    def greet(name: str) -> str:
        """Greet a person."""
        return f"hello {name}"

    agent = Agent(name="simple", instructions="Use tools.", tools=[greet])
    runtime = Runtime(
        agent=agent,
        model=StepStreamingScriptedModel(
            [
                [
                    ModelStreamEvent(
                        tool_call=ToolCallDelta(
                            index=0,
                            id="call_1",
                            name="greet",
                            arguments='{"name"',
                        )
                    ),
                    ModelStreamEvent(
                        tool_call=ToolCallDelta(index=0, arguments=':"Ada"}')
                    ),
                    ModelStreamEvent(
                        result=ModelResult(
                            tool_calls=[
                                ToolCall(
                                    id="call_1",
                                    name="greet",
                                    arguments={"name": "Ada"},
                                )
                            ]
                        )
                    ),
                ],
                [ModelStreamEvent(result=ModelResult(message="done"))],
            ]
        ),
    )

    events = await collect(runtime, "hi")

    assert [event.type for event in events] == [
        "tool_call_delta",
        "tool_call_delta",
        "tool_call",
        "tool_result",
        "message",
    ]
    assert events[0].data == {
        "index": 0,
        "id": "call_1",
        "name": "greet",
        "arguments": '{"name"',
    }
    assert events[1].data == {"index": 0, "arguments": ':"Ada"}'}
    assert events[2].data["name"] == "greet"
    assert events[3].data == {
        "name": "greet",
        "result": "hello Ada",
        "text": "hello Ada",
    }
    assert events[4].data == {"text": "done"}


async def test_runtime_reports_stream_without_final_result_as_error():
    class IncompleteStreamingModel:
        async def stream(self, messages, tools):
            yield ModelStreamEvent(delta="hel")

    agent = Agent(name="simple", instructions="Answer plainly.", tools=[])
    runtime = Runtime(agent=agent, model=IncompleteStreamingModel())

    events = await collect(runtime, "hi")

    assert [event.type for event in events] == ["message_delta", "error"]
    assert events[-1].data == {
        "message": "Streaming model ended without a final result",
        "reason": "stream_missing_result",
    }
    assert runtime.messages[-1].role == "assistant"
    assert runtime.messages[-1].content == "hel"


async def test_runtime_keeps_streamed_text_when_streaming_model_errors():
    class FailingStreamingModel:
        async def stream(self, messages, tools):
            yield ModelStreamEvent(delta="hel")
            yield ModelStreamEvent(delta="lo")
            raise RuntimeError("model unavailable")

    agent = Agent(name="simple", instructions="Answer plainly.", tools=[])
    runtime = Runtime(agent=agent, model=FailingStreamingModel())

    with pytest.raises(RuntimeError, match="model unavailable"):
        await collect(runtime, "hi")

    assert runtime.messages[-1].role == "assistant"
    assert runtime.messages[-1].content == "hello"


async def test_runtime_notifies_middleware_around_model_calls():
    seen = []

    class ObserveModel:
        async def before_model_call(self, state):
            seen.append(("before", state.step, len(state.messages), state.result))

        async def after_model_call(self, state):
            seen.append(("after", state.step, state.result.message))

    agent = Agent(name="simple", instructions="Answer plainly.", tools=[])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel([ModelResult(message="hello")]),
        middleware=[ObserveModel()],
    )

    events = await collect(runtime, "hi")

    assert [event.type for event in events] == ["message"]
    assert seen == [
        ("before", 1, 2, None),
        ("after", 1, "hello"),
    ]


async def test_runtime_notifies_middleware_on_model_error():
    seen = []

    class FailingModel:
        async def complete(self, messages, tools):
            raise RuntimeError("model unavailable")

    class ObserveModel:
        async def on_model_error(self, state, error):
            seen.append((state.step, type(error).__name__, str(error)))

    agent = Agent(name="simple", instructions="Answer plainly.", tools=[])
    runtime = Runtime(
        agent=agent,
        model=FailingModel(),
        middleware=[ObserveModel()],
    )

    with pytest.raises(RuntimeError, match="model unavailable"):
        await collect(runtime, "hi")

    assert seen == [(1, "RuntimeError", "model unavailable")]


async def test_runtime_lifecycle_ends_before_model_error_is_raised():
    class FailingModel:
        async def complete(self, messages, tools):
            raise RuntimeError("model unavailable")

    runtime = Runtime(
        agent=Agent(name="simple", instructions="Answer plainly."),
        model=FailingModel(),
    )
    stream = runtime.run("hi", lifecycle=True)

    first = await stream.__anext__()
    assert first.type == "run_start"

    second = await stream.__anext__()
    assert second.type == "run_end"
    assert second.data == {
        "agent": "simple",
        "status": "error",
        "reason": "model_error",
        "messages": 2,
    }

    with pytest.raises(RuntimeError, match="model unavailable"):
        await stream.__anext__()


async def test_runtime_executes_tool_call_then_continues_loop():
    @tool
    def greet(name: str) -> str:
        """Greet a person."""
        return f"hello {name}"

    agent = Agent(
        name="tool-user",
        instructions="Use tools when helpful.",
        tools=[greet],
    )
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(tool_calls=[ToolCall(name="greet", arguments={"name": "Ada"})]),
                ModelResult(message="done"),
            ]
        ),
    )

    events = await collect(runtime, "say hello")

    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "message",
    ]
    assert events[0].data["name"] == "greet"
    assert events[1].data == {
        "name": "greet",
        "result": "hello Ada",
        "text": "hello Ada",
    }
    assert events[2].data == {"text": "done"}
    assert [message.role for message in runtime.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert runtime.messages[2].tool_calls[0].name == "greet"
    assert runtime.messages[3].content == "hello Ada"


async def test_runtime_ask_can_use_temporary_tools():
    @tool
    def greet(name: str) -> str:
        """Greet a person."""
        return f"hello {name}"

    agent = Agent(name="tool-user", instructions="Use tools.", tools=[])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(tool_calls=[ToolCall(name="greet", arguments={"name": "Ada"})]),
                ModelResult(message="done"),
            ]
        ),
    )

    answer = await runtime.ask("say hello", tools=[greet])

    assert answer == "done"
    assert runtime.messages[3].content == "hello Ada"


async def test_runtime_keeps_assistant_text_with_tool_calls_and_continues_loop():
    @tool
    def inspect_project() -> str:
        """Inspect the project."""
        return "project summary"

    agent = Agent(
        name="tool-user",
        instructions="Use tools when helpful.",
        tools=[inspect_project],
    )
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    message="I will inspect the project.",
                    tool_calls=[ToolCall(name="inspect_project", arguments={})],
                ),
                ModelResult(message="The project is MiniADK."),
            ]
        ),
    )

    events = await collect(runtime, "read project")

    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "message",
    ]
    assert events[2].data == {"text": "The project is MiniADK."}
    assert [message.role for message in runtime.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert runtime.messages[2].content == "I will inspect the project."
    assert runtime.messages[2].tool_calls[0].name == "inspect_project"
    assert runtime.messages[4].content == "The project is MiniADK."


async def test_runtime_runs_concurrency_safe_tool_calls_together():
    started: set[str] = set()
    both_started = asyncio.Event()

    @tool(concurrency_safe=True)
    async def read_file(path: str) -> str:
        """Read a file."""
        started.add(path)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.2)
        return path

    agent = Agent(
        name="reader",
        instructions="Use tools when helpful.",
        tools=[read_file],
    )
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="read_file", arguments={"path": "a.py"}),
                        ToolCall(name="read_file", arguments={"path": "b.py"}),
                    ],
                ),
                ModelResult(message="done"),
            ]
        ),
    )

    events = await collect(runtime, "read both")

    assert [event.type for event in events] == [
        "tool_call",
        "tool_call",
        "tool_result",
        "tool_result",
        "message",
    ]
    assert [event.data["result"] for event in events if event.type == "tool_result"] == [
        "a.py",
        "b.py",
    ]
    assert [message.content for message in runtime.messages if message.role == "tool"] == [
        "a.py",
        "b.py",
    ]


async def test_runtime_cleans_up_concurrent_tool_tasks_on_error():
    seen = []

    class Audit:
        async def on_tool_error(self, tool, arguments, error):
            seen.append((tool.name, arguments["path"], type(error).__name__, str(error)))

    @tool(concurrency_safe=True)
    async def read_file(path: str, progress) -> str:
        """Read a file."""
        if path == "bad.py":
            await progress("starting")
            raise RuntimeError("cannot read")
        await asyncio.sleep(0)
        return path

    agent = Agent(
        name="reader",
        instructions="Use tools when helpful.",
        tools=[read_file],
    )
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="read_file", arguments={"path": "bad.py"}),
                        ToolCall(name="read_file", arguments={"path": "ok.py"}),
                    ],
                ),
                ModelResult(message="unused"),
            ]
        ),
        middleware=[Audit()],
    )

    events = await collect(runtime, "read both")

    assert [event.type for event in events] == [
        "tool_call",
        "tool_call",
        "tool_progress",
        "tool_error",
    ]
    assert events[-1].data["tool"] == "read_file"
    assert events[-1].data["message"] == "RuntimeError: cannot read"
    assert seen == [("read_file", "bad.py", "RuntimeError", "cannot read")]


async def test_runtime_keeps_unsafe_tool_calls_sequential():
    order = []

    @tool
    async def write_file(path: str) -> str:
        """Write a file."""
        order.append(f"start:{path}")
        await asyncio.sleep(0)
        order.append(f"end:{path}")
        return path

    agent = Agent(
        name="writer",
        instructions="Use tools when helpful.",
        tools=[write_file],
    )
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="write_file", arguments={"path": "a.py"}),
                        ToolCall(name="write_file", arguments={"path": "b.py"}),
                    ],
                ),
                ModelResult(message="done"),
            ]
        ),
    )

    await collect(runtime, "write both")

    assert order == [
        "start:a.py",
        "end:a.py",
        "start:b.py",
        "end:b.py",
    ]


async def test_runtime_uses_tool_format_for_model_text_and_keeps_raw_event_result():
    @tool(format=lambda result, path: f"{path}: {len(result)} lines")
    def read_lines(path: str) -> list[str]:
        """Read lines from a file."""
        return ["alpha", "beta"]

    agent = Agent(
        name="reader",
        instructions="Use tools when helpful.",
        tools=[read_lines],
    )
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="read_lines", arguments={"path": "notes.txt"})
                    ]
                ),
                ModelResult(message="done"),
            ]
        ),
    )

    events = await collect(runtime, "read notes")

    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "message",
    ]
    assert events[1].data == {
        "name": "read_lines",
        "result": ["alpha", "beta"],
        "text": "notes.txt: 2 lines",
    }
    assert runtime.messages[3].content == "notes.txt: 2 lines"


async def test_runtime_keeps_raw_result_when_tool_text_is_clipped():
    @tool(max_text=12)
    def read_file(path: str) -> str:
        """Read a file."""
        return "abcdefghijklmnopqrstuvwxyz"

    agent = Agent(
        name="reader",
        instructions="Use tools.",
        tools=[read_file],
    )
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="read_file", arguments={"path": "notes.txt"})
                    ]
                ),
                ModelResult(message="done"),
            ]
        ),
    )

    events = await collect(runtime, "read notes")

    assert events[1].data == {
        "name": "read_file",
        "result": "abcdefghijklmnopqrstuvwxyz",
        "text": "abcdefghijkl",
    }
    assert runtime.messages[3].content == "abcdefghijkl"


async def test_runtime_streams_tool_progress_events_before_result():
    @tool
    async def build(path: str, progress) -> str:
        """Build a target."""
        await progress("started", step=1)
        await progress("finished", step=2)
        return f"built {path}"

    agent = Agent(name="builder", instructions="Use tools.", tools=[build])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="build", arguments={"path": "app"})
                    ]
                ),
                ModelResult(message="done"),
            ]
        ),
    )

    events = await collect(runtime, "build")

    assert [event.type for event in events] == [
        "tool_call",
        "tool_progress",
        "tool_progress",
        "tool_result",
        "message",
    ]
    assert events[1].data == {
        "tool": "build",
        "message": "started",
        "data": {"step": 1},
    }
    assert events[2].data == {
        "tool": "build",
        "message": "finished",
        "data": {"step": 2},
    }


async def test_runtime_cancel_stops_running_tool_task():
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

    agent = Agent(name="worker", instructions="Use tools.", tools=[work])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [ModelResult(tool_calls=[ToolCall(name="work", arguments={})])]
        ),
    )

    seen = []
    async for event in runtime.run("work", lifecycle=True):
        seen.append(event.type)
        if event.type == "tool_progress":
            runtime.cancel("user_stop")

    assert seen == ["run_start", "tool_call", "tool_progress", "cancelled", "run_end"]
    await asyncio.wait_for(cancelled.wait(), timeout=0.2)


async def test_runtime_cancel_wakes_running_tool_without_progress():
    started = asyncio.Event()
    cancelled = asyncio.Event()

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

    agent = Agent(name="worker", instructions="Use tools.", tools=[wait])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [ModelResult(tool_calls=[ToolCall(name="wait", arguments={})])]
        ),
    )

    async def cancel_when_started():
        await started.wait()
        runtime.cancel("user_stop")

    cancel_task = asyncio.create_task(cancel_when_started())
    events = [
        event
        async for event in runtime.run("work", lifecycle=True)
    ]

    assert [event.type for event in events] == [
        "run_start",
        "tool_call",
        "cancelled",
        "run_end",
    ]
    await asyncio.wait_for(cancelled.wait(), timeout=0.2)
    await cancel_task


async def test_runtime_reports_tool_timeout_as_tool_error():
    cancelled = asyncio.Event()

    @tool(timeout=0.01)
    async def wait() -> str:
        """Wait too long."""
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    agent = Agent(name="worker", instructions="Use tools.", tools=[wait])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [ModelResult(tool_calls=[ToolCall(name="wait", arguments={})])]
        ),
    )

    events = await collect(runtime, "wait")

    assert [event.type for event in events] == ["tool_call", "tool_error"]
    assert events[1].data == {
        "tool": "wait",
        "message": "TimeoutError: wait timed out after 0.01 seconds",
    }
    assert runtime.messages[-1].content == (
        "TimeoutError: wait timed out after 0.01 seconds"
    )
    await asyncio.wait_for(cancelled.wait(), timeout=0.2)


async def test_runtime_cancel_stops_concurrent_tool_tasks():
    started: set[str] = set()
    both_started = asyncio.Event()
    cancelled: set[str] = set()

    @tool(concurrency_safe=True)
    async def wait(path: str) -> str:
        """Wait until cancelled."""
        started.add(path)
        if len(started) == 2:
            both_started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.add(path)
            raise
        return path

    agent = Agent(name="worker", instructions="Use tools.", tools=[wait])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="wait", arguments={"path": "a.py"}),
                        ToolCall(name="wait", arguments={"path": "b.py"}),
                    ],
                )
            ]
        ),
    )

    async def cancel_when_started():
        await both_started.wait()
        runtime.cancel("user_stop")

    cancel_task = asyncio.create_task(cancel_when_started())
    events = [
        event
        async for event in runtime.run("work", lifecycle=True)
    ]

    assert [event.type for event in events] == [
        "run_start",
        "tool_call",
        "tool_call",
        "cancelled",
        "run_end",
    ]
    assert events[-1].data["reason"] == "user_stop"
    assert cancelled == {"a.py", "b.py"}
    await cancel_task


async def test_runtime_stream_close_stops_concurrent_tool_tasks():
    started: set[str] = set()
    both_started = asyncio.Event()
    cancelled: set[str] = set()

    @tool(concurrency_safe=True)
    async def wait(path: str) -> str:
        """Wait until cancelled."""
        started.add(path)
        if len(started) == 2:
            both_started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.add(path)
            raise
        return path

    agent = Agent(name="worker", instructions="Use tools.", tools=[wait])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="wait", arguments={"path": "a.py"}),
                        ToolCall(name="wait", arguments={"path": "b.py"}),
                    ],
                )
            ]
        ),
    )

    stream = runtime.run("work")
    assert (await stream.__anext__()).type == "tool_call"
    assert (await stream.__anext__()).type == "tool_call"
    next_event = asyncio.create_task(stream.__anext__())
    await asyncio.wait_for(both_started.wait(), timeout=0.2)
    next_event.cancel()
    with pytest.raises(asyncio.CancelledError):
        await next_event

    assert cancelled == {"a.py", "b.py"}


async def test_agent_can_be_wrapped_as_a_tool():
    child = Agent(name="reviewer", instructions="Review code.", tools=[])
    child_model = ScriptedModel([ModelResult(message="looks good")])
    review = as_tool(child, model=child_model)

    assert review.input_schema == {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
        },
        "additionalProperties": False,
        "required": ["prompt"],
    }

    parent = Agent(name="parent", instructions="Delegate.", tools=[review])
    runtime = Runtime(
        agent=parent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="reviewer",
                            arguments={"prompt": "review app.py"},
                        )
                    ]
                ),
                ModelResult(message="done"),
            ]
        ),
    )

    events = await collect(runtime, "please review")

    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "message",
    ]
    assert events[1].data["result"] == "looks good"
    assert child_model.calls[0][0][-1].content == "review app.py"


async def test_agent_tool_can_reuse_child_session():
    child_session = Session()
    child = Agent(name="reviewer", instructions="Review code.", tools=[])
    child_model = ScriptedModel(
        [
            ModelResult(message="first"),
            ModelResult(message="second"),
        ]
    )
    review = as_tool(child, model=child_model, session=child_session)
    parent = Agent(name="parent", instructions="Delegate.", tools=[review])
    runtime = Runtime(
        agent=parent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="reviewer",
                            arguments={"prompt": "review app.py"},
                        )
                    ]
                ),
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="reviewer",
                            arguments={"prompt": "review tests.py"},
                        )
                    ]
                ),
                ModelResult(message="done"),
            ]
        ),
    )

    events = await collect(runtime, "please review")

    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "message",
    ]
    assert [message.content for message in child_session.messages] == [
        "Review code.",
        "review app.py",
        "first",
        "review tests.py",
        "second",
    ]


async def test_runtime_notifies_middleware_after_tool_success():
    audit = []

    class Audit:
        async def before_tool_call(self, tool, arguments):
            return PermissionDecision("allow")

        async def after_tool_call(self, tool, arguments, result, text):
            audit.append((tool.name, arguments, result, text))

    @tool(format=lambda result: f"total={result}")
    def add(left: int, right: int) -> int:
        """Add numbers."""
        return left + right

    agent = Agent(name="audited", instructions="Use tools.", tools=[add])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="add",
                            arguments={"left": 2, "right": 3},
                        )
                    ]
                ),
                ModelResult(message="done"),
            ]
        ),
        middleware=[Audit()],
    )

    await collect(runtime, "add")

    assert audit == [
        (
            "add",
            {"left": 2, "right": 3},
            5,
            "total=5",
        )
    ]


async def test_runtime_notifies_middleware_on_tool_error():
    errors = []

    class Audit:
        async def before_tool_call(self, tool, arguments):
            return PermissionDecision("allow")

        async def on_tool_error(self, tool, arguments, error):
            errors.append((tool.name, arguments, type(error).__name__, str(error)))

    @tool
    def fail(path: str) -> str:
        """Fail."""
        raise ValueError(f"bad {path}")

    agent = Agent(name="audited", instructions="Use tools.", tools=[fail])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[ToolCall(name="fail", arguments={"path": "a.py"})]
                )
            ]
        ),
        middleware=[Audit()],
    )

    events = await collect(runtime, "fail")

    assert [event.type for event in events] == ["tool_call", "tool_error"]
    assert events[-1].data == {
        "tool": "fail",
        "message": "ValueError: bad a.py",
    }
    assert errors == [("fail", {"path": "a.py"}, "ValueError", "bad a.py")]


async def test_runtime_reports_tool_formatter_error_as_tool_error():
    errors = []

    class Audit:
        async def on_tool_error(self, tool, arguments, error):
            errors.append((tool.name, arguments, type(error).__name__, str(error)))

    def format_result(result):
        raise ValueError(f"bad format: {result}")

    @tool(format=format_result)
    def read(path: str) -> str:
        """Read."""
        return path

    agent = Agent(name="reader", instructions="Use tools.", tools=[read])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[ToolCall(name="read", arguments={"path": "a.py"})]
                )
            ]
        ),
        middleware=[Audit()],
    )

    events = await collect(runtime, "read")

    assert [event.type for event in events] == ["tool_call", "tool_error"]
    assert events[-1].data == {
        "tool": "read",
        "message": "ValueError: bad format: a.py",
    }
    assert runtime.messages[-1].content == "ValueError: bad format: a.py"
    assert errors == [("read", {"path": "a.py"}, "ValueError", "bad format: a.py")]


async def test_runtime_reports_concurrent_tool_error_as_tool_error():
    @tool(concurrency_safe=True)
    def fail(path: str) -> str:
        """Fail."""
        raise ValueError(f"bad {path}")

    @tool(concurrency_safe=True)
    def read(path: str) -> str:
        """Read."""
        return path

    agent = Agent(name="reader", instructions="Use tools.", tools=[fail, read])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="fail", arguments={"path": "a.py"}),
                        ToolCall(name="read", arguments={"path": "b.py"}),
                    ]
                )
            ]
        ),
    )

    events = await collect(runtime, "read")

    assert [event.type for event in events] == [
        "tool_call",
        "tool_call",
        "tool_error",
    ]
    assert events[-1].data == {
        "tool": "fail",
        "message": "ValueError: bad a.py",
    }
    assert runtime.messages[-1].content == "ValueError: bad a.py"


async def test_runtime_preserves_model_content_blocks_in_history():
    blocks = [
        {"type": "thinking", "thinking": "private reasoning", "signature": "sig"},
        {"type": "text", "text": "hello"},
    ]
    agent = Agent(name="simple", instructions="Answer plainly.", tools=[])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel([ModelResult(message="hello", content_blocks=blocks)]),
    )

    events = await collect(runtime, "hi")

    assert [event.type for event in events] == ["message"]
    assert runtime.messages[-1].content_blocks == blocks


async def test_runtime_reports_unknown_tool_as_error_event():
    agent = Agent(name="broken", instructions="Use a missing tool.", tools=[])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel([ModelResult(tool_calls=[ToolCall(name="missing", arguments={})])]),
    )

    events = await collect(runtime, "use missing")

    assert [event.type for event in events] == ["error"]
    assert "Unknown tool: missing" in events[0].data["message"]
    assert runtime.messages[-1].role == "tool"
    assert runtime.messages[-1].content == "Unknown tool: missing"


async def test_runtime_reports_max_steps_exhaustion():
    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    agent = Agent(name="looper", instructions="Use tools.", tools=[ping])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(tool_calls=[ToolCall(name="ping", arguments={})]),
                ModelResult(tool_calls=[ToolCall(name="ping", arguments={})]),
            ]
        ),
        max_steps=1,
    )

    events = await collect(runtime, "loop")

    assert [event.type for event in events] == ["tool_call", "tool_result", "error"]
    assert events[-1].data == {
        "message": "Runtime stopped after 1 steps",
        "reason": "max_steps",
    }


async def test_runtime_matches_tool_calls_by_canonical_name():
    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return f"read {path}"

    agent = Agent(name="reader", instructions="Use tools.", tools=[read_file])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[ToolCall(name="Read", arguments={"path": "a.py"})]
                ),
                ModelResult(message="done"),
            ]
        ),
    )

    events = await collect(runtime, "read")

    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "message",
    ]
    assert events[0].data["name"] == "read_file"
    assert events[1].data["name"] == "read_file"
    assert runtime.messages[3].name == "read_file"


async def test_runtime_validates_tool_input_before_execution():
    calls = []

    def validate_count(count: int):
        if count < 1:
            return "count must be positive"
        return True

    @tool(validate=validate_count)
    def repeat(count: int) -> str:
        """Repeat something."""
        calls.append(count)
        return "ok"

    agent = Agent(name="validator", instructions="Use tools.", tools=[repeat])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [ModelResult(tool_calls=[ToolCall(name="repeat", arguments={"count": 0})])]
        ),
    )

    events = await collect(runtime, "repeat zero")

    assert calls == []
    assert [event.type for event in events] == ["tool_invalid"]
    assert events[0].data == {
        "tool": "repeat",
        "message": "count must be positive",
    }
    assert runtime.messages[-1].role == "tool"
    assert runtime.messages[-1].content == "count must be positive"


async def test_runtime_reports_validation_exceptions_as_invalid_tool_input():
    def broken_validate() -> bool:
        raise ValueError("bad validator")

    @tool(validate=broken_validate)
    def fragile() -> str:
        """A fragile tool."""
        return "ok"

    agent = Agent(name="validator", instructions="Use tools.", tools=[fragile])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel([ModelResult(tool_calls=[ToolCall(name="fragile", arguments={})])]),
    )

    events = await collect(runtime, "run fragile")

    assert [event.type for event in events] == ["tool_invalid"]
    assert events[0].data == {
        "tool": "fragile",
        "message": "ValueError: bad validator",
    }


async def test_runtime_rejects_model_tool_arguments_that_do_not_match_schema():
    calls = []

    @tool
    def search(mode: "Literal['files', 'content']") -> str:
        """Search something."""
        calls.append(mode)
        return mode

    agent = Agent(name="validator", instructions="Use tools.", tools=[search])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [ModelResult(tool_calls=[ToolCall(name="search", arguments={"mode": "shell"})])]
        ),
    )

    events = await collect(runtime, "search")

    assert calls == []
    assert [event.type for event in events] == ["tool_invalid"]
    assert events[0].data == {
        "tool": "search",
        "message": "Tool argument mode must be one of: files, content",
    }


async def test_runtime_does_not_prompt_or_run_invalid_standard_edit_tool(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    edit_file = make_edit_file(root=tmp_path)
    asked = []

    class Audit:
        async def before_tool_call(self, tool, arguments):
            asked.append((tool.name, arguments))
            return PermissionDecision("allow")

    agent = Agent(name="editor", instructions="Edit files.", tools=[edit_file])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="edit_file",
                            arguments={
                                "path": "a.txt",
                                "old": "missing",
                                "new": "hi",
                            },
                        )
                    ]
                )
            ]
        ),
        middleware=[Audit()],
    )

    events = await collect(runtime, "edit")

    assert asked == []
    assert [event.type for event in events] == ["tool_invalid"]
    assert events[0].data == {
        "tool": "edit_file",
        "message": "edit_file failed: 'missing' not found in a.txt",
    }
    assert target.read_text(encoding="utf-8") == "hello"


async def test_runtime_reports_invalid_standard_search_pattern(tmp_path):
    search_text = make_search_text(root=tmp_path)
    agent = Agent(name="searcher", instructions="Search files.", tools=[search_text])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="search_workspace_text",
                            arguments={"pattern": "["},
                        )
                    ]
                )
            ]
        ),
    )

    events = await collect(runtime, "search")

    assert [event.type for event in events] == ["tool_invalid"]
    assert events[0].data["tool"] == "search_workspace_text"
    assert events[0].data["message"].startswith(
        "search_text failed: invalid regex:"
    )
