from miniadk import (
    Agent,
    AskBeforeMiddleware,
    Guard,
    ModelResult,
    Runtime,
    ScriptedModel,
    ToolCall,
    ask_before,
    tool,
)
from miniadk.core.middleware import pop_ask_user, push_ask_user
from miniadk.stdtools import make_write_file
from miniadk.stdtools.shell import make_shell


async def test_ask_before_permission_can_deny_tool_execution():
    calls = []

    @tool(permission=ask_before("writing files"))
    def write_file(path: str, content: str) -> str:
        """Write a file."""
        calls.append((path, content))
        return "wrote"

    agent = Agent(
        name="guarded",
        instructions="Ask before risky actions.",
        tools=[write_file],
    )
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="write_file",
                            arguments={"path": "x.txt", "content": "hello"},
                        )
                    ]
                )
            ]
        ),
        middleware=[AskBeforeMiddleware(ask_user=lambda request: False)],
    )

    events = [event async for event in runtime.run("write")]

    assert calls == []
    assert [event.type for event in events] == ["permission_request", "tool_denied"]
    assert events[0].data["tool"] == "write_file"
    assert events[0].data["reason"] == "writing files"
    assert "denied" in events[1].data["message"].lower()
    assert runtime.messages[-1].role == "tool"
    assert "denied" in runtime.messages[-1].content.lower()


async def test_ask_before_permission_can_allow_tool_execution():
    calls = []

    @tool(permission=ask_before("running shell commands"))
    def shell(command: str) -> str:
        """Run a command."""
        calls.append(command)
        return "ok"

    agent = Agent(name="guarded", instructions="Run commands.", tools=[shell])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(tool_calls=[ToolCall(name="shell", arguments={"command": "pytest"})]),
                ModelResult(message="finished"),
            ]
        ),
        middleware=[AskBeforeMiddleware(ask_user=lambda request: True)],
    )

    events = [event async for event in runtime.run("test")]

    assert calls == ["pytest"]
    assert [event.type for event in events] == [
        "permission_request",
        "tool_call",
        "tool_result",
        "message",
    ]


async def test_ask_before_permission_uses_runtime_bound_callback():
    prompts = []
    calls = []

    @tool(permission=ask_before("writing files"))
    def write_file(path: str) -> str:
        """Write a file."""
        calls.append(path)
        return "wrote"

    runtime = Runtime(
        agent=Agent(name="guarded", instructions="Write files.", tools=[write_file]),
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="write_file", arguments={"path": "a.py"})
                    ]
                ),
                ModelResult(message="done"),
            ]
        ),
        middleware=[AskBeforeMiddleware()],
    )

    token = push_ask_user(
        lambda request: prompts.append((request.tool.name, request.reason)) or True
    )
    try:
        events = [event async for event in runtime.run("write")]
    finally:
        pop_ask_user(token)

    assert prompts == [("write_file", "writing files")]
    assert calls == ["a.py"]
    assert [event.type for event in events] == [
        "permission_request",
        "tool_call",
        "tool_result",
        "message",
    ]


async def test_runtime_skips_permission_for_read_only_sensitive_tool_call(tmp_path):
    tool = make_write_file(root=tmp_path)
    runtime = Runtime(
        agent=Agent(name="guarded", instructions="Preview writes.", tools=[tool]),
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="write_file",
                            arguments={
                                "path": "notes.txt",
                                "content": "hello",
                                "dry": True,
                            },
                        )
                    ]
                ),
                ModelResult(message="previewed"),
            ]
        ),
        middleware=[AskBeforeMiddleware(ask_user=lambda request: False)],
    )

    events = [event async for event in runtime.run("preview")]

    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "message",
    ]
    assert events[1].data["text"] == "would create notes.txt: 5 chars"
    assert not (tmp_path / "notes.txt").exists()


async def test_read_only_sensitive_tool_call_still_runs_policy_middleware(tmp_path):
    tool = make_write_file(root=tmp_path)
    runtime = Runtime(
        agent=Agent(name="guarded", instructions="Preview writes.", tools=[tool]),
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="write_file",
                            arguments={
                                "path": "notes.txt",
                                "content": "hello",
                                "dry": True,
                            },
                        )
                    ]
                ),
            ]
        ),
        middleware=[Guard("trust", deny="write_file:*")],
    )

    events = [event async for event in runtime.run("preview")]

    assert [event.type for event in events] == ["permission_request", "tool_denied"]
    assert events[0].data["reason"] == "blocked by guard policy"
    assert not (tmp_path / "notes.txt").exists()


async def test_read_only_tool_call_can_be_denied_by_guard_policy():
    calls = []

    @tool(read_only=True)
    def read_file(path: str) -> str:
        """Read a file."""
        calls.append(path)
        return "contents"

    runtime = Runtime(
        agent=Agent(name="guarded", instructions="Read files.", tools=[read_file]),
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="read_file", arguments={"path": "secret.txt"})
                    ]
                ),
            ]
        ),
        middleware=[Guard("trust", deny="read_file:secret*")],
    )

    events = [event async for event in runtime.run("read")]

    assert calls == []
    assert [event.type for event in events] == ["permission_request", "tool_denied"]
    assert events[0].data["reason"] == "blocked by guard policy"


async def test_read_only_shell_command_skips_prompt_but_keeps_guard_policy(tmp_path):
    shell = make_shell(cwd=tmp_path, read="git status*")
    runtime = Runtime(
        agent=Agent(name="guarded", instructions="Run checks.", tools=[shell]),
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="shell", arguments={"command": "git status --short"})
                    ]
                ),
                ModelResult(message="checked"),
            ]
        ),
        middleware=[Guard("ask", ask_user=lambda request: False)],
    )

    events = [event async for event in runtime.run("check")]

    assert [event.type for event in events] == ["tool_call", "tool_result", "message"]
    assert "fatal: not a git repository" in events[1].data["text"]


async def test_read_only_shell_command_can_still_be_denied_by_guard(tmp_path):
    shell = make_shell(cwd=tmp_path, read="git status*")
    runtime = Runtime(
        agent=Agent(name="guarded", instructions="Run checks.", tools=[shell]),
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="shell", arguments={"command": "git status --short"})
                    ]
                ),
            ]
        ),
        middleware=[Guard("trust", deny="shell:git status*")],
    )

    events = [event async for event in runtime.run("check")]

    assert [event.type for event in events] == ["permission_request", "tool_denied"]
    assert events[0].data["reason"] == "blocked by guard policy"


async def test_guard_read_mode_allows_read_only_and_denies_destructive_tools():
    calls = []

    @tool(read_only=True)
    def read_file(path: str) -> str:
        """Read a file."""
        calls.append(("read", path))
        return "contents"

    @tool(destructive=True)
    def write_file(path: str, content: str) -> str:
        """Write a file."""
        calls.append(("write", path, content))
        return "wrote"

    agent = Agent(
        name="guarded",
        instructions="Use tools.",
        tools=[read_file, write_file],
    )
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="read_file", arguments={"path": "a.py"})
                    ]
                ),
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="write_file",
                            arguments={"path": "a.py", "content": "x"},
                        )
                    ]
                ),
            ]
        ),
        middleware=[Guard("read")],
    )

    events = [event async for event in runtime.run("work")]

    assert calls == [("read", "a.py")]
    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "permission_request",
        "tool_denied",
    ]
    assert events[2].data["reason"] == "destructive tool use"


async def test_guard_ask_mode_uses_callback_for_sensitive_tools():
    calls = []

    @tool(permission=ask_before("running shell commands"))
    def shell(command: str) -> str:
        """Run a command."""
        calls.append(command)
        return "ok"

    agent = Agent(name="guarded", instructions="Run commands.", tools=[shell])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[ToolCall(name="shell", arguments={"command": "pytest"})]
                ),
                ModelResult(message="finished"),
            ]
        ),
        middleware=[
            Guard(
                "ask",
                ask_user=lambda request: request.reason == "running shell commands",
            )
        ],
    )

    events = [event async for event in runtime.run("test")]

    assert calls == ["pytest"]
    assert [event.type for event in events] == [
        "permission_request",
        "tool_call",
        "tool_result",
        "message",
    ]
    assert events[0].data["reason"] == "running shell commands"


async def test_guard_trust_mode_allows_sensitive_tools_without_prompt():
    calls = []

    @tool(destructive=True)
    def write_file(path: str) -> str:
        """Write a file."""
        calls.append(path)
        return "wrote"

    agent = Agent(name="trusted", instructions="Write files.", tools=[write_file])
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="write_file", arguments={"path": "a.py"})
                    ]
                ),
                ModelResult(message="finished"),
            ]
        ),
        middleware=[Guard("trust")],
    )

    events = [event async for event in runtime.run("write")]

    assert calls == ["a.py"]
    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "message",
    ]


async def test_guard_ask_mode_allows_plain_tools_without_prompt():
    calls = []

    @tool
    def echo(text: str) -> str:
        """Echo text."""
        calls.append(text)
        return text

    runtime = Runtime(
        agent=Agent(name="plain", instructions="Echo.", tools=[echo]),
        model=ScriptedModel(
            [
                ModelResult(tool_calls=[ToolCall(name="echo", arguments={"text": "ok"})]),
                ModelResult(message="finished"),
            ]
        ),
        middleware=[Guard("ask", ask_user=lambda request: False)],
    )

    events = [event async for event in runtime.run("echo")]

    assert calls == ["ok"]
    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "message",
    ]


async def test_guard_ask_mode_can_remember_allowed_tool_decisions():
    prompts = []
    calls = []

    @tool(permission=ask_before("running shell commands"))
    def shell(command: str) -> str:
        """Run a command."""
        calls.append(command)
        return "ok"

    guard = Guard(
        "ask",
        ask_user=lambda request: prompts.append(request.arguments["command"]) or True,
        remember=True,
    )
    runtime = Runtime(
        agent=Agent(name="trusted", instructions="Run commands.", tools=[shell]),
        model=ScriptedModel(
            [
                ModelResult(tool_calls=[ToolCall(name="shell", arguments={"command": "pytest"})]),
                ModelResult(tool_calls=[ToolCall(name="shell", arguments={"command": "ruff check"})]),
                ModelResult(message="finished"),
            ]
        ),
        middleware=[guard],
    )

    events = [event async for event in runtime.run("check")]

    assert prompts == ["pytest"]
    assert calls == ["pytest", "ruff check"]
    assert [event.type for event in events] == [
        "permission_request",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "message",
    ]


async def test_guard_remembers_runtime_bound_permission_decisions():
    prompts = []
    calls = []

    @tool(permission=ask_before("running shell commands"))
    def shell(command: str) -> str:
        """Run a command."""
        calls.append(command)
        return "ok"

    guard = Guard("ask", remember=True)
    runtime = Runtime(
        agent=Agent(name="trusted", instructions="Run commands.", tools=[shell]),
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="shell", arguments={"command": "pytest"})
                    ]
                ),
                ModelResult(
                    tool_calls=[
                        ToolCall(name="shell", arguments={"command": "ruff check"})
                    ]
                ),
                ModelResult(message="finished"),
            ]
        ),
        middleware=[guard],
    )

    token = push_ask_user(
        lambda request: prompts.append(request.arguments["command"]) or True
    )
    try:
        events = [event async for event in runtime.run("check")]
    finally:
        pop_ask_user(token)

    assert prompts == ["pytest"]
    assert calls == ["pytest", "ruff check"]
    assert [event.type for event in events] == [
        "permission_request",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "message",
    ]


async def test_guard_remember_reason_keeps_different_reasons_separate():
    prompts = []
    calls = []

    @tool(permission=ask_before("writing files"))
    def write_file(path: str) -> str:
        """Write a file."""
        calls.append(("write", path))
        return "wrote"

    @tool(permission=ask_before("running shell commands"))
    def shell(command: str) -> str:
        """Run a command."""
        calls.append(("shell", command))
        return "ok"

    guard = Guard(
        "ask",
        ask_user=lambda request: prompts.append(request.reason) or True,
        remember="reason",
    )
    runtime = Runtime(
        agent=Agent(
            name="trusted",
            instructions="Work.",
            tools=[write_file, shell],
        ),
        model=ScriptedModel(
            [
                ModelResult(tool_calls=[ToolCall(name="write_file", arguments={"path": "a.py"})]),
                ModelResult(tool_calls=[ToolCall(name="shell", arguments={"command": "pytest"})]),
                ModelResult(tool_calls=[ToolCall(name="write_file", arguments={"path": "b.py"})]),
                ModelResult(message="finished"),
            ]
        ),
        middleware=[guard],
    )

    events = [event async for event in runtime.run("work")]

    assert prompts == ["writing files", "running shell commands"]
    assert calls == [
        ("write", "a.py"),
        ("shell", "pytest"),
        ("write", "b.py"),
    ]
    assert [event.type for event in events].count("permission_request") == 2


async def test_guard_remembered_permissions_can_be_cleared():
    prompts = []
    calls = []

    @tool(permission=ask_before("running shell commands"))
    def shell(command: str) -> str:
        """Run a command."""
        calls.append(command)
        return "ok"

    guard = Guard(
        "ask",
        ask_user=lambda request: prompts.append(request.arguments["command"]) or True,
        remember=True,
    )

    async def run_once(command: str):
        runtime = Runtime(
            agent=Agent(name="trusted", instructions="Run.", tools=[shell]),
            model=ScriptedModel(
                [
                    ModelResult(tool_calls=[ToolCall(name="shell", arguments={"command": command})]),
                    ModelResult(message="finished"),
                ]
            ),
            middleware=[guard],
        )
        return [event async for event in runtime.run(command)]

    await run_once("pytest")
    guard.clear()
    await run_once("ruff check")

    assert prompts == ["pytest", "ruff check"]
    assert calls == ["pytest", "ruff check"]


async def test_guard_allow_rule_skips_prompt_for_matching_tools():
    prompts = []
    calls = []

    @tool(permission=ask_before("running shell commands"))
    def shell(command: str) -> str:
        """Run a command."""
        calls.append(command)
        return "ok"

    runtime = Runtime(
        agent=Agent(name="trusted", instructions="Run.", tools=[shell]),
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[ToolCall(name="shell", arguments={"command": "pytest"})]
                ),
                ModelResult(message="finished"),
            ]
        ),
        middleware=[
            Guard(
                "ask",
                ask_user=lambda request: prompts.append(request.tool.name) or False,
                allow=["bash"],
            )
        ],
    )

    events = [event async for event in runtime.run("test")]

    assert prompts == []
    assert calls == ["pytest"]
    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "message",
    ]


async def test_guard_deny_rule_blocks_matching_arguments():
    prompts = []
    calls = []

    @tool(permission=ask_before("running shell commands"))
    def shell(command: str) -> str:
        """Run a command."""
        calls.append(command)
        return "ok"

    runtime = Runtime(
        agent=Agent(name="guarded", instructions="Run.", tools=[shell]),
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="shell", arguments={"command": "rm -rf build"})
                    ]
                ),
            ]
        ),
        middleware=[
            Guard(
                "ask",
                ask_user=lambda request: prompts.append(request.tool.name) or True,
                deny=lambda request: request.arguments["command"].startswith("rm "),
            )
        ],
    )

    events = [event async for event in runtime.run("clean")]

    assert prompts == []
    assert calls == []
    assert [event.type for event in events] == [
        "permission_request",
        "tool_denied",
    ]
    assert events[0].data["reason"] == "blocked by guard policy"
    assert "blocked by guard policy" in events[1].data["message"]


async def test_guard_string_rule_can_deny_tool_argument_pattern():
    calls = []

    @tool(permission=ask_before("running shell commands"))
    def shell(command: str) -> str:
        """Run a command."""
        calls.append(command)
        return "ok"

    runtime = Runtime(
        agent=Agent(name="guarded", instructions="Run.", tools=[shell]),
        model=ScriptedModel(
            [
                ModelResult(tool_calls=[ToolCall(name="shell", arguments={"command": "rm -rf build"})]),
            ]
        ),
        middleware=[Guard("trust", deny="shell:rm *")],
    )

    events = [event async for event in runtime.run("clean")]

    assert calls == []
    assert [event.type for event in events] == ["permission_request", "tool_denied"]
    assert events[-1].data["message"] == (
        "Permission denied for shell: blocked by guard policy"
    )


async def test_guard_string_rule_can_allow_tool_argument_pattern():
    calls = []

    @tool(permission=ask_before("running shell commands"))
    def shell(command: str) -> str:
        """Run a command."""
        calls.append(command)
        return "ok"

    runtime = Runtime(
        agent=Agent(name="guarded", instructions="Run.", tools=[shell]),
        model=ScriptedModel(
            [
                ModelResult(tool_calls=[ToolCall(name="shell", arguments={"command": "pytest tests"})]),
                ModelResult(tool_calls=[ToolCall(name="shell", arguments={"command": "rm -rf build"})]),
            ]
        ),
        middleware=[Guard("read", allow="shell:pytest*")],
    )

    events = [event async for event in runtime.run("check")]

    assert calls == ["pytest tests"]
    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "permission_request",
        "tool_denied",
    ]


async def test_guard_deny_rule_takes_priority_over_allow_rule():
    calls = []

    @tool(permission=ask_before("running shell commands"))
    def shell(command: str) -> str:
        """Run a command."""
        calls.append(command)
        return "ok"

    runtime = Runtime(
        agent=Agent(name="guarded", instructions="Run.", tools=[shell]),
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[ToolCall(name="shell", arguments={"command": "pytest"})]
                ),
            ]
        ),
        middleware=[
            Guard(
                "ask",
                allow=["shell"],
                deny=lambda request: request.arguments["command"] == "pytest",
            )
        ],
    )

    events = [event async for event in runtime.run("test")]

    assert calls == []
    assert [event.type for event in events] == [
        "permission_request",
        "tool_denied",
    ]


def test_guard_rejects_unknown_remember_mode():
    try:
        Guard("ask", remember="forever")
    except ValueError as error:
        assert "remember" in str(error)
    else:
        raise AssertionError("Guard should reject invalid remember mode")
