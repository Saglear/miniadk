import miniadk.adapters.cli as cli_adapter
import miniadk.adapters._cli_input as cli_input
from miniadk import (
    Agent,
    CLIRenderer,
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
    make_edit_file,
    make_write_file,
    run_cli,
    sessions,
    tool,
)
from miniadk.presets import coder


class StreamingScriptedModel:
    async def complete(self, messages, tools):
        raise AssertionError("streaming model should use stream")

    async def stream(self, messages, tools):
        yield ModelStreamEvent(delta="hel")
        yield ModelStreamEvent(delta="lo")
        yield ModelStreamEvent(result=ModelResult(message="hello"))


class InspectingStreamingModel:
    async def complete(self, messages, tools):
        raise AssertionError("streaming model should use stream")

    async def stream(self, messages, tools):
        yield ModelStreamEvent(thinking="checking context")
        yield ModelStreamEvent(
            tool_call=ToolCallDelta(
                index=0,
                id="call_1",
                name="read_file",
                arguments='{"path":"README.md"}',
            )
        )
        yield ModelStreamEvent(delta="done")
        yield ModelStreamEvent(result=ModelResult(message="done"))


def test_run_cli_renders_runtime_messages():
    inputs = iter(["hello", "/exit"])
    outputs = []

    agent = Agent(name="cli", instructions="Answer.", tools=[])
    model = ScriptedModel([ModelResult(message="hi from agent")])

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
    )

    assert outputs == ["hi from agent"]


def test_run_cli_can_use_default_model_helper(monkeypatch):
    inputs = iter(["hello", "/exit"])
    outputs = []
    built = ScriptedModel([ModelResult(message="hi from default")])

    monkeypatch.setattr(cli_adapter, "default_model", lambda: built)

    run_cli(
        Agent(name="cli", instructions="Answer.", tools=[]),
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
    )

    assert outputs == ["hi from default"]
    assert built.calls


def test_run_cli_passes_max_steps_to_runtime():
    inputs = iter(["keep going", "/exit"])
    outputs = []

    run_cli(
        Agent(name="cli", instructions="Answer.", tools=[]),
        model=ScriptedModel([ModelResult()]),
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        max_steps=1,
    )

    assert outputs == ["error: Runtime stopped after 1 steps"]


def test_run_cli_accepts_custom_middleware():
    inputs = iter(["hello", "/exit"])
    seen = []

    class Audit:
        async def before_model_call(self, state):
            seen.append(("before", state.step, len(state.messages)))

    run_cli(
        Agent(name="cli", instructions="Answer.", tools=[]),
        model=ScriptedModel([ModelResult(message="hi")]),
        input_func=lambda prompt: next(inputs),
        output_func=lambda text: None,
        middleware=[Audit()],
    )

    assert seen == [("before", 1, 2)]


def test_run_cli_combines_preset_and_custom_middleware(tmp_path):
    target = tmp_path / "notes.txt"
    inputs = iter(["write", "n", "/exit"])
    seen = []

    class Audit:
        async def before_model_call(self, state):
            seen.append(("before", state.step))

    run_cli(
        coder(tmp_path, shell=False, chat=False),
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="write_file",
                            arguments={"path": "notes.txt", "content": "hello"},
                        )
                    ]
                )
            ]
        ),
        input_func=lambda prompt: next(inputs),
        output_func=lambda text: None,
        middleware=[Audit()],
    )

    assert seen == [("before", 1)]
    assert not target.exists()


def test_run_cli_coder_passes_permission_prompt_to_child_agents(tmp_path):
    calls = []

    @tool(destructive=True)
    def write_note(path: str) -> str:
        """Write a note."""
        calls.append(path)
        return f"wrote {path}"

    helper = Agent(name="helper", instructions="Help.")
    parent_model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[
                    ToolCall(
                        name="spawn_agent",
                        arguments={"agent": "helper", "prompt": "write"},
                    )
                ]
            ),
            ModelResult(message="done"),
        ]
    )
    child_model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[ToolCall(name="write_note", arguments={"path": "a.txt"})]
            ),
            ModelResult(message="child done"),
        ]
    )
    inputs = iter(["run helper", "y", "/exit"])
    prompts = []

    def input_func(prompt):
        prompts.append(prompt)
        return next(inputs)

    run_cli(
        coder(
            tmp_path,
            files=False,
            shell=False,
            chat=False,
            agents=[helper],
            agent_models={"helper": child_model},
            agent_tools={"helper": [write_note]},
        ),
        model=parent_model,
        input_func=input_func,
        output_func=lambda text: None,
    )

    assert calls == ["a.txt"]
    assert any("Allow write_note" in prompt for prompt in prompts)


def test_run_cli_does_not_mutate_shared_guard_when_binding_prompt():
    prompts = []

    @tool(destructive=True)
    def write_file(path: str) -> str:
        """Write a file."""
        return f"wrote {path}"

    agent = Agent(name="cli", instructions="Use tools.", tools=[write_file])
    guard = Guard("ask")

    def run_once(label: str):
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
        inputs = iter(["write", "y", "/exit"])

        def input_func(prompt):
            prompts.append((label, prompt))
            return next(inputs)

        run_cli(
            agent,
            model=model,
            input_func=input_func,
            output_func=lambda text: None,
            middleware=[guard],
        )

    run_once("first")
    run_once("second")

    assert guard.ask_user is None
    permission_prompts = [
        item
        for item in prompts
        if "Allow write_file" in item[1]
    ]
    assert [label for label, _ in permission_prompts] == ["first", "second"]


def test_run_cli_renders_streaming_message_without_duplicate_final_text():
    inputs = iter(["hello", "/exit"])
    outputs = []

    agent = Agent(name="cli", instructions="Answer.", tools=[])

    run_cli(
        agent,
        model=StreamingScriptedModel(),
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
    )

    assert outputs == ["hel", "lo", "\n"]


def test_run_cli_plain_mode_ignores_internal_streaming_deltas():
    inputs = iter(["hello", "/exit"])
    outputs = []

    run_cli(
        Agent(name="cli", instructions="Answer.", tools=[]),
        model=InspectingStreamingModel(),
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="plain",
    )

    assert outputs == ["done", "\n"]


def test_run_cli_pretty_mode_renders_internal_streaming_deltas():
    inputs = iter(["hello", "/exit"])
    outputs = []

    run_cli(
        Agent(name="cli", instructions="Answer.", tools=[]),
        model=InspectingStreamingModel(),
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="pretty",
    )

    rendered = "\n".join(outputs)
    assert "◇ thinking" in rendered
    assert "checking context" in rendered
    assert "◇ preparing" in rendered
    assert "read_file" in rendered
    assert "path" in rendered


def test_run_cli_expands_user_invoked_skill():
    inputs = iter(["/review src/app.py", "/exit"])
    outputs = []

    agent = Agent(
        name="cli",
        instructions="Answer.",
        skills=SkillRegistry(
            [
                Skill(
                    name="review",
                    description="Review code.",
                    body="Review this file: $ARGUMENTS",
                )
            ]
        ),
    )
    model = ScriptedModel([ModelResult(message="reviewed")])

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
    )

    assert outputs == ["reviewed"]
    messages, tools = model.calls[0]
    assert messages[-1].content == "Review this file: src/app.py"
    assert [tool.name for tool in tools] == ["skill"]


def test_run_cli_accepts_temporary_tools():
    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    inputs = iter(["ping", "/exit"])
    outputs = []
    model = ScriptedModel(
        [
            ModelResult(tool_calls=[ToolCall(name="ping", arguments={})]),
            ModelResult(message="done"),
        ]
    )

    run_cli(
        Agent(name="cli", instructions="Use tools."),
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        tools=[ping],
    )

    assert outputs == ["tool: ping", "pong", "done"]
    assert [item.name for item in model.calls[0][1]] == ["ping"]


def test_run_cli_temporary_tools_override_agent_tools_by_name():
    @tool
    def base_check(path: str) -> str:
        """Check a path."""
        return f"base:{path}"

    @tool
    def extra_check(path: str) -> str:
        """Check a path."""
        return f"extra:{path}"

    base_check.name = "check"
    extra_check.name = "check"
    inputs = iter(["check", "/exit"])
    outputs = []
    model = ScriptedModel(
        [
            ModelResult(tool_calls=[ToolCall(name="check", arguments={"path": "app.py"})]),
            ModelResult(message="done"),
        ]
    )

    run_cli(
        Agent(name="cli", instructions="Use tools.", tools=[base_check]),
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        tools=[extra_check],
    )

    assert outputs == ["tool: check", "extra:app.py", "done"]
    assert [item.name for item in model.calls[0][1]] == ["check"]


def test_run_cli_user_invoked_skill_can_use_temporary_tools():
    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    inputs = iter(["/review src/app.py", "/exit"])
    outputs = []
    agent = Agent(
        name="cli",
        instructions="Answer.",
        skills=SkillRegistry(
            [
                Skill(
                    name="review",
                    description="Review code.",
                    body="Review this file: $ARGUMENTS",
                    allowed_tools=["ping"],
                )
            ]
        ),
    )
    model = ScriptedModel(
        [
            ModelResult(tool_calls=[ToolCall(name="ping", arguments={})]),
            ModelResult(message="reviewed"),
        ]
    )

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        tools=[ping],
    )

    assert outputs == ["tool: ping", "pong", "reviewed"]
    assert [item.name for item in model.calls[0][1]] == ["skill", "ping"]


def test_run_cli_pretty_mode_renders_status_and_events():
    @tool
    async def echo(text: str, progress) -> str:
        await progress("working", step=1)
        return text

    inputs = iter(["hello", "/exit"])
    outputs = []

    agent = Agent(name="cli", instructions="Answer.", tools=[echo])
    model = ScriptedModel(
        [
            ModelResult(tool_calls=[ToolCall(name="echo", arguments={"text": "ok"})]),
            ModelResult(message="done"),
        ]
    )

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="pretty",
    )

    rendered = "\n".join(outputs)
    assert "miniadk" in rendered
    assert "cli" in rendered
    assert "ScriptedModel" in rendered
    assert "user #1" in rendered
    assert "/usage" in rendered
    assert "/undo" in rendered
    assert "/retry" in rendered
    assert "◇ tool" in rendered
    assert "◇ progress" in rendered
    assert "working" in rendered
    assert "echo" in rendered
    assert "assistant" in rendered
    assert "done" in rendered
    assert "◇ working" in rendered
    assert "◇ ready" in rendered


def test_run_cli_accepts_custom_renderer_for_adk_products():
    class RecordingRenderer(CLIRenderer):
        def __init__(self):
            super().__init__(lambda text: None, mode="plain")
            self.events = []

        def intro(self, status):
            self.events.append(("intro", status.agent_name))

        def user(self, text):
            self.events.append(("user", text))

        def event(self, event):
            self.events.append(("event", event.type))

        def run_start(self):
            self.events.append(("run_start", ""))

        def run_end(self):
            self.events.append(("run_end", ""))

        def prompt(self, text):
            return "custom> "

    inputs = iter(["hello", "/exit"])
    renderer = RecordingRenderer()

    run_cli(
        Agent(name="cli", instructions="Answer.", tools=[]),
        model=ScriptedModel([ModelResult(message="hi")]),
        input_func=lambda prompt: next(inputs),
        output_func=lambda text: None,
        renderer=renderer,
    )

    assert renderer.events == [
        ("intro", "cli"),
        ("user", "hello"),
        ("run_start", ""),
        ("event", "message"),
        ("run_end", ""),
    ]


def test_run_cli_uses_prompt_toolkit_input_for_real_tty(monkeypatch):
    created = []

    class FakeCLIInput:
        def __init__(self, *, prompt, commands, history_path=None):
            created.append((prompt, commands, history_path))
            self.items = iter(["/exit"])

        def __call__(self, prompt=None):
            return next(self.items)

    monkeypatch.setattr(cli_adapter, "should_use_prompt_toolkit", lambda input_func, output_func: True)
    monkeypatch.setattr(cli_adapter, "CLIInput", FakeCLIInput)

    run_cli(
        Agent(name="cli", instructions="Answer.", tools=[]),
        model=ScriptedModel([]),
    )

    assert created
    assert "/help" in created[0][1]
    assert "/exit" in created[0][1]
    assert "/undo" in created[0][1]
    assert "/retry" in created[0][1]
    assert "/new" in created[0][1]


def test_run_cli_intro_mentions_prompt_toolkit_input_features(monkeypatch):
    outputs = []

    class FakeCLIInput:
        def __init__(self, *, prompt, commands, history_path=None):
            self.items = iter(["/exit"])

        def __call__(self, prompt=None):
            return next(self.items)

    monkeypatch.setattr(cli_adapter, "should_use_prompt_toolkit", lambda input_func, output_func: True)
    monkeypatch.setattr(cli_adapter, "CLIInput", FakeCLIInput)

    run_cli(
        Agent(name="cli", instructions="Answer.", tools=[]),
        model=ScriptedModel([]),
        output_func=outputs.append,
        output_mode="pretty",
    )

    rendered = "\n".join(outputs)
    assert "history" in rendered
    assert "slash completion" in rendered
    assert "multiline input" in rendered


def test_run_cli_uses_prompt_toolkit_single_line_ask_for_permissions(monkeypatch):
    calls = []

    class FakeCLIInput:
        def __init__(self, *, prompt, commands, history_path=None):
            self.items = iter(["write", "/exit"])

        def __call__(self, prompt=None):
            calls.append(("main", prompt))
            return next(self.items)

        def ask(self, prompt):
            calls.append(("ask", prompt))
            return "y"

    @tool(destructive=True)
    def write_note(path: str) -> str:
        """Write a note."""
        return path

    monkeypatch.setattr(cli_adapter, "should_use_prompt_toolkit", lambda input_func, output_func: True)
    monkeypatch.setattr(cli_adapter, "CLIInput", FakeCLIInput)

    run_cli(
        Agent(name="cli", instructions="Use tools.", tools=[write_note]),
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="write_note", arguments={"path": "a.txt"})
                    ]
                ),
                ModelResult(message="done"),
            ]
        ),
    )

    assert [kind for kind, _ in calls] == ["main", "ask", "main"]
    assert "allow" in calls[1][1].lower()


def test_should_use_prompt_toolkit_respects_custom_io(monkeypatch):
    monkeypatch.setattr(cli_input.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli_input.sys.stdout, "isatty", lambda: True)

    assert cli_input.should_use_prompt_toolkit(input, print) is True
    assert cli_input.should_use_prompt_toolkit(lambda prompt: "", print) is False
    assert cli_input.should_use_prompt_toolkit(input, lambda text: None) is False


def test_run_cli_pretty_mode_uses_custom_prompt_text():
    inputs = iter(["/exit"])
    prompts = []

    agent = Agent(name="cli", instructions="Answer.", tools=[])
    model = ScriptedModel([])

    def input_func(prompt):
        prompts.append(prompt)
        return next(inputs)

    run_cli(
        agent,
        model=model,
        input_func=input_func,
        output_func=lambda text: None,
        prompt="claude > ",
        output_mode="pretty",
    )

    assert "claude" in prompts[0]
    assert "miniadk" not in prompts[0]


def test_run_cli_handles_builtin_help_without_model_call():
    inputs = iter(["/help", "/exit"])
    outputs = []

    agent = Agent(
        name="cli",
        instructions="Answer.",
        skills=SkillRegistry(
            [
                Skill(
                    name="review",
                    description="Review code.",
                    body="Review this file: $ARGUMENTS",
                )
            ]
        ),
    )
    model = ScriptedModel([ModelResult(message="should not be used")])

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="plain",
    )

    rendered = "\n".join(outputs)
    assert "Inspect" in rendered
    assert "Capabilities" in rendered
    assert "Session" in rendered
    assert "/status" in rendered
    assert "/usage" in rendered
    assert "/theme" in rendered
    assert "/undo" in rendered
    assert "/retry" in rendered
    assert "/new" in rendered
    assert "/review" in rendered
    assert model.calls == []


def test_run_cli_status_uses_session_stats():
    inputs = iter(["hello", "/status", "/exit"])
    outputs = []

    agent = Agent(name="cli", instructions="Answer.", tools=[])
    model = ScriptedModel([ModelResult(message="hi")])

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="plain",
    )

    rendered = "\n".join(outputs)
    assert "messages: 3" in rendered
    assert "tool calls: 0" in rendered
    assert "chars:" in rendered
    assert "last role: assistant" in rendered


def test_run_cli_usage_shows_session_counts_without_model_call():
    inputs = iter(["hello", "/usage", "/exit"])
    outputs = []

    agent = Agent(name="cli", instructions="Answer.", tools=[])
    model = ScriptedModel([ModelResult(message="hi")])

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="plain",
    )

    rendered = "\n".join(outputs)
    assert "Usage" in rendered
    assert "messages: 3" in rendered
    assert "tool calls: 0" in rendered
    assert "chars:" in rendered
    assert len(model.calls) == 1


def test_run_cli_theme_describes_active_theme_without_model_call():
    inputs = iter(["/theme", "/exit"])
    outputs = []

    run_cli(
        Agent(name="cli", instructions="Answer.", tools=[]),
        model=ScriptedModel([ModelResult(message="should not be used")]),
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="plain",
    )

    rendered = "\n".join(outputs)
    assert "Theme" in rendered
    assert "name: miniadk" in rendered
    assert "accent: \\033" in rendered


def test_run_cli_todos_shows_agentic_todo_store_without_model_call(tmp_path):
    built = coder(tmp_path, files=False, shell=False)
    built.todos.replace(
        [
            {"content": "Write code", "status": "completed"},
            {"content": "Run tests", "status": "pending"},
        ]
    )
    inputs = iter(["/todos", "/exit"])
    outputs = []
    model = ScriptedModel([ModelResult(message="should not be used")])

    run_cli(
        built,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="plain",
    )

    rendered = "\n".join(outputs)
    assert "Todos" in rendered
    assert "- 1. [completed] Write code" in rendered
    assert "- 2. [pending] Run tests" in rendered
    assert model.calls == []


def test_run_cli_todos_reports_missing_todo_store_without_model_call():
    inputs = iter(["/todos", "/exit"])
    outputs = []
    model = ScriptedModel([ModelResult(message="should not be used")])

    run_cli(
        Agent(name="cli", instructions="Answer.", tools=[]),
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="plain",
    )

    assert outputs == ["no todo store"]
    assert model.calls == []


def test_run_cli_persists_session_to_path(tmp_path):
    path = tmp_path / "session.json"
    inputs = iter(["hello", "/exit"])

    agent = Agent(name="cli", instructions="Answer.", tools=[])
    model = ScriptedModel([ModelResult(message="saved")])

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=lambda text: None,
        output_mode="plain",
        session=path,
    )

    loaded = Session.load(path)
    assert [message.role for message in loaded.messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert loaded.messages[-1].content == "saved"


def test_run_cli_resumes_session_from_path(tmp_path):
    path = tmp_path / "session.json"
    Session(
        [
            Message("system", "Answer."),
            Message("user", "old"),
            Message("assistant", "old answer"),
        ]
    ).save(path)
    inputs = iter(["again", "/exit"])

    agent = Agent(name="cli", instructions="Answer.", tools=[])
    model = ScriptedModel([ModelResult(message="new answer")])

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=lambda text: None,
        output_mode="plain",
        session=path,
    )

    messages, _ = model.calls[0]
    assert [message.content for message in messages] == [
        "Answer.",
        "old",
        "old answer",
        "again",
    ]
    loaded = Session.load(path)
    assert loaded.messages[-1].content == "new answer"


def test_run_cli_can_use_session_store_path(tmp_path):
    store = sessions(tmp_path / "sessions")
    inputs = iter(["hello", "/exit"])

    agent = Agent(name="cli", instructions="Answer.", tools=[])
    model = ScriptedModel([ModelResult(message="stored")])

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=lambda text: None,
        output_mode="plain",
        session=store.path("main"),
    )

    assert store.names() == ["main"]
    assert store.load("main").messages[-1].content == "stored"


def test_run_cli_can_use_default_session_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inputs = iter(["hello", "/exit"])

    agent = Agent(name="cli", instructions="Answer.", tools=[])
    model = ScriptedModel([ModelResult(message="stored")])

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=lambda text: None,
        output_mode="plain",
        session=True,
    )

    loaded = Session.load(tmp_path / ".miniadk" / "sessions" / "cli.json")
    assert loaded.messages[-1].content == "stored"


def test_run_cli_resumes_default_session_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    default_path = tmp_path / ".miniadk" / "sessions" / "cli.json"
    Session(
        [
            Message("system", "Answer."),
            Message("user", "old"),
            Message("assistant", "old answer"),
        ]
    ).save(default_path)
    inputs = iter(["again", "/exit"])

    agent = Agent(name="cli", instructions="Answer.", tools=[])
    model = ScriptedModel([ModelResult(message="new answer")])

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=lambda text: None,
        output_mode="plain",
        session=True,
    )

    messages, _ = model.calls[0]
    assert [message.content for message in messages] == [
        "Answer.",
        "old",
        "old answer",
        "again",
    ]
    assert Session.load(default_path).messages[-1].content == "new answer"


def test_run_cli_lists_builtin_tools_without_model_call():
    @tool
    def echo(text: str) -> str:
        """Echo text."""
        return text

    @tool(read_only=True, concurrency_safe=True)
    def read_note(path: str) -> str:
        """Read note."""
        return path

    @tool(destructive=True)
    def delete_note(path: str) -> str:
        """Delete note."""
        return path

    @tool(read_only=lambda path: path.endswith(".md"))
    def dynamic_read(path: str) -> str:
        """Dynamic read."""
        return path

    inputs = iter(["/tools", "/exit"])
    outputs = []

    agent = Agent(
        name="cli",
        instructions="Answer.",
        tools=[echo, read_note, delete_note, dynamic_read],
    )
    model = ScriptedModel([ModelResult(message="should not be used")])

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="plain",
    )

    rendered = "\n".join(outputs)
    assert "Tools" in rendered
    assert "echo: Echo text." in rendered
    assert "read_note [read-only, safe]: Read note." in rendered
    assert "delete_note [destructive]: Delete note." in rendered
    assert "dynamic_read: Dynamic read." in rendered
    assert "dynamic_read [" not in rendered
    assert model.calls == []


def test_run_cli_lists_skills_and_marks_invocation_modes():
    inputs = iter(["/skills", "/exit"])
    outputs = []

    agent = Agent(
        name="cli",
        instructions="Answer.",
        skills=SkillRegistry(
            [
                Skill(
                    name="review",
                    description="Review code.",
                    body="Review this file: $ARGUMENTS",
                ),
                Skill(
                    name="internal",
                    description="Internal planning.",
                    body="Plan.",
                    user_invocable=False,
                ),
            ]
        ),
    )
    model = ScriptedModel([ModelResult(message="should not be used")])

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="plain",
    )

    rendered = "\n".join(outputs)
    assert "/review [user]: Review code." in rendered
    assert "/internal [model]: Internal planning." in rendered
    assert model.calls == []


def test_run_cli_lists_skill_problems_without_model_call(tmp_path):
    skill_dir = tmp_path / "skills" / "broken"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: broken\nunknown: value\n---\n",
        encoding="utf-8",
    )
    inputs = iter(["/skills", "/exit"])
    outputs = []

    agent = Agent(
        name="cli",
        instructions="Answer.",
        skills=SkillRegistry.from_paths(tmp_path / "skills"),
    )
    model = ScriptedModel([ModelResult(message="should not be used")])

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="plain",
    )

    rendered = "\n".join(outputs)
    assert "Skill problems" in rendered
    assert "broken: body is empty" in rendered
    assert "broken: unknown metadata key: unknown" in rendered
    assert model.calls == []


def test_run_cli_reports_model_only_skill_without_model_call():
    inputs = iter(["/internal", "/exit"])
    outputs = []

    agent = Agent(
        name="cli",
        instructions="Answer.",
        skills=SkillRegistry(
            [
                Skill(
                    name="internal",
                    description="Internal planning.",
                    body="Plan.",
                    user_invocable=False,
                )
            ]
        ),
    )
    model = ScriptedModel([ModelResult(message="should not be used")])

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="plain",
    )

    assert outputs == ["skill /internal is model-only"]
    assert model.calls == []


def test_run_cli_compact_shows_recent_transcript_without_model_call():
    inputs = iter(["hello", "/compact", "/exit"])
    outputs = []

    agent = Agent(name="cli", instructions="Answer.", tools=[])
    model = ScriptedModel([ModelResult(message="hi\nthere")])

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="plain",
    )

    rendered = "\n".join(outputs)
    assert "hi\nthere" in rendered
    assert "user: hello" in rendered
    assert "assistant: hi there" in rendered
    assert len(model.calls) == 1


def test_run_cli_compact_can_summarize_and_compact_session():
    inputs = iter(["old", "recent", "/compact", "/status", "/exit"])
    outputs = []

    agent = Agent(name="cli", instructions="Answer.", tools=[])
    model = ScriptedModel(
        [
            ModelResult(message="old answer"),
            ModelResult(message="recent answer"),
            ModelResult(message="Old turn summary."),
        ]
    )

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="plain",
        compact=True,
        compact_keep=2,
    )

    rendered = "\n".join(outputs)
    assert "Old turn summary." in rendered
    assert "messages: 4" in rendered
    assert "Summarize the conversation" in model.calls[-1][0][0].content
    assert model.calls[-1][0][1].content == "user: old\nassistant: old answer"


def test_run_cli_can_auto_compact_session_after_run():
    inputs = iter(["hello", "/status", "/exit"])
    outputs = []

    agent = Agent(name="cli", instructions="Answer.", tools=[])
    model = ScriptedModel(
        [
            ModelResult(message="hi"),
            ModelResult(message="Conversation summary."),
        ]
    )

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="plain",
        compact=Compact(chars=1, keep=1),
        compact_keep=1,
    )

    rendered = "\n".join(outputs)
    assert "messages: 3" in rendered
    assert "Summarize the conversation" in model.calls[-1][0][0].content
    assert model.calls[-1][0][1].content == "user: hello"


def test_run_cli_clear_is_adapter_only():
    inputs = iter(["/clear", "/exit"])
    outputs = []

    agent = Agent(name="cli", instructions="Answer.", tools=[])
    model = ScriptedModel([ModelResult(message="should not be used")])

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="plain",
    )

    assert outputs == [""]
    assert model.calls == []


def test_run_cli_reset_clears_conversation_history_without_model_call():
    inputs = iter(["hello", "/reset", "/status", "/exit"])
    outputs = []

    agent = Agent(name="cli", instructions="Answer.", tools=[])
    model = ScriptedModel([ModelResult(message="hi")])

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="plain",
    )

    rendered = "\n".join(outputs)
    assert "session reset" in rendered
    assert "messages: 1" in rendered
    assert "last role: system" in rendered
    assert len(model.calls) == 1


def test_run_cli_new_alias_clears_conversation_history_without_model_call():
    inputs = iter(["hello", "/new", "/status", "/exit"])
    outputs = []

    agent = Agent(name="cli", instructions="Answer.", tools=[])
    model = ScriptedModel([ModelResult(message="hi")])

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="plain",
    )

    rendered = "\n".join(outputs)
    assert "session reset" in rendered
    assert "messages: 1" in rendered
    assert len(model.calls) == 1


def test_run_cli_undo_removes_last_turn_without_model_call():
    inputs = iter(["first", "second", "/undo", "/status", "/exit"])
    outputs = []

    agent = Agent(name="cli", instructions="Answer.", tools=[])
    model = ScriptedModel(
        [
            ModelResult(message="one"),
            ModelResult(message="two"),
        ]
    )

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="plain",
    )

    rendered = "\n".join(outputs)
    assert "removed 2 messages" in rendered
    assert "messages: 3" in rendered
    assert len(model.calls) == 2


def test_run_cli_retry_reruns_last_turn():
    inputs = iter(["hello", "/retry", "/status", "/exit"])
    outputs = []

    agent = Agent(name="cli", instructions="Answer.", tools=[])
    model = ScriptedModel(
        [
            ModelResult(message="first"),
            ModelResult(message="second"),
        ]
    )

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=outputs.append,
        output_mode="plain",
    )

    rendered = "\n".join(outputs)
    assert "retrying last turn" in rendered
    assert "first" in outputs
    assert "second" in outputs
    assert "messages: 3" in rendered
    assert len(model.calls) == 2


def test_run_cli_reset_persists_cleared_session_to_path(tmp_path):
    path = tmp_path / "session.json"
    inputs = iter(["hello", "/reset", "/exit"])

    agent = Agent(name="cli", instructions="Answer.", tools=[])
    model = ScriptedModel([ModelResult(message="hi")])

    run_cli(
        agent,
        model=model,
        input_func=lambda prompt: next(inputs),
        output_func=lambda text: None,
        output_mode="plain",
        session=path,
    )

    assert Session.load(path).messages == [Message("system", "Answer.")]


def test_run_cli_can_allow_write_permission_prompt(tmp_path):
    target = tmp_path / "notes.txt"
    inputs = iter(["write", "y", "/exit"])
    prompts = []
    outputs = []

    agent = Agent(
        name="cli",
        instructions="Answer.",
        tools=[make_write_file(root=tmp_path)],
    )
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
            ModelResult(message="done"),
        ]
    )

    def input_func(prompt):
        prompts.append(prompt)
        return next(inputs)

    run_cli(
        agent,
        model=model,
        input_func=input_func,
        output_func=outputs.append,
        output_mode="plain",
    )

    assert "Allow write_file (writing files)? [y/N]" in prompts[1]
    assert target.read_text(encoding="utf-8") == "hello"
    assert "wrote notes.txt" in outputs
    assert outputs[-1] == "done"


def test_run_cli_can_deny_edit_permission_prompt(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("hello", encoding="utf-8")
    inputs = iter(["edit", "n", "/exit"])
    prompts = []
    outputs = []

    agent = Agent(
        name="cli",
        instructions="Answer.",
        tools=[make_edit_file(root=tmp_path)],
    )
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[
                    ToolCall(
                        name="edit_file",
                        arguments={"path": "notes.txt", "old": "hello", "new": "bye"},
                    )
                ]
            )
        ]
    )

    def input_func(prompt):
        prompts.append(prompt)
        return next(inputs)

    run_cli(
        agent,
        model=model,
        input_func=input_func,
        output_func=outputs.append,
        output_mode="plain",
    )

    assert "Allow edit_file (editing files)? [y/N]" in prompts[1]
    assert target.read_text(encoding="utf-8") == "hello"
    assert outputs == ["Permission denied for edit_file: editing files"]
    assert len(model.calls) == 1
