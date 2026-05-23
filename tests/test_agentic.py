from miniadk import (
    Agent,
    Agentic,
    AgenticPolicy,
    Message,
    ModelResult,
    Runtime,
    ScriptedModel,
    TodoStore,
    ToolCall,
    agentic,
    make_todo_read,
    make_todo_tool,
    tool,
)


async def collect(runtime: Runtime, text: str):
    return [event async for event in runtime.run(text)]


async def test_agentic_policy_continues_when_todos_are_not_complete():
    todo_store = TodoStore()
    todo_write = make_todo_tool(todo_store)
    agent = Agent(
        name="coder",
        instructions="Use todos for multi-step work.",
        tools=[todo_write],
    )
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="todo_write",
                            arguments={
                                "todos": [
                                    {"content": "Write code", "status": "in_progress"},
                                    {"content": "Run tests", "status": "pending"},
                                ]
                            },
                        )
                    ]
                ),
                ModelResult(message="Done too early."),
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="todo_write",
                            arguments={
                                "todos": [
                                    {"content": "Write code", "status": "completed"},
                                    {"content": "Run tests", "status": "completed"},
                                ]
                            },
                        )
                    ]
                ),
                ModelResult(message="Now done."),
            ]
        ),
        policy=AgenticPolicy(todo_store=todo_store),
    )

    events = await collect(runtime, "implement and test")

    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "message",
    ]
    assert events[-1].data == {"text": "Now done."}
    assert any(
        message.role == "user" and "not done yet" in message.content
        for message in runtime.messages
    )


async def test_agentic_policy_requires_todo_for_multistep_project_work():
    todo_store = TodoStore()
    runtime = Runtime(
        agent=Agent(name="coder", instructions="Use todos.", tools=[]),
        model=ScriptedModel(
            [
                ModelResult(message="Done too early."),
                ModelResult(message="Still too early."),
            ]
        ),
        policy=AgenticPolicy(todo_store=todo_store, max_stop_retries=1),
    )

    events = await collect(runtime, "fix the bug and run tests")

    assert [event.type for event in events] == ["message"]
    assert events[0].data == {"text": "Stopped after repeated policy continuations."}
    assert any(
        message.role == "user" and "Start by writing a short todo list" in message.content
        for message in runtime.messages
    )


async def test_agentic_policy_does_not_require_todo_for_simple_chat():
    todo_store = TodoStore()
    runtime = Runtime(
        agent=Agent(name="coder", instructions="Use todos.", tools=[]),
        model=ScriptedModel([ModelResult(message="Hello.")]),
        policy=AgenticPolicy(todo_store=todo_store),
    )

    events = await collect(runtime, "hello")

    assert [event.type for event in events] == ["message"]
    assert events[0].data == {"text": "Hello."}
    assert not any(
        message.role == "user" and "todo_write" in message.content
        for message in runtime.messages
    )


async def test_agentic_policy_can_stop_when_todos_are_blocked():
    todo_store = TodoStore()
    todo_write = make_todo_tool(todo_store)
    agent = Agent(
        name="coder",
        instructions="Use todos for multi-step work.",
        tools=[todo_write],
    )
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="todo_write",
                            arguments={
                                "todos": [
                                    {
                                        "content": "Wait for credentials",
                                        "status": "blocked",
                                    },
                                ]
                            },
                        )
                    ]
                ),
                ModelResult(message="Blocked on credentials."),
            ]
        ),
        policy=AgenticPolicy(todo_store=todo_store),
    )

    events = await collect(runtime, "deploy")

    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "message",
    ]
    assert events[-1].data == {"text": "Blocked on credentials."}
    assert not any(
        message.role == "user" and "not done yet" in message.content
        for message in runtime.messages
    )


async def test_agentic_helper_builds_agent_policy_and_todo_store():
    base = Agent(name="coder", instructions="Write code.")
    kit = agentic(base)

    assert isinstance(kit, Agentic)
    assert kit.agent.name == "coder"
    assert "keep a todo list" in kit.agent.instructions
    assert [tool.name for tool in kit.agent.tools] == ["todo_read", "todo_write"]

    runtime = Runtime(
        agent=kit.agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="todo_write",
                            arguments={
                                "todos": [
                                    {"content": "Write code", "status": "completed"},
                                ]
                            },
                        )
                    ]
                ),
                ModelResult(message="done"),
            ]
        ),
        policy=kit.policy,
    )

    events = await collect(runtime, "implement")

    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "message",
    ]
    assert kit.todos.items == [
        {"content": "Write code", "status": "completed"},
    ]


async def test_todo_tool_exposes_typed_schema_and_normalizes_items():
    store = TodoStore()
    todo_write = make_todo_tool(store)

    assert todo_write.input_schema["properties"]["todos"] == {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "minLength": 1},
                "status": {
                    "enum": ["pending", "in_progress", "completed", "blocked"],
                    "type": "string",
                },
            },
            "additionalProperties": False,
            "required": ["content"],
        },
    }

    result = await todo_write.run(
        todos=[
            {"content": " Write code ", "status": "completed"},
            {"content": "Run tests"},
        ],
    )

    assert result == "todo list updated: 2 items, 1 not completed"
    assert store.items == [
        {"content": "Write code", "status": "completed"},
        {"content": "Run tests", "status": "pending"},
    ]


async def test_todo_read_returns_current_summary():
    store = TodoStore(
        [
            {"content": "Write code", "status": "completed"},
            {"content": "Run tests", "status": "pending"},
            {"content": "Wait for API key", "status": "blocked"},
        ]
    )
    todo_read = make_todo_read(store)

    result = await todo_read.run()

    assert result == (
        "1. [completed] Write code\n"
        "2. [pending] Run tests\n"
        "3. [blocked] Wait for API key"
    )
    assert todo_read.is_read_only() is True
    assert todo_read.is_concurrency_safe() is True


async def test_todo_tool_rejects_invalid_items_without_replacing_store():
    store = TodoStore([{"content": "Keep", "status": "pending"}])
    todo_write = make_todo_tool(store)

    empty = await todo_write.validate(todos=[])
    blank = await todo_write.validate(todos=[{"content": ""}])
    invalid_status = await todo_write.validate(
        todos=[{"content": "Write code", "status": "skipped"}]
    )

    assert empty.ok is False
    assert empty.message == "Tool argument todos must have at least 1 items"
    assert blank.ok is False
    assert blank.message == "Tool argument todos[0].content must be at least 1 chars"
    assert invalid_status.ok is False
    assert invalid_status.message == (
        "Tool argument todos[0].status must be one of: pending, in_progress, "
        "completed, blocked"
    )
    assert await todo_write.run(todos=[]) == "todo list needs at least one item"
    assert await todo_write.run(todos=[{"content": "   "}]) == "todo 1 needs content"
    assert store.items == [{"content": "Keep", "status": "pending"}]

    assert await todo_write.run(
        todos=[{"content": "Write code", "status": "skipped"}]
    ) == (
        "todo 1 has invalid status: skipped. "
        "Use pending, in_progress, completed, or blocked."
    )
    assert store.items == [{"content": "Keep", "status": "pending"}]


async def test_todo_tool_rejects_multiple_in_progress_items():
    store = TodoStore([{"content": "Keep", "status": "pending"}])
    todo_write = make_todo_tool(store)

    result = await todo_write.run(
        todos=[
            {"content": "Write code", "status": "in_progress"},
            {"content": "Run tests", "status": "in_progress"},
        ]
    )
    validation = await todo_write.validate(
        todos=[
            {"content": "Write code", "status": "in_progress"},
            {"content": "Run tests", "status": "in_progress"},
        ]
    )

    assert result == "only one todo can be in_progress: 1, 2"
    assert validation.ok is False
    assert validation.message == "only one todo can be in_progress: 1, 2"
    assert store.items == [{"content": "Keep", "status": "pending"}]


async def test_agentic_chat_policy_stops_tool_use_for_low_intent_chat():
    calls = []

    def inspect_repo() -> str:
        calls.append("ran")
        return "files"

    inspect_repo_tool = tool(inspect_repo)
    agent = Agent(
        name="coder",
        instructions="Help with code.",
        tools=[inspect_repo_tool],
    )
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="inspect_repo", arguments={}),
                    ],
                ),
            ]
        ),
        policy=AgenticPolicy(chat=True),
    )

    events = await collect(runtime, "hello")

    assert calls == []
    assert [event.type for event in events] == ["message"]
    assert events[0].data == {"text": "Hello. What would you like to work on?"}
    assert runtime.messages[-2].role == "tool"
    assert runtime.messages[-2].name == "inspect_repo"
    assert runtime.messages[-2].content.startswith("Skipped by chat policy")
    assert runtime.messages[-1] == Message(
        "assistant",
        "Hello. What would you like to work on?",
    )


async def test_agentic_chat_policy_handles_chinese_low_intent_chat():
    calls = []

    def search_workspace_text(query: str) -> str:
        calls.append(query)
        return "matches"

    agent = Agent(
        name="coder",
        instructions="Help with code.",
        tools=[tool(search_workspace_text)],
    )
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="search_workspace_text",
                            arguments={"query": "你好"},
                        ),
                    ],
                ),
            ]
        ),
        policy=AgenticPolicy(chat=True),
    )

    events = await collect(runtime, "你好")

    assert calls == []
    assert [event.type for event in events] == ["message"]
    assert events[0].data == {"text": "你好。你想处理什么？"}


async def test_agentic_chat_policy_allows_project_work_tool_use():
    calls = []

    def read_file(path: str) -> str:
        calls.append(path)
        return "content"

    read_file_tool = tool(read_file)
    agent = Agent(
        name="coder",
        instructions="Help with code.",
        tools=[read_file_tool],
    )
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="read_file", arguments={"path": "README.md"}),
                    ],
                ),
                ModelResult(message="read it"),
            ]
        ),
        policy=AgenticPolicy(chat=True),
    )

    events = await collect(runtime, "please read README.md")

    assert calls == ["README.md"]
    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "message",
    ]
    assert events[-1].data == {"text": "read it"}


async def test_agentic_chat_policy_allows_mixed_language_project_work():
    calls = []

    def read_file(path: str) -> str:
        calls.append(path)
        return "content"

    agent = Agent(
        name="coder",
        instructions="Help with code.",
        tools=[tool(read_file)],
    )
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(name="read_file", arguments={"path": "README.md"}),
                    ],
                ),
                ModelResult(message="read it"),
            ]
        ),
        policy=AgenticPolicy(chat=True),
    )

    events = await collect(runtime, "你好 read README.md")

    assert calls == ["README.md"]
    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "message",
    ]


def test_agentic_helper_replaces_existing_todo_tool():
    store = TodoStore()
    base = Agent(
        name="coder",
        instructions="Write code.",
        tools=[make_todo_read(store), make_todo_tool(store)],
    )

    kit = agentic(base, todos=store)

    assert [tool.name for tool in kit.agent.tools] == ["todo_read", "todo_write"]
    assert kit.todos is store
