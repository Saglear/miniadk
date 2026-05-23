from miniadk import (
    Agent,
    Message,
    ModelResult,
    Runtime,
    ScriptedModel,
    Session,
    ToolCall,
    tool,
)


def test_session_round_trips_messages_with_tool_calls_and_content_blocks(tmp_path):
    session = Session(
        [
            Message("system", "Answer."),
            Message(
                "assistant",
                "I will inspect.",
                tool_calls=[
                    ToolCall(
                        name="read_file",
                        arguments={"path": "a.py"},
                        id="call_1",
                    )
                ],
                content_blocks=[
                    {"type": "text", "text": "I will inspect."},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "read_file",
                        "input": {"path": "a.py"},
                    },
                ],
            ),
            Message("tool", "print(1)", name="read_file", tool_call_id="call_1"),
        ]
    )
    path = tmp_path / "session.json"

    session.save(path)
    loaded = Session.load(path)

    assert loaded.messages == session.messages


def test_session_save_is_atomic_when_replace_fails(tmp_path, monkeypatch):
    path = tmp_path / "session.json"
    path.write_text("old session", encoding="utf-8")
    session = Session([Message("user", "new session")])

    def fail_replace(self, target):
        raise OSError("replace failed")

    monkeypatch.setattr("pathlib.Path.replace", fail_replace)

    try:
        session.save(path)
    except OSError as error:
        assert str(error) == "replace failed"
    else:
        raise AssertionError("session save should surface replace failures")

    assert path.read_text(encoding="utf-8") == "old session"
    assert list(tmp_path.glob(".session.json.*.tmp")) == []


def test_session_compacts_old_messages_with_summary():
    session = Session(
        [
            Message("system", "Answer."),
            Message("user", "one"),
            Message("assistant", "two"),
            Message("user", "three"),
            Message("assistant", "four"),
        ]
    )

    session.compact("Earlier summary.", keep=2)

    assert session.messages == [
        Message("system", "Answer."),
        Message("system", "Earlier summary."),
        Message("user", "three"),
        Message("assistant", "four"),
    ]


def test_session_transcript_skips_initial_system_and_formats_tools():
    session = Session(
        [
            Message("system", "Answer."),
            Message("user", "Inspect."),
            Message(
                "assistant",
                "I will inspect.",
                tool_calls=[ToolCall(name="read_file", arguments={"path": "a.py"})],
            ),
            Message("tool", "print(1)", name="read_file"),
            Message("assistant", "Done."),
        ]
    )

    assert session.transcript(exclude_recent=1) == "\n".join(
        [
            "user: Inspect.",
            "assistant: I will inspect.\n[tool calls: read_file]",
            "tool:read_file: print(1)",
        ]
    )


async def test_session_summarize_compacts_with_model_summary():
    session = Session(
        [
            Message("system", "Answer."),
            Message("user", "old question"),
            Message("assistant", "old answer"),
            Message("user", "recent question"),
            Message("assistant", "recent answer"),
        ]
    )
    model = ScriptedModel([ModelResult(message="Old exchange summary.")])

    summary = await session.summarize(model=model, keep=2)

    assert summary == "Old exchange summary."
    assert session.messages == [
        Message("system", "Answer."),
        Message("system", "Old exchange summary."),
        Message("user", "recent question"),
        Message("assistant", "recent answer"),
    ]
    assert "Summarize the conversation" in model.calls[0][0][0].content
    assert model.calls[0][0][1].content == "user: old question\nassistant: old answer"
    assert model.calls[0][1] == []


async def test_session_summarize_skips_model_when_no_old_transcript():
    session = Session(
        [
            Message("system", "Answer."),
            Message("user", "recent question"),
            Message("assistant", "recent answer"),
        ]
    )
    model = ScriptedModel([ModelResult(message="should not be used")])

    summary = await session.summarize(model=model, keep=10)

    assert summary == ""
    assert model.calls == []
    assert session.messages == [
        Message("system", "Answer."),
        Message("user", "recent question"),
        Message("assistant", "recent answer"),
    ]


def test_session_stats_counts_messages_tool_calls_and_chars():
    session = Session(
        [
            Message("system", "Answer."),
            Message("user", "Inspect app."),
            Message(
                "assistant",
                "",
                tool_calls=[
                    ToolCall(name="read_file", arguments={"path": "app.py"}),
                    ToolCall(name="search_text", arguments={"pattern": "main"}),
                ],
            ),
            Message("tool", "print(1)", name="read_file"),
            Message("assistant", "Done."),
        ]
    )

    stats = session.stats

    assert stats.messages == 5
    assert stats.system == 1
    assert stats.user == 1
    assert stats.assistant == 2
    assert stats.tool == 1
    assert stats.tool_calls == 2
    assert stats.chars == len("Answer.Inspect app.print(1)Done.")
    assert stats.to_dict()["tool_calls"] == 2


def test_session_compact_can_drop_all_recent_messages():
    session = Session(
        [
            Message("system", "Answer."),
            Message("user", "one"),
            Message("assistant", "two"),
        ]
    )

    session.compact("Summary only.", keep=0)

    assert session.messages == [
        Message("system", "Answer."),
        Message("system", "Summary only."),
    ]


def test_session_branch_copies_messages_without_sharing_mutable_state():
    session = Session(
        [
            Message("system", "Answer."),
            Message(
                "assistant",
                "I will inspect.",
                tool_calls=[
                    ToolCall(
                        name="read_file",
                        arguments={"path": "a.py"},
                        id="call_1",
                    )
                ],
                content_blocks=[
                    {"type": "tool_use", "input": {"path": "a.py"}},
                ],
            ),
        ]
    )

    branch = session.branch()
    branch.messages[1].tool_calls[0].arguments["path"] = "b.py"
    branch.messages[1].content_blocks[0]["input"]["path"] = "b.py"
    branch.messages.append(Message("user", "new branch question"))

    assert session.messages[1].tool_calls[0].arguments == {"path": "a.py"}
    assert session.messages[1].content_blocks[0]["input"] == {"path": "a.py"}
    assert [message.content for message in session.messages] == [
        "Answer.",
        "I will inspect.",
    ]


def test_session_branch_can_keep_recent_messages():
    session = Session(
        [
            Message("system", "Answer."),
            Message("user", "one"),
            Message("assistant", "two"),
            Message("user", "three"),
        ]
    )

    branch = session.branch(keep=2)

    assert branch.messages == [
        Message("system", "Answer."),
        Message("assistant", "two"),
        Message("user", "three"),
    ]


def test_session_trim_keeps_initial_system_and_recent_messages():
    session = Session(
        [
            Message("system", "Answer."),
            Message("user", "one"),
            Message("assistant", "two"),
            Message("user", "three"),
            Message("assistant", "four"),
        ]
    )

    session.trim(keep=2)

    assert session.messages == [
        Message("system", "Answer."),
        Message("user", "three"),
        Message("assistant", "four"),
    ]


def test_session_trim_can_drop_all_history_after_system():
    session = Session(
        [
            Message("system", "Answer."),
            Message("user", "one"),
            Message("assistant", "two"),
        ]
    )

    session.trim(keep=0)

    assert session.messages == [Message("system", "Answer.")]


def test_session_trim_without_initial_system_keeps_recent_messages():
    session = Session(
        [
            Message("user", "one"),
            Message("assistant", "two"),
            Message("user", "three"),
        ]
    )

    session.trim(keep=1)

    assert session.messages == [Message("user", "three")]


async def test_runtime_can_resume_from_session_messages():
    @tool
    def remember() -> str:
        """Return remembered context."""
        return "remembered"

    session = Session(
        [
            Message("system", "Answer."),
            Message("user", "old question"),
            Message("assistant", "old answer"),
        ]
    )
    model = ScriptedModel([ModelResult(message="new answer")])
    runtime = Runtime(
        agent=Agent(name="resumer", instructions="Answer.", tools=[remember]),
        model=model,
        session=session,
    )

    events = [event async for event in runtime.run("new question")]

    assert events[-1].data == {"text": "new answer"}
    assert [message.content for message in model.calls[0][0]] == [
        "Answer.",
        "old question",
        "old answer",
        "new question",
    ]
    assert runtime.session is session


async def test_runtime_uses_compacted_session_messages():
    session = Session(
        [
            Message("system", "Answer."),
            Message("user", "old question"),
            Message("assistant", "old answer"),
            Message("user", "recent question"),
        ]
    )
    session.compact("Old exchange summary.", keep=1)
    model = ScriptedModel([ModelResult(message="new answer")])
    runtime = Runtime(
        agent=Agent(name="compact", instructions="Answer.", tools=[]),
        model=model,
        session=session,
    )

    await collect_events(runtime, "new question")

    assert [message.content for message in model.calls[0][0]] == [
        "Answer.",
        "Old exchange summary.",
        "recent question",
        "new question",
    ]


def test_runtime_starts_empty_session_with_system_message():
    session = Session()
    runtime = Runtime(
        agent=Agent(name="starter", instructions="Answer.", tools=[]),
        model=ScriptedModel([]),
        session=session,
    )

    assert runtime.session is session
    assert session.messages == [Message("system", "Answer.")]


async def collect_events(runtime: Runtime, text: str):
    return [event async for event in runtime.run(text)]
