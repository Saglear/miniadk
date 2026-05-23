from miniadk import (
    Agent,
    ModelResult,
    Runtime,
    ScriptedModel,
    Skill,
    SkillProblem,
    SkillRegistry,
    ToolCall,
    skill,
    tool,
)
from miniadk.skills import resolve_agent


def test_loads_project_skill_md(tmp_path):
    skill_dir = tmp_path / ".claude" / "skills" / "review"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        """---
name: review
description: Review code for defects.
when_to_use: Use when the user asks for a code review.
allowed-tools: Read, Grep, Glob
user-invocable: true
disable-model-invocation: false
---
Review this request: $ARGUMENTS
Use ${CLAUDE_SKILL_DIR} for references.
""",
        encoding="utf-8",
    )

    registry = SkillRegistry.from_claude_roots(project_root=tmp_path, user_root=tmp_path / "none")
    skill = registry.get("review")

    assert skill is not None
    assert skill.description == "Review code for defects."
    assert skill.when_to_use == "Use when the user asks for a code review."
    assert skill.allowed_tools == ["Read", "Grep", "Glob"]
    assert skill.user_invocable is True
    assert skill.model_invocable is True

    invocation = skill.render("src/app.py")
    assert "Review this request: src/app.py" in invocation.text
    assert skill_dir.as_posix() in invocation.text


def test_inline_skill_helper_builds_short_skill_definition():
    review = skill(
        "review",
        "Review this file: $ARGUMENTS",
        desc="Review code.",
        tools="Read, Grep",
        when="Use for code review.",
    )

    assert isinstance(review, Skill)
    assert review.name == "review"
    assert review.description == "Review code."
    assert review.allowed_tools == ["Read", "Grep"]
    assert review.when_to_use == "Use for code review."
    assert review.render("app.py").text == "Review this file: app.py"


def test_inline_skill_helper_can_declare_named_arguments():
    review = skill(
        "review",
        "Review $path for $focus.",
        desc="Review code.",
        args="path, focus",
    )

    assert review.arguments == ["path", "focus"]
    assert review.render("src/app.py security").text == "Review src/app.py for security."


def test_inline_skill_helper_uses_body_first_line_as_description():
    built = skill("fix", "Fix the bug.\nUse tests.", tools=["Read"])

    assert built.description == "Fix the bug."
    assert built.allowed_tools == ["Read"]


def test_skill_render_replaces_named_arguments_from_frontmatter(tmp_path):
    skill_dir = tmp_path / "skills" / "review"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        """---
name: review
description: Review code.
arguments: path, focus
---
Review $path for ${focus}. Also inspect {{path}}.
""",
        encoding="utf-8",
    )

    review = Skill.from_markdown(skill_file)

    assert review.arguments == ["path", "focus"]
    assert "Review src/app.py for security. Also inspect src/app.py." in (
        review.render("src/app.py security").text
    )


def test_skill_render_keeps_missing_named_arguments_unexpanded():
    review = Skill(
        name="review",
        description="Review.",
        body="Review $path for $focus.",
        arguments=["path", "focus"],
    )

    assert review.render("src/app.py").text == "Review src/app.py for $focus."


def test_skill_render_accepts_named_argument_mapping():
    review = Skill(
        name="review",
        description="Review.",
        body="Review $path for $focus. Args: $ARGUMENTS.",
        arguments=["path", "focus"],
    )

    rendered = review.render({"focus": "security", "path": "src/app.py"}).text

    assert rendered == "Review src/app.py for security. Args: src/app.py security."


async def test_agent_resolve_adds_model_invocable_skill_tool():
    registry = SkillRegistry(
        [
            Skill(
                name="review",
                description="Review code.",
                when_to_use="Use for code review.",
                body="Review: $ARGUMENTS",
            )
        ]
    )
    agent = Agent(name="dev", instructions="Code carefully.", skills=registry)

    resolved = await resolve_agent(agent)

    assert "Available skills:" in resolved.instructions
    assert "- review: Review code. - Use for code review." in resolved.instructions
    assert [tool.name for tool in resolved.tools] == ["skill"]
    assert resolved.tools[0].input_schema["properties"]["skill"] == {
        "type": "string",
        "description": "Skill name to invoke.",
        "enum": ["review"],
    }
    assert resolved.tools[0].input_schema["properties"]["args"] == {
        "description": "Skill arguments as plain text or named argument object.",
        "oneOf": [
            {"type": "string"},
            {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        ],
    }


async def test_skill_tool_schema_lists_only_model_invocable_skills():
    registry = SkillRegistry(
        [
            Skill(name="review", description="Review code.", body="review"),
            Skill(
                name="internal",
                description="Internal only.",
                body="internal",
                model_invocable=False,
            ),
        ]
    )
    agent = Agent(name="dev", instructions="Code carefully.", skills=registry)

    resolved = await resolve_agent(agent)

    assert resolved.tools[0].input_schema["properties"]["skill"]["enum"] == ["review"]


async def test_skill_tool_expands_prompt_for_model_invocation():
    calls: list[str] = []

    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        calls.append(path)
        return "content"

    @tool
    def shell(command: str) -> str:
        """Run a command."""
        return command

    registry = SkillRegistry(
        [
            Skill(
                name="read-only",
                description="Read files.",
                body="Read $ARGUMENTS",
                allowed_tools=["Read"],
            )
        ]
    )
    agent = await resolve_agent(Agent(
        name="dev",
        instructions="Use skills.",
        tools=[read_file, shell],
        skills=registry,
    ))
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(tool_calls=[ToolCall(name="skill", arguments={"skill": "read-only", "args": "a.py"})]),
                ModelResult(tool_calls=[ToolCall(name="read_file", arguments={"path": "a.py"})]),
                ModelResult(message="done"),
            ]
        ),
    )

    events = [event async for event in runtime.run("review a.py")]

    assert [event.type for event in events] == ["tool_call", "tool_result", "tool_call", "tool_result", "message"]
    assert calls == ["a.py"]
    assert "Read a.py" in runtime.messages[3].content


async def test_skill_tool_expands_named_arguments_for_model_invocation():
    registry = SkillRegistry(
        [
            Skill(
                name="review",
                description="Review.",
                body="Review $path for $focus.",
                arguments=["path", "focus"],
            )
        ]
    )
    agent = await resolve_agent(Agent(
        name="dev",
        instructions="Use skills.",
        skills=registry,
    ))
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="skill",
                            arguments={
                                "skill": "review",
                                "args": "src/app.py security",
                            },
                        )
                    ]
                ),
                ModelResult(message="done"),
            ]
        ),
    )

    events = [event async for event in runtime.run("review")]

    assert [event.type for event in events] == ["tool_call", "tool_result", "message"]
    assert "Review src/app.py for security." in runtime.messages[3].content


async def test_skill_tool_accepts_structured_args_for_model_invocation():
    registry = SkillRegistry(
        [
            Skill(
                name="review",
                description="Review.",
                body="Review $path for $focus. Args: $ARGUMENTS.",
                arguments=["path", "focus"],
            )
        ]
    )
    agent = await resolve_agent(Agent(
        name="dev",
        instructions="Use skills.",
        skills=registry,
    ))
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="skill",
                            arguments={
                                "skill": "review",
                                "args": {
                                    "focus": "security",
                                    "path": "src/app.py",
                                },
                            },
                        )
                    ]
                ),
                ModelResult(message="done"),
            ]
        ),
    )

    events = [event async for event in runtime.run("review")]

    assert [event.type for event in events] == ["tool_call", "tool_result", "message"]
    assert (
        "Review src/app.py for security. Args: src/app.py security."
        in runtime.messages[3].content
    )


async def test_model_invoked_skill_limits_followup_tools():
    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return "content"

    @tool
    def shell(command: str) -> str:
        """Run a command."""
        return command

    registry = SkillRegistry(
        [
            Skill(
                name="read-only",
                description="Read files.",
                body="Read $ARGUMENTS",
                allowed_tools=["Read"],
            )
        ]
    )
    agent = await resolve_agent(Agent(
        name="dev",
        instructions="Use skills.",
        tools=[read_file, shell],
        skills=registry,
    ))
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(tool_calls=[ToolCall(name="skill", arguments={"skill": "read-only", "args": "a.py"})]),
                ModelResult(tool_calls=[ToolCall(name="shell", arguments={"command": "cat a.py"})]),
            ]
        ),
    )

    events = [event async for event in runtime.run("review a.py")]

    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "error",
    ]
    assert events[-1].data == {"message": "Unknown tool: shell"}
    assert runtime.messages[-1].content == "Unknown tool: shell"


async def test_skill_allowed_tools_accepts_webfetch_alias():
    calls: list[str] = []

    @tool
    def fetch_url(url: str) -> str:
        """Fetch a URL."""
        calls.append(url)
        return "docs"

    @tool
    def shell(command: str) -> str:
        """Run a command."""
        return command

    registry = SkillRegistry(
        [
            Skill(
                name="read-docs",
                description="Read web docs.",
                body="Read $ARGUMENTS",
                allowed_tools=["WebFetch"],
            )
        ]
    )
    agent = await resolve_agent(Agent(
        name="dev",
        instructions="Use skills.",
        tools=[fetch_url, shell],
        skills=registry,
    ))
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(tool_calls=[ToolCall(name="skill", arguments={"skill": "read-docs", "args": "https://docs.example.test"})]),
                ModelResult(tool_calls=[ToolCall(name="fetch_url", arguments={"url": "https://docs.example.test"})]),
                ModelResult(message="done"),
            ]
        ),
    )

    events = [event async for event in runtime.run("read docs")]

    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "message",
    ]
    assert calls == ["https://docs.example.test"]


async def test_skill_allowed_tools_accepts_workspace_tool_aliases():
    calls: list[str] = []

    @tool
    def glob_workspace_files(pattern: str) -> list[str]:
        """Find files."""
        calls.append(f"glob:{pattern}")
        return ["app.py"]

    @tool
    def search_workspace_text(pattern: str) -> list[str]:
        """Search text."""
        calls.append(f"grep:{pattern}")
        return ["app.py:1: needle"]

    @tool
    def shell(command: str) -> str:
        """Run a command."""
        return command

    registry = SkillRegistry(
        [
            Skill(
                name="scan",
                description="Scan code.",
                body="Scan $ARGUMENTS",
                allowed_tools=["Glob", "Grep"],
            )
        ]
    )
    agent = await resolve_agent(Agent(
        name="dev",
        instructions="Use skills.",
        tools=[glob_workspace_files, search_workspace_text, shell],
        skills=registry,
    ))
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(tool_calls=[ToolCall(name="skill", arguments={"skill": "scan", "args": "needle"})]),
                ModelResult(tool_calls=[ToolCall(name="glob_workspace_files", arguments={"pattern": "*.py"})]),
                ModelResult(tool_calls=[ToolCall(name="search_workspace_text", arguments={"pattern": "needle"})]),
                ModelResult(tool_calls=[ToolCall(name="shell", arguments={"command": "pytest"})]),
            ]
        ),
    )

    events = [event async for event in runtime.run("scan code")]

    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "error",
    ]
    assert calls == ["glob:*.py", "grep:needle"]
    assert events[-1].data == {"message": "Unknown tool: shell"}


async def test_skill_allowed_tools_accepts_agentic_and_multi_edit_aliases():
    calls: list[str] = []

    @tool
    def edit_files(path: str, edits: list) -> str:
        """Apply edits."""
        calls.append(f"edit:{path}:{len(edits)}")
        return "edited"

    @tool
    def todo_write(todos: list) -> str:
        """Write todos."""
        calls.append(f"todos:{len(todos)}")
        return "updated"

    @tool
    def spawn_agent(agent: str, prompt: str) -> str:
        """Run a helper."""
        calls.append(f"task:{agent}:{prompt}")
        return "done"

    @tool
    def shell(command: str) -> str:
        """Run a command."""
        return command

    registry = SkillRegistry(
        [
            Skill(
                name="fix",
                description="Fix code.",
                body="Fix $ARGUMENTS",
                allowed_tools=["MultiEdit", "TodoWrite", "Task"],
            )
        ]
    )
    agent = await resolve_agent(Agent(
        name="dev",
        instructions="Use skills.",
        tools=[edit_files, todo_write, spawn_agent, shell],
        skills=registry,
    ))
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(tool_calls=[ToolCall(name="skill", arguments={"skill": "fix", "args": "bug"})]),
                ModelResult(tool_calls=[ToolCall(name="todo_write", arguments={"todos": [{"content": "fix"}]})]),
                ModelResult(tool_calls=[ToolCall(name="edit_files", arguments={"path": "app.py", "edits": [{"old": "a", "new": "b"}]})]),
                ModelResult(tool_calls=[ToolCall(name="spawn_agent", arguments={"agent": "reviewer", "prompt": "review"})]),
                ModelResult(tool_calls=[ToolCall(name="shell", arguments={"command": "pytest"})]),
            ]
        ),
    )

    events = [event async for event in runtime.run("fix bug")]

    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "error",
    ]
    assert calls == ["todos:1", "edit:app.py:1", "task:reviewer:review"]
    assert events[-1].data == {"message": "Unknown tool: shell"}


async def test_skill_allowed_tools_accepts_claude_style_parenthesized_rules():
    calls: list[str] = []

    @tool
    def shell(command: str) -> str:
        """Run a command."""
        calls.append(command)
        return "ok"

    @tool
    def write_file(path: str, content: str) -> str:
        """Write a file."""
        return f"wrote {path}"

    registry = SkillRegistry(
        [
            Skill(
                name="check",
                description="Check repo.",
                body="Check $ARGUMENTS",
                allowed_tools=["Bash(git diff:*)"],
            )
        ]
    )
    agent = await resolve_agent(
        Agent(
            name="dev",
            instructions="Use skills.",
            tools=[shell, write_file],
            skills=registry,
        )
    )
    runtime = Runtime(
        agent=agent,
        model=ScriptedModel(
            [
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="skill",
                            arguments={"skill": "check", "args": "changes"},
                        )
                    ]
                ),
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="shell",
                            arguments={"command": "git diff --stat"},
                        )
                    ]
                ),
                ModelResult(
                    tool_calls=[
                        ToolCall(
                            name="write_file",
                            arguments={"path": "notes.txt", "content": "bad"},
                        )
                    ]
                ),
            ]
        ),
    )

    events = [event async for event in runtime.run("check changes")]

    assert [event.type for event in events] == [
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "error",
    ]
    assert calls == ["git diff --stat"]
    assert events[-1].data == {"message": "Unknown tool: write_file"}


async def test_slash_only_skill_is_not_added_to_model_skill_catalog():
    registry = SkillRegistry(
        [
            Skill(
                name="debug",
                description="Debug session.",
                body="debug",
                model_invocable=False,
            )
        ]
    )
    resolved = await resolve_agent(Agent(name="dev", instructions="hi", skills=registry))

    assert resolved.tools == []
    assert "Available skills:" not in resolved.instructions


def test_skill_registry_reports_parse_problems(tmp_path):
    skill_dir = tmp_path / "skills" / "broken"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        """---
name: broken
unknown: value
---
""",
        encoding="utf-8",
    )

    registry = SkillRegistry.from_paths(tmp_path / "skills")
    problems = registry.problems()

    assert [problem.message for problem in problems] == [
        "body is empty",
        "unknown metadata key: unknown",
    ]
    assert all(isinstance(problem, SkillProblem) for problem in problems)
    assert all(problem.skill == "broken" for problem in problems)
    assert all(problem.path == skill_file for problem in problems)


def test_skill_registry_reports_unclosed_frontmatter(tmp_path):
    skill_dir = tmp_path / "skills" / "broken"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        """---
name: broken
description: Missing closing fence.
Body is parsed conservatively.
""",
        encoding="utf-8",
    )

    registry = SkillRegistry.from_paths(tmp_path / "skills")
    skill = registry.get("broken")

    assert skill is not None
    assert skill.name == "broken"
    assert [problem.message for problem in registry.problems()] == [
        "frontmatter starts with --- but has no closing ---"
    ]


def test_skill_registry_reports_duplicate_names_without_dropping_manual_skills():
    registry = SkillRegistry(
        [
            Skill(name="review", description="One.", body="one"),
            Skill(name="Review", description="Two.", body="two"),
        ]
    )

    problems = registry.problems()

    assert len(problems) == 1
    assert problems[0].skill == "Review"
    assert problems[0].message == "duplicate skill name also used by review"


def test_skill_registry_from_skills_dedupes_with_last_wins():
    registry = SkillRegistry.from_skills(
        Skill(name="review", description="One.", body="one"),
        Skill(name="Review", description="Two.", body="two"),
    )

    assert registry.get("review").description == "Two."
    assert [skill.name for skill in registry.all()] == ["Review"]
    assert [problem.message for problem in registry.problems()] == [
        "duplicate skill name also used by review"
    ]


def test_skill_registry_reports_duplicate_names_from_paths_while_last_wins(tmp_path):
    first = tmp_path / "a" / "review"
    second = tmp_path / "b" / "Review"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "SKILL.md").write_text(
        "---\nname: review\ndescription: First.\n---\nFirst body\n",
        encoding="utf-8",
    )
    (second / "SKILL.md").write_text(
        "---\nname: Review\ndescription: Second.\n---\nSecond body\n",
        encoding="utf-8",
    )

    registry = SkillRegistry.from_paths(tmp_path / "a", tmp_path / "b")

    assert registry.get("review").description == "Second."
    assert [skill.name for skill in registry.all()] == ["Review"]
    assert [problem.message for problem in registry.problems()] == [
        "duplicate skill name also used by review"
    ]
