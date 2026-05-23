import asyncio
from pathlib import Path

from miniadk import (
    Agent,
    Agentic,
    Guard,
    ModelResult,
    OpenAIModel,
    ScriptedModel,
    SkillRegistry,
    run_cli,
    tool,
)
from miniadk.adapters import run_cli as adapter_run_cli
from miniadk.presets import CODER_INSTRUCTIONS, CODER_READ_SHELL, coder


async def _run_tool(tools, name: str, **kwargs):
    tool = next(tool for tool in tools if tool.name == name)
    return await tool.run(**kwargs)


def test_coder_preset_builds_capable_agent_with_short_path(tmp_path):
    built = coder(tmp_path, shell_env={"OPENAI_API_KEY": None})

    assert isinstance(built, Agentic)
    assert built.agent.name == "coder"
    assert CODER_INSTRUCTIONS in built.agent.instructions
    assert "only a greeting" in built.agent.instructions
    assert [tool.name for tool in built.agent.tools] == [
        "read_file",
        "list_workspace_files",
        "glob_workspace_files",
        "search_workspace_text",
        "write_file",
        "edit_file",
        "edit_files",
        "delete_file",
        "move_file",
        "copy_file",
        "shell",
        "todo_read",
        "todo_write",
    ]
    assert built.agent.skills is None
    assert built.agent.mcp is None
    assert built.policy.todo_store is built.todos
    assert built.policy.chat is True
    assert len(built.middleware) == 1
    assert isinstance(built.middleware[0], Guard)


async def test_coder_preset_filters_common_model_keys_from_shell_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example.test")
    monkeypatch.setenv("OPENAI_MODEL", "openai-demo")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example.test")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-demo")
    monkeypatch.setenv("MINIADK_MODEL_KEY", "generic-secret")
    monkeypatch.setenv("MINIADK_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("MINIADK_MODEL_URL", "https://generic.example.test")
    monkeypatch.setenv("MINIADK_MODEL_NAME", "generic-demo")
    monkeypatch.setenv("MINIADK_MODEL_TEMPERATURE", "0.2")

    built = coder(tmp_path)

    result = await _run_tool(
        built.agent.tools,
        "shell",
        command=(
            "python -c \"import os; "
            "print(os.getenv('OPENAI_API_KEY', 'missing')); "
            "print(os.getenv('OPENAI_BASE_URL', 'missing')); "
            "print(os.getenv('OPENAI_MODEL', 'missing')); "
            "print(os.getenv('ANTHROPIC_API_KEY', 'missing')); "
            "print(os.getenv('ANTHROPIC_BASE_URL', 'missing')); "
            "print(os.getenv('ANTHROPIC_MODEL', 'missing')); "
            "print(os.getenv('MINIADK_MODEL_KEY', 'missing')); "
            "print(os.getenv('MINIADK_MODEL_PROVIDER', 'missing')); "
            "print(os.getenv('MINIADK_MODEL_URL', 'missing')); "
            "print(os.getenv('MINIADK_MODEL_NAME', 'missing')); "
            "print(os.getenv('MINIADK_MODEL_TEMPERATURE', 'missing'))\""
        ),
    )

    assert result.stdout.splitlines() == ["missing"] * 11


async def test_coder_preset_allows_explicit_shell_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    built = coder(tmp_path, shell_env={"OPENAI_API_KEY": "visible"})

    result = await _run_tool(
        built.agent.tools,
        "shell",
        command="python -c \"import os; print(os.getenv('OPENAI_API_KEY', 'missing'))\"",
    )

    assert result.stdout.strip() == "visible"


async def test_coder_preset_passes_file_scan_options(tmp_path):
    (tmp_path / "a.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "lib.txt").write_text("needle\n", encoding="utf-8")

    built = coder(
        tmp_path,
        write=False,
        shell=False,
        list_limit=1,
        search_limit=1,
        ignore={"vendor"},
    )

    assert await _run_tool(built.agent.tools, "list_workspace_files", pattern="*.txt") == [
        "a.txt"
    ]
    assert await _run_tool(built.agent.tools, "search_workspace_text", pattern="needle") == [
        "a.txt:1: needle"
    ]


def test_coder_preset_can_enable_web_fetch_tool(tmp_path):
    built = coder(tmp_path, files=False, shell=False, web=True)

    assert [tool.name for tool in built.agent.tools] == [
        "fetch_url",
        "todo_read",
        "todo_write",
    ]


def test_coder_preset_can_disable_default_guard(tmp_path):
    built = coder(tmp_path, files=False, shell=False, guard=False)

    assert built.middleware == []


async def test_coder_preset_passes_shell_validation(tmp_path):
    built = coder(
        tmp_path,
        files=False,
        validate_shell=lambda command: "blocked" if command == "bad" else True,
    )
    shell = next(tool for tool in built.agent.tools if tool.name == "shell")

    denied = await shell.validate(command="bad")
    allowed = await shell.validate(command="printf ok")

    assert denied.ok is False
    assert denied.message == "blocked"
    assert allowed.ok is True


def test_coder_preset_can_mark_read_only_shell_commands(tmp_path):
    built = coder(
        tmp_path,
        files=False,
        read_shell=["pytest --collect-only*"],
    )
    shell = next(tool for tool in built.agent.tools if tool.name == "shell")

    assert shell.is_read_only(command="pytest --collect-only tests") is True
    assert shell.is_destructive(command="pytest tests") is True


def test_coder_preset_marks_common_inspection_shell_commands_read_only(tmp_path):
    built = coder(tmp_path, files=False)
    shell = next(tool for tool in built.agent.tools if tool.name == "shell")

    assert "git status*" in CODER_READ_SHELL
    assert shell.is_read_only(command="git status --short") is True
    assert shell.is_read_only(command="rg needle src") is True
    assert shell.is_read_only(command="sed -n '1,20p' README.md") is True
    assert shell.is_destructive(command="pytest tests") is True
    assert shell.is_destructive(command="python build.py") is True


async def test_coder_preset_adds_spawn_tool_for_child_agents(tmp_path):
    reviewer = Agent(name="reviewer", instructions="Review code.")
    model = ScriptedModel([ModelResult(message="looks good")])

    built = coder(
        tmp_path,
        files=False,
        shell=False,
        agents=[reviewer],
        agent_model=model,
        keep_agent_session=True,
    )

    assert [tool.name for tool in built.agent.tools] == [
        "spawn_agent",
        "todo_read",
        "todo_write",
    ]

    result = await _run_tool(
        built.agent.tools,
        "spawn_agent",
        agent="reviewer",
        prompt="review app.py",
    )

    assert result.answer == "looks good"
    assert result.session is not None
    assert [message.content for message in result.session.messages] == [
        "Review code.",
        "review app.py",
        "looks good",
    ]


async def test_coder_preset_can_add_background_work_tools(tmp_path):
    reviewer = Agent(name="reviewer", instructions="Review code.")
    model = ScriptedModel([ModelResult(message="done")])

    built = coder(
        tmp_path,
        files=False,
        shell=False,
        agents=[reviewer],
        work=True,
        agent_model=model,
    )

    assert [tool.name for tool in built.agent.tools] == [
        "spawn_agent",
        "start_work",
        "list_work",
        "read_work",
        "cancel_work",
        "todo_read",
        "todo_write",
    ]

    started = await _run_tool(
        built.agent.tools,
        "start_work",
        agent="reviewer",
        prompt="review app.py",
    )
    await asyncio.sleep(0)
    result = await _run_tool(built.agent.tools, "read_work", id=started.id)

    assert result.status == "done"
    assert result.answer == "done"


def test_coder_preset_accepts_custom_pieces(tmp_path):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review code\n---\nReview $ARGUMENTS\n",
        encoding="utf-8",
    )

    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    built = coder(
        tmp_path,
        name="repo",
        instructions="Help this repo.",
        files=False,
        shell=False,
        chat=False,
        extra=[ping],
        skills=skills_dir,
    )

    assert built.agent.name == "repo"
    assert built.agent.instructions.startswith("Help this repo.")
    assert "only a greeting" not in built.agent.instructions
    assert [tool.name for tool in built.agent.tools] == ["ping", "todo_read", "todo_write"]
    assert isinstance(built.agent.skills, SkillRegistry)
    assert built.agent.skills.get("review").description == "Review code"
    assert built.policy.chat is False


def test_coder_preset_auto_loads_project_claude_skills(tmp_path):
    skill_dir = tmp_path / ".claude" / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review code\n---\nReview $ARGUMENTS\n",
        encoding="utf-8",
    )

    built = coder(tmp_path, files=False, shell=False)

    assert isinstance(built.agent.skills, SkillRegistry)
    assert built.agent.skills.get("review").description == "Review code"


def test_coder_preset_can_load_skills_from_multiple_paths(tmp_path):
    project_skills = tmp_path / "project-skills"
    team_skills = tmp_path / "team-skills"
    review = project_skills / "review"
    debug = team_skills / "debug"
    review.mkdir(parents=True)
    debug.mkdir(parents=True)
    (review / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review code\n---\nReview $ARGUMENTS\n",
        encoding="utf-8",
    )
    (debug / "SKILL.md").write_text(
        "---\nname: debug\ndescription: Debug issue\n---\nDebug $ARGUMENTS\n",
        encoding="utf-8",
    )

    built = coder(
        tmp_path,
        files=False,
        shell=False,
        skills=[project_skills, team_skills],
    )

    assert isinstance(built.agent.skills, SkillRegistry)
    assert [skill.name for skill in built.agent.skills.all()] == ["review", "debug"]


def test_coder_preset_treats_empty_skill_paths_as_no_registry(tmp_path):
    built = coder(tmp_path, files=False, shell=False, skills=[])

    assert built.agent.skills is None


def test_coder_preset_can_disable_auto_skill_loading(tmp_path):
    skill_dir = tmp_path / ".claude" / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review code\n---\nReview $ARGUMENTS\n",
        encoding="utf-8",
    )

    built = coder(tmp_path, files=False, shell=False, skills=None)

    assert built.agent.skills is None


def test_run_cli_accepts_agentic_preset(tmp_path):
    built = coder(tmp_path, files=False, shell=False)
    outputs: list[str] = []
    inputs = iter(["/status", "/exit"])

    run_cli(
        built,
        model=ScriptedModel([]),
        input_func=lambda _prompt: next(inputs),
        output_func=outputs.append,
    )

    assert any("agent" in output for output in outputs)
    assert any("coder" in output for output in outputs)


def test_coder_short_usage_shape_is_available():
    built = coder(".", files=False, shell=False)

    assert built.agent.name == "coder"
    assert callable(adapter_run_cli)
    assert OpenAIModel is not None
