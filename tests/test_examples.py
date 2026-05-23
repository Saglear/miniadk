import importlib.util
from pathlib import Path

from miniadk import Agent, SkillRegistry, Tool


def test_repo_cli_example_skills_load():
    skills_dir = Path(__file__).parents[1] / "examples" / "repo_cli_skills"

    registry = SkillRegistry.from_paths(skills_dir)

    assert {skill.name for skill in registry.all()} == {
        "fix",
        "plan",
        "review",
        "tests",
    }
    assert registry.get("fix").allowed_tools == [
        "Read",
        "Grep",
        "Glob",
        "Search",
        "Edit",
        "Write",
        "Bash",
    ]


def test_cli_interaction_lab_builds_agent_without_model_call():
    module = _load_example("cli_interaction_lab.py")
    agent = module.build_agent(Path.cwd())

    assert agent.name == "cli-lab"
    assert [tool.name for tool in agent.tools] == [
        "read_file",
        "search_workspace_text",
        "shell",
    ]
    assert {skill.name for skill in agent.skills.all()} == {
        "fix",
        "plan",
        "review",
        "tests",
    }


def test_file_assistant_examples_build_agents_without_model_call(tmp_path):
    for filename in ["openai_file_assistant.py", "anthropic_file_assistant.py"]:
        module = _load_example(filename)
        agent = module.build_agent(tmp_path)

        assert isinstance(agent, Agent)
        assert agent.name == "file-assistant"
        assert [tool.name for tool in agent.tools] == [
            "read_file",
            "write_file",
            "list_workspace_files",
            "shell",
        ]
        assert all(isinstance(tool, Tool) for tool in agent.tools)


def test_scripted_tiny_product_keeps_one_screen_shape():
    module = _load_example("scripted_tiny_product.py")

    assert module.add.name == "add"
    assert callable(module.main)


def test_coder_preset_example_keeps_tiny_shape():
    module = _load_example("coder_preset.py")
    source = (Path(__file__).parents[1] / "examples" / "coder_preset.py").read_text(
        encoding="utf-8"
    )

    assert callable(module.main)
    assert "load_env_upwards" not in source


def test_compact_coder_example_builds_agent_without_model_call(tmp_path):
    module = _load_example("compact_coder.py")
    source = (Path(__file__).parents[1] / "examples" / "compact_coder.py").read_text(
        encoding="utf-8"
    )

    kit = module.build(tmp_path)

    assert kit.agent.name == "coder"
    assert [tool.name for tool in kit.agent.tools] == [
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
        "spawn_agent",
        "start_work",
        "list_work",
        "read_work",
        "cancel_work",
        "todo_read",
        "todo_write",
    ]
    assert {skill.name for skill in kit.agent.skills.all()} == {
        "fix",
        "plan",
        "review",
        "tests",
    }
    assert "run_cli(build(\".\"), session=True)" in source
    assert len(source.splitlines()) <= 25


def test_smoke_llm_uses_single_default_model_path():
    module = _load_example("smoke_llm.py")
    source = (Path(__file__).parents[1] / "examples" / "smoke_llm.py").read_text(
        encoding="utf-8"
    )

    assert callable(module.main)
    assert "load_env_upwards" not in source
    assert "OpenAIModel" not in source
    assert "AnthropicModel" not in source


def _load_example(filename: str):
    path = Path(__file__).parents[1] / "examples" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
