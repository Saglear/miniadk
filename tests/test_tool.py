import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import Annotated, Literal, NotRequired, Optional, TypedDict

import pytest

from miniadk import ToolValidation, tool


def test_tool_decorator_builds_readable_tool_metadata():
    @tool
    def add(left: int, right: int) -> int:
        """Add two numbers."""
        return left + right

    assert add.name == "add"
    assert add.description == "Add two numbers."
    assert add.input_schema == {
        "type": "object",
        "properties": {
            "left": {"type": "integer"},
            "right": {"type": "integer"},
        },
        "additionalProperties": False,
        "required": ["left", "right"],
    }


def test_tool_schema_hides_runtime_progress_argument():
    @tool
    async def build(path: str, progress) -> str:
        """Build something."""
        await progress("building")
        return path

    assert build.input_schema == {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
        },
        "additionalProperties": False,
        "required": ["path"],
    }


async def test_tool_runs_sync_functions():
    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"hello {name}"

    assert await greet.run(name="Ada") == "hello Ada"


async def test_tool_validate_rejects_missing_required_arguments():
    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"hello {name}"

    assert await greet.validate() == ToolValidation.deny(
        "Missing required tool argument: name"
    )


async def test_tool_validate_rejects_unknown_arguments_by_default():
    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"hello {name}"

    assert await greet.validate(name="Ada", title="Dr") == ToolValidation.deny(
        "Unknown tool argument: title"
    )


async def test_tool_validate_rejects_wrong_basic_types():
    @tool
    def repeat(count: int, exact: bool, tags: list[str], scores: dict[str, int]) -> str:
        """Repeat something."""
        return str(count)

    assert await repeat.validate(
        count="3",
        exact=True,
        tags=["a"],
        scores={"a": 1},
    ) == ToolValidation.deny("Tool argument count must be integer")
    assert await repeat.validate(
        count=3,
        exact=1,
        tags=["a"],
        scores={"a": 1},
    ) == ToolValidation.deny("Tool argument exact must be boolean")
    assert await repeat.validate(
        count=3,
        exact=True,
        tags=["a", 1],
        scores={"a": 1},
    ) == ToolValidation.deny("Tool argument tags[1] must be string")
    assert await repeat.validate(
        count=3,
        exact=True,
        tags=["a"],
        scores={"a": "1"},
    ) == ToolValidation.deny("Tool argument scores.a must be integer")


async def test_tool_validate_rejects_values_outside_enum_schema():
    @tool
    def search(mode: Literal["files", "content"]) -> str:
        """Search something."""
        return mode

    assert await search.validate(mode="shell") == ToolValidation.deny(
        "Tool argument mode must be one of: files, content"
    )


async def test_tool_validate_accepts_one_of_schema_matches():
    @tool(
        schema={
            "value": {
                "oneOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                ]
            }
        }
    )
    def use(value: object) -> str:
        """Use a value."""
        return str(value)

    assert await use.validate(value="text") == ToolValidation.allow()
    assert await use.validate(value={"path": "app.py"}) == ToolValidation.allow()
    assert await use.validate(value=["bad"]) == ToolValidation.deny(
        "Tool argument value must match oneOf schema"
    )


async def test_tool_validate_accepts_any_of_schema_matches():
    @tool(schema={"value": {"anyOf": [{"type": "integer"}, {"type": "string"}]}})
    def use(value: object) -> str:
        """Use a value."""
        return str(value)

    assert await use.validate(value=1) == ToolValidation.allow()
    assert await use.validate(value="one") == ToolValidation.allow()
    assert await use.validate(value=False) == ToolValidation.deny(
        "Tool argument value must match anyOf schema"
    )


async def test_tool_validate_checks_string_schema_constraints():
    @tool(
        schema={
            "name": {
                "type": "string",
                "minLength": 2,
                "maxLength": 4,
                "pattern": "^[a-z]+$",
            }
        }
    )
    def greet(name: str) -> str:
        """Greet someone."""
        return name

    assert await greet.validate(name="ada") == ToolValidation.allow()
    assert await greet.validate(name="a") == ToolValidation.deny(
        "Tool argument name must be at least 2 chars"
    )
    assert await greet.validate(name="adams") == ToolValidation.deny(
        "Tool argument name must be at most 4 chars"
    )
    assert await greet.validate(name="Ada") == ToolValidation.deny(
        "Tool argument name must match pattern: ^[a-z]+$"
    )


async def test_tool_validate_checks_number_schema_constraints():
    @tool(schema={"count": {"type": "integer", "minimum": 1, "maximum": 3}})
    def repeat(count: int) -> str:
        """Repeat."""
        return str(count)

    assert await repeat.validate(count=2) == ToolValidation.allow()
    assert await repeat.validate(count=0) == ToolValidation.deny(
        "Tool argument count must be >= 1"
    )
    assert await repeat.validate(count=4) == ToolValidation.deny(
        "Tool argument count must be <= 3"
    )


async def test_tool_validate_checks_extended_number_schema_constraints():
    @tool(
        schema={
            "ratio": {
                "type": "number",
                "exclusiveMinimum": 0,
                "exclusiveMaximum": 1,
                "multipleOf": 0.25,
            }
        }
    )
    def scale(ratio: float) -> str:
        """Scale something."""
        return str(ratio)

    assert await scale.validate(ratio=0.5) == ToolValidation.allow()
    assert await scale.validate(ratio=0) == ToolValidation.deny(
        "Tool argument ratio must be > 0"
    )
    assert await scale.validate(ratio=1) == ToolValidation.deny(
        "Tool argument ratio must be < 1"
    )
    assert await scale.validate(ratio=0.3) == ToolValidation.deny(
        "Tool argument ratio must be a multiple of 0.25"
    )


async def test_tool_validate_checks_array_schema_constraints():
    @tool(schema={"items": {"type": "array", "minItems": 1, "maxItems": 2}})
    def save(items: list[str]) -> str:
        """Save items."""
        return ",".join(items)

    assert await save.validate(items=["a"]) == ToolValidation.allow()
    assert await save.validate(items=[]) == ToolValidation.deny(
        "Tool argument items must have at least 1 items"
    )
    assert await save.validate(items=["a", "b", "c"]) == ToolValidation.deny(
        "Tool argument items must have at most 2 items"
    )


async def test_tool_validate_checks_unique_array_items():
    @tool(schema={"items": {"type": "array", "uniqueItems": True}})
    def save(items: list[str]) -> str:
        """Save items."""
        return ",".join(items)

    assert await save.validate(items=["a", "b"]) == ToolValidation.allow()
    assert await save.validate(items=["a", "a"]) == ToolValidation.deny(
        "Tool argument items must have unique items"
    )


async def test_tool_validate_checks_object_property_count_schema_constraints():
    @tool(
        schema={
            "labels": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "minProperties": 1,
                "maxProperties": 2,
            }
        }
    )
    def tag(labels: dict[str, str]) -> str:
        """Tag something."""
        return str(labels)

    assert await tag.validate(labels={"env": "dev"}) == ToolValidation.allow()
    assert await tag.validate(labels={}) == ToolValidation.deny(
        "Tool argument labels must have at least 1 properties"
    )
    assert await tag.validate(labels={"a": "1", "b": "2", "c": "3"}) == (
        ToolValidation.deny("Tool argument labels must have at most 2 properties")
    )


async def test_tool_validate_checks_const_schema_constraints():
    @tool(schema={"kind": {"type": "string", "const": "patch"}})
    def apply(kind: str) -> str:
        """Apply something."""
        return kind

    assert await apply.validate(kind="patch") == ToolValidation.allow()
    assert await apply.validate(kind="write") == ToolValidation.deny(
        "Tool argument kind must be patch"
    )


def test_claude_tool_aliases_match_stdtools_names():
    from miniadk import canonical_tool_name

    assert canonical_tool_name("Read") == "read_file"
    assert canonical_tool_name("Delete") == "delete_file"
    assert canonical_tool_name("RM") == "delete_file"
    assert canonical_tool_name("Move") == "move_file"
    assert canonical_tool_name("MV") == "move_file"
    assert canonical_tool_name("Rename") == "move_file"
    assert canonical_tool_name("Copy") == "copy_file"
    assert canonical_tool_name("CP") == "copy_file"
    assert canonical_tool_name("Grep") == "search_text"
    assert canonical_tool_name("search_workspace_text") == "search_text"
    assert canonical_tool_name("Glob") == "glob_files"
    assert canonical_tool_name("glob_workspace_files") == "glob_files"
    assert canonical_tool_name("LS") == "list_files"
    assert canonical_tool_name("list_workspace_files") == "list_files"
    assert canonical_tool_name("Bash") == "shell"
    assert canonical_tool_name("MultiEdit") == "edit_files"
    assert canonical_tool_name("TodoWrite") == "todo_write"
    assert canonical_tool_name("TodoRead") == "todo_read"
    assert canonical_tool_name("Task") == "spawn_agent"
    assert canonical_tool_name("StartWork") == "start_work"
    assert canonical_tool_name("ListWork") == "list_work"
    assert canonical_tool_name("WorkList") == "list_work"
    assert canonical_tool_name("ReadWork") == "read_work"
    assert canonical_tool_name("CancelWork") == "cancel_work"
    assert canonical_tool_name("WebFetch") == "fetch_url"
    assert canonical_tool_name("URL Fetch") == "fetch_url"


def test_future_annotations_still_build_json_schema():
    from miniadk import Skill, SkillRegistry

    registry = SkillRegistry(
        [Skill(name="review", description="Review", body="body")]
    )
    tool = registry.tool()

    assert tool is not None
    assert tool.input_schema["properties"]["skill"]["type"] == "string"
    assert tool.input_schema["properties"]["args"]["oneOf"][0]["type"] == "string"


def test_tool_schema_handles_deferred_optional_annotations():
    @tool
    def read(path: "str", limit: "int | None" = None, tags: "list[str] | None" = None) -> str:
        """Read something."""
        return path

    assert read.input_schema["properties"]["path"]["type"] == "string"
    assert read.input_schema["properties"]["limit"]["type"] == "integer"
    assert read.input_schema["properties"]["tags"] == {
        "type": "array",
        "items": {"type": "string"},
        "default": None,
    }


def test_tool_schema_preserves_optional_inner_type():
    @tool
    def read(path: str, offset: int = 1, limit: int | None = None, tag: Optional[str] = None) -> str:
        """Read something."""
        return path

    assert read.input_schema["properties"]["offset"]["type"] == "integer"
    assert read.input_schema["properties"]["limit"]["type"] == "integer"
    assert read.input_schema["properties"]["tag"]["type"] == "string"
    assert read.input_schema["required"] == ["path"]


def test_tool_schema_preserves_json_safe_parameter_defaults():
    @tool
    def read(path: str, limit: int = 100, tags: list[str] = ["py"]) -> str:
        """Read something."""
        return path

    assert read.input_schema["properties"]["limit"] == {
        "type": "integer",
        "default": 100,
    }
    assert read.input_schema["properties"]["tags"] == {
        "type": "array",
        "items": {"type": "string"},
        "default": ["py"],
    }


def test_tool_schema_preserves_collection_inner_types():
    @tool
    def batch(paths: list[str], scores: dict[str, int]) -> str:
        """Batch something."""
        return ",".join(paths)

    assert batch.input_schema["properties"]["paths"] == {
        "type": "array",
        "items": {"type": "string"},
    }
    assert batch.input_schema["properties"]["scores"] == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }


def test_tool_schema_preserves_annotated_descriptions():
    @tool
    def read(path: Annotated[str, "Workspace-relative file path"]) -> str:
        """Read something."""
        return path

    assert read.input_schema["properties"]["path"] == {
        "type": "string",
        "description": "Workspace-relative file path",
    }


def test_tool_schema_merges_annotated_schema_metadata():
    @tool
    def read(
        path: Annotated[
            str,
            "Workspace-relative file path",
            {"minLength": 1, "pattern": r"^[^/].*"},
        ],
        limit: Annotated[int, {"minimum": 1, "maximum": 1000}] = 100,
    ) -> str:
        """Read something."""
        return path

    assert read.input_schema["properties"]["path"] == {
        "type": "string",
        "minLength": 1,
        "pattern": r"^[^/].*",
        "description": "Workspace-relative file path",
    }
    assert read.input_schema["properties"]["limit"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 1000,
        "default": 100,
    }


async def test_tool_validate_uses_annotated_schema_metadata():
    @tool
    def read(path: Annotated[str, {"minLength": 1}]) -> str:
        """Read something."""
        return path

    assert await read.validate(path="notes.txt") == ToolValidation.allow()
    assert await read.validate(path="") == ToolValidation.deny(
        "Tool argument path must be at least 1 chars"
    )


def test_tool_schema_preserves_nested_annotated_descriptions():
    @dataclass
    class Edit:
        old: Annotated[str, "Text to replace"]
        new: Annotated[str, "Replacement text"]

    @tool
    def edit_file(edit: Edit) -> str:
        """Edit a file."""
        return edit.new

    assert edit_file.input_schema["properties"]["edit"]["properties"] == {
        "old": {"type": "string", "description": "Text to replace"},
        "new": {"type": "string", "description": "Replacement text"},
    }


def test_tool_schema_preserves_dataclass_field_metadata_descriptions():
    @dataclass
    class Edit:
        old: str = field(metadata={"description": "Text to replace"})
        new: str = field(metadata={"doc": "Replacement text"})

    @tool
    def edit_file(edit: Edit) -> str:
        """Edit a file."""
        return edit.new

    assert edit_file.input_schema["properties"]["edit"]["properties"] == {
        "old": {"type": "string", "description": "Text to replace"},
        "new": {"type": "string", "description": "Replacement text"},
    }


def test_tool_schema_preserves_dataclass_fields():
    @dataclass
    class Edit:
        old: str
        new: str
        replace_all: bool = False

    @tool
    def edit_file(path: str, edit: Edit) -> str:
        """Edit a file."""
        return path

    assert edit_file.input_schema["properties"]["edit"] == {
        "type": "object",
        "properties": {
            "old": {"type": "string"},
            "new": {"type": "string"},
            "replace_all": {"type": "boolean", "default": False},
        },
        "additionalProperties": False,
        "required": ["old", "new"],
    }


def test_tool_schema_preserves_json_safe_dataclass_defaults():
    @dataclass
    class ReadOptions:
        limit: int = 100
        include_hidden: bool = False

    @tool
    def read(options: ReadOptions) -> str:
        """Read something."""
        return str(options.limit)

    assert read.input_schema["properties"]["options"]["properties"] == {
        "limit": {"type": "integer", "default": 100},
        "include_hidden": {"type": "boolean", "default": False},
    }


def test_tool_schema_preserves_typeddict_fields():
    class Edit(TypedDict):
        old: Annotated[str, "Text to replace"]
        new: str
        replace_all: NotRequired[bool]

    @tool
    def edit_file(edit: Edit) -> str:
        """Edit a file."""
        return edit["new"]

    assert edit_file.input_schema["properties"]["edit"] == {
        "type": "object",
        "properties": {
            "old": {"type": "string", "description": "Text to replace"},
            "new": {"type": "string"},
            "replace_all": {"type": "boolean"},
        },
        "additionalProperties": False,
        "required": ["old", "new"],
    }


def test_tool_schema_preserves_deferred_typeddict_notrequired_fields():
    Edit = TypedDict(
        "Edit",
        {
            "old": "str",
            "replace_all": "NotRequired[bool]",
        },
    )

    @tool
    def edit_file(edit: Edit) -> str:
        """Edit a file."""
        return str(edit["old"])

    assert edit_file.input_schema["properties"]["edit"]["required"] == ["old"]


async def test_tool_validate_checks_dataclass_fields():
    @dataclass
    class Edit:
        old: str
        new: str

    @tool
    def edit_file(edit: Edit) -> str:
        """Edit a file."""
        return edit.new

    assert await edit_file.validate(edit={"old": "a"}) == ToolValidation.deny(
        "Missing required tool argument: edit.new"
    )
    assert await edit_file.validate(
        edit={"old": "a", "new": "b", "extra": "c"}
    ) == ToolValidation.deny("Unknown tool argument: edit.extra")
    assert await edit_file.validate(edit={"old": "a", "new": 1}) == ToolValidation.deny(
        "Tool argument edit.new must be string"
    )


async def test_tool_validate_checks_typeddict_fields():
    class Edit(TypedDict):
        old: str
        new: str
        replace_all: NotRequired[bool]

    @tool
    def edit_file(edit: Edit) -> str:
        """Edit a file."""
        return edit["new"]

    assert await edit_file.validate(edit={"old": "a"}) == ToolValidation.deny(
        "Missing required tool argument: edit.new"
    )
    assert await edit_file.validate(
        edit={"old": "a", "new": "b", "extra": "c"}
    ) == ToolValidation.deny("Unknown tool argument: edit.extra")
    assert await edit_file.validate(
        edit={"old": "a", "new": "b", "replace_all": "yes"}
    ) == ToolValidation.deny("Tool argument edit.replace_all must be boolean")


async def test_tool_run_coerces_dataclass_arguments():
    @dataclass
    class Edit:
        old: str
        new: str

    seen = []

    @tool
    def edit_file(edit: Edit) -> str:
        """Edit a file."""
        seen.append(edit)
        return f"{edit.old}->{edit.new}"

    assert await edit_file.run(edit={"old": "a", "new": "b"}) == "a->b"
    assert seen == [Edit(old="a", new="b")]


def test_tool_direct_call_coerces_dataclass_arguments():
    @dataclass
    class Edit:
        old: str
        new: str

    @tool
    def edit_file(edit: Edit) -> str:
        """Edit a file."""
        return f"{edit.old}->{edit.new}"

    assert edit_file(edit={"old": "a", "new": "b"}) == "a->b"


async def test_tool_run_coerces_nested_dataclass_arguments():
    @dataclass
    class Edit:
        old: str
        new: str

    @dataclass
    class Patch:
        path: str
        edits: list[Edit]

    seen = []

    @tool
    def edit_file(patch: Patch) -> str:
        """Edit a file."""
        seen.append(patch)
        return patch.edits[0].new

    assert await edit_file.run(
        patch={"path": "a.txt", "edits": [{"old": "a", "new": "b"}]}
    ) == "b"
    assert seen == [Patch(path="a.txt", edits=[Edit(old="a", new="b")])]


async def test_tool_run_coerces_tuple_and_set_arguments():
    seen = []

    @tool
    def collect(names: tuple[str, ...], tags: set[str]) -> str:
        """Collect items."""
        seen.append((names, tags))
        return names[0]

    assert await collect.run(names=["Ada", "Grace"], tags=["py", "ai", "py"]) == "Ada"
    assert seen == [(("Ada", "Grace"), {"py", "ai"})]


def test_tool_direct_call_coerces_fixed_tuple_arguments():
    @tool
    def point(xy: tuple[int, int]) -> int:
        """Read a point."""
        assert isinstance(xy, tuple)
        return xy[0] + xy[1]

    assert point(xy=[2, 3]) == 5


def test_tool_schema_preserves_literal_choices():
    @tool
    def search(query: str, mode: Literal["files", "content"] = "content") -> str:
        """Search something."""
        return query

    assert search.input_schema["properties"]["mode"] == {
        "enum": ["files", "content"],
        "type": "string",
        "default": "content",
    }
    assert search.input_schema["required"] == ["query"]


def test_tool_schema_preserves_enum_values():
    class Mode(Enum):
        FAST = "fast"
        SAFE = "safe"

    @tool
    def run(mode: Mode) -> str:
        """Run something."""
        return mode.value

    assert run.input_schema["properties"]["mode"] == {
        "enum": ["fast", "safe"],
        "type": "string",
    }
    assert run.input_schema["required"] == ["mode"]


async def test_tool_run_coerces_enum_arguments():
    class Mode(Enum):
        FAST = "fast"
        SAFE = "safe"

    seen = []

    @tool
    def run(mode: Mode) -> str:
        """Run something."""
        seen.append(mode)
        return mode.name

    assert await run.run(mode="safe") == "SAFE"
    assert seen == [Mode.SAFE]


def test_tool_direct_call_coerces_enum_arguments():
    class Mode(Enum):
        FAST = "fast"
        SAFE = "safe"

    @tool
    def run(mode: Mode) -> str:
        """Run something."""
        return mode.name

    assert run(mode="safe") == "SAFE"


async def test_tool_run_coerces_nested_enum_arguments():
    class Mode(Enum):
        FAST = "fast"
        SAFE = "safe"

    @dataclass
    class Job:
        mode: Mode

    seen = []

    @tool
    def run(job: Job) -> str:
        """Run something."""
        seen.append(job)
        return job.mode.value

    assert await run.run(job={"mode": "fast"}) == "fast"
    assert seen == [Job(mode=Mode.FAST)]


def test_tool_schema_resolves_deferred_literal_annotations():
    @tool
    def search(mode: "Literal['files', 'content']") -> str:
        """Search something."""
        return mode

    assert search.input_schema["properties"]["mode"] == {
        "enum": ["files", "content"],
        "type": "string",
    }


def test_tool_schema_can_be_overridden_per_parameter():
    @tool(
        schema={
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            }
        }
    )
    def save(items: list) -> str:
        """Save items."""
        return str(len(items))

    assert save.input_schema["properties"]["items"] == {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    }
    assert save.input_schema["required"] == ["items"]


async def test_tool_runs_async_functions():
    @tool
    async def greet(name: str) -> str:
        """Greet someone."""
        return f"hello {name}"

    assert await greet.run(name="Ada") == "hello Ada"


def test_tool_rejects_non_positive_timeout():
    with pytest.raises(ValueError, match="timeout must be > 0"):

        @tool(timeout=0)
        async def wait() -> str:
            """Wait."""
            return "done"


async def test_tool_timeout_cancels_async_function():
    cancelled = asyncio.Event()

    @tool(timeout=0.01)
    async def wait() -> str:
        """Wait."""
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with pytest.raises(TimeoutError, match="wait timed out after 0.01 seconds"):
        await wait.run()

    await asyncio.wait_for(cancelled.wait(), timeout=0.2)


async def test_tool_supports_optional_capability_metadata():
    def validate_path(path: str):
        if path.startswith("/"):
            return "path must be relative"
        return True

    @tool(read_only=True, concurrency_safe=True, validate=validate_path)
    def read_file(path: str) -> str:
        """Read a file."""
        return path

    assert read_file.is_read_only(path="notes.txt") is True
    assert read_file.is_destructive(path="notes.txt") is False
    assert read_file.is_concurrency_safe(path="notes.txt") is True
    assert await read_file.validate(path="notes.txt") == ToolValidation.allow()
    assert await read_file.validate(path="/etc/passwd") == ToolValidation.deny(
        "path must be relative"
    )


async def test_tool_metadata_functions_ignore_unknown_arguments():
    seen = []

    def validate(path: str):
        seen.append(("validate", path))
        return True

    @tool(
        read_only=lambda path: path.endswith(".md"),
        destructive=lambda path: path.endswith(".py"),
        concurrency_safe=lambda path: path.startswith("docs/"),
        validate=validate,
    )
    def edit_file(path: str, content: str, dry: bool = False) -> str:
        """Edit a file."""
        return content

    assert await edit_file.validate(
        path="docs/readme.md",
        content="hello",
        dry=True,
    ) == ToolValidation.allow()
    assert edit_file.is_read_only(
        path="docs/readme.md",
        content="hello",
        dry=True,
    ) is True
    assert edit_file.is_destructive(
        path="src/app.py",
        content="hello",
        dry=True,
    ) is True
    assert edit_file.is_concurrency_safe(
        path="docs/readme.md",
        content="hello",
        dry=True,
    ) is True
    assert seen == [("validate", "docs/readme.md")]


async def test_tool_supports_callable_safety_flags():
    @tool(
        read_only=lambda command: command.startswith("git status"),
        destructive=lambda command: command.startswith("rm "),
    )
    def shell(command: str) -> str:
        """Run a command."""
        return command

    assert shell.is_read_only(command="git status --short") is True
    assert shell.is_read_only(command="pytest") is False
    assert shell.is_destructive(command="rm -rf build") is True


async def test_tool_formats_model_facing_text_without_changing_result():
    @tool(format=lambda result, path: f"{path}: {len(result)} lines")
    def read_lines(path: str) -> list[str]:
        """Read lines."""
        return ["a", "b"]

    result = await read_lines.run(path="notes.txt")

    assert result == ["a", "b"]
    assert await read_lines.text(result, path="notes.txt") == "notes.txt: 2 lines"


async def test_tool_format_can_ignore_tool_arguments():
    @tool(format=lambda result: f"total={result}")
    def add(left: int, right: int) -> int:
        """Add numbers."""
        return left + right

    result = await add.run(left=2, right=3)

    assert await add.text(result, left=2, right=3) == "total=5"


async def test_tool_can_clip_model_facing_text():
    @tool(max_text=12)
    def read_file(path: str) -> str:
        """Read a file."""
        return "abcdefghijklmnopqrstuvwxyz"

    result = await read_file.run(path="notes.txt")

    assert result == "abcdefghijklmnopqrstuvwxyz"
    assert await read_file.text(result, path="notes.txt") == "abcdefghijkl"


async def test_tool_can_clip_formatted_model_text_with_marker():
    @tool(format=lambda result: f"result={result}", max_text=18)
    def read_file(path: str) -> str:
        """Read a file."""
        return "abcdefghijklmnopqrstuvwxyz"

    result = await read_file.run(path="notes.txt")

    assert await read_file.text(result, path="notes.txt") == "res\n...[truncated]"
