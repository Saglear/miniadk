import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import pytest

import miniadk.stdtools.agents as agent_tools
from miniadk.stdtools import (
    glob_files,
    list_files,
    make_copy_file,
    make_delete_file,
    make_edit_file,
    make_edit_files,
    make_fetch_url,
    make_glob_files,
    make_list_files,
    make_move_file,
    make_read_file,
    make_search_text,
    make_shell,
    make_spawn,
    make_tools,
    make_work,
    make_write_file,
    search_text,
)
from miniadk import (
    Agent,
    FetchResult,
    Guard,
    Message,
    ModelResult,
    PermissionDecision,
    RunDecision,
    ScriptedModel,
    Skill,
    SkillRegistry,
    Spawn as PublicSpawn,
    SpawnResult as PublicSpawnResult,
    ToolCall,
    Work as PublicWork,
    WorkResult as PublicWorkResult,
    make_spawn as public_make_spawn,
    make_tools as public_make_tools,
    make_work as public_make_work,
    tool,
)
from miniadk.stdtools import Spawn, SpawnResult, Work, WorkResult


async def test_file_tools_are_rooted_to_workspace(tmp_path):
    read_file = make_read_file(root=tmp_path)
    write_file = make_write_file(root=tmp_path)

    assert await write_file.run(path="notes/hello.txt", content="hello") == "wrote notes/hello.txt"
    assert await read_file.run(path="notes/hello.txt") == "hello"
    assert read_file.is_read_only(path="notes/hello.txt") is True
    assert write_file.is_destructive(path="notes/hello.txt", content="hello") is True


async def test_write_file_can_preview_create_without_writing(tmp_path):
    write_file = make_write_file(root=tmp_path)

    result = await write_file.run(path="notes/hello.txt", content="hello", dry=True)

    assert result == "would create notes/hello.txt: 5 chars"
    assert not (tmp_path / "notes").exists()
    assert write_file.input_schema["properties"]["dry"]["type"] == "boolean"
    assert write_file.is_read_only(path="notes/hello.txt", content="hello", dry=True)
    assert not write_file.is_destructive(
        path="notes/hello.txt",
        content="hello",
        dry=True,
    )


async def test_write_file_can_preview_overwrite_without_writing(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("old", encoding="utf-8")
    write_file = make_write_file(root=tmp_path)

    result = await write_file.run(path="notes.txt", content="new content", dry=True)

    assert result == "would overwrite notes.txt: 11 chars"
    assert target.read_text(encoding="utf-8") == "old"


async def test_file_tools_reject_path_escape(tmp_path):
    read_file = make_read_file(root=tmp_path)

    result = await read_file.run(path="../secret.txt")

    assert "outside root" in result


async def test_read_file_can_limit_large_output(tmp_path):
    (tmp_path / "long.txt").write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
    read_file = make_read_file(root=tmp_path, max_text=12)

    result = await read_file.run(path="long.txt")
    text = await read_file.text(result, path="long.txt")

    assert result == "abcdefghijkl"
    assert text == "abcdefghijkl"


async def test_read_file_formats_empty_file_clearly(tmp_path):
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    read_file = make_read_file(root=tmp_path)

    result = await read_file.run(path="empty.txt")
    text = await read_file.text(result, path="empty.txt")

    assert result == ""
    assert text == "empty file"


async def test_read_file_can_read_a_line_range(tmp_path):
    (tmp_path / "notes.txt").write_text(
        "one\ntwo\nthree\nfour\n",
        encoding="utf-8",
    )
    read_file = make_read_file(root=tmp_path)

    result = await read_file.run(path="notes.txt", offset=2, limit=2)
    empty = await read_file.run(path="notes.txt", offset=10, limit=2)
    empty_text = await read_file.text(
        empty,
        path="notes.txt",
        offset=10,
        limit=2,
    )

    assert result == "two\nthree\n"
    assert empty == ""
    assert empty_text == "no lines in range"
    assert read_file.input_schema["properties"]["offset"] == {
        "type": "integer",
        "minimum": 1,
        "default": 1,
    }
    assert read_file.input_schema["properties"]["limit"] == {
        "type": "integer",
        "minimum": 1,
    }


async def test_read_file_can_include_line_numbers(tmp_path):
    (tmp_path / "notes.txt").write_text(
        "one\ntwo\nthree\n",
        encoding="utf-8",
    )
    read_file = make_read_file(root=tmp_path)

    full = await read_file.run(path="notes.txt", numbers=True)
    partial = await read_file.run(path="notes.txt", offset=2, limit=2, numbers=True)

    assert full == "1: one\n2: two\n3: three\n"
    assert partial == "2: two\n3: three\n"
    assert read_file.input_schema["properties"]["numbers"]["type"] == "boolean"


async def test_read_file_validates_line_range(tmp_path):
    (tmp_path / "notes.txt").write_text("one\n", encoding="utf-8")
    read_file = make_read_file(root=tmp_path)
    offset_validation = await read_file.validate(path="notes.txt", offset=0)
    limit_validation = await read_file.validate(path="notes.txt", limit=0)

    assert await read_file.run(path="notes.txt", offset=0) == (
        "read_file failed: offset must be >= 1"
    )
    assert await read_file.run(path="notes.txt", limit=0) == (
        "read_file failed: limit must be >= 1"
    )
    assert offset_validation.ok is False
    assert offset_validation.message == "Tool argument offset must be >= 1"
    assert limit_validation.ok is False
    assert limit_validation.message == "Tool argument limit must be >= 1"


async def test_read_file_rejects_binary_files(tmp_path):
    (tmp_path / "image.bin").write_bytes(b"hello\0world")
    read_file = make_read_file(root=tmp_path)

    result = await read_file.run(path="image.bin")

    assert result == "read_file failed: binary files are not supported: image.bin"


async def test_read_file_reports_invalid_utf8_text(tmp_path):
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe")
    read_file = make_read_file(root=tmp_path)

    result = await read_file.run(path="bad.txt")

    assert result == "read_file failed: file is not valid UTF-8 text: bad.txt"


async def test_list_files_returns_relative_paths(tmp_path):
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.py").write_text("", encoding="utf-8")

    result = await list_files(root=tmp_path, pattern="*.py")

    assert result == ["a.py", "nested/b.py"]


async def test_list_files_matches_relative_path_patterns(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("", encoding="utf-8")

    result = await list_files(root=tmp_path, pattern="src/*.py")

    assert result == ["src/app.py"]


async def test_list_files_can_ignore_directories(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.py").write_text("", encoding="utf-8")

    result = await list_files(root=tmp_path, pattern="*.py", ignore={"node_modules"})

    assert result == ["src/app.py"]


async def test_list_files_can_ignore_glob_patterns(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.js").write_text("", encoding="utf-8")
    (tmp_path / "src" / "app.min.js").write_text("", encoding="utf-8")
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "lib.js").write_text("", encoding="utf-8")

    result = await list_files(
        root=tmp_path,
        pattern="*.js",
        ignore={"*.min.js", "vendor/*"},
    )

    assert result == ["src/app.js"]


async def test_list_files_can_limit_scanned_files(tmp_path):
    for index in range(5):
        (tmp_path / f"{index}.txt").write_text("", encoding="utf-8")
    (tmp_path / "z.py").write_text("", encoding="utf-8")

    result = await list_files(root=tmp_path, pattern="*.py", max_files=3)

    assert result == []


async def test_list_files_prunes_nested_ignored_directories(tmp_path):
    (tmp_path / "src" / "node_modules").mkdir(parents=True)
    (tmp_path / "src" / "node_modules" / "lib.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "app.py").write_text("", encoding="utf-8")

    result = await list_files(root=tmp_path, pattern="*.py", ignore={"node_modules"})

    assert result == ["src/app.py"]


async def test_list_files_skips_symlink_escape(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    try:
        (outside / "secret.txt").write_text("secret\n", encoding="utf-8")
        (tmp_path / "safe.txt").write_text("safe\n", encoding="utf-8")
        (tmp_path / "link").symlink_to(outside)

        result = await list_files(root=tmp_path)

        assert result == ["safe.txt"]
    finally:
        (outside / "secret.txt").unlink(missing_ok=True)
        outside.rmdir()


async def test_search_text_skips_symlink_escape(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    try:
        (outside / "secret.txt").write_text("needle\n", encoding="utf-8")
        (tmp_path / "safe.txt").write_text("needle\n", encoding="utf-8")
        (tmp_path / "link").symlink_to(outside)

        result = await search_text(root=tmp_path, pattern="needle")

        assert result == ["safe.txt:1: needle"]
    finally:
        (outside / "secret.txt").unlink(missing_ok=True)
        outside.rmdir()


async def test_make_list_files_returns_a_tool(tmp_path):
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    list_files_tool = make_list_files(root=tmp_path)

    result = await list_files_tool.run(pattern="*.py")
    text = await list_files_tool.text(result, pattern="*.py")
    empty = await list_files_tool.run(pattern="*.md")
    empty_text = await list_files_tool.text(empty, pattern="*.md")

    assert result == ["a.py"]
    assert text == "a.py"
    assert empty == []
    assert empty_text == "no files found"
    assert list_files_tool.is_read_only(pattern="*.py") is True
    assert list_files_tool.is_concurrency_safe(pattern="*.py") is True


async def test_make_list_files_passes_ignore(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("", encoding="utf-8")
    (tmp_path / "app.py").write_text("", encoding="utf-8")
    list_files_tool = make_list_files(root=tmp_path, ignore={".git"})

    result = await list_files_tool.run(pattern="*")

    assert result == ["app.py"]


async def test_make_list_files_passes_scanned_file_limit(tmp_path):
    for index in range(5):
        (tmp_path / f"{index}.txt").write_text("", encoding="utf-8")
    (tmp_path / "z.py").write_text("", encoding="utf-8")
    list_files_tool = make_list_files(root=tmp_path, max_files=3)

    result = await list_files_tool.run(pattern="*.py")

    assert result == []


async def test_glob_and_search_tools_cover_claude_style_file_work(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hello')\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("hello docs\n", encoding="utf-8")

    assert await glob_files(root=tmp_path, pattern="*.md") == ["README.md"]
    assert await search_text(root=tmp_path, pattern="hello", glob="*.py") == [
        "src/app.py:1: print('hello')"
    ]

    glob_tool = make_glob_files(root=tmp_path)
    search_tool = make_search_text(root=tmp_path)

    assert await glob_tool.run(pattern="*.md") == ["README.md"]
    glob_empty = await glob_tool.run(pattern="*.txt")
    search_result = await search_tool.run(pattern="docs", glob="*.md")
    search_empty = await search_tool.run(pattern="missing")

    assert await glob_tool.text(glob_empty, pattern="*.txt") == "no files found"
    assert search_result == ["README.md:1: hello docs"]
    assert await search_tool.text(search_result, pattern="docs", glob="*.md") == (
        "README.md:1: hello docs"
    )
    assert search_empty == []
    assert await search_tool.text(search_empty, pattern="missing") == "no matches found"


async def test_glob_files_can_limit_scanned_files(tmp_path):
    for index in range(5):
        (tmp_path / f"{index}.txt").write_text("", encoding="utf-8")
    (tmp_path / "z.py").write_text("", encoding="utf-8")

    result = await glob_files(root=tmp_path, pattern="*.py", max_files=3)

    assert result == []


async def test_search_text_can_skip_large_files(tmp_path):
    (tmp_path / "small.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "large.txt").write_text("needle\n" * 100, encoding="utf-8")

    result = await search_text(root=tmp_path, pattern="needle", max_file=20)

    assert result == ["small.txt:1: needle"]


async def test_search_text_skips_binary_files(tmp_path):
    (tmp_path / "app.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "image.bin").write_bytes(b"needle\0")

    result = await search_text(root=tmp_path, pattern="needle")

    assert result == ["app.txt:1: needle"]


async def test_search_text_can_limit_scanned_files(tmp_path):
    for index in range(5):
        (tmp_path / f"{index}.txt").write_text("miss\n", encoding="utf-8")
    (tmp_path / "z.txt").write_text("needle\n", encoding="utf-8")

    result = await search_text(
        root=tmp_path,
        pattern="needle",
        max_files=3,
    )

    assert result == []


async def test_search_text_streams_until_limit_without_reading_whole_file(tmp_path, monkeypatch):
    big = tmp_path / "big.txt"
    big.write_text(("miss\n" * 1000) + "needle\n" + ("miss\n" * 1000), encoding="utf-8")

    def fail_read_text(*args, **kwargs):
        raise AssertionError("search_text should stream files instead of read_text")

    monkeypatch.setattr(type(big), "read_text", fail_read_text)

    result = await search_text(root=tmp_path, pattern="needle", limit=1)

    assert result == ["big.txt:1001: needle"]


async def test_search_text_can_include_context_lines(tmp_path):
    (tmp_path / "notes.txt").write_text(
        "one\ntwo\nneedle\nfour\nfive\n",
        encoding="utf-8",
    )

    result = await search_text(root=tmp_path, pattern="needle", context=1)

    assert result == [
        "notes.txt:2:- two",
        "notes.txt:3:> needle",
        "notes.txt:4:- four",
    ]


async def test_search_text_context_limit_counts_matches_not_context_lines(tmp_path):
    (tmp_path / "notes.txt").write_text(
        "one\ntwo\nneedle\nfour\nfive\n",
        encoding="utf-8",
    )

    result = await search_text(root=tmp_path, pattern="needle", context=1, limit=1)

    assert result == [
        "notes.txt:2:- two",
        "notes.txt:3:> needle",
        "notes.txt:4:- four",
    ]


async def test_search_text_context_marks_overlapping_matches(tmp_path):
    (tmp_path / "notes.txt").write_text(
        "one\nneedle\nneedle\nfour\n",
        encoding="utf-8",
    )

    result = await search_text(root=tmp_path, pattern="needle", context=1, limit=2)

    assert result == [
        "notes.txt:1:- one",
        "notes.txt:2:> needle",
        "notes.txt:3:> needle",
        "notes.txt:4:- four",
    ]


async def test_search_text_can_ignore_directories(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("needle\n", encoding="utf-8")

    result = await search_text(root=tmp_path, pattern="needle", ignore={".venv"})

    assert result == ["src/app.py:1: needle"]


async def test_search_text_can_ignore_glob_patterns(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "src" / "generated.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "lib.py").write_text("needle\n", encoding="utf-8")

    result = await search_text(
        root=tmp_path,
        pattern="needle",
        ignore={"*/generated.py", "vendor/*"},
    )

    assert result == ["src/app.py:1: needle"]


async def test_search_text_prunes_nested_ignored_directories(tmp_path):
    (tmp_path / "src" / "dist").mkdir(parents=True)
    (tmp_path / "src" / "dist" / "bundle.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "src" / "app.py").write_text("needle\n", encoding="utf-8")

    result = await search_text(root=tmp_path, pattern="needle", ignore={"dist"})

    assert result == ["src/app.py:1: needle"]


async def test_make_search_text_passes_large_file_limit(tmp_path):
    (tmp_path / "small.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "large.txt").write_text("needle\n" * 100, encoding="utf-8")
    search_tool = make_search_text(root=tmp_path, max_file=20)

    result = await search_tool.run(pattern="needle")

    assert result == ["small.txt:1: needle"]


async def test_make_search_text_passes_scanned_file_limit(tmp_path):
    for index in range(5):
        (tmp_path / f"{index}.txt").write_text("miss\n", encoding="utf-8")
    (tmp_path / "z.txt").write_text("needle\n", encoding="utf-8")
    search_tool = make_search_text(root=tmp_path, max_files=3)

    result = await search_tool.run(pattern="needle")

    assert result == []


async def test_make_search_text_passes_ignore(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "bundle.py").write_text("needle\n", encoding="utf-8")
    search_tool = make_search_text(root=tmp_path, ignore={"dist"})

    result = await search_tool.run(pattern="needle")

    assert result == ["src/app.py:1: needle"]


async def test_make_search_text_validates_invalid_regex(tmp_path):
    search_tool = make_search_text(root=tmp_path)

    validation = await search_tool.validate(pattern="[")

    assert validation.ok is False
    assert validation.message.startswith("search_text failed: invalid regex:")


async def test_make_search_text_passes_context_and_validates_it(tmp_path):
    (tmp_path / "notes.txt").write_text(
        "one\ntwo\nneedle\nfour\n",
        encoding="utf-8",
    )
    search_tool = make_search_text(root=tmp_path)

    result = await search_tool.run(pattern="needle", context=1)
    validation = await search_tool.validate(pattern="needle", context=-1)

    assert result == [
        "notes.txt:2:- two",
        "notes.txt:3:> needle",
        "notes.txt:4:- four",
    ]
    assert validation.ok is False
    assert validation.message == "Tool argument context must be >= 0"
    assert search_tool.input_schema["properties"]["context"] == {
        "type": "integer",
        "minimum": 0,
        "default": 0,
    }


async def test_edit_file_replaces_one_occurrence(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello world", encoding="utf-8")

    edit_file = make_edit_file(root=tmp_path)
    result = await edit_file.run(path="a.txt", old="hello", new="hi")

    assert result == "edited a.txt"
    assert target.read_text(encoding="utf-8") == "hi world"


async def test_edit_file_rejects_ambiguous_text_by_default(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello world hello", encoding="utf-8")

    edit_file = make_edit_file(root=tmp_path)
    validation = await edit_file.validate(path="a.txt", old="hello", new="hi")
    result = await edit_file.run(path="a.txt", old="hello", new="hi")

    assert validation.ok is False
    assert validation.message == (
        "edit_file failed: 'hello' appears 2 times in a.txt; "
        "pass all=True to replace every match"
    )
    assert result == validation.message
    assert target.read_text(encoding="utf-8") == "hello world hello"


async def test_edit_file_can_replace_all_matches_when_requested(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello world hello", encoding="utf-8")

    edit_file = make_edit_file(root=tmp_path)
    result = await edit_file.run(path="a.txt", old="hello", new="hi", all=True)

    assert result == "edited a.txt: 2 replacements"
    assert target.read_text(encoding="utf-8") == "hi world hi"
    assert edit_file.input_schema["properties"]["all"]["type"] == "boolean"


async def test_edit_file_can_preview_without_writing(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello world hello", encoding="utf-8")

    edit_file = make_edit_file(root=tmp_path)
    result = await edit_file.run(
        path="a.txt",
        old="hello",
        new="hi",
        dry=True,
        all=True,
    )

    assert result == "would edit a.txt: 2 replacements"
    assert target.read_text(encoding="utf-8") == "hello world hello"
    assert edit_file.input_schema["properties"]["dry"]["type"] == "boolean"
    assert edit_file.is_read_only(
        path="a.txt",
        old="hello",
        new="hi",
        dry=True,
        all=True,
    )
    assert not edit_file.is_destructive(
        path="a.txt",
        old="hello",
        new="hi",
        dry=True,
        all=True,
    )


async def test_edit_file_validates_missing_text_before_running(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    edit_file = make_edit_file(root=tmp_path)

    validation = await edit_file.validate(path="a.txt", old="missing", new="hi")

    assert validation.ok is False
    assert validation.message == "edit_file failed: 'missing' not found in a.txt"


async def test_edit_file_rejects_empty_old_text(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    edit_file = make_edit_file(root=tmp_path)

    validation = await edit_file.validate(path="a.txt", old="", new="prefix")
    result = await edit_file.run(path="a.txt", old="", new="prefix")

    assert validation.ok is False
    assert validation.message == "edit_file failed: old text is required"
    assert result == "edit_file failed: old text is required"
    assert target.read_text(encoding="utf-8") == "hello"


async def test_edit_file_keeps_original_when_atomic_replace_fails(tmp_path, monkeypatch):
    target = tmp_path / "a.txt"
    target.write_text("hello world", encoding="utf-8")
    original_replace = Path.replace

    def fail_replace(self, target_path):
        if self.name.startswith(".a.txt.") and self.name.endswith(".tmp"):
            raise OSError("replace failed")
        return original_replace(self, target_path)

    monkeypatch.setattr(Path, "replace", fail_replace)
    edit_file = make_edit_file(root=tmp_path)

    result = await edit_file.run(path="a.txt", old="hello", new="hi")

    assert result == "edit_file failed: replace failed"
    assert target.read_text(encoding="utf-8") == "hello world"
    assert not list(tmp_path.glob(".a.txt.*.tmp"))


async def test_edit_files_applies_multiple_replacements_atomically(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    edit_files = make_edit_files(root=tmp_path)
    result = await edit_files.run(
        path="a.txt",
        edits=[
            {"old": "alpha", "new": "one"},
            {"old": "gamma", "new": "three"},
        ],
    )

    assert result == "edited a.txt: 2 replacements"
    assert target.read_text(encoding="utf-8") == "one\nbeta\nthree\n"
    assert edit_files.is_destructive(path="a.txt", edits=[]) is True
    assert edit_files.input_schema["properties"]["edits"] == {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "old": {"type": "string"},
                "new": {"type": "string"},
            },
            "required": ["old", "new"],
            "additionalProperties": False,
        },
    }
    assert edit_files.input_schema["properties"]["dry"]["type"] == "boolean"


async def test_edit_files_can_preview_without_writing(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    edit_files = make_edit_files(root=tmp_path)
    result = await edit_files.run(
        path="a.txt",
        edits=[
            {"old": "alpha", "new": "one"},
            {"old": "gamma", "new": "three"},
        ],
        dry=True,
    )

    assert result == "would edit a.txt: 2 replacements"
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"
    assert edit_files.is_read_only(
        path="a.txt",
        edits=[{"old": "alpha", "new": "one"}],
        dry=True,
    )
    assert not edit_files.is_destructive(
        path="a.txt",
        edits=[{"old": "alpha", "new": "one"}],
        dry=True,
    )


async def test_write_file_keeps_original_when_atomic_replace_fails(tmp_path, monkeypatch):
    target = tmp_path / "a.txt"
    target.write_text("old", encoding="utf-8")
    original_replace = Path.replace

    def fail_replace(self, target_path):
        if self.name.startswith(".a.txt.") and self.name.endswith(".tmp"):
            raise OSError("replace failed")
        return original_replace(self, target_path)

    monkeypatch.setattr(Path, "replace", fail_replace)
    write_file = make_write_file(root=tmp_path)

    result = await write_file.run(path="a.txt", content="new")

    assert result == "write_file failed: replace failed"
    assert target.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".a.txt.*.tmp"))


async def test_edit_files_does_not_write_when_any_replacement_is_missing(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    edit_files = make_edit_files(root=tmp_path)
    result = await edit_files.run(
        path="a.txt",
        edits=[
            {"old": "alpha", "new": "one"},
            {"old": "missing", "new": "two"},
        ],
    )

    assert result == "edit_files failed: edit 2 old text not found in a.txt"
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"


async def test_edit_files_validates_all_replacements_before_running(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    edit_files = make_edit_files(root=tmp_path)

    validation = await edit_files.validate(
        path="a.txt",
        edits=[
            {"old": "alpha", "new": "one"},
            {"old": "missing", "new": "two"},
        ],
    )

    assert validation.ok is False
    assert validation.message == "edit_files failed: edit 2 old text not found in a.txt"


async def test_edit_files_rejects_empty_old_text(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("alpha\n", encoding="utf-8")
    edit_files = make_edit_files(root=tmp_path)

    validation = await edit_files.validate(
        path="a.txt",
        edits=[{"old": "", "new": "prefix"}],
    )
    result = await edit_files.run(
        path="a.txt",
        edits=[{"old": "", "new": "prefix"}],
    )

    assert validation.ok is False
    assert validation.message == "edit_files failed: edit 1 old text is required"
    assert result == "edit_files failed: edit 1 old text is required"
    assert target.read_text(encoding="utf-8") == "alpha\n"


async def test_delete_file_removes_regular_file(tmp_path):
    target = tmp_path / "old.txt"
    target.write_text("old", encoding="utf-8")
    delete_file = make_delete_file(root=tmp_path)

    result = await delete_file.run(path="old.txt")

    assert result == "deleted old.txt"
    assert not target.exists()
    assert delete_file.is_destructive(path="old.txt") is True


async def test_delete_file_can_preview_without_deleting(tmp_path):
    target = tmp_path / "old.txt"
    target.write_text("old", encoding="utf-8")
    delete_file = make_delete_file(root=tmp_path)

    result = await delete_file.run(path="old.txt", dry=True)

    assert result == "would delete old.txt"
    assert target.read_text(encoding="utf-8") == "old"
    assert delete_file.input_schema["properties"]["dry"]["type"] == "boolean"
    assert delete_file.is_read_only(path="old.txt", dry=True) is True
    assert delete_file.is_destructive(path="old.txt", dry=True) is False


async def test_delete_file_validates_missing_file_and_directories(tmp_path):
    (tmp_path / "dir").mkdir()
    delete_file = make_delete_file(root=tmp_path)

    missing = await delete_file.validate(path="missing.txt")
    directory = await delete_file.validate(path="dir")
    missing_result = await delete_file.run(path="missing.txt")

    assert missing.ok is False
    assert missing.message == "delete_file failed: file not found: missing.txt"
    assert directory.ok is False
    assert directory.message == "delete_file failed: not a regular file: dir"
    assert missing_result == "delete_file failed: file not found: missing.txt"


async def test_delete_file_rejects_path_escape_and_symlinks(tmp_path):
    outside = tmp_path.parent / "outside-delete.txt"
    outside.write_text("outside", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        return
    delete_file = make_delete_file(root=tmp_path)

    escaped = await delete_file.validate(path="../outside-delete.txt")
    linked = await delete_file.validate(path="link.txt")

    assert escaped.ok is False
    assert "outside root" in escaped.message
    assert linked.ok is False
    assert linked.message == "delete_file failed: not a regular file: link.txt"
    assert outside.exists()


async def test_move_file_moves_regular_file(tmp_path):
    source = tmp_path / "old.txt"
    target = tmp_path / "nested" / "new.txt"
    source.write_text("old", encoding="utf-8")
    move_file = make_move_file(root=tmp_path)

    result = await move_file.run(src="old.txt", dst="nested/new.txt")

    assert result == "moved old.txt -> nested/new.txt"
    assert not source.exists()
    assert target.read_text(encoding="utf-8") == "old"
    assert move_file.is_destructive(src="old.txt", dst="nested/new.txt") is True


async def test_move_file_can_preview_without_moving(tmp_path):
    source = tmp_path / "old.txt"
    source.write_text("old", encoding="utf-8")
    move_file = make_move_file(root=tmp_path)

    result = await move_file.run(src="old.txt", dst="new.txt", dry=True)

    assert result == "would move old.txt -> new.txt"
    assert source.read_text(encoding="utf-8") == "old"
    assert not (tmp_path / "new.txt").exists()
    assert move_file.input_schema["properties"]["dry"]["type"] == "boolean"
    assert move_file.input_schema["properties"]["overwrite"]["type"] == "boolean"
    assert move_file.is_read_only(src="old.txt", dst="new.txt", dry=True) is True
    assert move_file.is_destructive(src="old.txt", dst="new.txt", dry=True) is False


async def test_move_file_requires_overwrite_for_existing_destination(tmp_path):
    source = tmp_path / "old.txt"
    target = tmp_path / "new.txt"
    source.write_text("old", encoding="utf-8")
    target.write_text("new", encoding="utf-8")
    move_file = make_move_file(root=tmp_path)

    validation = await move_file.validate(src="old.txt", dst="new.txt")
    preview = await move_file.run(
        src="old.txt",
        dst="new.txt",
        dry=True,
        overwrite=True,
    )
    result = await move_file.run(src="old.txt", dst="new.txt", overwrite=True)

    assert validation.ok is False
    assert validation.message == "move_file failed: destination exists: new.txt"
    assert preview == "would overwrite old.txt -> new.txt"
    assert result == "moved old.txt -> new.txt"
    assert target.read_text(encoding="utf-8") == "old"


async def test_move_file_validates_missing_same_and_directory_paths(tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "dir").mkdir()
    move_file = make_move_file(root=tmp_path)

    missing = await move_file.validate(src="missing.txt", dst="x.txt")
    same = await move_file.validate(src="a.txt", dst="a.txt")
    source_dir = await move_file.validate(src="dir", dst="x.txt")
    target_dir = await move_file.validate(src="a.txt", dst="dir")

    assert missing.ok is False
    assert missing.message == "move_file failed: file not found: missing.txt"
    assert same.ok is False
    assert same.message == "move_file failed: source and destination are the same"
    assert source_dir.ok is False
    assert source_dir.message == "move_file failed: source is not a regular file: dir"
    assert target_dir.ok is False
    assert target_dir.message == "move_file failed: destination is not a regular file: dir"


async def test_move_file_rejects_path_escape_and_symlinks(tmp_path):
    outside = tmp_path.parent / "outside-move.txt"
    outside.write_text("outside", encoding="utf-8")
    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        return
    move_file = make_move_file(root=tmp_path)

    escaped_source = await move_file.validate(src="../outside-move.txt", dst="x.txt")
    escaped_target = await move_file.validate(src="source.txt", dst="../outside-move.txt")
    source_link = await move_file.validate(src="link.txt", dst="x.txt")
    target_link = await move_file.validate(src="source.txt", dst="link.txt")

    assert escaped_source.ok is False
    assert "outside root" in escaped_source.message
    assert escaped_target.ok is False
    assert "outside root" in escaped_target.message
    assert source_link.ok is False
    assert source_link.message == "move_file failed: source is not a regular file: link.txt"
    assert target_link.ok is False
    assert target_link.message == (
        "move_file failed: destination is not a regular file: link.txt"
    )
    assert outside.read_text(encoding="utf-8") == "outside"


async def test_copy_file_copies_regular_file(tmp_path):
    source = tmp_path / "old.txt"
    target = tmp_path / "nested" / "new.txt"
    source.write_text("old", encoding="utf-8")
    copy_file = make_copy_file(root=tmp_path)

    result = await copy_file.run(src="old.txt", dst="nested/new.txt")

    assert result == "copied old.txt -> nested/new.txt"
    assert source.read_text(encoding="utf-8") == "old"
    assert target.read_text(encoding="utf-8") == "old"
    assert copy_file.is_destructive(src="old.txt", dst="nested/new.txt") is True


async def test_copy_file_can_preview_without_copying(tmp_path):
    source = tmp_path / "old.txt"
    source.write_text("old", encoding="utf-8")
    copy_file = make_copy_file(root=tmp_path)

    result = await copy_file.run(src="old.txt", dst="new.txt", dry=True)

    assert result == "would copy old.txt -> new.txt"
    assert source.read_text(encoding="utf-8") == "old"
    assert not (tmp_path / "new.txt").exists()
    assert copy_file.input_schema["properties"]["dry"]["type"] == "boolean"
    assert copy_file.input_schema["properties"]["overwrite"]["type"] == "boolean"
    assert copy_file.is_read_only(src="old.txt", dst="new.txt", dry=True) is True
    assert copy_file.is_destructive(src="old.txt", dst="new.txt", dry=True) is False


async def test_copy_file_requires_overwrite_for_existing_destination(tmp_path):
    source = tmp_path / "old.txt"
    target = tmp_path / "new.txt"
    source.write_text("old", encoding="utf-8")
    target.write_text("new", encoding="utf-8")
    copy_file = make_copy_file(root=tmp_path)

    validation = await copy_file.validate(src="old.txt", dst="new.txt")
    preview = await copy_file.run(
        src="old.txt",
        dst="new.txt",
        dry=True,
        overwrite=True,
    )
    result = await copy_file.run(src="old.txt", dst="new.txt", overwrite=True)

    assert validation.ok is False
    assert validation.message == "copy_file failed: destination exists: new.txt"
    assert preview == "would overwrite old.txt -> new.txt"
    assert result == "copied old.txt -> new.txt"
    assert source.read_text(encoding="utf-8") == "old"
    assert target.read_text(encoding="utf-8") == "old"


async def test_copy_file_validates_missing_same_and_directory_paths(tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "dir").mkdir()
    copy_file = make_copy_file(root=tmp_path)

    missing = await copy_file.validate(src="missing.txt", dst="x.txt")
    same = await copy_file.validate(src="a.txt", dst="a.txt")
    source_dir = await copy_file.validate(src="dir", dst="x.txt")
    target_dir = await copy_file.validate(src="a.txt", dst="dir")

    assert missing.ok is False
    assert missing.message == "copy_file failed: file not found: missing.txt"
    assert same.ok is False
    assert same.message == "copy_file failed: source and destination are the same"
    assert source_dir.ok is False
    assert source_dir.message == "copy_file failed: source is not a regular file: dir"
    assert target_dir.ok is False
    assert target_dir.message == "copy_file failed: destination is not a regular file: dir"


async def test_copy_file_rejects_path_escape_and_symlinks(tmp_path):
    outside = tmp_path.parent / "outside-copy.txt"
    outside.write_text("outside", encoding="utf-8")
    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        return
    copy_file = make_copy_file(root=tmp_path)

    escaped_source = await copy_file.validate(src="../outside-copy.txt", dst="x.txt")
    escaped_target = await copy_file.validate(src="source.txt", dst="../outside-copy.txt")
    source_link = await copy_file.validate(src="link.txt", dst="x.txt")
    target_link = await copy_file.validate(src="source.txt", dst="link.txt")

    assert escaped_source.ok is False
    assert "outside root" in escaped_source.message
    assert escaped_target.ok is False
    assert "outside root" in escaped_target.message
    assert source_link.ok is False
    assert source_link.message == "copy_file failed: source is not a regular file: link.txt"
    assert target_link.ok is False
    assert target_link.message == (
        "copy_file failed: destination is not a regular file: link.txt"
    )
    assert outside.read_text(encoding="utf-8") == "outside"


async def test_shell_tool_runs_command_with_timeout(tmp_path):
    shell = make_shell(cwd=tmp_path, timeout_seconds=5)

    result = await shell.run(command="printf hello")

    assert result.stdout == "hello"
    assert result.stderr == ""
    assert result.returncode == 0
    assert shell.is_destructive(command="printf hello") is True


async def test_shell_tool_can_pass_stdin(tmp_path):
    shell = make_shell(cwd=tmp_path)

    result = await shell.run(
        command="python -c \"import sys; print(sys.stdin.read().upper())\"",
        input="hello",
    )

    assert result.returncode == 0
    assert result.stdout == "HELLO\n"
    assert shell.input_schema["properties"]["input"]["type"] == "string"


async def test_shell_tool_can_run_from_subdirectory(tmp_path):
    nested = tmp_path / "pkg"
    nested.mkdir()
    shell = make_shell(cwd=tmp_path)

    result = await shell.run(
        command="python -c \"import pathlib; print(pathlib.Path.cwd().name)\"",
        cwd="pkg",
    )

    assert result.returncode == 0
    assert result.stdout == "pkg\n"
    assert shell.input_schema["properties"]["cwd"] == {
        "type": "string",
        "default": ".",
    }


async def test_shell_tool_closes_stdin_by_default(tmp_path):
    shell = make_shell(cwd=tmp_path, timeout_seconds=1)

    result = await shell.run(
        command="python -c \"import sys; print(sys.stdin.read() == '')\""
    )

    assert result.returncode == 0
    assert result.stdout == "True\n"


async def test_shell_tool_can_mark_read_only_commands_by_pattern(tmp_path):
    shell = make_shell(cwd=tmp_path, read=["git status*", "pytest --collect-only*"])

    assert shell.is_read_only(command="git status --short") is True
    assert shell.is_destructive(command="git status --short") is False
    assert shell.is_read_only(command="pytest --collect-only tests") is True
    assert shell.is_destructive(command="pytest tests") is True


async def test_shell_tool_can_mark_read_only_commands_by_callable(tmp_path):
    shell = make_shell(
        cwd=tmp_path,
        read=lambda command: command.startswith("python -m pytest --collect-only"),
    )

    assert shell.is_read_only(command="python -m pytest --collect-only tests") is True
    assert shell.is_destructive(command="python -m pytest tests") is True


async def test_shell_tool_validates_required_command(tmp_path):
    shell = make_shell(cwd=tmp_path)

    validation = await shell.validate(command="  ")
    empty_validation = await shell.validate(command="")

    assert validation.ok is False
    assert validation.message == "shell command is required"
    assert empty_validation.ok is False
    assert empty_validation.message == "Tool argument command must be at least 1 chars"
    assert shell.input_schema["properties"]["command"] == {
        "type": "string",
        "minLength": 1,
    }


async def test_shell_tool_validates_cwd_exists(tmp_path):
    missing = tmp_path / "missing"
    shell = make_shell(cwd=missing)

    validation = await shell.validate(command="printf hello")

    assert validation.ok is False
    assert validation.message == f"shell cwd does not exist: {missing}"


async def test_shell_tool_validates_per_call_cwd_inside_workspace(tmp_path):
    shell = make_shell(cwd=tmp_path)

    escaped = await shell.validate(command="printf hello", cwd="..")
    missing = await shell.validate(command="printf hello", cwd="missing")
    file_path = tmp_path / "file.txt"
    file_path.write_text("hello", encoding="utf-8")
    file_cwd = await shell.validate(command="printf hello", cwd="file.txt")

    assert escaped.ok is False
    assert escaped.message == "shell cwd failed: path is outside root: .."
    assert missing.ok is False
    assert missing.message == "shell cwd does not exist: missing"
    assert file_cwd.ok is False
    assert file_cwd.message == "shell cwd is not a directory: file.txt"


async def test_shell_tool_validates_positive_timeout(tmp_path):
    shell = make_shell(cwd=tmp_path, timeout_seconds=0)

    validation = await shell.validate(command="printf hello")

    assert validation.ok is False
    assert validation.message.startswith("shell timeout must be > 0")


async def test_shell_tool_reports_timeout(tmp_path):
    shell = make_shell(cwd=tmp_path, timeout_seconds=0.01)

    result = await shell.run(command="sleep 1")

    assert result.returncode == -1
    assert "timed out" in result.stderr


async def test_shell_tool_reports_exit_code_when_command_fails_silently(tmp_path):
    shell = make_shell(cwd=tmp_path)

    result = await shell.run(command="exit 7")
    text = await shell.text(result, command="exit 7")

    assert result.stdout == ""
    assert result.stderr == ""
    assert result.returncode == 7
    assert str(result) == "exit code: 7"
    assert text == "exit code: 7"


async def test_shell_tool_text_includes_exit_code_with_stderr(tmp_path):
    shell = make_shell(cwd=tmp_path)

    result = await shell.run(
        command="python -c \"import sys; sys.stderr.write('bad\\n'); sys.exit(3)\""
    )
    text = await shell.text(result, command="fail")

    assert result.returncode == 3
    assert result.stderr == "bad\n"
    assert str(result) == "bad\n"
    assert text == "bad\nexit code: 3"


async def test_shell_tool_timeout_stops_child_processes(tmp_path):
    shell = make_shell(cwd=tmp_path, timeout_seconds=0.05)

    result = await shell.run(
        command="python -c \"import subprocess, time; subprocess.Popen(['python', '-c', 'import time, pathlib; time.sleep(0.3); pathlib.Path(\\'child.txt\\').write_text(\\'alive\\')']); time.sleep(2)\""
    )
    await asyncio.sleep(0.5)

    assert result.returncode == -1
    assert "timed out" in result.stderr
    assert not (tmp_path / "child.txt").exists()


async def test_shell_tool_cancellation_stops_process_tree(tmp_path):
    shell = make_shell(cwd=tmp_path, timeout_seconds=5)
    task = asyncio.create_task(
        shell.run(
            command="python -c \"import time, pathlib; time.sleep(0.3); pathlib.Path('cancelled.txt').write_text('alive')\""
        )
    )

    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("shell task should be cancellable")

    await asyncio.sleep(0.4)

    assert not (tmp_path / "cancelled.txt").exists()


async def test_shell_tool_validates_commands_before_running(tmp_path):
    shell = make_shell(
        cwd=tmp_path,
        validate=lambda command: "rm is disabled" if command.startswith("rm ") else True,
    )

    denied = await shell.validate(command="rm -rf build")
    allowed = await shell.validate(command="printf ok")

    assert denied.ok is False
    assert denied.message == "rm is disabled"
    assert allowed.ok is True


async def test_shell_tool_limits_output_text(tmp_path):
    shell = make_shell(cwd=tmp_path, max_output=12)

    result = await shell.run(command="printf abcdefghijklmnopqrstuvwxyz")
    text = await shell.text(result, command="printf abcdefghijklmnopqrstuvwxyz")

    assert result.stdout == "abcdefghijkl"
    assert str(result) == "abcdefghijkl"
    assert text == "abcdefghijkl"


async def test_shell_tool_limits_combined_stdout_and_stderr(tmp_path):
    shell = make_shell(cwd=tmp_path, max_output=12)

    result = await shell.run(
        command="python -c \"import sys; sys.stdout.write('abcdefgh'); sys.stderr.write('ijklmnop')\""
    )
    text = await shell.text(result, command="mixed")

    assert result.stdout == "abcdefgh"
    assert result.stderr == "ijkl"
    assert len(result.stdout + result.stderr) == 12
    assert text == "abcdefghijkl"


async def test_shell_tool_timeout_respects_combined_output_limit(tmp_path):
    shell = make_shell(cwd=tmp_path, timeout_seconds=0.01, max_output=30)

    result = await shell.run(
        command="python -c \"import sys, time; sys.stdout.write('abcdefghij'); sys.stdout.flush(); time.sleep(1)\""
    )

    assert result.returncode == -1
    assert len(result.stdout + result.stderr) <= 30
    assert "timed out" in result.stderr or result.stderr.endswith("...[truncated]")


async def test_shell_tool_can_override_and_remove_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIADK_SECRET", "hidden")
    shell = make_shell(
        cwd=tmp_path,
        env={
            "MINIADK_PUBLIC": "visible",
            "MINIADK_SECRET": None,
        },
    )

    result = await shell.run(
        command=(
            "python -c \"import os; "
            "print(os.getenv('MINIADK_PUBLIC', 'missing')); "
            "print(os.getenv('MINIADK_SECRET', 'missing'))\""
        )
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == ["visible", "missing"]


async def test_fetch_url_reads_http_text():
    with _http_server(b"hello docs", content_type="text/plain; charset=utf-8") as url:
        fetch = make_fetch_url()

        result = await fetch.run(url=url)

    assert isinstance(result, FetchResult)
    assert result.status == 200
    assert result.content_type.startswith("text/plain")
    assert result.text == "hello docs"
    assert str(result) == "hello docs"
    assert fetch.is_read_only(url="https://example.test") is True
    assert fetch.is_concurrency_safe(url="https://example.test") is True


async def test_fetch_url_truncates_large_responses():
    with _http_server(b"abcdefghijklmnopqrstuvwxyz") as url:
        fetch = make_fetch_url(max_bytes=5)

        result = await fetch.run(url=url)

    assert result.text == "abcde"
    assert result.truncated is True
    assert str(result) == "HTTP 200\nabcde\n...[truncated]"


async def test_fetch_url_formats_http_failure_status():
    fetch = make_fetch_url()

    result = FetchResult(
        url="https://example.test/missing",
        status=404,
        content_type="text/plain",
        text="not found",
    )

    assert await fetch.text(result, url=result.url) == "HTTP 404\nnot found"


async def test_fetch_url_validates_url_and_allow_rule():
    fetch = make_fetch_url(allow=lambda url: "blocked" if "blocked" in url else True)

    empty = await fetch.validate(url="")
    bad_scheme = await fetch.validate(url="file:///etc/passwd")
    blocked = await fetch.validate(url="https://blocked.example.test")
    allowed = await fetch.validate(url="https://docs.example.test")

    assert empty.ok is False
    assert empty.message == "Tool argument url must be at least 1 chars"
    assert bad_scheme.ok is False
    assert bad_scheme.message == "Tool argument url must match pattern: ^https?://"
    assert blocked.ok is False
    assert blocked.message == "blocked"
    assert allowed.ok is True
    assert fetch.input_schema["properties"]["url"] == {
        "type": "string",
        "minLength": 1,
        "pattern": r"^https?://",
    }


async def test_make_tools_returns_common_workspace_tools(tmp_path):
    tools = make_tools(root=tmp_path)

    assert [tool.name for tool in tools] == [
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
    ]
    assert tools[0].is_read_only(path="a.txt") is True
    assert tools[4].is_destructive(path="a.txt", content="x") is True
    assert tools[7].is_destructive(path="a.txt") is True
    assert tools[8].is_destructive(src="a.txt", dst="b.txt") is True
    assert tools[9].is_destructive(src="a.txt", dst="b.txt") is True
    assert tools[-1].is_destructive(command="printf ok") is True


async def test_make_tools_limits_read_file_by_default(tmp_path):
    (tmp_path / "long.txt").write_text("x" * 21000, encoding="utf-8")
    read_file = make_tools(root=tmp_path, write=False, shell=False)[0]

    result = await read_file.run(path="long.txt")

    assert len(result) == 20000
    assert result.endswith("\n...[truncated]")


async def test_make_tools_can_disable_read_limit(tmp_path):
    (tmp_path / "long.txt").write_text("x" * 21000, encoding="utf-8")
    read_file = make_tools(root=tmp_path, write=False, shell=False, max_read=None)[0]

    result = await read_file.run(path="long.txt")

    assert len(result) == 21000


async def test_make_tools_limits_search_file_size_by_default(tmp_path):
    (tmp_path / "huge.txt").write_text("needle\n" * 200000, encoding="utf-8")
    search_tool = make_tools(root=tmp_path, write=False, shell=False)[3]

    result = await search_tool.run(pattern="needle")

    assert result == []


async def test_make_tools_limits_search_scanned_files_by_default(tmp_path):
    for index in range(1001):
        (tmp_path / f"{index:04}.txt").write_text("miss\n", encoding="utf-8")
    (tmp_path / "zzzz.txt").write_text("needle\n", encoding="utf-8")
    search_tool = make_tools(root=tmp_path, write=False, shell=False)[3]

    result = await search_tool.run(pattern="needle")

    assert result == []


async def test_make_tools_limits_list_scanned_files_by_default(tmp_path):
    for index in range(5001):
        (tmp_path / f"{index:04}.txt").write_text("", encoding="utf-8")
    (tmp_path / "zzzz.py").write_text("", encoding="utf-8")
    list_tool = make_tools(root=tmp_path, write=False, shell=False)[1]
    glob_tool = make_tools(root=tmp_path, write=False, shell=False)[2]

    assert await list_tool.run(pattern="*.py") == []
    assert await glob_tool.run(pattern="*.py") == []


async def test_make_tools_ignores_common_large_directories_by_default(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.py").write_text("needle\n", encoding="utf-8")
    tools = make_tools(root=tmp_path, write=False, shell=False)
    list_tool = tools[1]
    search_tool = tools[3]

    assert await list_tool.run(pattern="*.py") == ["src/app.py"]
    assert await search_tool.run(pattern="needle") == ["src/app.py:1: needle"]


async def test_make_tools_can_disable_default_ignore(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.py").write_text("needle\n", encoding="utf-8")
    tools = make_tools(root=tmp_path, write=False, shell=False, ignore=None)

    assert await tools[1].run(pattern="*.py") == ["node_modules/lib.py"]
    assert await tools[3].run(pattern="needle") == [
        "node_modules/lib.py:1: needle"
    ]


async def test_make_tools_can_disable_search_file_limit(tmp_path):
    (tmp_path / "huge.txt").write_text("needle\n" * 200000, encoding="utf-8")
    search_tool = make_tools(
        root=tmp_path,
        write=False,
        shell=False,
        max_search_file=None,
        search_limit=1,
    )[3]

    result = await search_tool.run(pattern="needle")

    assert result == ["huge.txt:1: needle"]


async def test_make_tools_can_disable_search_file_count_limit(tmp_path):
    for index in range(5):
        (tmp_path / f"{index}.txt").write_text("miss\n", encoding="utf-8")
    (tmp_path / "z.txt").write_text("needle\n", encoding="utf-8")
    search_tool = make_tools(
        root=tmp_path,
        write=False,
        shell=False,
        max_search_files=None,
        search_limit=1,
    )[3]

    result = await search_tool.run(pattern="needle")

    assert result == ["z.txt:1: needle"]


async def test_make_tools_can_disable_list_file_count_limit(tmp_path):
    for index in range(5):
        (tmp_path / f"{index}.txt").write_text("", encoding="utf-8")
    (tmp_path / "z.py").write_text("", encoding="utf-8")
    tools = make_tools(
        root=tmp_path,
        write=False,
        shell=False,
        max_list_files=None,
    )

    assert await tools[1].run(pattern="*.py") == ["z.py"]
    assert await tools[2].run(pattern="*.py") == ["z.py"]


async def test_make_tools_can_create_read_only_file_kit(tmp_path):
    tools = make_tools(root=tmp_path, write=False, shell=False)

    assert [tool.name for tool in tools] == [
        "read_file",
        "list_workspace_files",
        "glob_workspace_files",
        "search_workspace_text",
    ]
    assert all(tool.is_read_only(path="a.txt") for tool in tools[:1])


async def test_make_tools_can_include_web_fetch_tool(tmp_path):
    tools = make_tools(root=tmp_path, files=False, shell=False, web=True)

    assert [tool.name for tool in tools] == ["fetch_url"]


async def test_make_tools_passes_shell_options(tmp_path):
    tools = make_tools(
        root=tmp_path,
        files=False,
        validate_shell=lambda command: "blocked" if command == "bad" else True,
        read_shell="printf *",
        max_shell=4,
    )
    shell = tools[0]

    denied = await shell.validate(command="bad")
    result = await shell.run(command="printf abcdef")

    assert [tool.name for tool in tools] == ["shell"]
    assert denied.ok is False
    assert denied.message == "blocked"
    assert result.stdout == "abcd"
    assert shell.is_read_only(command="printf abcdef") is True
    assert shell.is_destructive(command="python build.py") is True


async def test_make_tools_shell_validation_ignores_stdin_option(tmp_path):
    tools = make_tools(
        root=tmp_path,
        files=False,
        validate_shell=lambda command: command.startswith("python"),
    )
    shell = tools[0]

    validation = await shell.validate(
        command="python -c \"import sys; print(sys.stdin.read())\"",
        input="hello",
    )

    assert validation.ok is True


async def test_make_tools_passes_shell_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIADK_SECRET", "hidden")
    tools = make_tools(
        root=tmp_path,
        files=False,
        shell_env={"MINIADK_SECRET": None},
    )
    shell = tools[0]

    result = await shell.run(
        command="python -c \"import os; print(os.getenv('MINIADK_SECRET', 'missing'))\""
    )

    assert result.stdout.strip() == "missing"


async def test_spawn_tool_runs_named_agent_and_returns_answer():
    tester = Agent(name="tester", instructions="Test code.")
    model = ScriptedModel([ModelResult(message="tests pass")])
    spawn = make_spawn([tester], model=model)

    result = await spawn.run(agent="tester", prompt="run tests")

    assert isinstance(result, SpawnResult)
    assert str(result) == "tests pass"
    assert result.agent == "tester"
    assert result.answer == "tests pass"
    assert await spawn.text(result, agent="tester", prompt="run tests") == (
        "tester: tests pass"
    )
    assert result.session is None
    assert model.calls[0][0] == [
        Message("system", "Test code."),
        Message("user", "run tests"),
    ]
    assert spawn.name == "spawn_agent"
    assert "tester" in spawn.description
    assert spawn.input_schema["properties"]["agent"] == {
        "type": "string",
        "description": "Agent name to run.",
        "enum": ["tester"],
    }
    assert spawn.input_schema["properties"]["prompt"] == {
        "type": "string",
        "description": "Task prompt for the agent.",
        "minLength": 1,
    }


async def test_spawn_tool_can_use_default_model_helper(monkeypatch):
    tester = Agent(name="tester", instructions="Test code.")
    model = ScriptedModel([ModelResult(message="tests pass")])
    monkeypatch.setattr(agent_tools, "build_model", lambda: model)
    spawn = make_spawn([tester])

    result = await spawn.run(agent="tester", prompt="run tests")

    assert result.answer == "tests pass"
    assert model.calls[0][0] == [
        Message("system", "Test code."),
        Message("user", "run tests"),
    ]


async def test_spawn_tool_can_keep_child_session():
    reviewer = Agent(name="reviewer", instructions="Review code.")
    model = ScriptedModel([ModelResult(message="looks good")])
    spawn = make_spawn({"reviewer": reviewer}, model=model, keep_session=True)

    result = await spawn.run(agent="reviewer", prompt="review app.py")

    assert result.session is not None
    assert [message.content for message in result.session.messages] == [
        "Review code.",
        "review app.py",
        "looks good",
    ]


async def test_spawn_tool_uses_agent_specific_model_and_extra_tools():
    @tool
    def check(path: str) -> str:
        """Check a path."""
        return f"checked {path}"

    tester = Agent(name="tester", instructions="Test code.")
    tester_model = ScriptedModel(
        [
            ModelResult(tool_calls=[ToolCall(name="check", arguments={"path": "app.py"})]),
            ModelResult(message="done"),
        ]
    )
    unused_model = ScriptedModel([ModelResult(message="unused")])
    spawn = make_spawn(
        [tester],
        model=unused_model,
        models={"tester": tester_model},
        tools={"tester": [check]},
    )

    result = await spawn.run(agent="tester", prompt="test app")

    assert result.answer == "done"
    assert tester_model.calls[0][1] == [check]
    assert unused_model.calls == []


async def test_spawn_tool_resolves_child_agent_skills():
    helper = Agent(
        name="helper",
        instructions="Help.",
        skills=SkillRegistry(
            [
                Skill(
                    name="review",
                    description="Review code.",
                    body="Review $ARGUMENTS",
                )
            ]
        ),
    )
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[
                    ToolCall(
                        name="skill",
                        arguments={"skill": "review", "args": "app.py"},
                    )
                ]
            ),
            ModelResult(message="done"),
        ]
    )
    spawn = make_spawn([helper], model=model, keep_session=True)

    result = await spawn.run(agent="helper", prompt="review")

    assert result.answer == "done"
    assert model.calls[0][1][0].name == "skill"
    assert result.session is not None
    assert "Available skills:" in result.session.messages[0].content
    assert result.session.messages[3].content == "Review app.py"


async def test_spawn_tool_merges_resolved_child_tools_with_extra_tools():
    @tool
    def check(path: str) -> str:
        """Check a path."""
        return f"checked {path}"

    helper = Agent(
        name="helper",
        instructions="Help.",
        skills=SkillRegistry(
            [
                Skill(
                    name="review",
                    description="Review code.",
                    body="Review $ARGUMENTS",
                )
            ]
        ),
    )
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[
                    ToolCall(
                        name="skill",
                        arguments={"skill": "review", "args": "app.py"},
                    )
                ]
            ),
            ModelResult(tool_calls=[ToolCall(name="check", arguments={"path": "app.py"})]),
            ModelResult(message="done"),
        ]
    )
    spawn = make_spawn(
        [helper],
        model=model,
        tools={"helper": [check]},
        keep_session=True,
    )

    result = await spawn.run(agent="helper", prompt="review")

    assert result.answer == "done"
    assert [tool.name for tool in model.calls[0][1]] == ["skill", "check"]
    assert result.session is not None
    assert result.session.messages[3].content == "Review app.py"
    assert result.session.messages[5].content == "checked app.py"


async def test_spawn_tool_extra_tools_override_child_tools_by_name():
    @tool
    def child_check(path: str) -> str:
        """Check a path."""
        return f"child:{path}"

    @tool
    def extra_check(path: str) -> str:
        """Check a path."""
        return f"extra:{path}"

    child_check.name = "check"
    extra_check.name = "check"

    helper = Agent(name="helper", instructions="Help.", tools=[child_check])
    model = ScriptedModel(
        [
            ModelResult(tool_calls=[ToolCall(name="check", arguments={"path": "app.py"})]),
            ModelResult(message="done"),
        ]
    )
    spawn = make_spawn(
        [helper],
        model=model,
        tools={"helper": [extra_check]},
        keep_session=True,
    )

    result = await spawn.run(agent="helper", prompt="check")

    assert result.answer == "done"
    assert [tool.name for tool in model.calls[0][1]] == ["check"]
    assert result.session is not None
    assert result.session.messages[3].content == "extra:app.py"


async def test_spawn_tool_passes_child_middleware():
    class DenyCheck:
        def __init__(self):
            self.calls = 0

        async def before_tool_call(self, tool, arguments):
            self.calls += 1
            return PermissionDecision("deny", "child blocked")

    @tool
    def check(path: str) -> str:
        """Check a path."""
        return f"checked {path}"

    helper = Agent(name="helper", instructions="Help.")
    model = ScriptedModel(
        [
            ModelResult(tool_calls=[ToolCall(name="check", arguments={"path": "app.py"})]),
            ModelResult(message="unused"),
        ]
    )
    middleware = DenyCheck()
    spawn = make_spawn(
        [helper],
        model=model,
        tools={"helper": [check]},
        middleware=[middleware],
        keep_session=True,
    )

    result = await spawn.run(agent="helper", prompt="check app")

    assert result.answer == ""
    assert result.session is not None
    assert [message.content for message in result.session.messages] == [
        "Help.",
        "check app",
        "",
        "child blocked",
    ]
    assert len(result.session.messages[2].tool_calls) == 1
    assert middleware.calls == 1


async def test_spawn_tool_can_bind_guard_to_permission_callback():
    prompts = []

    @tool(destructive=True)
    def write_file(path: str) -> str:
        """Write a file."""
        return f"wrote {path}"

    helper = Agent(name="helper", instructions="Help.")
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[ToolCall(name="write_file", arguments={"path": "app.py"})]
            ),
            ModelResult(message="done"),
        ]
    )
    spawn = make_spawn(
        [helper],
        model=model,
        tools={"helper": [write_file]},
        middleware=[Guard("ask")],
        ask_user=lambda request: prompts.append(
            (request.tool.name, request.arguments["path"], request.reason)
        )
        or True,
    )

    result = await spawn.run(agent="helper", prompt="write")

    assert result.answer == "done"
    assert prompts == [("write_file", "app.py", "destructive tool use")]


async def test_spawn_tool_does_not_mutate_shared_guard_when_binding_permission_callback():
    prompts = []

    @tool(destructive=True)
    def write_file(path: str) -> str:
        """Write a file."""
        return f"wrote {path}"

    helper = Agent(name="helper", instructions="Help.")
    guard = Guard("ask")

    async def run_once(label: str):
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
        spawn = make_spawn(
            [helper],
            model=model,
            tools={"helper": [write_file]},
            middleware=[guard],
            ask_user=lambda request: prompts.append(
                (label, request.arguments["path"])
            )
            or True,
        )
        return await spawn.run(agent="helper", prompt="write")

    assert (await run_once("first")).answer == "done"
    assert (await run_once("second")).answer == "done"
    assert guard.ask_user is None
    assert prompts == [("first", "first.py"), ("second", "second.py")]


async def test_spawn_tool_passes_child_policy():
    class StopPolicy:
        async def after_model(self, state):
            return RunDecision.stop("policy answer", reason="policy_stop")

        async def after_tools(self, state):
            return RunDecision()

    helper = Agent(name="helper", instructions="Help.")
    spawn = make_spawn(
        [helper],
        model=ScriptedModel([ModelResult(message="raw answer")]),
        policy=StopPolicy(),
        keep_session=True,
    )

    result = await spawn.run(agent="helper", prompt="answer")

    assert result.answer == "policy answer"
    assert result.session is not None
    assert [message.content for message in result.session.messages] == [
        "Help.",
        "answer",
        "raw answer",
    ]


async def test_spawn_tool_validates_agent_and_prompt_before_running():
    tester = Agent(name="tester", instructions="Test code.")
    model = ScriptedModel([ModelResult(message="should not run")])
    spawn = make_spawn([tester], model=model)

    missing = await spawn.validate(agent="missing", prompt="run")
    empty_prompt = await spawn.validate(agent="tester", prompt=" ")
    empty_schema_prompt = await spawn.validate(agent="tester", prompt="")

    assert missing.ok is False
    assert missing.message == "Tool argument agent must be one of: tester"
    assert empty_prompt.ok is False
    assert empty_prompt.message == "prompt is required"
    assert empty_schema_prompt.ok is False
    assert empty_schema_prompt.message == (
        "Tool argument prompt must be at least 1 chars"
    )
    assert model.calls == []


def test_spawn_tool_rejects_duplicate_agent_names_from_list():
    first = Agent(name="helper", instructions="First.")
    second = Agent(name="helper", instructions="Second.")

    with pytest.raises(ValueError, match="duplicate agent name: helper"):
        make_spawn([first, second])


async def test_spawn_builder_can_customize_public_tool_name():
    tester = Agent(name="tester", instructions="Test code.")
    model = ScriptedModel([ModelResult(message="ok")])
    spawn = Spawn(agents={"tester": tester}, model=model)(
        name="ask_agent",
        description="Ask a helper.",
    )

    assert spawn.name == "ask_agent"
    assert spawn.description == "Ask a helper."


async def test_work_tools_start_and_read_background_agent_work():
    helper = Agent(name="helper", instructions="Help.")
    model = ScriptedModel([ModelResult(message="finished")])
    start_work, list_work, read_work, cancel_work = make_work(
        [helper],
        model=model,
        keep_session=True,
    )

    started = await start_work.run(agent="helper", prompt="do it")

    assert started.agent == "helper"
    assert started.status == "running"
    assert await start_work.text(started, agent="helper", prompt="do it") == (
        f"{started.id} [running] helper"
    )
    result = await read_work.run(id=started.id, wait=0.2)

    assert result.status == "done"
    assert result.answer == "finished"
    assert await read_work.text(result, id=started.id) == (
        f"{started.id} [done] helper: answer ready\nfinished"
    )
    assert result.session is not None
    assert [message.content for message in result.session.messages] == [
        "Help.",
        "do it",
        "finished",
    ]
    assert read_work.is_read_only(id=started.id) is True
    assert read_work.is_concurrency_safe(id=started.id) is True
    assert cancel_work.is_destructive(id=started.id) is True
    assert start_work.input_schema["properties"]["agent"] == {
        "type": "string",
        "description": "Agent name to run.",
        "enum": ["helper"],
    }
    assert start_work.input_schema["properties"]["prompt"] == {
        "type": "string",
        "description": "Task prompt for the agent.",
        "minLength": 1,
    }
    assert read_work.input_schema["properties"]["id"] == {
        "type": "string",
        "description": "Background work id.",
        "minLength": 1,
    }
    assert read_work.input_schema["properties"]["wait"] == {
        "type": "number",
        "minimum": 0,
        "default": 0,
    }
    assert cancel_work.input_schema["properties"]["id"] == {
        "type": "string",
        "description": "Background work id.",
        "minLength": 1,
    }
    assert list_work.name == "list_work"
    assert cancel_work.name == "cancel_work"


async def test_work_tools_can_list_background_agent_work():
    helper = Agent(name="helper", instructions="Help.")
    model = ScriptedModel(
        [
            ModelResult(message="first"),
            ModelResult(message="second"),
        ]
    )
    start_work, list_work, read_work, _ = make_work([helper], model=model)

    first = await start_work.run(agent="helper", prompt="first task")
    second = await start_work.run(agent="helper", prompt="second task")
    running = await list_work.run()
    running_text = await list_work.text(running)
    await asyncio.sleep(0)
    done = await list_work.run()
    done_text = await list_work.text(done)

    assert [item.id for item in running] == sorted([first.id, second.id])
    assert {item.status for item in running} == {"running"}
    assert running_text == "\n".join(
        f"{item.id} [running] helper" for item in running
    )
    by_id = {item.id: item for item in done}
    assert (by_id[first.id].agent, by_id[first.id].status, by_id[first.id].answer) == (
        "helper",
        "done",
        "first",
    )
    assert (by_id[second.id].agent, by_id[second.id].status, by_id[second.id].answer) == (
        "helper",
        "done",
        "second",
    )
    assert done_text == "\n".join(
        f"{item.id} [done] helper: answer ready" for item in done
    )
    assert (await read_work.run(id=first.id)).answer == "first"
    assert list_work.is_read_only() is True
    assert list_work.is_concurrency_safe() is True


async def test_work_read_can_wait_for_background_result():
    class SlowModel:
        async def complete(self, messages, tools):
            await asyncio.sleep(0.01)
            return ModelResult(message="ready")

    helper = Agent(name="helper", instructions="Help.")
    start_work, _, read_work, _ = make_work([helper], model=SlowModel())

    started = await start_work.run(agent="helper", prompt="do it")
    result = await read_work.run(id=started.id, wait=0.2)

    assert result.status == "done"
    assert result.answer == "ready"


async def test_work_read_wait_timeout_leaves_background_work_running():
    class SlowModel:
        def __init__(self):
            self.cancelled = False

        async def complete(self, messages, tools):
            try:
                await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            return ModelResult(message="late")

    helper = Agent(name="helper", instructions="Help.")
    model = SlowModel()
    start_work, _, read_work, cancel_work = make_work([helper], model=model)

    started = await start_work.run(agent="helper", prompt="do it")
    running = await read_work.run(id=started.id, wait=0.001)

    assert running.status == "running"
    assert model.cancelled is False
    await cancel_work.run(id=started.id)


async def test_work_list_tool_formats_empty_work_clearly():
    helper = Agent(name="helper", instructions="Help.")
    _, list_work, _, _ = make_work([helper], model=ScriptedModel([]))

    result = await list_work.run()
    text = await list_work.text(result)

    assert result == []
    assert text == "no background work"


async def test_work_tools_can_use_default_model_helper(monkeypatch):
    helper = Agent(name="helper", instructions="Help.")
    model = ScriptedModel([ModelResult(message="ok")])
    monkeypatch.setattr(agent_tools, "build_model", lambda: model)
    start_work, _, read_work, _ = make_work([helper])

    started = await start_work.run(agent="helper", prompt="do it")
    await asyncio.sleep(0)
    result = await read_work.run(id=started.id)

    assert result.status == "done"
    assert result.answer == "ok"
    assert model.calls[0][0] == [
        Message("system", "Help."),
        Message("user", "do it"),
    ]


async def test_work_tools_pass_agent_specific_middleware():
    class DenyCheck:
        def __init__(self):
            self.calls = 0

        async def before_tool_call(self, tool, arguments):
            self.calls += 1
            return PermissionDecision("deny", "child blocked")

    @tool
    def check(path: str) -> str:
        """Check a path."""
        return f"checked {path}"

    helper = Agent(name="helper", instructions="Help.")
    model = ScriptedModel(
        [
            ModelResult(tool_calls=[ToolCall(name="check", arguments={"path": "app.py"})]),
            ModelResult(message="unused"),
        ]
    )
    middleware = DenyCheck()
    start_work, _, read_work, _ = make_work(
        [helper],
        model=model,
        tools={"helper": [check]},
        middlewares={"helper": [middleware]},
        keep_session=True,
    )

    started = await start_work.run(agent="helper", prompt="check app")
    await asyncio.sleep(0)
    result = await read_work.run(id=started.id)

    assert result.answer == ""
    assert result.session is not None
    assert [message.content for message in result.session.messages] == [
        "Help.",
        "check app",
        "",
        "child blocked",
    ]
    assert len(result.session.messages[2].tool_calls) == 1
    assert middleware.calls == 1


async def test_work_tools_resolve_child_agent_skills():
    helper = Agent(
        name="helper",
        instructions="Help.",
        skills=SkillRegistry(
            [
                Skill(
                    name="review",
                    description="Review code.",
                    body="Review $ARGUMENTS",
                )
            ]
        ),
    )
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[
                    ToolCall(
                        name="skill",
                        arguments={"skill": "review", "args": "app.py"},
                    )
                ]
            ),
            ModelResult(message="done"),
        ]
    )
    start_work, _, read_work, _ = make_work(
        [helper],
        model=model,
        keep_session=True,
    )

    started = await start_work.run(agent="helper", prompt="review")
    result = await read_work.run(id=started.id, wait=0.2)

    assert result.answer == "done"
    assert model.calls[0][1][0].name == "skill"
    assert result.session is not None
    assert "Available skills:" in result.session.messages[0].content
    assert result.session.messages[3].content == "Review app.py"


async def test_work_tools_merge_resolved_child_tools_with_extra_tools():
    @tool
    def check(path: str) -> str:
        """Check a path."""
        return f"checked {path}"

    helper = Agent(
        name="helper",
        instructions="Help.",
        skills=SkillRegistry(
            [
                Skill(
                    name="review",
                    description="Review code.",
                    body="Review $ARGUMENTS",
                )
            ]
        ),
    )
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[
                    ToolCall(
                        name="skill",
                        arguments={"skill": "review", "args": "app.py"},
                    )
                ]
            ),
            ModelResult(tool_calls=[ToolCall(name="check", arguments={"path": "app.py"})]),
            ModelResult(message="done"),
        ]
    )
    start_work, _, read_work, _ = make_work(
        [helper],
        model=model,
        tools={"helper": [check]},
        keep_session=True,
    )

    started = await start_work.run(agent="helper", prompt="review")
    result = await read_work.run(id=started.id, wait=0.2)

    assert result.answer == "done"
    assert [tool.name for tool in model.calls[0][1]] == ["skill", "check"]
    assert result.session is not None
    assert result.session.messages[3].content == "Review app.py"
    assert result.session.messages[5].content == "checked app.py"


async def test_work_tools_extra_tools_override_child_tools_by_name():
    @tool
    def child_check(path: str) -> str:
        """Check a path."""
        return f"child:{path}"

    @tool
    def extra_check(path: str) -> str:
        """Check a path."""
        return f"extra:{path}"

    child_check.name = "check"
    extra_check.name = "check"

    helper = Agent(name="helper", instructions="Help.", tools=[child_check])
    model = ScriptedModel(
        [
            ModelResult(tool_calls=[ToolCall(name="check", arguments={"path": "app.py"})]),
            ModelResult(message="done"),
        ]
    )
    start_work, _, read_work, _ = make_work(
        [helper],
        model=model,
        tools={"helper": [extra_check]},
        keep_session=True,
    )

    started = await start_work.run(agent="helper", prompt="check")
    result = await read_work.run(id=started.id, wait=0.2)

    assert result.answer == "done"
    assert [tool.name for tool in model.calls[0][1]] == ["check"]
    assert result.session is not None
    assert result.session.messages[3].content == "extra:app.py"


async def test_work_tools_can_bind_guard_to_permission_callback():
    prompts = []

    @tool(destructive=True)
    def write_file(path: str) -> str:
        """Write a file."""
        return f"wrote {path}"

    helper = Agent(name="helper", instructions="Help.")
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[ToolCall(name="write_file", arguments={"path": "app.py"})]
            ),
            ModelResult(message="done"),
        ]
    )
    start_work, _, read_work, _ = make_work(
        [helper],
        model=model,
        tools={"helper": [write_file]},
        middleware=[Guard("ask")],
        ask_user=lambda request: prompts.append(
            (request.tool.name, request.arguments["path"], request.reason)
        )
        or True,
    )

    started = await start_work.run(agent="helper", prompt="write")
    result = await read_work.run(id=started.id, wait=0.2)

    assert result.answer == "done"
    assert prompts == [("write_file", "app.py", "destructive tool use")]


async def test_work_tools_can_deny_permission_callback():
    prompts = []

    @tool(destructive=True)
    def write_file(path: str) -> str:
        """Write a file."""
        return f"wrote {path}"

    helper = Agent(name="helper", instructions="Help.")
    model = ScriptedModel(
        [
            ModelResult(
                tool_calls=[ToolCall(name="write_file", arguments={"path": "app.py"})]
            ),
            ModelResult(message="unused"),
        ]
    )
    start_work, _, read_work, _ = make_work(
        [helper],
        model=model,
        tools={"helper": [write_file]},
        middleware=[Guard("ask")],
        ask_user=lambda request: prompts.append(request.arguments["path"]) or False,
        keep_session=True,
    )

    started = await start_work.run(agent="helper", prompt="write")
    await asyncio.sleep(0)
    result = await read_work.run(id=started.id)

    assert result.answer == ""
    assert result.session is not None
    assert result.session.messages[-1].content == (
        "Permission denied for write_file: destructive tool use"
    )
    assert prompts == ["app.py"]


async def test_work_tools_pass_default_child_policy():
    class StopPolicy:
        async def after_model(self, state):
            return RunDecision.stop("work policy answer", reason="policy_stop")

        async def after_tools(self, state):
            return RunDecision()

    helper = Agent(name="helper", instructions="Help.")
    start_work, _, read_work, _ = make_work(
        [helper],
        model=ScriptedModel([ModelResult(message="raw answer")]),
        policy=StopPolicy(),
        keep_session=True,
    )

    started = await start_work.run(agent="helper", prompt="answer")
    await asyncio.sleep(0)
    result = await read_work.run(id=started.id)

    assert result.answer == "work policy answer"
    assert result.session is not None
    assert [message.content for message in result.session.messages] == [
        "Help.",
        "answer",
        "raw answer",
    ]


async def test_work_tools_can_cancel_background_work():
    class SlowModel:
        def __init__(self):
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def complete(self, messages, tools):
            self.started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            return ModelResult(message="late")

    helper = Agent(name="helper", instructions="Help.")
    model = SlowModel()
    start_work, _, read_work, cancel_work = make_work([helper], model=model)

    started = await start_work.run(agent="helper", prompt="wait")
    await asyncio.wait_for(model.started.wait(), timeout=0.2)
    cancelled = await cancel_work.run(id=started.id)
    result = await read_work.run(id=started.id)

    assert cancelled.status == "cancelled"
    assert await cancel_work.text(cancelled, id=started.id) == (
        f"{started.id} [cancelled] helper"
    )
    assert result.status == "cancelled"
    assert model.cancelled.is_set()


async def test_work_cancel_returns_finished_result_when_already_done():
    helper = Agent(name="helper", instructions="Help.")
    start_work, _, _, cancel_work = make_work(
        [helper],
        model=ScriptedModel([ModelResult(message="done")]),
    )

    started = await start_work.run(agent="helper", prompt="finish")
    await asyncio.sleep(0)
    result = await cancel_work.run(id=started.id)

    assert result.status == "done"
    assert result.answer == "done"


async def test_work_tools_validate_agent_prompt_and_work_id():
    helper = Agent(name="helper", instructions="Help.")
    start_work, _, read_work, cancel_work = make_work(
        [helper],
        model=ScriptedModel([ModelResult(message="unused")]),
    )

    missing_agent = await start_work.validate(agent="missing", prompt="do it")
    empty_prompt = await start_work.validate(agent="helper", prompt=" ")
    empty_schema_prompt = await start_work.validate(agent="helper", prompt="")
    started = await start_work.run(agent="helper", prompt="do it")
    missing_read = await read_work.validate(id="missing")
    empty_read = await read_work.validate(id="")
    invalid_wait = await read_work.validate(id=started.id, wait=-1)
    missing_cancel = await cancel_work.validate(id="missing")
    empty_cancel = await cancel_work.validate(id="")

    assert missing_agent.ok is False
    assert missing_agent.message == "Tool argument agent must be one of: helper"
    assert empty_prompt.ok is False
    assert empty_prompt.message == "prompt is required"
    assert empty_schema_prompt.ok is False
    assert empty_schema_prompt.message == (
        "Tool argument prompt must be at least 1 chars"
    )
    assert missing_read.ok is False
    assert missing_read.message == "Unknown work id: missing"
    assert empty_read.ok is False
    assert empty_read.message == "Tool argument id must be at least 1 chars"
    assert invalid_wait.ok is False
    assert invalid_wait.message == "Tool argument wait must be >= 0"
    assert missing_cancel.ok is False
    assert missing_cancel.message == "Unknown work id: missing"
    assert empty_cancel.ok is False
    assert empty_cancel.message == "Tool argument id must be at least 1 chars"


def test_work_tools_reject_duplicate_agent_names_from_list():
    first = Agent(name="helper", instructions="First.")
    second = Agent(name="helper", instructions="Second.")

    with pytest.raises(ValueError, match="duplicate agent name: helper"):
        make_work([first, second])


async def test_work_builder_can_customize_tool_names():
    helper = Agent(name="helper", instructions="Help.")
    work = Work(agents={"helper": helper}, model=ScriptedModel([]))

    start = work.start_tool(name="begin_work", description="Begin.")
    list_ = work.list_tool(name="list_jobs", description="List.")
    read = work.read_tool(name="poll_work", description="Poll.")
    cancel = work.cancel_tool(name="stop_work", description="Stop.")

    assert start.name == "begin_work"
    assert start.description == "Begin."
    assert list_.name == "list_jobs"
    assert list_.description == "List."
    assert read.name == "poll_work"
    assert cancel.name == "stop_work"


def test_spawn_tool_is_available_from_top_level_package():
    tester = Agent(name="tester", instructions="Test code.")

    assert PublicSpawnResult(agent="tester", answer="ok").answer == "ok"
    assert PublicSpawn(agents={"tester": tester}, model=ScriptedModel([]))
    assert public_make_spawn([tester], model=ScriptedModel([])).name == "spawn_agent"
    assert PublicWorkResult(id="work_1", agent="tester", status="running").status == "running"
    assert PublicWork(agents={"tester": tester}, model=ScriptedModel([]))
    assert [tool.name for tool in public_make_work([tester], model=ScriptedModel([]))] == [
        "start_work",
        "list_work",
        "read_work",
        "cancel_work",
    ]
    assert public_make_tools(shell=False)


class _HTTPServer:
    def __init__(self, body: bytes, *, content_type: str = "text/plain"):
        self.body = body
        self.content_type = content_type
        self.server = None
        self.thread = None

    def __enter__(self):
        body = self.body
        content_type = self.content_type

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        return f"http://{host}:{port}/docs"

    def __exit__(self, exc_type, exc, tb):
        self.server.shutdown()
        self.server.server_close()


def _http_server(body: bytes, *, content_type: str = "text/plain") -> _HTTPServer:
    return _HTTPServer(body, content_type=content_type)
