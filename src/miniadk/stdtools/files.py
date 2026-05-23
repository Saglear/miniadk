from __future__ import annotations

import os
import re
import shutil
import tempfile
from fnmatch import fnmatch
from pathlib import Path

from ..core.middleware import ask_before
from ..core.tools import tool


def _resolve_inside(root: Path, path: str) -> Path:
    root = root.resolve()
    target = (root / path).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path is outside root: {path}")
    return target


def _resolve_inside_no_follow(root: Path, path: str) -> Path:
    root = root.resolve()
    target = root / path
    parent = target.parent.resolve()
    candidate = parent / target.name
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path is outside root: {path}")
    return candidate


async def list_files(
    *,
    root: str | Path = ".",
    pattern: str = "*",
    limit: int = 200,
    max_files: int | None = None,
    ignore: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[str]:
    root_path = Path(root).resolve()
    paths = []
    scanned = 0
    for path, relative in _walk_files(root_path, ignore):
        scanned += 1
        if max_files is not None and scanned > max_files:
            break
        if not _matches(relative, pattern):
            continue
        if path.is_file():
            paths.append(relative.as_posix())
        if len(paths) >= limit:
            break
    return sorted(paths)


async def glob_files(
    *,
    root: str | Path = ".",
    pattern: str = "*",
    limit: int = 200,
    max_files: int | None = None,
    ignore: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[str]:
    return await list_files(
        root=root,
        pattern=pattern,
        limit=limit,
        max_files=max_files,
        ignore=ignore,
    )


async def search_text(
    *,
    root: str | Path = ".",
    pattern: str,
    glob: str = "*",
    limit: int = 20,
    context: int = 0,
    max_file: int | None = None,
    max_files: int | None = None,
    ignore: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[str]:
    root_path = Path(root).resolve()
    if context < 0:
        raise ValueError("context must be >= 0")
    regex = re.compile(pattern)
    matches: list[str] = []
    matched = 0
    scanned = 0
    for path, relative in _walk_files(root_path, ignore):
        if not _matches(relative, glob):
            continue
        if not path.is_file():
            continue
        scanned += 1
        if max_files is not None and scanned > max_files:
            return matches
        if max_file is not None:
            try:
                if path.stat().st_size > max_file:
                    continue
            except OSError:
                continue
        if _is_binary(path):
            continue
        try:
            if context:
                context_matches, count = _search_file_with_context(
                    path,
                    relative,
                    regex,
                    context=context,
                    remaining=limit - matched,
                )
                for match in context_matches:
                    matches.append(match)
                matched += count
                if matched >= limit:
                    return matches
                continue
            with path.open("r", encoding="utf-8") as handle:
                for index, line in enumerate(handle, start=1):
                    line = line.rstrip("\n")
                    if regex.search(line):
                        matches.append(f"{relative.as_posix()}:{index}: {line}")
                        if len(matches) >= limit:
                            return matches
        except Exception:  # noqa: BLE001 - search skips unreadable files
            continue
    return matches


def make_read_file(*, root: str | Path = ".", max_text: int | None = None):
    root_path = Path(root)

    @tool(
        read_only=True,
        concurrency_safe=True,
        max_text=max_text,
        format=_format_read_file,
        schema={
            "offset": {"type": "integer", "minimum": 1, "default": 1},
            "limit": {"type": "integer", "minimum": 1},
        },
    )
    def read_file(
        path: str,
        offset: int = 1,
        limit: int | None = None,
        numbers: bool = False,
    ) -> str:
        """Read a UTF-8 text file inside the workspace."""
        try:
            target = _resolve_inside(root_path, path)
            if offset < 1:
                return "read_file failed: offset must be >= 1"
            if limit is not None and limit < 1:
                return "read_file failed: limit must be >= 1"
            if _is_binary(target):
                return f"read_file failed: binary files are not supported: {path}"
            if offset == 1 and limit is None:
                content = target.read_text(encoding="utf-8")
                if numbers:
                    content = _number_lines(content, start=1)
                return _clip(content, max_text)
            return _clip(
                _read_lines(target, offset=offset, limit=limit, numbers=numbers),
                max_text,
            )
        except UnicodeDecodeError:
            return f"read_file failed: file is not valid UTF-8 text: {path}"
        except Exception as error:  # noqa: BLE001 - tools report readable failures
            return f"read_file failed: {error}"

    return read_file


def make_write_file(*, root: str | Path = "."):
    root_path = Path(root)

    def validate_write_file(path: str, content: str, dry: bool = False) -> bool | str:
        try:
            _resolve_inside(root_path, path)
        except Exception as error:  # noqa: BLE001 - validation returns user-visible text
            return f"write_file failed: {error}"
        return True

    @tool(
        permission=ask_before("writing files"),
        read_only=lambda path, content, dry=False: dry,
        destructive=lambda path, content, dry=False: not dry,
        validate=validate_write_file,
    )
    def write_file(path: str, content: str, dry: bool = False) -> str:
        """Write a UTF-8 text file inside the workspace."""
        try:
            target = _resolve_inside(root_path, path)
            if dry:
                action = "overwrite" if target.exists() else "create"
                return f"would {action} {path}: {len(content)} chars"
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_text_atomic(target, content)
            return f"wrote {path}"
        except Exception as error:  # noqa: BLE001 - tools report readable failures
            return f"write_file failed: {error}"

    return write_file


def make_edit_file(*, root: str | Path = "."):
    root_path = Path(root)

    def validate_edit_file(
        path: str,
        old: str,
        new: str,
        dry: bool = False,
        all: bool = False,
    ) -> bool | str:
        if old == "":
            return "edit_file failed: old text is required"
        try:
            target = _resolve_inside(root_path, path)
            content = target.read_text(encoding="utf-8")
        except Exception as error:  # noqa: BLE001 - validation returns user-visible text
            return f"edit_file failed: {error}"
        if old not in content:
            return f"edit_file failed: '{old}' not found in {path}"
        count = content.count(old)
        if count > 1 and not all:
            return (
                f"edit_file failed: '{old}' appears {count} times in {path}; "
                "pass all=True to replace every match"
            )
        return True

    @tool(
        permission=ask_before("editing files"),
        read_only=lambda path, old, new, dry=False, all=False: dry,
        destructive=lambda path, old, new, dry=False, all=False: not dry,
        validate=validate_edit_file,
    )
    def edit_file(
        path: str,
        old: str,
        new: str,
        dry: bool = False,
        all: bool = False,
    ) -> str:
        """Replace text in a UTF-8 file inside the workspace."""
        try:
            target = _resolve_inside(root_path, path)
            content = target.read_text(encoding="utf-8")
            if old == "":
                return "edit_file failed: old text is required"
            if old not in content:
                return f"edit_file failed: '{old}' not found in {path}"
            count = content.count(old)
            if count > 1 and not all:
                return (
                    f"edit_file failed: '{old}' appears {count} times in {path}; "
                    "pass all=True to replace every match"
                )
            if dry:
                return f"would edit {path}: {count} replacement{'s' if count != 1 else ''}"
            _write_text_atomic(target, content.replace(old, new, -1 if all else 1))
            if count == 1:
                return f"edited {path}"
            return f"edited {path}: {count} replacements"
        except Exception as error:  # noqa: BLE001 - tools report readable failures
            return f"edit_file failed: {error}"

    return edit_file


def make_edit_files(*, root: str | Path = "."):
    root_path = Path(root)

    def validate_edit_files(path: str, edits: list, dry: bool = False) -> bool | str:
        try:
            target = _resolve_inside(root_path, path)
            content = target.read_text(encoding="utf-8")
            normalized = _normalize_edits(edits)
        except Exception as error:  # noqa: BLE001 - validation returns user-visible text
            return f"edit_files failed: {error}"
        if not normalized:
            return "edit_files failed: edits are required"
        updated = content
        for index, item in enumerate(normalized, start=1):
            old = item["old"]
            new = item["new"]
            if old == "":
                return f"edit_files failed: edit {index} old text is required"
            if old not in updated:
                return f"edit_files failed: edit {index} old text not found in {path}"
            updated = updated.replace(old, new, 1)
        return True

    @tool(
        permission=ask_before("editing files"),
        read_only=lambda path, edits, dry=False: dry,
        destructive=lambda path, edits, dry=False: not dry,
        validate=validate_edit_files,
        schema={
            "edits": {
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
        },
    )
    def edit_files(path: str, edits: list, dry: bool = False) -> str:
        """Apply multiple text replacements to one UTF-8 file atomically."""
        try:
            target = _resolve_inside(root_path, path)
            content = target.read_text(encoding="utf-8")
            normalized = _normalize_edits(edits)
            if not normalized:
                return "edit_files failed: edits are required"

            updated = content
            for index, item in enumerate(normalized, start=1):
                old = item["old"]
                new = item["new"]
                if old == "":
                    return f"edit_files failed: edit {index} old text is required"
                if old not in updated:
                    return f"edit_files failed: edit {index} old text not found in {path}"
                updated = updated.replace(old, new, 1)

            if dry:
                return f"would edit {path}: {len(normalized)} replacements"
            _write_text_atomic(target, updated)
            return f"edited {path}: {len(normalized)} replacements"
        except Exception as error:  # noqa: BLE001 - tools report readable failures
            return f"edit_files failed: {error}"

    return edit_files


def make_delete_file(*, root: str | Path = "."):
    root_path = Path(root)

    def validate_delete_file(path: str, dry: bool = False) -> bool | str:
        try:
            target = _resolve_inside_no_follow(root_path, path)
        except Exception as error:  # noqa: BLE001 - validation returns user-visible text
            return f"delete_file failed: {error}"
        if not target.exists():
            return f"delete_file failed: file not found: {path}"
        if not target.is_file() or target.is_symlink():
            return f"delete_file failed: not a regular file: {path}"
        return True

    @tool(
        permission=ask_before("deleting files"),
        read_only=lambda path, dry=False: dry,
        destructive=lambda path, dry=False: not dry,
        validate=validate_delete_file,
    )
    def delete_file(path: str, dry: bool = False) -> str:
        """Delete one regular file inside the workspace."""
        try:
            target = _resolve_inside_no_follow(root_path, path)
            if not target.exists():
                return f"delete_file failed: file not found: {path}"
            if not target.is_file() or target.is_symlink():
                return f"delete_file failed: not a regular file: {path}"
            if dry:
                return f"would delete {path}"
            target.unlink()
            return f"deleted {path}"
        except Exception as error:  # noqa: BLE001 - tools report readable failures
            return f"delete_file failed: {error}"

    return delete_file


def make_move_file(*, root: str | Path = "."):
    root_path = Path(root)

    def validate_move_file(
        src: str,
        dst: str,
        dry: bool = False,
        overwrite: bool = False,
    ) -> bool | str:
        try:
            source = _resolve_inside_no_follow(root_path, src)
            target = _resolve_inside_no_follow(root_path, dst)
        except Exception as error:  # noqa: BLE001 - validation returns user-visible text
            return f"move_file failed: {error}"
        if source == target:
            return "move_file failed: source and destination are the same"
        if not source.exists():
            return f"move_file failed: file not found: {src}"
        if source.is_symlink() or not source.is_file():
            return f"move_file failed: source is not a regular file: {src}"
        if target.is_symlink():
            return f"move_file failed: destination is not a regular file: {dst}"
        if target.exists():
            if not target.is_file():
                return f"move_file failed: destination is not a regular file: {dst}"
            if not overwrite:
                return f"move_file failed: destination exists: {dst}"
        return True

    @tool(
        permission=ask_before("moving files"),
        read_only=lambda src, dst, dry=False, overwrite=False: dry,
        destructive=lambda src, dst, dry=False, overwrite=False: not dry,
        validate=validate_move_file,
    )
    def move_file(
        src: str,
        dst: str,
        dry: bool = False,
        overwrite: bool = False,
    ) -> str:
        """Move or rename one regular file inside the workspace."""
        try:
            source = _resolve_inside_no_follow(root_path, src)
            target = _resolve_inside_no_follow(root_path, dst)
            if source == target:
                return "move_file failed: source and destination are the same"
            if not source.exists():
                return f"move_file failed: file not found: {src}"
            if source.is_symlink() or not source.is_file():
                return f"move_file failed: source is not a regular file: {src}"
            if target.is_symlink():
                return f"move_file failed: destination is not a regular file: {dst}"
            if target.exists():
                if not target.is_file():
                    return f"move_file failed: destination is not a regular file: {dst}"
                if not overwrite:
                    return f"move_file failed: destination exists: {dst}"
            if dry:
                action = "overwrite" if target.exists() else "move"
                return f"would {action} {src} -> {dst}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if overwrite:
                source.replace(target)
            else:
                source.rename(target)
            return f"moved {src} -> {dst}"
        except Exception as error:  # noqa: BLE001 - tools report readable failures
            return f"move_file failed: {error}"

    return move_file


def make_copy_file(*, root: str | Path = "."):
    root_path = Path(root)

    def validate_copy_file(
        src: str,
        dst: str,
        dry: bool = False,
        overwrite: bool = False,
    ) -> bool | str:
        try:
            source = _resolve_inside_no_follow(root_path, src)
            target = _resolve_inside_no_follow(root_path, dst)
        except Exception as error:  # noqa: BLE001 - validation returns user-visible text
            return f"copy_file failed: {error}"
        if source == target:
            return "copy_file failed: source and destination are the same"
        if not source.exists():
            return f"copy_file failed: file not found: {src}"
        if source.is_symlink() or not source.is_file():
            return f"copy_file failed: source is not a regular file: {src}"
        if target.is_symlink():
            return f"copy_file failed: destination is not a regular file: {dst}"
        if target.exists():
            if not target.is_file():
                return f"copy_file failed: destination is not a regular file: {dst}"
            if not overwrite:
                return f"copy_file failed: destination exists: {dst}"
        return True

    @tool(
        permission=ask_before("copying files"),
        read_only=lambda src, dst, dry=False, overwrite=False: dry,
        destructive=lambda src, dst, dry=False, overwrite=False: not dry,
        validate=validate_copy_file,
    )
    def copy_file(
        src: str,
        dst: str,
        dry: bool = False,
        overwrite: bool = False,
    ) -> str:
        """Copy one regular file inside the workspace."""
        try:
            source = _resolve_inside_no_follow(root_path, src)
            target = _resolve_inside_no_follow(root_path, dst)
            if source == target:
                return "copy_file failed: source and destination are the same"
            if not source.exists():
                return f"copy_file failed: file not found: {src}"
            if source.is_symlink() or not source.is_file():
                return f"copy_file failed: source is not a regular file: {src}"
            if target.is_symlink():
                return f"copy_file failed: destination is not a regular file: {dst}"
            if target.exists():
                if not target.is_file():
                    return f"copy_file failed: destination is not a regular file: {dst}"
                if not overwrite:
                    return f"copy_file failed: destination exists: {dst}"
            if dry:
                action = "overwrite" if target.exists() else "copy"
                return f"would {action} {src} -> {dst}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            return f"copied {src} -> {dst}"
        except Exception as error:  # noqa: BLE001 - tools report readable failures
            return f"copy_file failed: {error}"

    return copy_file


def make_list_files(
    *,
    root: str | Path = ".",
    limit: int = 200,
    max_files: int | None = None,
    ignore: list[str] | tuple[str, ...] | set[str] | None = None,
):
    root_path = Path(root)

    @tool(
        read_only=True,
        concurrency_safe=True,
        format=lambda result: _format_lines(result, empty="no files found"),
    )
    async def list_workspace_files(pattern: str = "*") -> list[str]:
        """List files in the workspace using a glob pattern."""
        return await list_files(
            root=root_path,
            pattern=pattern,
            limit=limit,
            max_files=max_files,
            ignore=ignore,
        )

    return list_workspace_files


def make_glob_files(
    *,
    root: str | Path = ".",
    limit: int = 200,
    max_files: int | None = None,
    ignore: list[str] | tuple[str, ...] | set[str] | None = None,
):
    root_path = Path(root)

    @tool(
        read_only=True,
        concurrency_safe=True,
        format=lambda result: _format_lines(result, empty="no files found"),
    )
    async def glob_workspace_files(pattern: str = "*") -> list[str]:
        """Find files in the workspace using a glob pattern."""
        return await glob_files(
            root=root_path,
            pattern=pattern,
            limit=limit,
            max_files=max_files,
            ignore=ignore,
        )

    return glob_workspace_files


def make_search_text(
    *,
    root: str | Path = ".",
    limit: int = 20,
    max_file: int | None = None,
    max_files: int | None = None,
    ignore: list[str] | tuple[str, ...] | set[str] | None = None,
):
    root_path = Path(root)

    def validate_search_text(pattern: str, glob: str = "*", context: int = 0) -> bool | str:
        if context < 0:
            return "search_text failed: context must be >= 0"
        try:
            re.compile(pattern)
        except re.error as error:
            return f"search_text failed: invalid regex: {error}"
        return True

    @tool(
        read_only=True,
        concurrency_safe=True,
        validate=validate_search_text,
        format=lambda result: _format_lines(result, empty="no matches found"),
        schema={"context": {"type": "integer", "minimum": 0, "default": 0}},
    )
    async def search_workspace_text(
        pattern: str,
        glob: str = "*",
        context: int = 0,
    ) -> list[str]:
        """Search for text in workspace files."""
        return await search_text(
            root=root_path,
            pattern=pattern,
            glob=glob,
            limit=limit,
            context=context,
            max_file=max_file,
            max_files=max_files,
            ignore=ignore,
        )

    return search_workspace_text


def _clip(text: str, limit: int | None) -> str:
    if limit is None or len(text) <= limit:
        return text
    if limit <= 0:
        return ""
    suffix = "\n...[truncated]"
    if limit <= len(suffix):
        return text[:limit]
    return text[: limit - len(suffix)] + suffix


def _format_lines(items: list[str], *, empty: str) -> str:
    if not items:
        return empty
    return "\n".join(items)


def _search_file_with_context(
    path: Path,
    relative: Path,
    regex: re.Pattern,
    *,
    context: int,
    remaining: int,
) -> tuple[list[str], int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:  # noqa: BLE001 - search skips unreadable files
        return [], 0
    if remaining <= 0:
        return [], 0
    emitted: set[int] = set()
    match_lines: set[int] = set()
    matches = 0
    for index, line in enumerate(lines, start=1):
        if not regex.search(line):
            continue
        matches += 1
        match_lines.add(index)
        start = max(1, index - context)
        end = min(len(lines), index + context)
        for line_number in range(start, end + 1):
            emitted.add(line_number)
        if matches >= remaining:
            break

    results = []
    for line_number in sorted(emitted):
        prefix = ">" if line_number in match_lines else "-"
        results.append(
            f"{relative.as_posix()}:{line_number}:{prefix} {lines[line_number - 1]}"
        )
    return results, matches


def _format_read_file(
    content: str,
    *,
    path: str,
    offset: int = 1,
    limit: int | None = None,
) -> str:
    if content:
        return content
    if offset != 1 or limit is not None:
        return "no lines in range"
    return "empty file"


def _read_lines(
    path: Path,
    *,
    offset: int,
    limit: int | None,
    numbers: bool = False,
) -> str:
    lines = []
    end = None if limit is None else offset + limit
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if index < offset:
                continue
            if end is not None and index >= end:
                break
            lines.append(f"{index}: {line}" if numbers else line)
    return "".join(lines)


def _number_lines(content: str, *, start: int) -> str:
    return "".join(
        f"{index}: {line}"
        for index, line in enumerate(content.splitlines(keepends=True), start=start)
    )


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return b"\0" in handle.read(8192)
    except OSError:
        return False


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _ignored(
    path: Path,
    ignore: list[str] | tuple[str, ...] | set[str] | None,
) -> bool:
    if not ignore:
        return False
    value = path.as_posix()
    parts = set(path.parts)
    for item in ignore:
        pattern = str(item).strip()
        if not pattern:
            continue
        pattern = pattern.rstrip("/")
        if pattern in parts:
            return True
        if fnmatch(value, pattern) or fnmatch(path.name, pattern):
            return True
    return False


def _walk_files(
    root: Path,
    ignore: list[str] | tuple[str, ...] | set[str] | None,
):
    root = root.resolve()
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError:
            continue
        for child in children:
            relative = child.relative_to(root)
            if _ignored(relative, ignore):
                continue
            if not _is_inside(root, child):
                continue
            if child.is_dir() and not child.is_symlink():
                stack.append(child)
            elif child.is_file():
                yield child, relative


def _is_inside(root: Path, path: Path) -> bool:
    try:
        target = path.resolve()
    except OSError:
        return False
    return target == root or root in target.parents


def _matches(path: Path, pattern: str) -> bool:
    value = path.as_posix()
    return fnmatch(value, pattern) or fnmatch(path.name, pattern)


def _normalize_edits(edits: list) -> list[dict[str, str]]:
    normalized = []
    for item in edits:
        if not isinstance(item, dict):
            raise ValueError("each edit must be an object with old and new")
        if "old" not in item or "new" not in item:
            raise ValueError("each edit must include old and new")
        normalized.append({"old": str(item["old"]), "new": str(item["new"])})
    return normalized
