from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

from ..core.tools import Tool
from .files import (
    make_copy_file,
    make_delete_file,
    make_edit_file,
    make_edit_files,
    make_glob_files,
    make_list_files,
    make_move_file,
    make_read_file,
    make_search_text,
    make_write_file,
)
from .shell import ReadRule, make_shell
from .web import make_fetch_url

DEFAULT_IGNORE = (
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
)


def make_tools(
    *,
    root: str | Path = ".",
    files: bool = True,
    shell: bool = True,
    write: bool = True,
    web: bool = False,
    search_limit: int = 20,
    list_limit: int = 200,
    max_list_files: int | None = 5000,
    max_read: int | None = 20000,
    max_search_file: int | None = 1_000_000,
    max_search_files: int | None = 1000,
    ignore: list[str] | tuple[str, ...] | set[str] | None = DEFAULT_IGNORE,
    timeout: float = 30,
    validate_shell: Callable[[str], bool | str | None] | None = None,
    read_shell: ReadRule = False,
    max_shell: int | None = None,
    shell_env: Mapping[str, str | None] | None = None,
    fetch_timeout: float = 10,
    max_fetch: int | None = 200_000,
    allow_url: Callable[[str], bool | str | None] | None = None,
) -> list[Tool]:
    tools: list[Tool] = []
    if files:
        tools.extend(
            [
                make_read_file(root=root, max_text=max_read),
                make_list_files(
                    root=root,
                    limit=list_limit,
                    max_files=max_list_files,
                    ignore=ignore,
                ),
                make_glob_files(
                    root=root,
                    limit=list_limit,
                    max_files=max_list_files,
                    ignore=ignore,
                ),
                make_search_text(
                    root=root,
                    limit=search_limit,
                    max_file=max_search_file,
                    max_files=max_search_files,
                    ignore=ignore,
                ),
            ]
        )
        if write:
            tools.extend(
                [
                    make_write_file(root=root),
                    make_edit_file(root=root),
                    make_edit_files(root=root),
                    make_delete_file(root=root),
                    make_move_file(root=root),
                    make_copy_file(root=root),
                ]
            )
    if shell:
        tools.append(
            make_shell(
                cwd=root,
                timeout_seconds=timeout,
                validate=validate_shell,
                read=read_shell,
                max_output=max_shell,
                env=shell_env,
            )
        )
    if web:
        tools.append(
            make_fetch_url(
                timeout=fetch_timeout,
                max_bytes=max_fetch,
                allow=allow_url,
            )
        )
    return tools
