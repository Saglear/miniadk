from __future__ import annotations

import asyncio
import os
import signal
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Mapping

from ..core.middleware import ask_before
from ..core.tools import tool

ReadRule = bool | str | list[str] | tuple[str, ...] | set[str] | Callable[[str], bool]


@dataclass(slots=True)
class ShellResult:
    command: str
    returncode: int
    stdout: str
    stderr: str

    def __str__(self) -> str:
        text = self.stdout + self.stderr
        if text:
            return text
        if self.returncode != 0:
            return f"exit code: {self.returncode}"
        return ""


def make_shell(
    *,
    cwd: str | Path = ".",
    # ``None`` means no timeout — the agent runs the command to
    # completion. This is the right default for an agent product:
    # build / test / install / long-running scripts are normal use,
    # and an arbitrary cap creates more pain than it prevents. Users
    # who need a hard ceiling pass an explicit number.
    timeout_seconds: float | None = None,
    validate: Callable[[str], bool | str | None] | None = None,
    read: ReadRule = False,
    max_output: int | None = None,
    env: Mapping[str, str | None] | None = None,
):
    cwd_path = Path(cwd).resolve()
    shell_env = _build_env(env)
    is_read = _read_matcher(read)

    def validate_command(command: str, cwd: str = "."):
        if not command.strip():
            return "shell command is required"
        if timeout_seconds is not None and timeout_seconds <= 0:
            return "shell timeout must be > 0 (or omit it for no timeout)"
        if not cwd_path.exists():
            return f"shell cwd does not exist: {cwd_path}"
        if not cwd_path.is_dir():
            return f"shell cwd is not a directory: {cwd_path}"
        try:
            run_cwd = _resolve_cwd(cwd_path, cwd)
        except Exception as error:  # noqa: BLE001 - validation returns user-visible text
            return f"shell cwd failed: {error}"
        if not run_cwd.exists():
            return f"shell cwd does not exist: {cwd}"
        if not run_cwd.is_dir():
            return f"shell cwd is not a directory: {cwd}"
        if validate is None:
            return True
        return validate(command)

    @tool(
        permission=ask_before("running shell commands"),
        read_only=is_read,
        destructive=lambda command: not is_read(command),
        validate=validate_command,
        format=_format_shell_result,
        max_text=max_output,
        schema={
            "command": {"type": "string", "minLength": 1},
            "cwd": {"type": "string", "default": "."},
        },
    )
    async def shell(
        command: str,
        input: str | None = None,
        cwd: str = ".",
    ) -> ShellResult:
        """Run a shell command in the workspace."""
        run_cwd = _resolve_cwd(cwd_path, cwd)
        return await _run_shell(
            command,
            run_cwd,
            timeout_seconds,
            max_output,
            shell_env,
            input,
        )

    return shell


def _resolve_cwd(root: Path, cwd: str) -> Path:
    target = (root / cwd).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path is outside root: {cwd}")
    return target


def _read_matcher(read: ReadRule) -> Callable[[str], bool]:
    if callable(read):
        return lambda command: bool(read(command))
    if isinstance(read, bool):
        return lambda command: read
    patterns = (read,) if isinstance(read, str) else tuple(read)
    return lambda command: any(fnmatch(command.strip(), pattern) for pattern in patterns)


async def _run_shell(
    command: str,
    cwd: Path,
    timeout_seconds: float | None,
    max_output: int | None,
    env: dict[str, str] | None,
    input: str | None,
) -> ShellResult:
    process = None
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        if timeout_seconds is None:
            stdout, stderr = await process.communicate(
                None if input is None else input.encode()
            )
        else:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(None if input is None else input.encode()),
                timeout=timeout_seconds,
            )
        stdout_text, stderr_text = _clip_streams(stdout, stderr, max_output)
        return ShellResult(
            command=command,
            returncode=process.returncode,
            stdout=stdout_text,
            stderr=stderr_text,
        )
    except TimeoutError:
        if process is not None:
            await _terminate_process_tree(process)
            stdout, stderr = await process.communicate()
        else:
            stdout = b""
            stderr = b""
        timeout_stderr = _join_stderr(
            stderr,
            f"command timed out after {timeout_seconds} seconds",
        )
        stdout_text, stderr_text = _clip_streams(stdout, timeout_stderr, max_output)
        return ShellResult(
            command=command,
            returncode=-1,
            stdout=stdout_text,
            stderr=stderr_text,
        )
    except asyncio.CancelledError:
        if process is not None:
            await _terminate_process_tree(process)
            await process.communicate()
        raise


async def _terminate_process_tree(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        process.terminate()

    try:
        await asyncio.wait_for(process.wait(), timeout=1)
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError:
            process.kill()
        await process.wait()


def _join_stderr(stderr: str | bytes, message: str) -> str:
    text = _clip(stderr, None).rstrip()
    if not text:
        return message
    return f"{text}\n{message}"


def _format_shell_result(result: ShellResult) -> str:
    text = str(result)
    if result.returncode == 0:
        return text
    status = f"exit code: {result.returncode}"
    if not text:
        return status
    if status in text:
        return text
    return f"{text.rstrip()}\n{status}"


def _clip(text: str | bytes, limit: int | None) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if limit is None or len(text) <= limit:
        return text
    if limit <= 0:
        return ""
    suffix = "\n...[truncated]"
    if limit <= len(suffix):
        return text[:limit]
    return text[: limit - len(suffix)] + suffix


def _clip_streams(
    stdout: str | bytes,
    stderr: str | bytes,
    limit: int | None,
) -> tuple[str, str]:
    stdout_text = _clip(stdout, None)
    stderr_text = _clip(stderr, None)
    if limit is None:
        return stdout_text, stderr_text
    if limit <= 0:
        return "", ""

    total = len(stdout_text) + len(stderr_text)
    if total <= limit:
        return stdout_text, stderr_text

    stdout_limit = min(len(stdout_text), limit)
    clipped_stdout = _clip(stdout_text, stdout_limit)
    remaining = limit - len(clipped_stdout)
    if remaining <= 0:
        return clipped_stdout, ""
    return clipped_stdout, _clip(stderr_text, remaining)


def _build_env(env: Mapping[str, str | None] | None) -> dict[str, str] | None:
    if env is None:
        return None
    built = dict(os.environ)
    for key, value in env.items():
        if value is None:
            built.pop(key, None)
        else:
            built[key] = str(value)
    return built
