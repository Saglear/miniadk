"""Bridge runtime events ↔ JSON line protocol over a TUI subprocess.

Architecture (single responsibility):

    +---------------------+
    |     TUIBridge       |   only knows runtime ↔ JSON line protocol
    +----------+----------+
               |
               |   sends event JSON         reads UpEvent JSON
               v
    +----------+----------+      +-------------------------+
    |    TUIProcess       | ---> |    miniadk-tui          |
    | (subprocess + pipes)|      |    (Bun + Ink)          |
    +---------------------+      +-------------------------+

The bridge is the only place that knows about JSON serialization. Higher-
level concerns (slash commands, sessions, skill expansion) live in
``run_cli`` so the bridge stays small and easy to reason about.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core.events import Event
from ...core.middleware import PermissionRequest


REPO_ROOT = Path(__file__).resolve().parents[4]
TUI_TS_ENTRY = REPO_ROOT / "tui-ts" / "src" / "index.tsx"


def find_tui_command(*, auto_fetch: bool = True) -> list[str] | None:
    """Locate a runnable command for the Ink TUI.

    Resolution order:

    1. ``MINIADK_TUI_BIN`` env override — a path to a built binary.
    2. Cached binary fetched by ``miniadk-tui-fetch`` at
       ``~/.cache/miniadk/tui/<release>/<asset>``.
    3. ``bun`` on ``$PATH`` (or ``~/.bun/bin/bun``) plus the in-repo
       source tree (dev mode).
    4. **Auto-fetch** — when ``auto_fetch`` is true (the default and
       what ``run_cli`` uses) and steps 1-3 fail, try downloading the
       binary on the fly. Set ``MINIADK_TUI_NO_FETCH=1`` to opt out.

    Returns ``None`` if none of the steps succeed.
    """

    override = os.environ.get("MINIADK_TUI_BIN")
    if override and Path(override).exists():
        return [override]

    # Cached prebuilt binary (PyPI install path).
    try:
        from ..._tui_fetch import detect_asset, cached_path, DEFAULT_RELEASE
        release = os.environ.get("MINIADK_TUI_RELEASE", DEFAULT_RELEASE)
        cached = cached_path(release, detect_asset())
        if cached.exists():
            return [str(cached)]
    except Exception:
        pass

    bun = shutil.which("bun") or str(Path.home() / ".bun" / "bin" / "bun")
    if Path(bun).exists() and TUI_TS_ENTRY.exists():
        return [bun, str(TUI_TS_ENTRY)]

    # Last resort: try to fetch the prebuilt binary now. This is what
    # makes ``pip install miniadk`` + ``run_cli`` work end-to-end on a
    # fresh machine without any extra steps.
    if auto_fetch:
        try:
            from ..._tui_fetch import ensure_binary

            fetched = ensure_binary()
            if fetched is not None:
                return [str(fetched)]
        except Exception:
            pass

    return None


# ── data shapes ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class IntroPayload:
    agent: str
    model: str
    cwd: str
    tool_count: int


# A function that accepts a user submission (text from the TUI) and
# yields runtime events. The bridge does not care what the caller does
# with the input — it could be a slash command, a skill, or a raw
# runtime turn. That separation keeps the bridge tiny.
TurnRunner = Callable[[str], AsyncIterator[Event]]

# A coroutine resolving permission decisions. The bridge owns the
# JSON round-trip; the caller plugs this into its middleware.
PermissionAsker = Callable[[PermissionRequest], Awaitable[bool]]

# A function that lists files matching a prefix relative to ``cwd``.
# Returns paths in display order (already truncated to ``limit``).
FileLister = Callable[[str, int], list[str]]

# Optional accounting hook called at the end of every turn. Returns
# ``(token_estimate, bell_threshold_ms)`` so the bridge can attach the
# token count to ``run_end`` and decide whether to ring the bell.
TurnAccountant = Callable[[], tuple[int | None, int]]


# ── transport: subprocess + pipes ───────────────────────────────────────


class _TUIProcess:
    """Owns the child process and its three pipes.

    Layout::

        parent →  child stdin   (DownEvents)
        parent ← read_fd ← child(write_fd)   (UpEvents)
        child  →  /dev/tty                   (visual rendering)
    """

    def __init__(self, argv: list[str]):
        self._argv = argv
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.StreamReader | None = None

    async def start(self) -> None:
        read_fd, write_fd = os.pipe()
        os.set_inheritable(write_fd, True)
        env = {**os.environ, "MINIADK_TUI_OUTPUT_FD": str(write_fd)}

        self._proc = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
            pass_fds=(write_fd,),
            env=env,
        )
        os.close(write_fd)

        loop = asyncio.get_event_loop()
        self._reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(self._reader)
        await loop.connect_read_pipe(lambda: protocol, os.fdopen(read_fd, "rb"))

    def write(self, message: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            return
        line = json.dumps(message, ensure_ascii=False) + "\n"
        try:
            self._proc.stdin.write(line.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            pass

    async def read_line(self) -> dict[str, Any] | None:
        if self._reader is None:
            return None
        line = await self._reader.readline()
        if not line:
            return None
        try:
            return json.loads(line.decode("utf-8").strip())
        except json.JSONDecodeError:
            return None

    async def stop(self) -> None:
        if self._proc is None:
            return
        self.write({"type": "quit", "data": {}})
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                self._proc.kill()


# ── event translation ───────────────────────────────────────────────────


_PASSTHROUGH = {
    "thinking_delta",
    "tool_call_delta",
    "tool_call",
    "tool_progress",
    "tool_result",
    "tool_denied",
    "tool_invalid",
    "tool_error",
    "error",
    "message_delta",
    "message",
    # Slash-command surfaces — emitted by the caller's turn_runner, not
    # the runtime itself, but they look just like any other downstream
    # event so we let them through.
    "notice",
    "clear",
    "quit",
}


def event_to_wire(event: Event) -> dict[str, Any] | None:
    """Translate a runtime ``Event`` into a downstream JSON message.

    Returns ``None`` if the event has no TUI representation.
    """

    if event.type not in _PASSTHROUGH:
        return None

    data = dict(event.data)
    # Tool result text may live under "text" or "result".
    if event.type == "tool_result":
        data = {
            "name": data.get("name") or "result",
            "text": str(data.get("text") or data.get("result") or ""),
        }
    return {"type": event.type, "data": data}


# ── bridge ─────────────────────────────────────────────────────────────


class TUIBridge:
    """Run the Ink TUI and relay runtime events.

    The bridge is intentionally small. Three things happen here:

    * ``intro`` is sent once on startup,
    * each user submission goes through ``turn_runner`` (provided by
      the caller) and its events stream back to the TUI,
    * permission requests round-trip through ``permission_asker``.

    Everything else — slash commands, skills, sessions — lives in the
    caller. The bridge does not import the runtime.
    """

    def __init__(
        self,
        *,
        intro: IntroPayload,
        turn_runner: TurnRunner,
        permission_asker: PermissionAsker | None = None,
        file_lister: FileLister | None = None,
        turn_accountant: TurnAccountant | None = None,
        on_mode_changed: Callable[[str], None] | None = None,
        pending_notices: list[str] | None = None,
    ):
        self._intro = intro
        self._turn_runner = turn_runner
        self._permission_asker = permission_asker
        self._file_lister = file_lister
        self._turn_accountant = turn_accountant
        self._on_mode_changed = on_mode_changed
        self._pending_notices: list[str] = list(pending_notices or [])
        self._process: _TUIProcess | None = None
        self._pending: dict[str, asyncio.Future[bool]] = {}
        self._quit = asyncio.Event()
        self._submissions: asyncio.Queue[str | None] = asyncio.Queue()
        self._reader_task: asyncio.Task[None] | None = None
        self._current_turn_task: asyncio.Task[None] | None = None
        self._permission_mode: str = "default"
        self._wire_log: Any = _open_wire_log()

    # ── public ────────────────────────────────────────────────────────

    async def run(self, argv: list[str] | None = None, *, process: Any | None = None) -> None:
        """Drive the TUI subprocess until it asks to quit.

        ``process`` is a hook for tests — pass an object that mimics
        :class:`_TUIProcess` (``start``/``write``/``read_line``/``stop``).
        When omitted the bridge spawns the real Bun subprocess.
        """
        if process is not None:
            self._process = process
        else:
            argv = argv or find_tui_command()
            if argv is None:
                raise RuntimeError(
                    "miniadk-tui not found. Install Bun (https://bun.sh) or "
                    "set MINIADK_TUI_BIN to a built binary path."
                )
            self._process = _TUIProcess(argv)
        await self._process.start()
        self._reader_task = asyncio.create_task(self._reader_loop())

        try:
            await self._wait_for_ready()
            self._send({"type": "intro", "data": {
                "agent": self._intro.agent,
                "model": self._intro.model,
                "cwd": self._intro.cwd,
                "tool_count": self._intro.tool_count,
                "permission_mode": self._permission_mode,
            }})
            for text in self._pending_notices:
                self._send({"type": "notice", "data": {"text": text}})
            self._pending_notices.clear()
            await self._main_loop()
        finally:
            self._quit.set()
            if self._reader_task is not None:
                self._reader_task.cancel()
            await self._process.stop()

    def emit(self, message: dict[str, Any]) -> None:
        """Send an arbitrary downstream message (e.g. ``notice``).

        Useful for slash-command output so the caller doesn't need its
        own JSON serializer.
        """
        self._send(message)

    async def ask_permission(self, request: PermissionRequest) -> bool:
        """Round-trip a permission decision through the TUI.

        Honors the current ``permission_mode``:

        * ``accept_edits`` auto-allows tools that look like file edits
          (``write_file``, ``edit_file``, ``create_file``, ...).
        * ``plan`` auto-denies anything the runtime flagged as
          destructive — the model is expected to plan, not act.
        * ``default`` always asks via modal.
        """
        mode = self._permission_mode
        if mode == "accept_edits" and _looks_like_edit(request.tool.name):
            return True
        if mode == "plan" and _looks_like_destructive(request):
            return False
        request_id = uuid.uuid4().hex
        future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future
        self._send({
            "type": "permission_request",
            "data": {
                "id": request_id,
                "tool": request.tool.name,
                "reason": request.reason,
                "arguments": dict(request.arguments) if request.arguments else {},
            },
        })
        return await future

    @property
    def permission_mode(self) -> str:
        return self._permission_mode

    # ── internals ─────────────────────────────────────────────────────

    async def _wait_for_ready(self) -> None:
        # The reader_loop will signal _quit if the child dies. We just
        # wait for the first submission slot to be free of a sentinel.
        # Simpler approach: poll a flag set by the reader.
        for _ in range(200):  # ~10s @ 50ms
            if self._ready:
                return
            await asyncio.sleep(0.05)
        raise RuntimeError("miniadk-tui did not signal ready")

    _ready: bool = False

    async def _reader_loop(self) -> None:
        """Single owner of the upstream pipe.

        Routes:
          * ``ready`` → set ``_ready``
          * ``submit`` → push to ``_submissions``
          * ``permission_response`` → resolve pending future
          * ``quit`` → push ``None`` sentinel and set ``_quit``
        """
        assert self._process is not None
        while True:
            event = await self._process.read_line()
            if event is None:
                self._quit.set()
                await self._submissions.put(None)
                return
            self._log_wire("up", event)
            kind = event.get("type")
            data = event.get("data") or {}
            if kind == "ready":
                self._ready = True
            elif kind == "submit":
                text = str(data.get("text") or "").strip()
                if text:
                    await self._submissions.put(text)
            elif kind == "permission_response":
                future = self._pending.pop(data.get("id"), None)
                if future is not None and not future.done():
                    future.set_result(bool(data.get("allow")))
            elif kind == "set_permission_mode":
                mode = str(data.get("mode") or "default")
                if mode in {"default", "accept_edits", "plan"}:
                    self._permission_mode = mode
                    self._send({
                        "type": "permission_mode_changed",
                        "data": {"mode": mode},
                    })
                    if self._on_mode_changed is not None:
                        try:
                            self._on_mode_changed(mode)
                        except Exception:
                            pass
            elif kind == "list_files":
                request_id = str(data.get("request_id") or "")
                prefix = str(data.get("prefix") or "")
                limit = int(data.get("limit") or 30)
                paths: list[str] = []
                if self._file_lister is not None:
                    try:
                        paths = self._file_lister(prefix, limit)
                    except Exception:
                        paths = []
                self._send({
                    "type": "files",
                    "data": {"request_id": request_id, "paths": paths},
                })
            elif kind == "interrupt":
                # Cancel the in-flight runtime turn. The task will raise
                # CancelledError, _run_turn catches it and emits a notice.
                # Permission round-trips are also resolved (deny) so we
                # don't deadlock if a modal was open.
                task = self._current_turn_task
                if task is not None and not task.done():
                    task.cancel()
                for future in list(self._pending.values()):
                    if not future.done():
                        future.set_result(False)
                self._pending.clear()
            elif kind == "quit":
                self._quit.set()
                await self._submissions.put(None)
                return

    async def _main_loop(self) -> None:
        while not self._quit.is_set():
            text = await self._submissions.get()
            if text is None:
                return
            # Wrap _run_turn in a Task so the reader-loop can cancel it
            # on interrupt without affecting the rest of the bridge.
            self._current_turn_task = asyncio.create_task(self._run_turn(text))
            try:
                await self._current_turn_task
            except asyncio.CancelledError:
                # The reader loop fired interrupt — the turn already
                # emitted its own 'cancelled' notice + run_end.
                pass
            finally:
                self._current_turn_task = None

    async def _run_turn(self, text: str) -> None:
        import time as _time

        start = _time.monotonic()
        is_slash = text.startswith("/")
        if not is_slash:
            self._send({"type": "user", "data": {"text": text, "turn": 0}})
        self._send({"type": "run_start", "data": {}})
        cancelled = False
        try:
            async for event in self._turn_runner(text):
                wire = event_to_wire(event)
                if wire is not None:
                    self._send(wire)
        except asyncio.CancelledError:
            cancelled = True
            self._send({"type": "notice", "data": {"text": "cancelled"}})
            raise
        except Exception as error:
            self._send({"type": "error", "data": {"message": str(error)}})
        finally:
            duration_ms = int((_time.monotonic() - start) * 1000)
            tokens: int | None = None
            bell_threshold_ms = 0
            if self._turn_accountant is not None:
                try:
                    tokens, bell_threshold_ms = self._turn_accountant()
                except Exception:
                    tokens, bell_threshold_ms = None, 0
            payload: dict[str, Any] = {"duration_ms": duration_ms, "cancelled": cancelled}
            if tokens is not None:
                payload["tokens"] = tokens
            self._send({"type": "run_end", "data": payload})
            if bell_threshold_ms and duration_ms >= bell_threshold_ms:
                # \a is the BEL character — most terminals beep or flash.
                # We write straight to the controlling tty so it survives
                # the JSON pipe.
                try:
                    import sys as _sys
                    _sys.stderr.write("\a")
                    _sys.stderr.flush()
                except Exception:
                    pass

    def _send(self, message: dict[str, Any]) -> None:
        if self._process is not None:
            self._log_wire("down", message)
            self._process.write(message)

    def _log_wire(self, direction: str, message: dict[str, Any]) -> None:
        if self._wire_log is None:
            return
        import time as _time
        try:
            line = json.dumps(
                {"t": round(_time.time(), 3), "dir": direction, **message},
                ensure_ascii=False,
            )
            self._wire_log.write(line + "\n")
            self._wire_log.flush()
        except Exception:
            pass


def _open_wire_log() -> Any:
    """Return a file handle when MINIADK_TUI_DEBUG is set, else None.

    Useful for repro: ``MINIADK_TUI_DEBUG=/tmp/tui.log uv run …`` produces
    a one-line-per-event log of both directions, suitable to attach to
    bug reports.
    """
    path = os.environ.get("MINIADK_TUI_DEBUG")
    if not path:
        return None
    try:
        return open(path, "w", encoding="utf-8")
    except OSError:
        return None


def _looks_like_edit(tool_name: str) -> bool:
    name = tool_name.lower()
    keywords = ("write", "edit", "create", "patch", "apply", "modify", "save", "format")
    return any(keyword in name for keyword in keywords)


def _looks_like_destructive(request: PermissionRequest) -> bool:
    """Plan-mode auto-deny heuristic."""
    name = request.tool.name.lower()
    if name in {"read_file", "search_text", "list_directory", "glob", "grep"}:
        return False
    if any(keyword in name for keyword in ("write", "edit", "create", "delete", "remove",
                                            "shell", "exec", "run", "patch", "modify")):
        return True
    # Tools whose declared schema marks them destructive — we trust the
    # framework's classification.
    is_destructive = getattr(request.tool, "is_destructive", None)
    if callable(is_destructive):
        try:
            return bool(is_destructive())
        except Exception:
            pass
    return False


__all__ = ["IntroPayload", "TUIBridge", "find_tui_command", "event_to_wire"]
