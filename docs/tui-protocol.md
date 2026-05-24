# MiniADK TUI Protocol (v0)

> JSON-RPC-style line protocol between the Python ADK process and the
> Ink/TS TUI subprocess. One JSON object per line on stdin/stdout.

## Architecture

```
+---------------------------+        +-------------------------+
|  Python user code         |        |  miniadk-tui binary     |
|  ┌─────────────────────┐  |        |  (Bun + Ink + React)    |
|  │ from miniadk import │  |        |                         |
|  │   run_cli           │  | spawn  | - owns stdin/stdout/tty |
|  │ run_cli(agent)      │--+--> tty | - renders frames        |
|  └─────────────────────┘  |        | - reads keystrokes      |
|                           |        |                         |
|  miniadk.adapters         |  pipe  |  parses JSON lines      |
|  .tui_bridge              |<------>|  emits JSON lines       |
+---------------------------+        +-------------------------+
       Python runtime                     Terminal UI
```

The TUI process owns the terminal. The Python process only knows about
events. This is the same architecture Zed's ACP uses.

## Wire format

* One UTF-8 JSON object per line, terminated by `\n`.
* Object shape: `{"type": "<kind>", "data": {...}}`.
* No request/response correlation IDs in v0 — events are one-way except
  for the explicit permission round-trip below.

## Direction: Python → TUI (downstream)

Events that update the TUI display.

| `type`               | `data` fields                                    | Meaning                                     |
| -------------------- | ------------------------------------------------ | ------------------------------------------- |
| `intro`              | `agent`, `model`, `cwd`, `tool_count`, `permission_mode?` | Sent once on startup so the TUI can render the welcome line. |
| `user`               | `text`, `turn`                                   | A user turn was committed (for transcript). |
| `thinking_delta`     | `text`                                           | The model is reasoning.                     |
| `tool_call_delta`    | `index`, `name?`, `arguments?`                   | Streaming tool args before the tool fires.  |
| `tool_call`          | `name`, `arguments`                              | Final tool invocation.                      |
| `tool_progress`      | `tool`, `message`, `data?`                       | In-flight tool progress update.             |
| `tool_result`        | `name`, `text`                                   | Tool returned a result.                     |
| `tool_denied`        | `message`                                        | A guard denied a tool.                      |
| `tool_invalid`       | `message`                                        | Schema validation failed.                   |
| `tool_error`         | `message`                                        | Tool raised.                                |
| `message_delta`      | `text`                                           | Streamed assistant token.                   |
| `message`            | `text`, `streamed`                               | Final assistant message (commit).           |
| `error`              | `message`                                        | Runtime / adapter error.                    |
| `run_start`          | `{}`                                             | A turn started — TUI may show busy state.   |
| `run_end`            | `tokens?`, `duration_ms?`                        | Turn finished; bridge attaches accounting.  |
| `permission_request` | `id`, `tool`, `reason`, `arguments`              | Ask the user to allow / deny.               |
| `permission_mode_changed` | `mode` (`default` / `accept_edits` / `plan`)  | Echo of an accepted `set_permission_mode`.  |
| `files`              | `request_id`, `paths`                            | Response to a `list_files` request.         |
| `notice`             | `text`                                           | Informational note from a slash command.    |
| `clear`              | `{}`                                             | Clear the transcript scrollback.            |
| `quit`               | `{}`                                             | Tell the TUI to exit gracefully.            |

## Direction: TUI → Python (upstream)

Events the TUI emits when the user interacts.

| `type`                | `data` fields                                | Meaning                                              |
| --------------------- | -------------------------------------------- | ---------------------------------------------------- |
| `submit`              | `text`                                       | User submitted a message (or slash command).         |
| `permission_response` | `id`, `allow` (`bool`)                       | Response to a `permission_request`.                  |
| `set_permission_mode` | `mode` (`default` / `accept_edits` / `plan`) | Shift+Tab cycle pushed a new mode.                   |
| `list_files`          | `request_id`, `prefix`, `limit?`             | `@`-completion lookup; bridge replies with `files`.  |
| `interrupt`           | `{}`                                         | User pressed esc / ctrl+c during a turn — cancel.    |
| `quit`                | `{}`                                         | User asked to exit; Python should tear down cleanly. |
| `ready`               | `{}`                                         | TUI finished mounting — Python may now send `intro`. |

## Lifecycle

1. Python spawns `miniadk-tui` with the user's tty inherited.
2. TUI mounts, sends `{"type":"ready"}`.
3. Python sends `{"type":"intro", ...}`.
4. Loop:
   * TUI sends `{"type":"submit","data":{"text":"..."}}` whenever the
     user hits enter.
   * Python runs the runtime turn; each event from the iterator becomes
     one downstream JSON line.
   * If a guard requires permission, Python sends `permission_request`
     and awaits a `permission_response` (matched by `id`).
5. When the user types `/exit` or hits ctrl+q, the TUI sends `quit` and
   exits. Python tears down the subprocess.

## Forward compatibility

* TUI **must ignore unknown `type` values** to allow Python to add new
  event kinds without bumping the protocol version.
* Python **must ignore unknown TUI events** for the same reason.
* When breaking changes are needed, bump the file name to `protocol-v1.md`
  and add a `protocol_version` to the `intro` event.

## Debug logging

Set `MINIADK_TUI_DEBUG=/path/to/log.jsonl` before launching the Python
side. The bridge will write one line per event, in both directions,
each prefixed with a UNIX timestamp. Useful for repro reports.

## Limits & timeouts

MiniADK is built for agents — task complexity comes from the task, not
the framework. We deliberately avoid arbitrary caps where they don't
buy safety. Defaults reflect 2026 LLM workloads (long-context reasoning,
build/test loops, file editing).

| Setting | Default | Where it lives | Override |
|---|---|---|---|
| HTTP timeout (model requests) | **600s** | `JsonHttpClient.timeout_seconds` | `ANTHROPIC_TIMEOUT`, `OPENAI_TIMEOUT`, `MINIADK_MODEL_TIMEOUT`, or `timeout=` arg |
| `max_tokens` per response | **per-model table** | `_default_max_tokens()` in `models/anthropic.py` | `ANTHROPIC_MAX_TOKENS`, `MINIADK_MODEL_MAX_TOKENS`, or `max_tokens=` arg |
| Web fetch timeout | **120s** | `make_fetch_url()` | `timeout=` arg |
| MCP server timeout | **120s** | `mcp.StdioServer.timeout_seconds` | constructor arg |
| Shell command timeout | **None (unlimited)** | `make_shell()` | pass `timeout_seconds=N` to enforce |
| Agent step cap (`max_steps`) | **None (unlimited)** | `Runtime`, all adapter `run_*` | pass `max_steps=N` to enforce |

**The two `None` defaults are intentional**: shell commands and step
counts vary wildly across legitimate agent tasks (`pytest` may run for
20 minutes; a refactor may need 80 steps). Capping them by default
creates more pain than it prevents — pass an explicit number when you
have a real reason (CI run, untrusted input, demo with budget).

Process-shutdown grace periods (1–2s for SIGTERM → SIGKILL) and TUI UX
timings (ctrl-c window 1.5s, bell threshold 3s, file-list popup 1s) are
deliberate UX/safety choices and aren't user-configurable.

## Slash commands

The Ink runner intercepts every input starting with `/` before it
reaches the LLM. Built-in commands (`/help`, `/status`, `/tools`,
`/skills`, `/clear`, `/reset`, `/new`, `/undo`, `/retry`, `/compact`,
`/exit`, `/quit`) are dispatched in `adapters/tui_ink/runner.py` and emit
plain `notice` / `clear` / `quit` events. Unknown commands surface a
`unknown command: /xxx (type /help)` notice — they never reach the
model.
