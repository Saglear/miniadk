# MiniADK Examples

A short tour. Each file is self-contained and meant to be read in
order — concepts stack from the simplest to the most useful.

| # | File | Teaches |
|---|------|---------|
| 01 | [`01_hello_agent.py`](01_hello_agent.py) | The smallest possible agent: instructions, `run`, done. |
| 02 | [`02_with_a_tool.py`](02_with_a_tool.py) | `@tool` — a Python function the model can call. |
| 03 | [`03_streaming.py`](03_streaming.py) | Iterate over `Runtime.run()` events: tokens, tool calls, results. |
| 04 | [`04_session_and_compaction.py`](04_session_and_compaction.py) | Persist a conversation across runs with `session=`. |
| 05 | [`05_middleware.py`](05_middleware.py) | Audit and gate tool calls via `before_tool_call`. |
| 06 | [`06_run_cli.py`](06_run_cli.py) | The default terminal UI in 15 lines. |
| 07 | [`07_repo_assistant.py`](07_repo_assistant.py) | A read-only repo helper using `make_tools(...)`. |
| 08 | [`08_custom_policy.py`](08_custom_policy.py) | A custom `RunPolicy` — bound the loop, write your own ReAct. |
| 09 | [`09_mcp_client.py`](09_mcp_client.py) | Borrow tools from an external MCP server over stdio. |
| 10 | [`10_skills_router.py`](10_skills_router.py) | Register slash-commands the user can launch from the CLI. |
| — | [`custom_tui/`](custom_tui/) | Replace the entire React/Ink UI in ~80 lines while reusing the bridge. |

## Running them

Most files just need an LLM key in your environment (or a nearby `.env`):

```bash
export ANTHROPIC_API_KEY=...           # or OPENAI_API_KEY=...
uv run python examples/01_hello_agent.py
```

For the CLI examples (`06`, `07`, `10`), MiniADK will lazily download the
Ink TUI binary on first run (~70 MB, cached at
`~/.cache/miniadk/tui/`). No extra steps needed. Set
`MINIADK_TUI_NO_FETCH=1` to skip the download and use the Textual
fallback (`pip install miniadk[tui-textual]`).

## What you should walk away with

- **Tools are functions.** No registry to learn — just `@tool`.
- **Composition lives on `Agent`.** `policy=`, `middleware=`, `tools=`
  is enough for any workflow you'd otherwise reach for a "framework"
  to do.
- **Adapters are thin.** `run`, `arun`, `run_cli`, `astream_json`,
  `ws_json` all take the same `Agent`; nothing about the runtime
  knows about preset shapes.
- **The Ink TUI is replaceable.** See `custom_tui/` — Python doesn't
  change at all when you swap the React tree.

These examples deliberately stop short of "look how powerful": the
point of MiniADK is that you write the smarts. The presets here are
example compositions, not the framework.
