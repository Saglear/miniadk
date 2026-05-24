# MiniADK Architecture

A short map of where things live and where you'd extend.

## Layers

```
┌──────────────────────────────────────────────────────┐
│  miniadk.adapters.*                                  │
│   tui_ink/   tui_textual/   json   web   ws          │
│   (TUI / API surfaces. Lazy-loaded; opt-in.)         │
└──────────────────────────────────────────────────────┘
                  ▲
                  │  resolve_composition / runtime events
                  ▼
┌──────────────────────────────────────────────────────┐
│  miniadk.core                                        │
│   Agent · Tool · Model · Runtime · Session           │
│   Event · RunPolicy · Middleware                     │
│   (Zero-dep building blocks. Always loaded.)         │
└──────────────────────────────────────────────────────┘
```

The line between **core** and **adapters** is a hard one:

- `import miniadk` loads only `miniadk.core` and pure-Python helpers.
  No Textual, no Ink, no React. Verified by
  `tests/test_import_isolation.py`.
- Anything that would pull a heavy dep is reached through
  `__getattr__` and only resolved on attribute access.

## Composition (presets, agentic, custom plugins)

`Agent` carries its own `policy` and `middleware`. Adapters do **not**
inspect preset-specific shapes. They call `resolve_composition` once,
which:

- accepts a bare `Agent` or any wrapper exposing `.agent` / `.policy` /
  `.middleware` (e.g. the legacy `Agentic` struct);
- prepends agent-level middleware so guards stay in front of caller
  decorators;
- returns `(agent, middleware, policy)` for the runtime.

This is the single extension point for "agent + behaviour" composition.
Custom presets should:

1. Construct an `Agent` with the right tools, instructions, policy,
   and middleware fields filled in.
2. Optionally wrap it in your own struct with `.agent` if you need to
   expose a side channel (e.g. a shared todo store, an observability
   hook).

If you find yourself inventing a new subclass or adding a special path
to an adapter, the abstraction is wrong — push the missing capability
into `Agent` / `Runtime` first, then rebuild the preset on top.

## Extension points

### Add a tool

```py
from miniadk import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
```

That's the whole API. Pass it to `Agent(tools=[add])`. There is no
"tool registry" you need to know about.

### Add middleware (before/after model + tool calls)

Implement any subset of:

- `before_model_call(state)`
- `after_model_call(state, result)`
- `before_tool_call(state, call)`
- `after_tool_call(state, call, result)`

Pass `Agent(middleware=[YourMW()])`. Middleware ordering is preserved.
Adapter callers can also pass extra `middleware=` to `arun` / `run_cli`
— they're appended after the agent's own.

### Customize the loop policy

Implement `RunPolicy.decide(state) -> RunDecision`. The default is
`DefaultRunPolicy` (run until the model stops or step cap hits). The
agentic preset's `AgenticPolicy` is an example of a more opinionated
loop.

### Customize the TUI

#### Slash commands (Textual backend)

```py
from miniadk.adapters.tui_textual import Command, builtin_commands
```

Register a `Command` on a fresh registry, pass to `MiniADKApp(commands=...)`.

#### Replace the React tree (Ink backend)

```tsx
import { mount } from "@miniadk/tui";
mount((bridge) => <YourApp {...bridge} />);
```

`bridge.send` pushes `UpEvent`s to Python; `bridge.subscribe` (or the
`useBridgeEvents` hook) receives `DownEvent`s. The wire protocol is in
[`docs/tui-protocol.md`](tui-protocol.md). Point Python at your binary
via `MINIADK_TUI_BIN` and `run_cli` does the rest. See
`examples/custom_tui/` for a working ~80-line replacement.

## What MiniADK does *not* provide

By design:

- No agent-loop preset zoo (ReAct, Plan-and-Execute, Tree-of-Thought…).
  Build them out of `RunPolicy` + middleware if you want them.
- No prompt template library. Strings work fine.
- No reflection / self-critique scaffolds.
- No retrieval / vector store integration. Wrap one in a `Tool` if you
  need it.

The `agentic()` preset is the **one** opinionated composition we ship.
It exists because it pulls its weight as a CLI starting point. Treat
it as an example you can read and copy, not as the canonical way.

## File map

| Path | What lives there |
|---|---|
| `src/miniadk/core/` | `Agent`, `Runtime`, `Tool`, `Model`, `Session`, `Event` — zero-dep core |
| `src/miniadk/models/` | Provider adapters (`anthropic.py`, `openai.py`, `factory.py`) |
| `src/miniadk/stdtools/` | Bundled tools (`shell`, `kit`, `web`, `agents`) |
| `src/miniadk/adapters/tui_ink/` | Ink TUI bridge + Python-side runner |
| `src/miniadk/adapters/tui_textual/` | Textual TUI app, screens, widgets |
| `src/miniadk/adapters/{json,web,ws}.py` | Streaming / web / WS surfaces |
| `src/miniadk/agentic.py` | The one preset — todo loop + chat instructions |
| `src/miniadk/skills.py`, `mcp.py` | Skill registration & MCP client |
| `tui-ts/src/` | Default Ink TUI; published as `@miniadk/tui` |
| `examples/custom_tui/` | Custom-TUI replacement template |
| `docs/tui-protocol.md` | Wire format between Python and the Ink subprocess |
