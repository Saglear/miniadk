# MiniADK

[简体中文](README.zh-CN.md)

A small Python Agent Development Kit. Build tool-using agents with a
compact, readable API. The core stays small; providers, tools,
policies, skills, MCP, CLI adapters, and presets live around it.

MiniADK is a development kit, **not a finished agent product**. We
ship the building blocks; you write the smarts.

## Install

```bash
pip install miniadk
```

That's it for the core. The default terminal UI (Ink, written in
TypeScript) is downloaded on first use of `run_cli` and cached in
`~/.cache/miniadk/tui/`. If you'd rather skip the network fetch:

```bash
pip install miniadk[tui-textual]   # pure-Python fallback TUI
```

Then set `MINIADK_TUI_NO_FETCH=1` in your environment, or pass
`backend="textual"` to `run_cli`.

For development:

```bash
git clone https://github.com/Saglear/miniadk.git
cd miniadk
uv sync --extra dev
uv run --extra dev pytest -q
```

## Three quickstarts

### 1. Headless agent (≤ 15 lines)

```py
from miniadk import Agent, run, tool

@tool
def add(a: int, b: int) -> int:
    "Return a + b."
    return a + b

agent = Agent("calc", "Use tools when they help.", tools=[add])
print(run(agent, "What is 17 + 25?"))
```

### 2. Default terminal UI (≤ 10 lines)

```py
from miniadk import Agent, make_tools, run_cli

run_cli(Agent(
    "repo",
    "Answer questions about this repository.",
    tools=make_tools(write=False, shell=False),
))
```

### 3. Custom React/Ink UI (~30 lines, see `examples/custom_tui/`)

```tsx
import { mount, BridgeProvider, useBridgeSend, useBridgeEvents,
         Markdown } from "@miniadk/tui";

function App() {
  const send = useBridgeSend();
  const [items, setItems] = useState([]);
  useBridgeEvents("message", (e) => setItems(p => [...p, e.data.text]));
  return /* …your layout… */;
}

mount((bridge) => <BridgeProvider bridge={bridge}><App/></BridgeProvider>);
```

Point Python at your binary via `MINIADK_TUI_BIN` and `run_cli` does
the rest.

## Models

`model()` reads provider settings from environment variables and
returns a configured adapter:

```txt
ANTHROPIC_API_KEY=...    # or ANTHROPIC_AUTH_TOKEN
ANTHROPIC_BASE_URL=...   # optional, for proxies / Anthropic-compatible APIs
ANTHROPIC_MODEL=claude-opus-4-7
```

```txt
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
OPENAI_MODEL=gpt-5-pro
```

When both are set, choose with `MINIADK_MODEL_PROVIDER=anthropic` (or
`openai`).

## Tools

Any typed Python function with a docstring becomes a tool:

```py
from miniadk import tool

@tool
def now_utc() -> str:
    "Return the current UTC time, ISO-8601."
    from datetime import datetime, UTC
    return datetime.now(UTC).isoformat(timespec="seconds")
```

Sync and async are both fine. The decorator reads the function name,
docstring, and type hints to build the JSON schema.

For batteries-included tools, `make_tools` returns a curated set:

```py
from miniadk import make_tools

tools = make_tools(
    files=True,    # read_file, list_files, glob_files, search_text
    shell=False,   # subprocess.run wrapper
    write=False,   # mutating file ops
    web=True,      # fetch_url
)
```

Each tool keeps its own path checks, permission prompts, and limits —
the runtime stays atomic.

## Composition

`Agent` carries its own `policy` and `middleware`. Adapters never
learn about preset shapes; they call `resolve_composition(agent)`
once.

```py
from miniadk import Agent, RunDecision

class StopAfterThreeTools:
    def __init__(self): self.rounds = 0
    async def after_model(self, state):
        r = state.result
        if r and r.message and not r.tool_calls:
            return RunDecision.stop(r.message)
        return RunDecision()
    async def after_tools(self, state):
        self.rounds += 1
        return RunDecision.stop("hit cap") if self.rounds >= 3 else RunDecision()

agent = Agent("bounded", "Answer briefly.", policy=StopAfterThreeTools())
```

Same pattern for middleware (`before_tool_call` to gate, `after_tool_call`
to log). See `examples/05_middleware.py` and
`examples/08_custom_policy.py`.

## Skills and MCP

Skills are slash-launchable playbooks (prompt + allowed tools). MCP
servers are external tool providers connected over stdio. Both
resolve to plain `Tool`s before the runtime loop runs.

```py
from miniadk import Agent, MCPServer, run_cli, skill
from miniadk.mcp import MCPHub
from miniadk.skills import SkillRegistry

run_cli(Agent(
    "assistant",
    "Use available capabilities.",
    skills=SkillRegistry.from_skills(
        skill("review", "Read $path and summarise.", tools=["read_file"], args=["path"]),
    ),
    mcp=MCPHub([MCPServer(name="docs", command="uvx", args=["some-mcp-server"])]),
))
```

## Examples

A teaching ladder, plus practical tools — see
[`examples/README.md`](examples/README.md) for the index. Highlights:

- `01–05` — Agent, tools, streaming, sessions, middleware (concepts).
- `06–07` — Default CLI; a read-only repo assistant.
- `08` — Write your own `RunPolicy` (this is how you'd build ReAct or
  Plan-and-Execute, instead of importing a preset).
- `09–10` — MCP client; slash-skill router.
- `custom_tui/` — Replace the entire terminal UI with your own React
  components while reusing the bridge.

```bash
uv run python examples/01_hello_agent.py
uv run python examples/06_run_cli.py
```

## Architecture

Two-layer separation:

```
adapters/      tui_ink   tui_textual   json   web   ws    (lazy, opt-in)
core/          Agent  Tool  Model  Runtime  Session  Event  RunPolicy
```

`import miniadk` loads only the core — no Textual, no Ink, no React.
TUI deps are resolved on attribute access. See
[`docs/architecture.md`](docs/architecture.md) for layering and
extension points, and [`docs/tui-protocol.md`](docs/tui-protocol.md)
for the JSON wire format between Python and the Ink subprocess.

## What MiniADK does *not* provide

By design — these belong to your application, not the framework:

- No agent-loop preset zoo (ReAct, Plan-and-Execute, Tree-of-Thought,
  reflection). Compose them out of `RunPolicy` + middleware.
- No prompt template library. Strings work fine.
- No retrieval / vector store integration. Wrap one in a `Tool` if
  you need it.

The `agentic()` preset is the **one** opinionated composition we
ship. Treat it as an example you can read and copy, not as the
canonical way.

## License

MIT
