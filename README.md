# MiniADK

[简体中文](README.zh-CN.md)

MiniADK is a small Python Agent Development Kit for building tool-using agents
with a compact, readable API.

It gives you the pieces needed to build agent products:

- `Agent` for instructions and capabilities
- `Model` adapters for LLM providers
- `Tool` wrappers for Python functions
- `Runtime` for the agent loop
- `Event` streams for adapters and UIs
- `Session` helpers for conversation state

MiniADK is a development kit, not a finished agent product. The core stays
small, while providers, tools, policies, skills, MCP, CLI adapters, and presets
live around it.

## Install

Clone the source and install it in editable mode:

```bash
git clone https://github.com/Saglear/miniadk.git
cd miniadk
uv sync --extra dev
```

Run the test suite:

```bash
uv run --extra dev pytest -q
```

## Quick Start

```py
from miniadk import Agent, model, run_cli, tool


@tool
def add(left: int, right: int) -> int:
    """Add two numbers."""
    return left + right


agent = Agent(
    "calc",
    "Use tools when they help.",
    tools=[add],
)

run_cli(agent, model=model())
```

Run it:

```bash
uv run python calc.py
```

## One-Shot Calls

```py
from miniadk import Agent, model, run, tool


@tool
def greet(name: str) -> str:
    """Return a greeting."""
    return f"hello {name}"


agent = Agent("hello", "Use tools when useful.", tools=[greet])
answer = run(agent, "Greet Ada", model=model())
print(answer)
```

## Models

`model()` reads provider settings from environment variables and returns a
configured adapter.

OpenAI-compatible settings:

```txt
OPENAI_KEY=...
OPENAI_URL=...
OPENAI_BASE_URL=...
OPENAI_MODEL=...
```

Anthropic settings:

```txt
ANTHROPIC_KEY=...
ANTHROPIC_URL=...
ANTHROPIC_BASE_URL=...
ANTHROPIC_MODEL=...
```

When more than one provider is configured, choose the default explicitly:

```txt
MINIADK_MODEL_PROVIDER=openai
```

or:

```txt
MINIADK_MODEL_PROVIDER=anthropic
```

## Tools

Any typed Python function can become a tool:

```py
from pathlib import Path

from miniadk import tool


@tool
def read_note(path: str) -> str:
    """Read a UTF-8 note."""
    return Path(path).read_text(encoding="utf-8")
```

MiniADK uses the function name, docstring, and type hints to build the tool
schema. Sync and async functions are both supported.

## Prebuilt Tools

Reusable tools are available from `miniadk.stdtools`:

```py
from pathlib import Path

from miniadk import Agent, model, run_cli
from miniadk.stdtools import make_list_files, make_read_file, make_search_text

root = Path.cwd()
agent = Agent(
    "repo",
    "Help inspect this repository.",
    tools=[
        make_list_files(root=root),
        make_read_file(root=root),
        make_search_text(root=root),
    ],
)

run_cli(agent, model=model())
```

File and shell tools keep path checks, permission prompts, limits, and timeouts
outside the atomic runtime.

## Skills And MCP

Skills and MCP servers are business-layer integrations. They resolve into
ordinary instructions and tools before the runtime loop runs.

```py
from miniadk import Agent, MCPHub, MCPServer, SkillRegistry, model, run_cli

agent = Agent(
    "assistant",
    "Use the configured project capabilities.",
    skills=SkillRegistry.from_paths(".miniadk/skills"),
    mcp=MCPHub([
        MCPServer(name="docs", command="uvx", args=["some-mcp-server"]),
    ]),
)

run_cli(agent, model=model())
```

CLI rendering is replaceable for products that want their own terminal style:

```py
from miniadk import CLIRenderer, run_cli

run_cli(agent, renderer=CLIRenderer(print, mode="pretty"))
```

In an interactive terminal, the default Python CLI uses history, slash-command
completion, multiline editing, and streaming event rendering. Products can keep
that default or replace the renderer/input layer when they need a custom
terminal experience.

## Core Shape

The runtime loop is intentionally direct:

```txt
user message
  -> model call
  -> optional tool calls
  -> tool results
  -> model response
  -> events
```

The core concepts are:

```txt
Message  - what the agent sees
Model    - how the agent asks an LLM what to do next
Tool     - what the agent can do
Agent    - instructions plus capabilities
Runtime  - the loop that connects everything
Event    - what adapters and UIs observe
Session  - persisted conversation state
```

## Package Layout

```txt
src/miniadk/core/       atomic runtime types and loop
src/miniadk/models/     provider adapters
src/miniadk/stdtools/   reusable file, shell, web, and agent tools
src/miniadk/adapters/   CLI, JSON, web, and WebSocket adapters
src/miniadk/skills.py   skill loading and invocation helpers
src/miniadk/mcp.py      MCP integration
src/miniadk/presets.py  optional high-level assembly helpers
```

Import the common user-facing API from `miniadk`. Use submodules when you need
advanced control.

## Examples

```bash
uv run --extra dev python examples/smoke_llm.py
uv run --extra dev python examples/scripted_tiny_product.py
uv run --extra dev python examples/coder_preset.py
uv run --extra dev python examples/compact_coder.py
uv run --extra dev python examples/repo_cli.py
uv run --extra dev python examples/cli_interaction_lab.py
```

## Development

```bash
uv sync --extra dev
uv run --extra dev pytest -q
uv build
```

## License

MIT
