# MiniADK Examples

These are small agent products built with MiniADK.

They are intentionally outside the core package. MiniADK is the ADK; examples
are products built on top of it.

## Environment

Examples that use `model()` read `.env` from this directory or any parent
directory automatically. Examples that instantiate provider adapters directly
load `.env` explicitly before construction.

Supported variables:

```txt
OPENAI_KEY=...
OPENAI_URL=...
OPENAI_BASE_URL=...
OPENAI_MODEL=...

ANTHROPIC_KEY=...
ANTHROPIC_URL=...
ANTHROPIC_BASE_URL=...
ANTHROPIC_MODEL=...
```

Examples that call `model()` pick Anthropic when Anthropic keys are present,
otherwise they use the OpenAI-compatible adapter. `smoke_llm.py` uses that
single default path, so only one provider needs to be configured for the smoke
test.

## Run

From the `miniadk` directory:

```bash
uv run --extra dev python examples/openai_file_assistant.py
uv run --extra dev python examples/anthropic_file_assistant.py
uv run --extra dev python examples/coder_preset.py
uv run --extra dev python examples/compact_coder.py
uv run --extra dev python examples/repo_cli.py
uv run --extra dev python examples/cli_interaction_lab.py
```

The file assistant examples are deliberately small. They show how a real
product can assemble an agent from MiniADK pieces without changing the ADK
itself.

`coder_preset.py` is the shortest preset path. It packages common defaults
while still letting users replace tools, skills, models, permissions, and
adapters.

`compact_coder.py` keeps a coding-agent product on one screen. It combines
workspace tools, project skills, foreground/background reviewer agents, todos,
and session persistence through the `coder()` preset.

`repo_cli.py` is a larger repository-assistant example. It uses:

- `stdtools` for file and shell actions
- `AgenticPolicy` and `todo_write` for continuation
- local `SKILL.md` files
- `/skill` style invocation in the CLI
- the short `model()` helper for OpenAI-compatible or Anthropic adapters
- the same atomic runtime core, with business-layer policy

`cli_interaction_lab.py` is the maintained CLI interaction playground. It uses
the real model configured in `.env` and should be updated whenever the CLI
surface changes.

## Boundary

These examples are products. MiniADK itself is not the product.

The package provides:

- runtime
- loop policy hooks
- models
- tools
- middleware
- event stream
- adapters

The examples choose:

- instructions
- which tools to expose
- which model adapter to use
- how to run the CLI
