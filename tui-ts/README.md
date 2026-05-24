# Ink TUI for MiniADK

This is the polished terminal UI shipped with MiniADK. It runs as a
subprocess of the Python ADK; communication happens over stdin/stdout
JSON lines (see `../docs/tui-protocol.md`).

## Why TS / Ink

We ran into hard ceilings hand-rolling a Python TUI on top of
`prompt_toolkit + rich` and again on top of `textual`. The look-and-feel
of Claude Code / opencode / codex / hermes / codewhale all comes from
the same place: a React-style virtual DOM diffed at 60fps over the
terminal. Ink is the React-for-terminal that powers Claude Code itself,
so we use it directly here and bridge to it from Python.

## Develop

```sh
bun install
bun run dev
```

This launches the TUI without a Python parent. Useful for visual review;
input/output JSON lines are echoed.

When the Python ADK launches the TUI, it spawns this process with the
user's tty inherited, then drives it with JSON-RPC events. The Python
side is in `src/miniadk/adapters/tui_bridge.py`.

## Build

```sh
bun run build           # current platform
bun run build-all       # all five platforms (linux x64/arm64, mac x64/arm64, windows x64)
```

Output goes to `dist/`. Each binary is ~50–90 MB (bun runtime is
embedded). The Python wheel ships the matching binary for the user's
platform via PEP 425 platform tags.
