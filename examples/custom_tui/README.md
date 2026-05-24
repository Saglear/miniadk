# Custom TUI example

This folder shows how to replace the default MiniADK TUI with your own
React/Ink components. The Python side is unchanged — only the TUI
binary swaps.

## Files

- `tui.tsx` — your custom UI. Imports from `@miniadk/tui` (the public
  API) and consumes the bridge through hooks.
- `package.json` — declares `@miniadk/tui` as a `file:` dependency.
- `run.sh` — small wrapper so `MINIADK_TUI_BIN` can point at a script.

## Run it

```bash
cd examples/custom_tui
bun install                           # one-time, links @miniadk/tui
chmod +x run.sh
MINIADK_TUI_BIN="$(realpath ./run.sh)" \
    uv run python ../real_cli.py
```

When MiniADK launches the TUI, it'll spawn `run.sh`, which `cd`s here
and execs `bun tui.tsx`. Your component receives every transcript /
streaming / permission event through `useBridgeEvents`, and emits user
input via `useBridgeSend`.

## What you'd typically customize

- **Layout**: replace the column-flexbox in `MinimalApp` with whatever
  shape you want.
- **Markdown**: drop the `<Markdown>` component for plain `<Text>` if
  you'd rather render raw output.
- **Theming**: import `theme` from `@miniadk/tui/theme` or pass your
  own colour tokens.
- **Slash commands**: handle them client-side before they reach
  Python by intercepting `/help`, `/exit`, etc. before calling
  `send({ type: "submit", ... })`.

## What you can't (currently) customize from the TS side

Anything that's policy on the ADK side: the model, available tools,
session storage, permission gates. Those live in the Python `Agent`
object you launch with `run_cli`.
