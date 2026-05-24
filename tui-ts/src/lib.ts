/**
 * Public API for ``@miniadk/tui``.
 *
 * Two ways to use this package:
 *
 * 1. **Default TUI as a binary.** ``pip install miniadk`` plus
 *    ``miniadk-tui-fetch`` gets you the prebuilt Bun-compiled CLI.
 *    Nothing to import.
 *
 * 2. **Custom TUI.** Build your own Bun script that mounts your own
 *    React tree on top of the bridge:
 *
 *    ```tsx
 *    import { mount, MiniADKApp } from "@miniadk/tui";
 *    mount((bridge) => <MiniADKApp {...bridge} />);
 *    ```
 *
 *    Or compose your UI from the building blocks:
 *
 *    ```tsx
 *    import {
 *      mount, BridgeProvider, useBridgeEvents,
 *      Transcript, PromptInput, StatusBar,
 *    } from "@miniadk/tui";
 *    ```
 *
 * Then point Python at your binary via ``MINIADK_TUI_BIN`` and call
 * ``run_cli`` as usual.
 */

// ── bootstrap & bridge ───────────────────────────────────────────────
export { mount } from "./bootstrap.js";
export type { BridgeApi, MountOptions, DownEvent, UpEvent } from "./bootstrap.js";
export {
  BridgeProvider,
  useBridge,
  useBridgeSend,
  useBridgeEvents,
} from "./hooks/useBridge.js";

// ── default app & components ─────────────────────────────────────────
export { App as MiniADKApp } from "./App.js";
export type { TranscriptItem } from "./protocol.js";

export { Welcome } from "./components/Welcome.js";
export { Transcript } from "./components/Transcript.js";
export { PromptInput } from "./components/PromptInput.js";
export { StatusBar } from "./components/StatusBar.js";
export { ActivityLine } from "./components/ActivityLine.js";
export { PermissionModal } from "./components/PermissionModal.js";
export { Autocomplete } from "./components/Autocomplete.js";
export { CommandPalette } from "./components/CommandPalette.js";
export { Markdown } from "./components/Markdown.js";

// ── hooks ────────────────────────────────────────────────────────────
export { useAutocomplete } from "./hooks/useAutocomplete.js";

// ── theme tokens ─────────────────────────────────────────────────────
export { theme, type Theme } from "./theme.js";
