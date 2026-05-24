/**
 * Minimal custom TUI built on @miniadk/tui's public API.
 *
 * Demonstrates how a front-end-experienced user can replace the
 * default UI with ~50 lines of Ink. All transcript / streaming /
 * permission logic comes from the bridge — we only describe layout.
 *
 * Run from this folder:
 *
 *     bun install
 *     MINIADK_TUI_BIN="$(realpath .)/run.sh" \
 *         uv run python ../real_cli.py
 *
 * (run.sh just `cd`s here and execs `bun src/tui.tsx`.)
 */

import React, { useRef, useState } from "react";
import { Box, Text, useInput } from "ink";
import TextInput from "ink-text-input";
import {
  mount,
  BridgeProvider,
  useBridgeSend,
  useBridgeEvents,
  Markdown,
  type TranscriptItem,
} from "@miniadk/tui";

interface IntroData {
  agent: string;
  model: string;
}

function MinimalApp() {
  const send = useBridgeSend();
  const [intro, setIntro] = useState<IntroData | null>(null);
  const [items, setItems] = useState<TranscriptItem[]>([]);
  const [streaming, setStreaming] = useState("");
  const [draft, setDraftState] = useState("");
  const [busy, setBusy] = useState(false);

  // ink-text-input's onSubmit captures `originalValue` in a useInput
  // closure at render time. When the kernel merges keystrokes
  // ("hello\r" arriving as one chunk), the second hook callback fires
  // before React re-renders, so onSubmit sees the stale empty value.
  // We mirror state into a ref that's updated synchronously in
  // setDraft so the submit handler always reads the live text.
  const draftRef = useRef("");
  const setDraft = (next: string) => {
    draftRef.current = next;
    setDraftState(next);
  };

  useBridgeEvents("intro", (e) => setIntro(e.data));
  useBridgeEvents("user", (e) => {
    setItems((prev) => [...prev, { kind: "user", text: e.data.text }]);
  });
  useBridgeEvents("message_delta", (e) => setStreaming((p) => p + e.data.text));
  useBridgeEvents("message", (e) => {
    setStreaming("");
    setItems((prev) => [...prev, { kind: "assistant", text: e.data.text }]);
  });
  useBridgeEvents("run_start", () => setBusy(true));
  useBridgeEvents("run_end", () => setBusy(false));
  useBridgeEvents("error", (e) =>
    setItems((prev) => [...prev, { kind: "error", text: e.data.message }]),
  );

  useInput((_, key) => {
    if (key.ctrl && key.delete) send({ type: "quit", data: {} });
  });

  const submit = () => {
    const text = draftRef.current;
    if (!text.trim() || busy) return;
    send({ type: "submit", data: { text } });
    setDraft("");
  };

  return (
    <Box flexDirection="column" padding={1}>
      <Text color="magenta" bold>
        ▎ {intro ? `${intro.agent} · ${intro.model}` : "loading…"}
      </Text>
      <Box flexDirection="column" marginTop={1}>
        {items.map((item, i) => (
          <Box key={i} marginBottom={1}>
            <Text
              color={
                item.kind === "user" ? "cyan" : item.kind === "error" ? "red" : "white"
              }
              bold={item.kind !== "assistant"}
            >
              {item.kind === "user" ? "› " : item.kind === "error" ? "× " : "‹ "}
            </Text>
            <Box flexGrow={1}>
              {item.kind === "assistant" ? (
                <Markdown text={item.text} />
              ) : (
                <Text>{item.text}</Text>
              )}
            </Box>
          </Box>
        ))}
        {streaming ? (
          <Box>
            <Text color="white">‹ </Text>
            <Markdown text={streaming} />
          </Box>
        ) : null}
      </Box>
      <Box marginTop={1}>
        <Text color={busy ? "yellow" : "green"}>{busy ? "⏳ " : "❯ "}</Text>
        <TextInput value={draft} onChange={setDraft} onSubmit={submit} />
      </Box>
    </Box>
  );
}

mount((bridge) => (
  <BridgeProvider bridge={bridge}>
    <MinimalApp />
  </BridgeProvider>
));
