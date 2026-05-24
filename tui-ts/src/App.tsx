import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Box, Text, useApp, useInput } from "ink";
import {
  DownEvent,
  Intro,
  PendingPermission,
  PermissionMode,
  TranscriptItem,
  UpEvent,
} from "./protocol.js";
import { Transcript } from "./components/Transcript.js";
import { PromptInput } from "./components/PromptInput.js";
import { ActivityLine } from "./components/ActivityLine.js";
import { StatusBar } from "./components/StatusBar.js";
import { PermissionModal } from "./components/PermissionModal.js";
import { Welcome } from "./components/Welcome.js";
import { Autocomplete } from "./components/Autocomplete.js";
import { CommandCatalogItem, useAutocomplete } from "./hooks/useAutocomplete.js";

interface Props {
  send: (event: UpEvent) => void;
  subscribe: (handler: (event: DownEvent) => void) => () => void;
}

const BUILTIN_COMMANDS: CommandCatalogItem[] = [
  { name: "help", description: "show available commands", category: "command" },
  { name: "status", description: "show session info", category: "command" },
  { name: "tools", description: "list tools", category: "command" },
  { name: "skills", description: "list skills", category: "command" },
  { name: "clear", description: "clear transcript", category: "command" },
  { name: "reset", description: "clear conversation history", category: "command" },
  { name: "undo", description: "remove last user turn", category: "command" },
  { name: "retry", description: "rerun last user turn", category: "command" },
  { name: "compact", description: "summarise older turns", category: "command" },
  { name: "exit", description: "leave the session", category: "command" },
];

const PERMISSION_MODE_ORDER: PermissionMode[] = ["default", "accept_edits", "plan"];

// Two-stage ctrl+c: first press clears the draft / cancels the running
// turn; a second press within this many ms truly exits. Mirrors Claude
// Code / standard shells. Long enough that an accidental double-press
// doesn't kill the session, short enough that an intentional second
// press feels immediate.
const CTRL_C_EXIT_WINDOW_MS = 1500;

export const App: React.FC<Props> = ({ send, subscribe }) => {
  const { exit } = useApp();
  const [intro, setIntro] = useState<Intro | null>(null);
  const [items, setItems] = useState<TranscriptItem[]>([]);
  const [activity, setActivity] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [draft, setDraftState] = useState("");
  // Keep an always-current ref to the draft. ink-text-input captures
  // `originalValue` in its useInput closure at render time, so when two
  // keystroke chunks ("hello" + "\r") arrive in the same Ink read loop
  // the second hook call still sees the empty pre-render value. We
  // bypass that by reading from this ref in handleSubmit instead of
  // trusting the argument ink-text-input passes us.
  const draftRef = useRef("");
  const setDraft = useCallback((next: string) => {
    draftRef.current = next;
    setDraftState(next);
  }, []);
  const [pending, setPending] = useState<PendingPermission | null>(null);
  const [streamingAssistant, setStreamingAssistant] = useState<string | null>(null);
  const [showWelcome, setShowWelcome] = useState(true);
  const [tokens, setTokens] = useState<number | null>(null);
  const [permissionMode, setPermissionMode] = useState<PermissionMode>("default");
  const [expandLast, setExpandLast] = useState(false);
  const [history, setHistory] = useState<string[]>([]);
  const [historyCursor, setHistoryCursor] = useState<number | null>(null);
  const [exitArmed, setExitArmed] = useState(false);
  const exitArmedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const toolBuffersRef = useRef<Record<number, { name: string; args: string }>>({});

  // ── files RPC ────────────────────────────────────────────────────────
  const filePromisesRef = useRef<Map<string, (paths: string[]) => void>>(new Map());
  const filesProvider = useCallback(
    async (prefix: string): Promise<string[]> => {
      const requestId = `f${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      return new Promise<string[]>((resolve) => {
        filePromisesRef.current.set(requestId, resolve);
        send({ type: "list_files", data: { request_id: requestId, prefix, limit: 12 } });
        setTimeout(() => {
          if (filePromisesRef.current.delete(requestId)) resolve([]);
        }, 1000);
      });
    },
    [send],
  );

  const autocomplete = useAutocomplete(draft, draft.length, BUILTIN_COMMANDS, filesProvider);

  // The ``ready`` handshake is emitted by ``bootstrap.mount`` so every
  // custom TUI gets it for free — no per-component lifecycle to wire.

  const append = useCallback((item: TranscriptItem) => {
    setItems((prev) => [...prev, item]);
    if (item.kind === "tool_result") setExpandLast(false);
  }, []);

  const handleEvent = useCallback(
    (event: DownEvent) => {
      switch (event.type) {
        case "intro":
          setIntro({
            agent: event.data.agent,
            model: event.data.model,
            cwd: event.data.cwd,
            toolCount: event.data.tool_count,
          });
          if (event.data.permission_mode) setPermissionMode(event.data.permission_mode);
          return;
        case "user":
          setShowWelcome(false);
          append({ kind: "user", text: event.data.text, turn: event.data.turn });
          return;
        case "thinking_delta":
          setActivity("thinking");
          return;
        case "tool_call_delta": {
          const idx = event.data.index ?? 0;
          const cur = toolBuffersRef.current[idx] ?? { name: "", args: "" };
          const name = event.data.name ?? cur.name;
          const args = (cur.args + (event.data.arguments ?? "")).replace(/\n/g, " ");
          toolBuffersRef.current = { ...toolBuffersRef.current, [idx]: { name, args } };
          const label = name || `#${idx}`;
          setActivity(`preparing ${label} ${args.slice(-60)}`);
          return;
        }
        case "tool_call":
          setActivity(null);
          toolBuffersRef.current = {};
          append({
            kind: "tool_call",
            name: event.data.name,
            arguments: event.data.arguments,
          });
          return;
        case "tool_progress":
          append({
            kind: "tool_progress",
            name: event.data.tool,
            message: event.data.message,
          });
          return;
        case "tool_result":
          setActivity(null);
          append({
            kind: "tool_result",
            name: event.data.name,
            text: event.data.text,
          });
          return;
        case "tool_denied":
          append({ kind: "tool_denied", message: event.data.message });
          return;
        case "tool_invalid":
        case "tool_error":
        case "error":
          append({ kind: "error", message: event.data.message });
          return;
        case "message_delta":
          setStreamingAssistant((prev) => (prev ?? "") + event.data.text);
          return;
        case "message": {
          setActivity(null);
          setStreamingAssistant(null);
          if (event.data.text) append({ kind: "assistant", text: event.data.text });
          return;
        }
        case "run_start":
          setBusy(true);
          return;
        case "run_end":
          setBusy(false);
          setActivity(null);
          if (typeof event.data.tokens === "number") setTokens(event.data.tokens);
          return;
        case "permission_request":
          setPending({
            id: event.data.id,
            tool: event.data.tool,
            reason: event.data.reason,
            arguments: event.data.arguments,
          });
          return;
        case "permission_mode_changed":
          setPermissionMode(event.data.mode);
          return;
        case "files": {
          const resolver = filePromisesRef.current.get(event.data.request_id);
          if (resolver) {
            filePromisesRef.current.delete(event.data.request_id);
            resolver(event.data.paths);
          }
          return;
        }
        case "notice":
          append({ kind: "notice", text: event.data.text });
          return;
        case "clear":
          setItems([]);
          setShowWelcome(true);
          return;
        case "quit":
          exit();
          return;
      }
    },
    [append, exit],
  );

  useEffect(() => {
    const off = subscribe(handleEvent);
    return off;
  }, [subscribe, handleEvent]);

  // ── two-stage ctrl+c helpers ─────────────────────────────────────────
  const armExit = useCallback(() => {
    setExitArmed(true);
    if (exitArmedTimerRef.current) clearTimeout(exitArmedTimerRef.current);
    exitArmedTimerRef.current = setTimeout(() => {
      setExitArmed(false);
      exitArmedTimerRef.current = null;
    }, CTRL_C_EXIT_WINDOW_MS);
  }, []);

  const disarmExit = useCallback(() => {
    setExitArmed(false);
    if (exitArmedTimerRef.current) {
      clearTimeout(exitArmedTimerRef.current);
      exitArmedTimerRef.current = null;
    }
  }, []);

  // ── global key bindings ──────────────────────────────────────────────
  useInput((input, key) => {
    // ctrl+d: hard exit, no second confirmation needed (it's the
    // "I really mean it" key).
    if (key.ctrl && input === "d") {
      send({ type: "quit", data: {} });
      exit();
      return;
    }

    // ctrl+c: two-stage. First press cancels in-flight work / clears the
    // draft. Second press within CTRL_C_EXIT_WINDOW_MS exits.
    if (key.ctrl && input === "c") {
      if (exitArmed) {
        send({ type: "quit", data: {} });
        exit();
        return;
      }
      // First press: clear or cancel.
      if (busy) {
        send({ type: "interrupt", data: {} });
      }
      if (draft.length > 0) {
        setDraft("");
      }
      armExit();
      return;
    }

    // Esc: cancel a running turn, or close autocomplete.
    if (key.escape) {
      if (autocomplete.state.visible) {
        // Let the input swallow Esc to close the dropdown — but we don't
        // have a hook here, so just clear by clearing the trigger via
        // typing a space then deleting. Simplest: do nothing; user can
        // backspace the trigger char.
        return;
      }
      if (busy) {
        send({ type: "interrupt", data: {} });
        return;
      }
    }

    if (key.shift && key.tab) {
      const i = PERMISSION_MODE_ORDER.indexOf(permissionMode);
      const next = PERMISSION_MODE_ORDER[(i + 1) % PERMISSION_MODE_ORDER.length];
      setPermissionMode(next);
      send({ type: "set_permission_mode", data: { mode: next } });
      disarmExit();
      return;
    }

    if (key.ctrl && input === "r") {
      setExpandLast((v) => !v);
      disarmExit();
      return;
    }

    if (key.ctrl && input === "l") {
      // ctrl+l clears the visible transcript locally. Conversation
      // history on the Python side is unaffected.
      setItems([]);
      setShowWelcome(true);
      disarmExit();
      return;
    }

    // Autocomplete navigation — only when visible.
    if (autocomplete.state.visible && !pending) {
      if (key.upArrow) {
        autocomplete.setCursor(autocomplete.state.cursor - 1);
        disarmExit();
        return;
      }
      if (key.downArrow) {
        autocomplete.setCursor(autocomplete.state.cursor + 1);
        disarmExit();
        return;
      }
      if (key.tab) {
        const next = autocomplete.apply();
        if (next !== null) setDraft(next);
        disarmExit();
        return;
      }
    }

    // History recall — Up/Down on an empty draft (or while drafting
    // something we previously sent). Only applies when not busy and
    // autocomplete isn't visible.
    if (!busy && !pending && !autocomplete.state.visible) {
      if (key.upArrow && history.length > 0) {
        const next = historyCursor === null ? history.length - 1 : Math.max(0, historyCursor - 1);
        setHistoryCursor(next);
        setDraft(history[next] ?? "");
        disarmExit();
        return;
      }
      if (key.downArrow && historyCursor !== null) {
        const next = historyCursor + 1;
        if (next >= history.length) {
          setHistoryCursor(null);
          setDraft("");
        } else {
          setHistoryCursor(next);
          setDraft(history[next] ?? "");
        }
        disarmExit();
        return;
      }
    }

    // Any other meaningful keystroke disarms the pending exit.
    if (input || key.return || key.delete || key.backspace) {
      disarmExit();
    }
  });

  const handleSubmit = useCallback(
    (text: string) => {
      // Enter always submits. Tab accepts the highlighted completion —
      // see the autocomplete branch in `useInput`. Earlier we tried to
      // make Enter accept-then-submit, but that traps fully-typed
      // commands like `/help` because `apply()` returns the same string,
      // so Enter would loop without ever sending.
      //
      // We trust `draftRef` over the `text` arg because ink-text-input's
      // useInput captures originalValue in a closure at render time. If
      // the user's chunk arrives merged ("hello\r"), Ink runs both the
      // text-insert and the key.return callbacks before React commits a
      // re-render, so `text` is the stale pre-insert value (empty).
      const live = draftRef.current || text;
      const trimmed = live.trim();
      if (!trimmed || busy) return;
      send({ type: "submit", data: { text: trimmed } });
      setHistory((prev) => {
        const next = prev.filter((entry) => entry !== trimmed);
        next.push(trimmed);
        // Cap history to keep memory predictable.
        if (next.length > 200) next.shift();
        return next;
      });
      setHistoryCursor(null);
      setDraft("");
      disarmExit();
    },
    [busy, send, disarmExit],
  );

  const handlePermission = useCallback(
    (allow: boolean) => {
      if (!pending) return;
      send({ type: "permission_response", data: { id: pending.id, allow } });
      setPending(null);
    },
    [pending, send],
  );

  const handleDraftChange = useCallback((next: string) => {
    setDraft(next);
    setHistoryCursor(null);
  }, [setDraft]);

  if (!intro) {
    return (
      <Box paddingX={2} paddingY={1}>
        <Text dimColor>booting…</Text>
      </Box>
    );
  }

  const busyHint = busy
    ? "running… esc to cancel · ctrl+c to clear"
    : pending
    ? "waiting for permission decision"
    : undefined;

  return (
    <Box flexDirection="column">
      {showWelcome && items.every((item) => item.kind === "notice") && (
        <Welcome intro={intro} />
      )}
      <Transcript
        items={items}
        streamingAssistant={streamingAssistant}
        expandLast={expandLast}
      />
      <ActivityLine text={activity} />
      {pending && (
        <PermissionModal request={pending} onResolve={handlePermission} />
      )}
      <Autocomplete
        items={autocomplete.state.items}
        cursor={autocomplete.state.cursor}
        visible={autocomplete.state.visible && !pending}
      />
      <PromptInput
        value={draft}
        onChange={handleDraftChange}
        onSubmit={handleSubmit}
        disabled={busy || !!pending}
        busyHint={busyHint}
        focused={!pending}
      />
      <StatusBar
        agent={intro.agent}
        model={intro.model}
        cwd={intro.cwd}
        permissionMode={permissionMode}
        tokens={tokens}
        exitHint={exitArmed ? "press ctrl+c again to exit" : null}
      />
    </Box>
  );
};
