/**
 * Subprocess bootstrap helpers for a MiniADK Ink TUI.
 *
 * A custom TUI typically just wants:
 *
 *     import { mount } from "@miniadk/tui/bootstrap";
 *     import { App } from "./MyApp";
 *     mount((bridge) => <App {...bridge} />);
 *
 * `mount` handles the JSON-over-stdio wiring with the Python parent
 * (DownEvent stream in, UpEvent stream out), the ``/dev/tty`` raw-mode
 * setup for keystrokes, and the chunk-splitter that prevents merged
 * input chunks (e.g. ``"你好\r"``) from breaking ``ink-text-input``.
 *
 * What it does NOT do: anything visual. Layout, theming, command
 * handling, transcript rendering — all up to the React tree you
 * provide.
 */

import React from "react";
import { render } from "ink";
import chalk from "chalk";
import { createInterface } from "node:readline";
import { createWriteStream, openSync } from "node:fs";
import { ReadStream as TTYReadStream, WriteStream as TTYWriteStream } from "node:tty";
import { PassThrough } from "node:stream";
import type { DownEvent, UpEvent } from "./protocol.js";

export type { DownEvent, UpEvent };

// Force chalk to a real colour level. When the TUI is launched as a
// subprocess by the Python parent, ``process.stdout`` is the JSON pipe
// (not a TTY) so chalk's auto-detection falls back to level 0 — which
// silently strips every ANSI sequence we generate. Ink itself writes
// to a separate ``/dev/tty`` stream, so the terminal does support
// colours. We fix the asymmetry once at bootstrap so chalk-driven
// styles (markdown bold, inverse cursor, syntax highlighting) all
// render correctly. Users that genuinely want plain output can set
// ``FORCE_COLOR=0``.
if (process.env.FORCE_COLOR === undefined) {
  chalk.level = 3;
}

/** Bridge to the Python ADK parent. Pass to your root component. */
export interface BridgeApi {
  /** Send an UpEvent (user input, permission response, etc.) upstream. */
  send: (event: UpEvent) => void;
  /** Subscribe to DownEvents from Python. Returns an unsubscribe fn. */
  subscribe: (handler: (event: DownEvent) => void) => () => void;
}

export interface MountOptions {
  /**
   * If true (default), synthesise a minimal ``intro`` event when the
   * process is launched standalone (``bun src/index.tsx``) so the UI
   * has something to show without a Python parent. Set to false to opt
   * out — useful when your custom UI handles dev-mode itself.
   */
  devIntro?: boolean | (() => DownEvent);
  /**
   * Override the TTY input stream used for keystrokes. Defaults to
   * ``process.stdin`` when it's a TTY, or a freshly-opened
   * ``/dev/tty`` otherwise.
   */
  ttyIn?: NodeJS.ReadStream;
  /**
   * Override the TTY output stream used for rendering. Defaults to
   * ``process.stdout`` when it's a TTY, or a freshly-opened
   * ``/dev/tty`` otherwise.
   */
  ttyOut?: NodeJS.WriteStream;
}

/**
 * Mount a React tree as a MiniADK TUI subprocess.
 *
 * @param renderRoot Receives the bridge and returns the React element
 *                   to render. The bridge can be threaded into your
 *                   own components or passed to ``BridgeProvider`` if
 *                   you prefer the context-based API.
 * @param options    Optional bootstrap overrides.
 */
export function mount(
  renderRoot: (bridge: BridgeApi) => React.ReactElement,
  options: MountOptions = {},
): void {
  const handlers = new Set<(event: DownEvent) => void>();

  // ── output channel: explicit fd from parent, else stdout ───────────
  const outputFdEnv = process.env.MINIADK_TUI_OUTPUT_FD;
  const outputFd = outputFdEnv ? parseInt(outputFdEnv, 10) : NaN;
  const outChannel: NodeJS.WritableStream =
    Number.isFinite(outputFd) && outputFd > 0
      ? createWriteStream("", { fd: outputFd })
      : process.stdout;

  const send: BridgeApi["send"] = (event) => {
    outChannel.write(JSON.stringify(event) + "\n");
  };

  const subscribe: BridgeApi["subscribe"] = (handler) => {
    handlers.add(handler);
    return () => {
      handlers.delete(handler);
    };
  };

  // ── stdin reader: only when stdin is a pipe (subprocess mode) ──────
  const stdinIsTTY = Boolean((process.stdin as { isTTY?: boolean }).isTTY);
  if (!stdinIsTTY) {
    const rl = createInterface({ input: process.stdin });
    rl.on("line", (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      let event: DownEvent;
      try {
        event = JSON.parse(trimmed) as DownEvent;
      } catch {
        process.stderr.write(`miniadk-tui: bad json: ${trimmed.slice(0, 80)}\n`);
        return;
      }
      for (const handler of handlers) handler(event);
    });
  }

  // ── tty streams (raw-mode capable) ─────────────────────────────────
  const ttyIn = options.ttyIn ?? defaultTtyIn(stdinIsTTY);
  const ttyOut = options.ttyOut ?? defaultTtyOut();
  wireResize(ttyOut);
  const inkStdin = chunkSplittingProxy(ttyIn);

  // ── dev-mode synthetic intro ───────────────────────────────────────
  if (stdinIsTTY && options.devIntro !== false) {
    setTimeout(() => {
      const event =
        typeof options.devIntro === "function"
          ? options.devIntro()
          : ({
              type: "intro",
              data: {
                agent: "demo",
                model: "DemoModel",
                cwd: process.cwd(),
                tool_count: 0,
              },
            } satisfies DownEvent);
      for (const handler of handlers) handler(event);
    }, 50);
  }

  render(renderRoot({ send, subscribe }), {
    stdin: inkStdin,
    stdout: ttyOut,
    exitOnCtrlC: false,
    patchConsole: false,
  });

  // Emit ``ready`` once the React tree has had a chance to mount its
  // event subscribers. The Python parent waits for this before sending
  // ``intro`` — without it custom TUIs would hang on startup.
  setImmediate(() => {
    send({ type: "ready", data: {} });
  });
}

// ── tty stream helpers ────────────────────────────────────────────────
function defaultTtyIn(stdinIsTTY: boolean): NodeJS.ReadStream {
  if (stdinIsTTY) return process.stdin;
  try {
    const fd = openSync("/dev/tty", "r");
    return new TTYReadStream(fd) as unknown as NodeJS.ReadStream;
  } catch {
    return process.stdin;
  }
}

function defaultTtyOut(): NodeJS.WriteStream {
  if ((process.stdout as { isTTY?: boolean }).isTTY) return process.stdout;
  try {
    const fd = openSync("/dev/tty", "w");
    return new TTYWriteStream(fd) as unknown as NodeJS.WriteStream;
  } catch {
    return process.stdout;
  }
}

/**
 * Wire the host process's SIGWINCH to the tty write stream Ink is
 * rendering on.
 *
 * Node only auto-installs the SIGWINCH handler for the standard
 * ``process.stdout`` / ``process.stderr`` streams. When we render onto
 * a manually-opened ``/dev/tty`` (because stdout is the JSON pipe to
 * the Python parent), no handler exists and Ink never learns the
 * window resized — so the rendered frame stays at the original
 * dimensions until the user kills the TUI.
 *
 * We fix that here by:
 *   1. listening for SIGWINCH ourselves;
 *   2. asking the kernel for the current size with ``TIOCGWINSZ``;
 *   3. updating ``columns`` / ``rows`` on the stream and emitting
 *      ``"resize"`` so Ink's listener (added in ``ink.js`` via
 *      ``options.stdout.on("resize", this.resized)``) reflows.
 */
function wireResize(stream: NodeJS.WriteStream): void {
  if (stream === process.stdout || stream === process.stderr) {
    // Node already wires these; double-subscribing would just cause a
    // redundant render per resize.
    return;
  }
  const fd = (stream as unknown as { fd?: number }).fd;
  if (typeof fd !== "number") return;
  let pending: NodeJS.Immediate | null = null;
  const refresh = () => {
    pending = null;
    try {
      // ``getWindowSize`` is the public API on a tty.WriteStream and
      // wraps the underlying ioctl(TIOCGWINSZ). Returns [cols, rows].
      const dims = (stream as unknown as { getWindowSize?: () => [number, number] }).getWindowSize?.();
      if (dims) {
        (stream as unknown as { columns: number }).columns = dims[0];
        (stream as unknown as { rows: number }).rows = dims[1];
      }
    } catch {
      // If the ioctl fails (rare — tty went away), let Ink keep its
      // last-known size rather than crashing.
      return;
    }
    stream.emit("resize");
  };
  // SIGWINCH can fire several times during a single drag; coalesce
  // bursts onto one render via setImmediate so we don't redraw 60×.
  const onSignal = () => {
    if (pending !== null) return;
    pending = setImmediate(refresh);
  };
  process.on("SIGWINCH", onSignal);
}

/**
 * Ink's ``parseKeypress`` matches whole input chunks against literal
 * byte patterns (``s === "\r"``). In raw mode the kernel can merge
 * adjacent keystrokes — fast typing, IME commits, paste — into one
 * read. A merged chunk like ``"你好\r"`` then has ``key.name === ""``
 * and ``key.return === false``, so ``ink-text-input`` misses Enter and
 * the message never submits.
 *
 * Fix: split chunks on ``\r`` / ``\n`` / control-byte boundaries
 * (preserving multi-byte ANSI sequences) and emit one piece per tick
 * so each keypress goes through ``parseKeypress`` independently.
 */
function splitInputChunks(text: string): string[] {
  const out: string[] = [];
  let buf = "";
  let i = 0;
  while (i < text.length) {
    const c = text[i] ?? "";
    if (c === "\x1b") {
      if (buf) {
        out.push(buf);
        buf = "";
      }
      let j = i + 1;
      if (text[j] === "[" || text[j] === "O") {
        j++;
        while (j < text.length) {
          const k = text[j] ?? "";
          j++;
          if (/[A-Za-z~^$]/.test(k)) break;
        }
      }
      out.push(text.slice(i, j));
      i = j;
      continue;
    }
    if (c === "\r" || c === "\n" || c === "\x03" || c === "\x04" || c === "\x7f" || c === "\b" || c === "\t") {
      if (buf) {
        out.push(buf);
        buf = "";
      }
      out.push(c);
      i++;
      continue;
    }
    buf += c;
    i++;
  }
  if (buf) out.push(buf);
  return out;
}

function chunkSplittingProxy(source: NodeJS.ReadStream): NodeJS.ReadStream {
  const proxy = new PassThrough() as unknown as NodeJS.ReadStream & PassThrough;
  (proxy as unknown as { isTTY: boolean }).isTTY = true;
  (proxy as unknown as { setRawMode: (v: boolean) => void }).setRawMode = (
    enabled: boolean,
  ) => {
    if (typeof (source as unknown as { setRawMode?: (v: boolean) => void }).setRawMode === "function") {
      (source as unknown as { setRawMode: (v: boolean) => void }).setRawMode(enabled);
    }
    return proxy;
  };
  (proxy as unknown as { ref: () => void }).ref = () => source.ref?.();
  (proxy as unknown as { unref: () => void }).unref = () => source.unref?.();

  const queue: string[] = [];
  let flushScheduled = false;
  const flush = () => {
    flushScheduled = false;
    const piece = queue.shift();
    if (piece === undefined) return;
    proxy.write(piece);
    if (queue.length > 0) {
      flushScheduled = true;
      setImmediate(flush);
    }
  };

  source.setEncoding("utf8");
  source.on("data", (data: string | Buffer) => {
    const text = typeof data === "string" ? data : data.toString("utf8");
    for (const piece of splitInputChunks(text)) queue.push(piece);
    if (!flushScheduled && queue.length > 0) {
      flushScheduled = true;
      setImmediate(flush);
    }
  });
  return proxy;
}
