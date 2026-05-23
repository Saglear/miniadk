#!/usr/bin/env node
import net from "node:net";
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
type BridgeMessage = { type: string; data?: Record<string, Json>; [key: string]: Json | undefined };
type RuntimeEvent = { type: string; data?: Record<string, Json> };
type ToolInfo = {
  name: string;
  description?: string;
  readOnly?: boolean;
  destructive?: boolean;
  safe?: boolean;
};
type SkillInfo = {
  name: string;
  description?: string;
  userInvocable?: boolean;
  modelInvocable?: boolean;
};

const port = Number(process.argv[2]);
const token = process.argv[3] || "";

const ansi = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[2m",
  cyan: "\x1b[38;5;81m",
  green: "\x1b[38;5;120m",
  yellow: "\x1b[38;5;215m",
  red: "\x1b[38;5;203m",
  gray: "\x1b[38;5;244m",
  blue: "\x1b[38;5;111m",
};

let promptText = "mini";
let busy = false;
let streaming = false;
let rl: readline.Interface | undefined;

const socket = net.createConnection({ host: "127.0.0.1", port });
socket.setEncoding("utf8");

socket.on("connect", () => send({ type: "hello", token }));
socket.on("error", (error: Error) => {
  console.error(`${color("MiniADK CLI bridge failed:", "red")} ${error.message}`);
  process.exitCode = 1;
});
socket.on("close", () => rl?.close());

let buffer = "";
socket.on("data", (chunk: string) => {
  buffer += chunk;
  while (true) {
    const index = buffer.indexOf("\n");
    if (index < 0) break;
    const lineText = buffer.slice(0, index);
    buffer = buffer.slice(index + 1);
    if (!lineText.trim()) continue;
    void handle(JSON.parse(lineText) as BridgeMessage);
  }
});

process.on("SIGINT", () => {
  if (busy) {
    line("");
    line(color("Current step is running. Press Ctrl-D or type /exit after it returns.", "gray"));
    return;
  }
  send({ type: "exit" });
  process.exit(0);
});

function send(message: Record<string, Json | undefined>): void {
  socket.write(JSON.stringify(message) + "\n");
}

async function handle(message: BridgeMessage): Promise<void> {
  const { type, data = {} } = message;
  if (type === "ready") {
    promptText = promptLabel(stringOf(data.prompt), stringOf(data.theme) || "mini");
    renderIntro(data);
    rl = readline.createInterface({ input, output });
    await promptLoop();
    return;
  }
  if (type === "run_start") {
    busy = true;
    streaming = false;
    line("");
    status("working", "model is thinking");
    return;
  }
  if (type === "event") {
    renderEvent(data as RuntimeEvent);
    return;
  }
  if (type === "run_end" || type === "idle") {
    finishStream();
    busy = false;
    return;
  }
  if (type === "permission") {
    const allow = await askPermission(data);
    send({ type: "permission", allow });
    return;
  }
  if (type === "notice") {
    line(color(`• ${stringOf(data.text)}`, "gray"));
    return;
  }
  if (type === "clear") {
    output.write("\x1b[2J\x1b[H");
    return;
  }
  if (type === "command") renderCommand(data);
}

async function promptLoop(): Promise<void> {
  while (rl) {
    const text = await rl.question(`${color(promptText, "cyan")} ${color("›", "gray")} `);
    if (!text.trim()) continue;
    busy = true;
    send({ type: "input", text });
    if (["/exit", "/quit"].includes(text.trim())) {
      rl.close();
      socket.end();
      return;
    }
    while (busy) await sleep(20);
  }
}

async function askPermission(data: Record<string, Json>): Promise<boolean> {
  finishStream();
  const tool = stringOf(data.tool);
  const reason = stringOf(data.reason) || "tool use";
  const args = compactArgs(data.arguments);
  panel("permission", [
    `${color(tool, "yellow")} ${color(reason, "gray")}`,
    args ? color(args, "gray") : "",
  ].filter(Boolean));
  const answer = await rl!.question(`${color("allow?", "yellow")} ${color("[y/N]", "gray")} `);
  return ["y", "yes"].includes(answer.trim().toLowerCase());
}

function renderIntro(data: Record<string, Json>): void {
  const tools = arrayOf<ToolInfo>(data.tools);
  const skills = arrayOf<SkillInfo>(data.skills);
  const model = stringOf(data.model) || "model";
  const cwd = stringOf(data.cwd);
  const width = Math.min(Math.max(64, model.length + 32), Math.max(64, output.columns || 96));

  line(color("╭" + "─".repeat(width - 2) + "╮", "cyan"));
  line(color(`│ ${pad(`MiniADK · ${stringOf(data.agent)}`, width - 4)} │`, "cyan"));
  line(color(`│ ${pad(`${model} · ${tools.length} tools · ${skills.length} skills`, width - 4)} │`, "gray"));
  line(color(`│ ${pad(cwd, width - 4)} │`, "gray"));
  line(color("╰" + "─".repeat(width - 2) + "╯", "cyan"));
  line(color("Commands: /help /status /tools /skills /clear /exit", "gray"));
}

function renderEvent(event: RuntimeEvent): void {
  const data = event.data || {};
  if (event.type === "message_delta") {
    if (!streaming) {
      streaming = true;
      line(color("assistant", "green"));
      output.write(color("│ ", "green"));
    }
    output.write(stringOf(data.text));
    return;
  }
  if (event.type === "message") {
    if (data.streamed) {
      finishStream();
      return;
    }
    finishStream();
    block("assistant", stringOf(data.text), "green");
    return;
  }
  if (event.type === "thinking_delta") {
    finishStream();
    status("thinking", stringOf(data.text));
    return;
  }
  if (event.type === "tool_call" || event.type === "tool_call_delta") {
    finishStream();
    const name = stringOf(data.name) || `#${stringOf(data.index) || "0"}`;
    status("tool", `${name} ${compactArgs(data.arguments)}`.trim(), "yellow");
    return;
  }
  if (event.type === "tool_progress") {
    finishStream();
    status("progress", `${stringOf(data.tool)} ${stringOf(data.message)}`.trim(), "yellow");
    return;
  }
  if (event.type === "tool_result") {
    finishStream();
    const result = stringOf(data.text) || stringOf(data.result);
    if (result) block("result", clip(result, 1200), "gray");
    else status("done", "", "gray");
    return;
  }
  if (event.type === "permission_request") {
    finishStream();
    status("permission", `${stringOf(data.tool)} ${stringOf(data.reason)}`.trim(), "yellow");
    return;
  }
  if (["tool_denied", "tool_invalid", "tool_error", "error"].includes(event.type)) {
    finishStream();
    line(color(`× ${stringOf(data.message) || event.type}`, "red"));
  }
}

function renderCommand(data: Record<string, Json>): void {
  finishStream();
  if (data.rows) {
    section(stringOf(data.name));
    for (const row of arrayOf<Json[]>(data.rows)) {
      const left = stringOf(row[0]);
      const right = stringOf(row[1]);
      line(color(pad(left, 12), "yellow") + " " + color(right, "gray"));
    }
    return;
  }
  if (data.items) {
    section(stringOf(data.name));
    for (const item of arrayOf<Json>(data.items)) line(color("• ", "yellow") + stringOf(item));
    return;
  }
  if (data.tools) {
    section("tools");
    for (const tool of arrayOf<ToolInfo>(data.tools)) {
      const flags = [tool.readOnly && "read", tool.destructive && "write", tool.safe && "safe"].filter(Boolean).join(",");
      line(color(tool.name, "yellow") + color(flags ? ` [${flags}]` : "", "gray") + color(` ${tool.description || ""}`, "gray"));
    }
    return;
  }
  if (data.skills) {
    section("skills");
    for (const skill of arrayOf<SkillInfo>(data.skills)) {
      const mode = skill.userInvocable ? "user" : "model";
      line(color(`/${skill.name}`, "yellow") + color(` [${mode}] ${skill.description || ""}`, "gray"));
    }
    return;
  }
  if (data.text) block(stringOf(data.name) || "output", stringOf(data.text), "gray");
}

function panel(label: string, lines: string[]): void {
  section(label);
  for (const item of lines) line(color("│ ", "yellow") + item);
}

function block(label: string, text: string, tone: keyof typeof ansi): void {
  section(label, tone);
  for (const raw of (text || "").split("\n")) {
    const wrapped = wrap(raw, Math.max(40, Math.min(104, (output.columns || 96) - 4)));
    for (const item of wrapped) line(color("│ ", tone) + item);
  }
}

function section(label: string, tone: keyof typeof ansi = "cyan"): void {
  line("");
  line(color(label || "output", tone));
}

function status(label: string, text: string, tone: keyof typeof ansi = "gray"): void {
  line(color("◇ ", tone) + color(label, tone) + (text ? color(` · ${text}`, "gray") : ""));
}

function finishStream(): void {
  if (streaming) {
    output.write("\n");
    streaming = false;
  }
}

function compactArgs(value: Json | undefined): string {
  if (value === undefined || value === null || value === "") return "";
  if (typeof value === "string") return clip(value.replace(/\n/g, "\\n"), 140);
  try {
    return clip(JSON.stringify(value), 160);
  } catch {
    return String(value);
  }
}

function promptLabel(text: string, fallback: string): string {
  const label = text.trim().replace(/[>›]+$/, "").trim();
  return label || fallback || "mini";
}

function wrap(text: string, width: number): string[] {
  if (text.length <= width) return [text];
  const lines: string[] = [];
  let rest = text;
  while (rest.length > width) {
    let at = rest.lastIndexOf(" ", width);
    if (at < width * 0.5) at = width;
    lines.push(rest.slice(0, at));
    rest = rest.slice(at).trimStart();
  }
  if (rest) lines.push(rest);
  return lines;
}

function clip(text: string, limit: number): string {
  if (text.length <= limit) return text;
  return `${text.slice(0, limit).trimEnd()}\n... (${text.length - limit} more chars)`;
}

function color(text: string, name: keyof typeof ansi): string {
  return `${ansi[name] || ""}${text}${ansi.reset}`;
}

function pad(text: string, width: number): string {
  const value = String(text);
  if (value.length >= width) return value.slice(0, width);
  return value + " ".repeat(width - value.length);
}

function line(text = ""): void {
  output.write(text + "\n");
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function stringOf(value: Json | undefined): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  return String(value);
}

function arrayOf<T>(value: Json | undefined): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}
