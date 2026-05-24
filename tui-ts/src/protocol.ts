/**
 * Wire-protocol types for stdin/stdout JSON-RPC messages.
 * Mirror of docs/tui-protocol.md.
 */

export type PermissionMode = "default" | "accept_edits" | "plan";

export type DownEvent =
  | { type: "intro"; data: { agent: string; model: string; cwd: string; tool_count: number; permission_mode?: PermissionMode } }
  | { type: "user"; data: { text: string; turn: number } }
  | { type: "thinking_delta"; data: { text: string } }
  | { type: "tool_call_delta"; data: { index: number; name?: string; arguments?: string } }
  | { type: "tool_call"; data: { name: string; arguments: Record<string, unknown> } }
  | { type: "tool_progress"; data: { tool: string; message: string; data?: Record<string, unknown> } }
  | { type: "tool_result"; data: { name: string; text: string } }
  | { type: "tool_denied"; data: { message: string } }
  | { type: "tool_invalid"; data: { message: string } }
  | { type: "tool_error"; data: { message: string } }
  | { type: "message_delta"; data: { text: string } }
  | { type: "message"; data: { text: string; streamed: boolean } }
  | { type: "error"; data: { message: string } }
  | { type: "run_start"; data: Record<string, never> }
  | { type: "run_end"; data: { tokens?: number; duration_ms?: number } }
  | { type: "permission_request"; data: { id: string; tool: string; reason: string; arguments: Record<string, unknown> } }
  | { type: "permission_mode_changed"; data: { mode: PermissionMode } }
  | { type: "notice"; data: { text: string } }
  | { type: "files"; data: { request_id: string; paths: string[] } }
  | { type: "clear"; data: Record<string, never> }
  | { type: "quit"; data: Record<string, never> };

export type UpEvent =
  | { type: "ready"; data: Record<string, never> }
  | { type: "submit"; data: { text: string } }
  | { type: "permission_response"; data: { id: string; allow: boolean } }
  | { type: "set_permission_mode"; data: { mode: PermissionMode } }
  | { type: "list_files"; data: { request_id: string; prefix: string; limit?: number } }
  | { type: "interrupt"; data: Record<string, never> }
  | { type: "quit"; data: Record<string, never> };

export interface Intro {
  agent: string;
  model: string;
  cwd: string;
  toolCount: number;
}

/** A persisted scrollback entry. */
export type TranscriptItem =
  | { kind: "user"; text: string; turn: number }
  | { kind: "assistant"; text: string }
  | { kind: "tool_call"; name: string; arguments: Record<string, unknown> }
  | { kind: "tool_result"; name: string; text: string }
  | { kind: "tool_progress"; name: string; message: string }
  | { kind: "tool_denied"; message: string }
  | { kind: "error"; message: string }
  | { kind: "notice"; text: string };

/** Pending permission decision. */
export interface PendingPermission {
  id: string;
  tool: string;
  reason: string;
  arguments: Record<string, unknown>;
}
