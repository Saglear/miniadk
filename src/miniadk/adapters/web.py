from __future__ import annotations

import inspect
import json
from html import escape
from pathlib import Path
from typing import Any

from ..core.agent import Agent
from ..core.middleware import Middleware, PermissionRequest
from ..core.model import Model
from ..core.policy import RunPolicy
from ..core.session import Session
from ..core.tools import Tool
from ..sessions import CompactSpec
from .ws import ws_json


def web_html(
    *,
    title: str = "MiniADK",
    ws_path: str = "/ws",
) -> str:
    safe_title = escape(title)
    safe_ws_path = escape(ws_path)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #171717;
      --muted: #64706b;
      --line: #d8dedb;
      --panel: #f3f7f5;
      --accent: #0f766e;
      --tool: #9a3412;
      --error: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: #f9fbfa;
      color: var(--ink);
      font: 15px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    main {{
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: 100vh;
      max-width: 980px;
      margin: 0 auto;
      padding: 24px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 16px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 14px;
    }}
    h1 {{
      margin: 0;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    #state {{ color: var(--muted); font-size: 13px; }}
    #log {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      overflow: auto;
      padding: 18px 0;
    }}
    .event {{
      border-left: 3px solid var(--line);
      padding: 8px 0 8px 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
    .user {{ border-color: var(--accent); }}
    .assistant {{ border-color: var(--accent); }}
    .tool {{ border-color: var(--tool); color: var(--tool); }}
    .error {{ border-color: var(--error); color: var(--error); }}
    .label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 3px;
    }}
    form {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      border-top: 1px solid var(--line);
      padding-top: 14px;
    }}
    input {{
      min-width: 0;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      padding: 12px 13px;
      font: inherit;
      outline: none;
    }}
    input:focus {{ border-color: var(--accent); }}
    button {{
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      padding: 0 18px;
      font: inherit;
      cursor: pointer;
    }}
    button:disabled {{ opacity: .5; cursor: wait; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{safe_title}</h1>
      <span id="state">connecting</span>
    </header>
    <section id="log" aria-live="polite"></section>
    <form id="chat">
      <input id="text" autocomplete="off" placeholder="Ask the agent">
      <button id="send" type="submit">Send</button>
    </form>
  </main>
  <script>
    const state = document.querySelector("#state");
    const log = document.querySelector("#log");
    const form = document.querySelector("#chat");
    const input = document.querySelector("#text");
    const button = document.querySelector("#send");
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${{proto}}//${{location.host}}{safe_ws_path}`);

    socket.onopen = () => state.textContent = "ready";
    socket.onclose = () => state.textContent = "closed";
    socket.onerror = () => state.textContent = "error";
    socket.onmessage = (message) => render(JSON.parse(message.data));

    form.addEventListener("submit", (event) => {{
      event.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      write("user", text, "user");
      socket.send(JSON.stringify({{ text }}));
      input.value = "";
      button.disabled = true;
    }});

    function render(event) {{
      if (event.type === "message_delta") {{
        appendDelta(event.data.text);
        return;
      }}
      if (event.type === "permission_prompt") {{
        const detail = `${{event.data.tool}} · ${{event.data.reason}}`;
        write("permission", detail, "tool");
        const allow = window.confirm(`Allow ${{detail}}?`);
        socket.send(JSON.stringify({{ allow }}));
        return;
      }}
      if (event.type === "message") {{
        button.disabled = false;
        if (!event.data.streamed) {{
          write("assistant", event.data.text || "", "assistant");
        }}
      }} else if (event.type === "tool_call") {{
        write("tool", `${{event.data.name}} ${{JSON.stringify(event.data.arguments)}}`, "tool");
      }} else if (event.type === "tool_result") {{
        write("result", event.data.text || String(event.data.result || ""), "tool");
      }} else if (event.type === "permission_request") {{
        write("permission", `${{event.data.tool}} · ${{event.data.reason}}`, "tool");
      }} else if (event.type === "tool_denied") {{
        button.disabled = false;
        write("denied", event.data.message || "denied", "error");
      }} else if (event.type === "run_end") {{
        button.disabled = false;
      }} else if (event.type === "error" || event.type === "tool_invalid" || event.type === "tool_error") {{
        button.disabled = false;
        write("error", event.data.message || "error", "error");
      }}
    }}

    function appendDelta(text) {{
      let last = log.lastElementChild;
      if (!last || !last.classList.contains("stream")) {{
        last = write("assistant", "", "assistant stream");
      }}
      last.lastChild.textContent += text;
      log.scrollTop = log.scrollHeight;
    }}

    function write(label, text, kind) {{
      const item = document.createElement("div");
      item.className = `event ${{kind}}`;
      const head = document.createElement("span");
      head.className = "label";
      head.textContent = label;
      const body = document.createElement("span");
      body.textContent = text;
      item.append(head, body);
      log.append(item);
      log.scrollTop = log.scrollHeight;
      return item;
    }}
  </script>
</body>
</html>
"""


async def ws_chat(
    ws: Any,
    agent: Agent,
    *,
    model: Model | None = None,
    middleware: list[Middleware] | None = None,
    policy: RunPolicy | None = None,
    session: Session | str | Path | bool | None = None,
    tools: list[Tool] | None = None,
    max_steps: int = 20,
    lifecycle: bool = False,
    trace: bool = False,
    resolve: bool = True,
    compact: CompactSpec = None,
) -> None:
    accept = getattr(ws, "accept", None)
    if accept is not None:
        result = accept()
        if inspect.isawaitable(result):
            await result

    active_session = session if session is not None else Session()
    while True:
        text = await _recv_text(ws)
        if text is None:
            return
        if not text.strip():
            continue
        await ws_json(
            ws,
            agent,
            text,
            model=model,
            middleware=middleware,
            policy=policy,
            session=active_session,
            tools=tools,
            max_steps=max_steps,
            lifecycle=lifecycle,
            trace=trace,
            resolve=resolve,
            compact=compact,
            ask_user=_ask_permission(ws),
        )


async def _recv_text(ws: Any) -> str | None:
    return _text_from_message(await _recv_message(ws))


def _ask_permission(ws: Any):
    async def ask(request: PermissionRequest) -> bool:
        await _send_permission_prompt(ws, request)
        while True:
            message = await _recv_message(ws)
            if message is None:
                return False
            answer = _permission_answer(message)
            if answer is not None:
                return answer

    return ask


async def _send_permission_prompt(ws: Any, request: PermissionRequest) -> None:
    event = {
        "type": "permission_prompt",
        "data": {
            "tool": request.tool.name,
            "arguments": request.arguments,
            "reason": request.reason,
        },
    }
    send_json = getattr(ws, "send_json", None)
    if send_json is not None:
        result = send_json(event)
    else:
        send = getattr(ws, "send", None)
        if send is None:
            raise TypeError("websocket must provide send_json or send")
        result = send(json.dumps(event, ensure_ascii=False))
    if inspect.isawaitable(result):
        await result


async def _recv_message(ws: Any) -> Any:
    receive_json = getattr(ws, "receive_json", None)
    if receive_json is not None:
        message = receive_json()
        if inspect.isawaitable(message):
            message = await message
        return message

    receive_text = getattr(ws, "receive_text", None)
    if receive_text is not None:
        message = receive_text()
        if inspect.isawaitable(message):
            message = await message
        return message

    recv = getattr(ws, "recv", None)
    if recv is not None:
        message = recv()
        if inspect.isawaitable(message):
            message = await message
        return message

    raise TypeError("websocket must provide receive_json, receive_text, or recv")


def _text_from_message(message: Any) -> str | None:
    if message is None:
        return None
    if isinstance(message, dict):
        value = message.get("text")
        return None if value is None else str(value)
    if isinstance(message, str):
        try:
            parsed = json.loads(message)
        except json.JSONDecodeError:
            return message
        return _text_from_message(parsed)
    return str(message)


def _permission_answer(message: Any) -> bool | None:
    if isinstance(message, dict):
        for key in ("allow", "allowed", "yes", "approve"):
            if key in message:
                value = message[key]
                if isinstance(value, str):
                    return _permission_answer(value)
                return bool(value)
        if message.get("type") in {"permission", "permission_response"}:
            text = message.get("text")
            if text is not None:
                return _permission_answer(str(text))
        return None
    if isinstance(message, str):
        try:
            parsed = json.loads(message)
        except json.JSONDecodeError:
            text = message.strip().lower()
            if text in {"y", "yes", "allow", "approve", "approved", "true", "1"}:
                return True
            if text in {"n", "no", "deny", "denied", "false", "0"}:
                return False
            return None
        return _permission_answer(parsed)
    return None
