import asyncio
import os
import sys

from miniadk import (
    Agent,
    MCPHub,
    MCPNotice,
    MCPServer,
    MCPToolError,
    ModelResult,
    Runtime,
    ScriptedModel,
    Skill,
    SkillRegistry,
    ToolCall,
)
from miniadk.skills import resolve_agent


MCP_SERVER_SCRIPT = r'''
import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    body = sys.stdin.buffer.read(int(headers["content-length"]))
    return json.loads(body.decode())


def write_message(message):
    data = json.dumps(message).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    sys.stdout.buffer.flush()


while True:
    message = read_message()
    method = message.get("method")
    if method == "initialize":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "fake", "version": "1.0"},
                "capabilities": {"tools": {}, "resources": {}, "prompts": {}}
            }
        })
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "tools": [{
                    "name": "echo",
                    "description": "Echo text.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"]
                    }
                }]
            }
        })
    elif method == "tools/call":
        text = message["params"]["arguments"]["text"]
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {"content": [{"type": "text", "text": "echo:" + text}]}
        })
    elif method == "resources/list":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "resources": [{
                    "uri": "docs://intro",
                    "name": "Intro",
                    "description": "Intro docs",
                    "mimeType": "text/plain"
                }]
            }
        })
    elif method == "resources/read":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "contents": [{
                    "uri": message["params"]["uri"],
                    "mimeType": "text/plain",
                    "text": "hello docs"
                }]
            }
        })
    elif method == "prompts/list":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "prompts": [{
                    "name": "review-docs",
                    "description": "Review docs.",
                    "arguments": []
                }]
            }
        })
    elif method == "prompts/get":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "text": "Review the docs carefully."}]
                }]
            }
        })
'''


ANNOTATED_MCP_SERVER_SCRIPT = r'''
import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    body = sys.stdin.buffer.read(int(headers["content-length"]))
    return json.loads(body.decode())


def write_message(message):
    data = json.dumps(message).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    sys.stdout.buffer.flush()


while True:
    message = read_message()
    method = message.get("method")
    if method == "initialize":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {"capabilities": {"tools": {}}}
        })
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "tools": [
                    {
                        "name": "inspect",
                        "description": "Inspect state.",
                        "inputSchema": {"type": "object", "properties": {}},
                        "annotations": {
                            "readOnlyHint": True,
                            "idempotentHint": True,
                            "openWorldHint": False
                        }
                    },
                    {
                        "name": "mutate",
                        "description": "Mutate state.",
                        "inputSchema": {"type": "object", "properties": {}},
                        "annotations": {
                            "destructiveHint": True,
                            "idempotentHint": False
                        }
                    }
                ]
            }
        })
    elif method == "tools/call":
        name = message["params"]["name"]
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {"content": [{"type": "text", "text": name + ":ok"}]}
        })
'''


NOISY_MCP_SERVER_SCRIPT = r'''
import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    body = sys.stdin.buffer.read(int(headers["content-length"]))
    return json.loads(body.decode())


def write_message(message):
    data = json.dumps(message).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    sys.stdout.buffer.flush()


sys.stderr.buffer.write(b"x" * 200000)
sys.stderr.buffer.flush()

while True:
    message = read_message()
    method = message.get("method")
    if method == "initialize":
        write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"capabilities": {"tools": {}}}})
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"tools": []}})
'''


HANGING_MCP_SERVER_SCRIPT = r'''
import time

time.sleep(60)
'''


SLOW_TOOLS_MCP_SERVER_SCRIPT = r'''
import json
import sys
import time


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    body = sys.stdin.buffer.read(int(headers["content-length"]))
    return json.loads(body.decode())


def write_message(message):
    data = json.dumps(message).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    sys.stdout.buffer.flush()


while True:
    message = read_message()
    method = message.get("method")
    if method == "initialize":
        write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"capabilities": {"tools": {}}}})
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        time.sleep(60)
'''


MALFORMED_MCP_SERVER_SCRIPT = r'''
import sys

sys.stdout.buffer.write(b"Content-Type: application/json\r\n\r\n{}")
sys.stdout.buffer.flush()
'''


CACHED_LOOKUP_MCP_SERVER_SCRIPT = r'''
import json
import sys

resource_lists = 0
prompt_lists = 0


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    body = sys.stdin.buffer.read(int(headers["content-length"]))
    return json.loads(body.decode())


def write_message(message):
    data = json.dumps(message).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    sys.stdout.buffer.flush()


while True:
    message = read_message()
    method = message.get("method")
    if method == "initialize":
        write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"capabilities": {"resources": {}, "prompts": {}}}})
    elif method == "notifications/initialized":
        pass
    elif method == "resources/list":
        resource_lists += 1
        if resource_lists > 1:
            write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"resources": []}})
        else:
            write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"resources": [{"uri": "docs://cached", "name": "Cached"}]}})
    elif method == "resources/read":
        write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"contents": [{"text": "cached docs"}]}})
    elif method == "prompts/list":
        prompt_lists += 1
        if prompt_lists > 1:
            write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"prompts": []}})
        else:
            write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"prompts": [{"name": "cached-prompt", "description": "Cached prompt."}]}})
    elif method == "prompts/get":
        write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"messages": [{"role": "user", "content": [{"type": "text", "text": "cached prompt"}]}]}})
'''


ERROR_MCP_SERVER_SCRIPT = r'''
import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    body = sys.stdin.buffer.read(int(headers["content-length"]))
    return json.loads(body.decode())


def write_message(message):
    data = json.dumps(message).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    sys.stdout.buffer.flush()


while True:
    message = read_message()
    method = message.get("method")
    if method == "initialize":
        write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"capabilities": {"tools": {}}}})
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "tools": [{
                    "name": "fail",
                    "description": "Fail clearly.",
                    "inputSchema": {"type": "object", "properties": {}}
                }]
            }
        })
    elif method == "tools/call":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "isError": True,
                "content": [{"type": "text", "text": "remote tool failed"}]
            }
        })
'''


OBJECT_TOOL_MCP_SERVER_SCRIPT = r'''
import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    body = sys.stdin.buffer.read(int(headers["content-length"]))
    return json.loads(body.decode())


def write_message(message):
    data = json.dumps(message).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    sys.stdout.buffer.flush()


while True:
    message = read_message()
    method = message.get("method")
    if method == "initialize":
        write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"capabilities": {"tools": {}}}})
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "tools": [{
                    "name": "object",
                    "description": "Return object content.",
                    "inputSchema": {"type": "object", "properties": {}}
                }]
            }
        })
    elif method == "tools/call":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {"content": {"type": "text", "text": "object tool text"}}
        })
'''


STRUCTURED_TOOL_MCP_SERVER_SCRIPT = r'''
import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    body = sys.stdin.buffer.read(int(headers["content-length"]))
    return json.loads(body.decode())


def write_message(message):
    data = json.dumps(message).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    sys.stdout.buffer.flush()


while True:
    message = read_message()
    method = message.get("method")
    if method == "initialize":
        write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"capabilities": {"tools": {}}}})
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "tools": [{
                    "name": "structured",
                    "description": "Return structured data.",
                    "inputSchema": {"type": "object", "properties": {}}
                }]
            }
        })
    elif method == "tools/call":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {"items": [1, 2], "ok": True}
        })
'''


BAD_SCHEMA_MCP_SERVER_SCRIPT = r'''
import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    body = sys.stdin.buffer.read(int(headers["content-length"]))
    return json.loads(body.decode())


def write_message(message):
    data = json.dumps(message).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    sys.stdout.buffer.flush()


while True:
    message = read_message()
    method = message.get("method")
    if method == "initialize":
        write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"capabilities": {"tools": {}}}})
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "tools": [
                    {"name": "array_schema", "inputSchema": {"type": "array"}},
                    {
                        "name": "bad_fields",
                        "inputSchema": {
                            "type": "object",
                            "properties": [],
                            "required": "path"
                        }
                    }
                ]
            }
        })
'''


OBJECT_PROMPT_MCP_SERVER_SCRIPT = r'''
import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    body = sys.stdin.buffer.read(int(headers["content-length"]))
    return json.loads(body.decode())


def write_message(message):
    data = json.dumps(message).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    sys.stdout.buffer.flush()


while True:
    message = read_message()
    method = message.get("method")
    if method == "initialize":
        write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"capabilities": {"prompts": {}}}})
    elif method == "notifications/initialized":
        pass
    elif method == "prompts/list":
        write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"prompts": [{"name": "object-prompt"}]}})
    elif method == "prompts/get":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "messages": [
                    {"role": "user", "content": {"type": "text", "text": "Object text."}},
                    {"role": "assistant", "content": {"type": "image", "data": "abc"}}
                ]
            }
        })
'''


ARGUMENT_PROMPT_MCP_SERVER_SCRIPT = r'''
import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    body = sys.stdin.buffer.read(int(headers["content-length"]))
    return json.loads(body.decode())


def write_message(message):
    data = json.dumps(message).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    sys.stdout.buffer.flush()


while True:
    message = read_message()
    method = message.get("method")
    if method == "initialize":
        write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"capabilities": {"prompts": {}}}})
    elif method == "notifications/initialized":
        pass
    elif method == "prompts/list":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "prompts": [{
                    "name": "review-file",
                    "description": "Review a file.",
                    "arguments": [{"name": "path"}, {"name": "focus"}]
                }]
            }
        })
    elif method == "prompts/get":
        args = message["params"].get("arguments") or {}
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "text": "Review " + args.get("path", "file")}]
                }]
            }
        })
'''


ENV_MCP_SERVER_SCRIPT = r'''
import json
import os
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    body = sys.stdin.buffer.read(int(headers["content-length"]))
    return json.loads(body.decode())


def write_message(message):
    data = json.dumps(message).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    sys.stdout.buffer.flush()


while True:
    message = read_message()
    method = message.get("method")
    if method == "initialize":
        write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"capabilities": {"tools": {}}}})
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "tools": [{
                    "name": "env",
                    "description": "Return selected environment variables.",
                    "inputSchema": {"type": "object", "properties": {}}
                }]
            }
        })
    elif method == "tools/call":
        values = [
            os.getenv("MINIADK_PARENT_SECRET", "missing"),
            os.getenv("MINIADK_ALLOWED", "missing"),
            os.getenv("PATH", "missing"),
        ]
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {"content": [{"type": "text", "text": "|".join(values)}]}
        })
'''


PAGED_MCP_SERVER_SCRIPT = r'''
import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    body = sys.stdin.buffer.read(int(headers["content-length"]))
    return json.loads(body.decode())


def write_message(message):
    data = json.dumps(message).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    sys.stdout.buffer.flush()


def paged(params, key, first, second):
    cursor = params.get("cursor")
    if cursor is None:
        return {key: [first], "nextCursor": ""}
    if cursor == "":
        return {key: [second], "nextCursor": "done"}
    return {key: []}


while True:
    message = read_message()
    method = message.get("method")
    params = message.get("params") or {}
    if method == "initialize":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "capabilities": {"tools": {}, "resources": {}, "prompts": {}}
            }
        })
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": paged(
                params,
                "tools",
                {
                    "name": "first",
                    "description": "First tool.",
                    "inputSchema": {"type": "object", "properties": {}}
                },
                {
                    "name": "second",
                    "description": "Second tool.",
                    "inputSchema": {"type": "object", "properties": {}}
                },
            )
        })
    elif method == "resources/list":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": paged(
                params,
                "resources",
                {"uri": "docs://first", "name": "First"},
                {"uri": "docs://second", "name": "Second"},
            )
        })
    elif method == "prompts/list":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": paged(
                params,
                "prompts",
                {"name": "first-prompt", "description": "First prompt."},
                {"name": "second-prompt", "description": "Second prompt."},
            )
        })
'''


NO_RESOURCES_MCP_SERVER_SCRIPT = r'''
import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    body = sys.stdin.buffer.read(int(headers["content-length"]))
    return json.loads(body.decode())


def write_message(message):
    data = json.dumps(message).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    sys.stdout.buffer.flush()


while True:
    message = read_message()
    method = message.get("method")
    if method == "initialize":
        write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"serverInfo": {"name": "tools-only"}, "capabilities": {"tools": {}}}})
    elif method == "notifications/initialized":
        pass
    elif method in {"resources/list", "prompts/list"}:
        write_message({"jsonrpc": "2.0", "id": message["id"], "error": {"code": 123, "message": "should not list unsupported capability"}})
    elif method == "tools/list":
        write_message({"jsonrpc": "2.0", "id": message["id"], "result": {"tools": []}})
'''


NOTIFYING_MCP_SERVER_SCRIPT = r'''
import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    body = sys.stdin.buffer.read(int(headers["content-length"]))
    return json.loads(body.decode())


def write_message(message):
    data = json.dumps(message).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    sys.stdout.buffer.flush()


while True:
    message = read_message()
    method = message.get("method")
    if method == "initialize":
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {"capabilities": {"tools": {}}}
        })
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        write_message({
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {"message": "listing"}
        })
        write_message({
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {"tools": []}
        })
'''


async def test_mcp_hub_discovers_and_calls_stdio_tools(tmp_path):
    server = tmp_path / "fake_mcp_server.py"
    server.write_text(MCP_SERVER_SCRIPT, encoding="utf-8")

    hub = MCPHub([MCPServer(name="local", command=sys.executable, args=[str(server)])])

    tools = await hub.tools()
    try:
        assert [tool.name for tool in tools] == ["mcp__local__echo"]
        assert tools[0].input_schema["required"] == ["text"]
        assert await tools[0].run(text="hello") == "echo:hello"
    finally:
        await hub.close()


async def test_mcp_tool_annotations_drive_tool_safety_metadata(tmp_path):
    server = tmp_path / "annotated_mcp_server.py"
    server.write_text(ANNOTATED_MCP_SERVER_SCRIPT, encoding="utf-8")

    hub = MCPHub([MCPServer(name="local", command=sys.executable, args=[str(server)])])

    try:
        tools = await hub.tools()
        inspect_tool, mutate_tool = tools

        assert [tool.name for tool in tools] == [
            "mcp__local__inspect",
            "mcp__local__mutate",
        ]
        assert inspect_tool.is_read_only() is True
        assert inspect_tool.is_destructive() is False
        assert inspect_tool.is_concurrency_safe() is True
        assert inspect_tool.permission is None
        assert mutate_tool.is_read_only() is False
        assert mutate_tool.is_destructive() is True
        assert mutate_tool.is_concurrency_safe() is False
        assert mutate_tool.permission is not None
        assert await inspect_tool.run() == "inspect:ok"
        assert await mutate_tool.run() == "mutate:ok"
    finally:
        await hub.close()


async def test_mcp_hub_rejects_duplicate_server_names(tmp_path):
    first = tmp_path / "first_mcp_server.py"
    second = tmp_path / "second_mcp_server.py"
    first.write_text(MCP_SERVER_SCRIPT, encoding="utf-8")
    second.write_text(MCP_SERVER_SCRIPT, encoding="utf-8")

    try:
        MCPHub(
            [
                MCPServer(name="local", command=sys.executable, args=[str(first)]),
                MCPServer(name="local", command=sys.executable, args=[str(second)]),
            ]
        )
    except ValueError as error:
        assert str(error) == "duplicate MCP server name: local"
    else:
        raise AssertionError("duplicate MCP server names should fail")


def test_mcp_server_validates_basic_configuration():
    cases = [
        ({"name": "", "command": "python"}, "MCP server name is required"),
        ({"name": "docs", "command": ""}, "MCP server command is required"),
        (
            {"name": "docs", "command": "python", "timeout_seconds": 0},
            "MCP server timeout_seconds must be > 0",
        ),
    ]

    for kwargs, message in cases:
        try:
            MCPServer(**kwargs)
        except ValueError as error:
            assert str(error) == message
        else:
            raise AssertionError(f"invalid MCPServer should fail: {kwargs}")


async def test_mcp_hub_async_context_closes_started_clients(tmp_path):
    server = tmp_path / "fake_mcp_server.py"
    server.write_text(MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub([MCPServer(name="local", command=sys.executable, args=[str(server)])])

    async with hub as active:
        assert active is hub
        assert [tool.name for tool in await active.tools()] == ["mcp__local__echo"]
        process = active._clients["local"].process
        assert process is not None
        assert process.returncode is None

    assert hub._clients == {}
    assert process.returncode is not None


async def test_mcp_hub_exposes_server_info(tmp_path):
    server = tmp_path / "fake_mcp_server.py"
    server.write_text(MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub([MCPServer(name="local", command=sys.executable, args=[str(server)])])

    try:
        infos = await hub.info()

        assert len(infos) == 1
        assert infos[0].name == "local"
        assert infos[0].protocol_version == "2024-11-05"
        assert infos[0].server == {"name": "fake", "version": "1.0"}
        assert sorted(infos[0].capabilities) == ["prompts", "resources", "tools"]
    finally:
        await hub.close()


async def test_mcp_hub_skips_unsupported_resource_and_prompt_lists(tmp_path):
    server = tmp_path / "tools_only_mcp_server.py"
    server.write_text(NO_RESOURCES_MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub([MCPServer(name="local", command=sys.executable, args=[str(server)])])

    try:
        assert await hub.resources() == []
        assert await hub.prompts() == []
    finally:
        await hub.close()


async def test_mcp_tool_error_result_raises_and_reaches_runtime(tmp_path):
    server = tmp_path / "error_mcp_server.py"
    server.write_text(ERROR_MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub([MCPServer(name="bad", command=sys.executable, args=[str(server)])])

    try:
        tools = await hub.tools()
        assert [tool.name for tool in tools] == ["mcp__bad__fail"]

        try:
            await tools[0].run()
        except MCPToolError as error:
            assert str(error) == "remote tool failed"
        else:
            raise AssertionError("MCP isError results should raise")

        runtime = Runtime(
            Agent(name="agent", instructions="Use MCP.", tools=tools),
            model=ScriptedModel(
                [ModelResult(tool_calls=[ToolCall(name="mcp__bad__fail", arguments={})])]
            ),
        )
        events = [event async for event in runtime.run("try")]

        assert [event.type for event in events] == ["tool_call", "tool_error"]
        assert events[-1].data == {
            "tool": "mcp__bad__fail",
            "message": "MCPToolError: remote tool failed",
        }
        assert runtime.messages[-1].content == "MCPToolError: remote tool failed"
    finally:
        await hub.close()


async def test_mcp_tool_content_accepts_object_blocks(tmp_path):
    server = tmp_path / "object_tool_mcp_server.py"
    server.write_text(OBJECT_TOOL_MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub([MCPServer(name="local", command=sys.executable, args=[str(server)])])

    try:
        tools = await hub.tools()

        assert [tool.name for tool in tools] == ["mcp__local__object"]
        assert await tools[0].run() == "object tool text"
    finally:
        await hub.close()


async def test_mcp_tool_structured_result_falls_back_to_json_text(tmp_path):
    server = tmp_path / "structured_tool_mcp_server.py"
    server.write_text(STRUCTURED_TOOL_MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub([MCPServer(name="local", command=sys.executable, args=[str(server)])])

    try:
        tools = await hub.tools()

        assert [tool.name for tool in tools] == ["mcp__local__structured"]
        assert await tools[0].run() == '{"items": [1, 2], "ok": true}'
    finally:
        await hub.close()


async def test_mcp_tool_schemas_are_normalized_to_object_contracts(tmp_path):
    server = tmp_path / "bad_schema_mcp_server.py"
    server.write_text(BAD_SCHEMA_MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub([MCPServer(name="local", command=sys.executable, args=[str(server)])])

    try:
        tools = await hub.tools()

        assert [tool.name for tool in tools] == [
            "mcp__local__array_schema",
            "mcp__local__bad_fields",
        ]
        assert tools[0].input_schema == {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        assert tools[1].input_schema == {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
    finally:
        await hub.close()


async def test_mcp_server_env_can_remove_inherited_values(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIADK_PARENT_SECRET", "hidden")
    monkeypatch.setenv("MINIADK_ALLOWED", "parent")
    server = tmp_path / "env_mcp_server.py"
    server.write_text(ENV_MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub(
        [
            MCPServer(
                name="env",
                command=sys.executable,
                args=[str(server)],
                env={"MINIADK_PARENT_SECRET": None, "MINIADK_ALLOWED": "server"},
            )
        ]
    )

    try:
        tools = await hub.tools()

        assert await tools[0].run() == "missing|server|" + os.environ["PATH"]
    finally:
        await hub.close()


async def test_mcp_server_env_can_disable_parent_inheritance(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIADK_PARENT_SECRET", "hidden")
    monkeypatch.setenv("MINIADK_ALLOWED", "parent")
    server = tmp_path / "env_mcp_server.py"
    server.write_text(ENV_MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub(
        [
            MCPServer(
                name="env",
                command=sys.executable,
                args=[str(server)],
                env={"MINIADK_ALLOWED": "server", "PATH": os.environ["PATH"]},
                inherit_env=False,
            )
        ]
    )

    try:
        tools = await hub.tools()

        assert await tools[0].run() == "missing|server|" + os.environ["PATH"]
    finally:
        await hub.close()


async def test_mcp_hub_follows_list_pagination(tmp_path):
    server = tmp_path / "paged_mcp_server.py"
    server.write_text(PAGED_MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub([MCPServer(name="paged", command=sys.executable, args=[str(server)])])

    try:
        assert [tool.name for tool in await hub.tools()] == [
            "mcp__paged__first",
            "mcp__paged__second",
        ]
        assert [resource.uri for resource in await hub.resources()] == [
            "docs://first",
            "docs://second",
        ]
        assert [prompt.name for prompt in await hub.prompts()] == [
            "first-prompt",
            "second-prompt",
        ]
    finally:
        await hub.close()


async def test_mcp_hub_records_server_notifications(tmp_path):
    server = tmp_path / "notifying_mcp_server.py"
    server.write_text(NOTIFYING_MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub([MCPServer(name="notify", command=sys.executable, args=[str(server)])])

    try:
        assert await hub.tools() == []

        notices = hub.notices()
        assert notices == [
            MCPNotice(
                server="notify",
                method="notifications/progress",
                params={"message": "listing"},
            )
        ]

        hub.clear_notices()
        assert hub.notices() == []
    finally:
        await hub.close()


async def test_mcp_hub_drains_server_stderr(tmp_path):
    server = tmp_path / "noisy_mcp_server.py"
    server.write_text(NOISY_MCP_SERVER_SCRIPT, encoding="utf-8")

    hub = MCPHub(
        [MCPServer(name="noisy", command=sys.executable, args=[str(server)])]
    )

    try:
        assert await hub.tools() == []
    finally:
        await hub.close()


async def test_mcp_hub_closes_process_when_initialize_fails(tmp_path):
    server = tmp_path / "hanging_mcp_server.py"
    server.write_text(HANGING_MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub(
        [
            MCPServer(
                name="hanging",
                command=sys.executable,
                args=[str(server)],
                timeout_seconds=0.01,
            )
        ]
    )

    try:
        try:
            await hub.tools()
        except TimeoutError:
            pass
        else:
            raise AssertionError("MCP initialize should time out")

        client = hub._clients["hanging"]
        assert client.process is None
        assert client._stderr_task is None
    finally:
        await hub.close()


async def test_mcp_hub_closes_process_when_request_is_cancelled(tmp_path):
    server = tmp_path / "slow_tools_mcp_server.py"
    server.write_text(SLOW_TOOLS_MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub(
        [
            MCPServer(
                name="slow",
                command=sys.executable,
                args=[str(server)],
                timeout_seconds=5,
            )
        ]
    )

    task = asyncio.create_task(hub.tools())
    await asyncio.sleep(0.1)
    client = hub._clients["slow"]
    process = client.process
    assert process is not None
    assert process.returncode is None

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("MCP tools request should be cancellable")

    assert client.process is None
    assert client._stderr_task is None
    assert process.returncode is not None
    await hub.close()


async def test_mcp_hub_reports_missing_server_command():
    hub = MCPHub(
        [
            MCPServer(
                name="missing",
                command="definitely-not-a-miniadk-mcp-server",
            )
        ]
    )

    try:
        try:
            await hub.tools()
        except FileNotFoundError as error:
            assert "definitely-not-a-miniadk-mcp-server" in str(error)
        else:
            raise AssertionError("MCP startup should fail for missing command")

        assert hub._clients["missing"].process is None
    finally:
        await hub.close()


async def test_mcp_hub_reports_malformed_server_framing(tmp_path):
    server = tmp_path / "malformed_mcp_server.py"
    server.write_text(MALFORMED_MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub([MCPServer(name="bad", command=sys.executable, args=[str(server)])])

    try:
        try:
            await hub.tools()
        except RuntimeError as error:
            assert str(error) == "Invalid MCP message: missing Content-Length"
        else:
            raise AssertionError("MCP startup should fail for malformed framing")

        client = hub._clients["bad"]
        assert client.process is None
        assert client._stderr_task is None
    finally:
        await hub.close()


async def test_mcp_hub_reuses_resource_and_prompt_owner_cache(tmp_path):
    server = tmp_path / "cached_lookup_mcp_server.py"
    server.write_text(CACHED_LOOKUP_MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub([MCPServer(name="cached", command=sys.executable, args=[str(server)])])

    try:
        assert [resource.uri for resource in await hub.resources()] == ["docs://cached"]
        assert await hub.read_resource("docs://cached") == "cached docs"
        assert [prompt.name for prompt in await hub.prompts()] == ["cached-prompt"]
        assert (await hub.get_prompt("cached-prompt")).text() == "user: cached prompt"
    finally:
        await hub.close()


async def test_mcp_hub_refreshes_discovery_caches(tmp_path):
    server = tmp_path / "cached_lookup_mcp_server.py"
    server.write_text(CACHED_LOOKUP_MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub([MCPServer(name="cached", command=sys.executable, args=[str(server)])])

    try:
        assert [resource.uri for resource in await hub.resources()] == ["docs://cached"]
        assert [prompt.name for prompt in await hub.prompts()] == ["cached-prompt"]

        hub.refresh()

        assert await hub.resources() == []
        assert await hub.prompts() == []
    finally:
        await hub.close()


async def test_mcp_hub_uses_discovery_cache_for_resource_and_prompt_lookup(tmp_path):
    server = tmp_path / "cached_lookup_mcp_server.py"
    server.write_text(CACHED_LOOKUP_MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub([MCPServer(name="cached", command=sys.executable, args=[str(server)])])

    try:
        assert await hub.read_resource("docs://cached") == "cached docs"
        assert (await hub.get_prompt("cached-prompt")).text() == "user: cached prompt"

        hub.refresh()

        try:
            await hub.read_resource("docs://cached")
        except RuntimeError as error:
            assert "Unknown MCP resource: docs://cached" in str(error)
        else:
            raise AssertionError("stale resource owner should not survive refresh")

        try:
            await hub.get_prompt("cached-prompt")
        except RuntimeError as error:
            assert "Unknown MCP prompt: cached-prompt" in str(error)
        else:
            raise AssertionError("stale prompt owner should not survive refresh")
    finally:
        await hub.close()


async def test_mcp_hub_reads_resources_and_loads_prompts_as_skills(tmp_path):
    server = tmp_path / "fake_mcp_server.py"
    server.write_text(MCP_SERVER_SCRIPT, encoding="utf-8")

    hub = MCPHub([MCPServer(name="local", command=sys.executable, args=[str(server)])])

    try:
        resources = await hub.resources()
        prompts = await hub.prompts()
        resource_text = await hub.read_resource("docs://intro")
        prompt = await hub.get_prompt("review-docs")
        skills = await hub.skills()

        assert [resource.uri for resource in resources] == ["docs://intro"]
        assert resource_text == "hello docs"
        assert [prompt.name for prompt in prompts] == ["review-docs"]
        assert prompt.text() == "user: Review the docs carefully."
        assert skills.get("review-docs") is not None
        assert "Review the docs carefully." in skills.get("review-docs").body
    finally:
        await hub.close()


async def test_mcp_prompt_content_accepts_object_blocks(tmp_path):
    server = tmp_path / "object_prompt_mcp_server.py"
    server.write_text(OBJECT_PROMPT_MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub([MCPServer(name="local", command=sys.executable, args=[str(server)])])

    try:
        prompt = await hub.get_prompt("object-prompt")

        assert prompt.text() == (
            "user: Object text.\n"
            'assistant: {"type": "image", "data": "abc"}'
        )
    finally:
        await hub.close()


async def test_mcp_prompt_arguments_are_preserved_on_loaded_skills(tmp_path):
    server = tmp_path / "argument_prompt_mcp_server.py"
    server.write_text(ARGUMENT_PROMPT_MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub([MCPServer(name="local", command=sys.executable, args=[str(server)])])

    try:
        prompts = await hub.prompts()
        skills = await hub.skills()

        loaded = skills.get("review-file")
        assert [prompt.arguments for prompt in prompts] == [["path", "focus"]]
        assert loaded is not None
        assert loaded.arguments == ["path", "focus"]
        assert loaded.metadata == {"source": "mcp"}
    finally:
        await hub.close()


async def test_agent_resolve_adds_mcp_resource_tool_and_prompt_skill(tmp_path):
    server = tmp_path / "fake_mcp_server.py"
    server.write_text(MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub([MCPServer(name="local", command=sys.executable, args=[str(server)])])

    try:
        agent = Agent(name="repo", instructions="Use integrations.", mcp=hub)
        resolved = await resolve_agent(agent)

        assert "review-docs" in resolved.instructions
        assert [tool.name for tool in resolved.tools] == [
            "skill",
            "mcp__local__echo",
            "read_mcp_resource",
        ]
        assert resolved.tools[-1].input_schema["properties"]["uri"] == {
            "type": "string",
            "description": "MCP resource URI to read.",
            "enum": ["docs://intro"],
        }
        assert resolved.tools[-1].is_read_only(uri="docs://intro") is True
        assert resolved.tools[-1].is_concurrency_safe(uri="docs://intro") is True
        assert resolved.tools[-1].is_destructive(uri="docs://intro") is False
    finally:
        await hub.close()


async def test_agent_resolve_dedupes_local_and_mcp_prompt_skills(tmp_path):
    server = tmp_path / "fake_mcp_server.py"
    server.write_text(MCP_SERVER_SCRIPT, encoding="utf-8")
    hub = MCPHub([MCPServer(name="local", command=sys.executable, args=[str(server)])])
    registry = SkillRegistry(
        [
            Skill(
                name="review-docs",
                description="Local review.",
                body="Local review body.",
            )
        ]
    )

    try:
        resolved = await resolve_agent(
            Agent(
                name="repo",
                instructions="Use integrations.",
                skills=registry,
                mcp=hub,
            )
        )

        assert resolved.skills is not None
        assert [skill.name for skill in resolved.skills.all()] == ["review-docs"]
        assert resolved.skills.get("review-docs").description == "Review docs."
        assert [problem.message for problem in resolved.skills.problems()] == [
            "duplicate skill name also used by review-docs"
        ]
        assert resolved.tools[0].input_schema["properties"]["skill"]["enum"] == [
            "review-docs"
        ]
        assert resolved.instructions.count("- review-docs:") == 1
    finally:
        await hub.close()
