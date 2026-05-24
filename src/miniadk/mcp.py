from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core.middleware import ask_before
from .core.tools import Tool, ToolMeta, canonical_tool_name


def _safe_tool_name(value: str) -> str:
    pieces = []
    for char in value.lower():
        if char.isalnum():
            pieces.append(char)
        else:
            pieces.append("_")
    name = "".join(pieces).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return name or "tool"


class MCPError(RuntimeError):
    def __init__(self, error: dict[str, Any]):
        self.error = error
        self.code = error.get("code")
        message = error.get("message", str(error))
        super().__init__(str(message))


class MCPToolError(RuntimeError):
    """Raised when an MCP tool response marks its result as an error."""


@dataclass(slots=True)
class MCPServer:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    cwd: str | Path | None = None
    env: dict[str, str | None] = field(default_factory=dict)
    inherit_env: bool = True
    # 2 minutes — covers MCP server startup (some load big models or
    # warm caches) and slow tool calls. Override on the StdioServer
    # for stricter cases.
    timeout_seconds: float = 120

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("MCP server name is required")
        if not self.command.strip():
            raise ValueError("MCP server command is required")
        if self.timeout_seconds <= 0:
            raise ValueError("MCP server timeout_seconds must be > 0")


@dataclass(slots=True)
class MCPResource:
    uri: str
    name: str
    description: str | None = None
    mime_type: str | None = None


@dataclass(slots=True)
class MCPPrompt:
    name: str
    description: str | None = None
    arguments: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MCPPromptMessage:
    role: str
    content: str


@dataclass(slots=True)
class MCPPromptResult:
    name: str
    messages: list[MCPPromptMessage] = field(default_factory=list)

    def text(self) -> str:
        parts = []
        for message in self.messages:
            parts.append(f"{message.role}: {message.content}")
        return "\n".join(parts).strip()


@dataclass(slots=True)
class MCPInfo:
    name: str
    protocol_version: str | None = None
    server: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MCPNotice:
    server: str
    method: str
    params: dict[str, Any] = field(default_factory=dict)


class _MCPClient:
    def __init__(self, server: MCPServer, protocol_version: str = "2024-11-05"):
        self.server = server
        self.protocol_version = protocol_version
        self.process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._stderr_task: asyncio.Task | None = None
        self._message_id = 0
        self._initialized = False
        self.info = MCPInfo(name=server.name)
        self.notices: list[MCPNotice] = []

    async def start(self) -> None:
        if self.process is not None:
            return
        self.process = await asyncio.create_subprocess_exec(
            self.server.command,
            *self.server.args,
            cwd=str(self.server.cwd) if self.server.cwd is not None else None,
            env=_build_env(self.server.env, inherit=self.server.inherit_env),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        self._reader = self.process.stdout
        self._writer = self.process.stdin
        self._stderr_task = asyncio.create_task(self.process.stderr.read())
        try:
            await self._initialize()
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        if self.process is None:
            return
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=2)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        if self._stderr_task is not None:
            if not self._stderr_task.done():
                self._stderr_task.cancel()
            if self._stderr_task is not asyncio.current_task():
                try:
                    await self._stderr_task
                except asyncio.CancelledError:
                    pass
        self.process = None
        self._reader = None
        self._writer = None
        self._stderr_task = None
        self._initialized = False

    async def list_tools(self) -> list[Tool]:
        await self.start()
        try:
            raw_tools = await self._list_paginated("tools/list", "tools")
        except MCPError as error:
            if error.code == -32601:
                return []
            raise
        tools: list[Tool] = []
        for raw_tool in raw_tools:
            tools.append(self._build_tool(raw_tool))
        return tools

    async def list_resources(self) -> list[MCPResource]:
        await self.start()
        if not self.has("resources"):
            return []
        try:
            raw_resources = await self._list_paginated("resources/list", "resources")
        except MCPError as error:
            if error.code == -32601:
                return []
            raise
        resources: list[MCPResource] = []
        for raw in raw_resources:
            resources.append(
                MCPResource(
                    uri=str(raw["uri"]),
                    name=str(raw.get("name") or raw["uri"]),
                    description=str(raw.get("description")) if raw.get("description") else None,
                    mime_type=str(raw.get("mimeType")) if raw.get("mimeType") else None,
                )
            )
        return resources

    async def read_resource(self, uri: str) -> str:
        await self.start()
        response = await self._request("resources/read", {"uri": uri})
        parts: list[str] = []
        for block in response.get("contents", []):
            if not isinstance(block, dict):
                continue
            if "text" in block:
                parts.append(str(block["text"]))
            elif "blob" in block:
                parts.append(str(block["blob"]))
        if parts:
            return "\n".join(parts).strip()
        if "text" in response:
            return str(response["text"])
        return json.dumps(response, ensure_ascii=False)

    async def list_prompts(self) -> list[MCPPrompt]:
        await self.start()
        if not self.has("prompts"):
            return []
        try:
            raw_prompts = await self._list_paginated("prompts/list", "prompts")
        except MCPError as error:
            if error.code == -32601:
                return []
            raise
        prompts: list[MCPPrompt] = []
        for raw in raw_prompts:
            prompts.append(
                MCPPrompt(
                    name=str(raw["name"]),
                    description=str(raw.get("description")) if raw.get("description") else None,
                    arguments=[
                        str(item.get("name"))
                        for item in raw.get("arguments", [])
                        if isinstance(item, dict) and item.get("name")
                    ],
                )
            )
        return prompts

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> MCPPromptResult:
        await self.start()
        response = await self._request(
            "prompts/get",
            {"name": name, "arguments": arguments or {}},
        )
        messages: list[MCPPromptMessage] = []
        for raw in response.get("messages", []):
            if not isinstance(raw, dict):
                continue
            messages.append(
                MCPPromptMessage(
                    role=str(raw.get("role") or "user"),
                    content=_mcp_content_text(raw.get("content")),
                )
            )
        return MCPPromptResult(name=name, messages=messages)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        await self.start()
        response = await self._request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        text = self._content_text(response)
        if response.get("isError") is True:
            raise MCPToolError(text or f"MCP tool failed: {name}")
        return text or json.dumps(response, ensure_ascii=False)

    @staticmethod
    def _content_text(response: dict[str, Any]) -> str:
        content = _mcp_content_text(response.get("content"))
        if content:
            return content
        if "text" in response:
            return str(response["text"])
        return ""

    async def _initialize(self) -> None:
        if self._initialized:
            return
        response = await self._request(
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {},
                },
                "clientInfo": {"name": "miniadk", "version": "0.1.0"},
            },
        )
        self.info = MCPInfo(
            name=self.server.name,
            protocol_version=str(response.get("protocolVersion"))
            if response.get("protocolVersion") is not None
            else None,
            server=dict(response.get("serverInfo") or {}),
            capabilities=dict(response.get("capabilities") or {}),
        )
        await self._notify("notifications/initialized", {})
        self._initialized = True

    def has(self, capability: str) -> bool:
        return capability in self.info.capabilities

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        message_id = self._next_id()
        try:
            await self._send({"jsonrpc": "2.0", "id": message_id, "method": method, "params": params})
            while True:
                response = await asyncio.wait_for(
                    self._read(),
                    timeout=self.server.timeout_seconds,
                )
                if response.get("id") == message_id:
                    if "error" in response:
                        raise MCPError(response["error"])
                    return response.get("result", {})
                if "id" not in response and "method" in response:
                    self.notices.append(
                        MCPNotice(
                            server=self.server.name,
                            method=str(response["method"]),
                            params=dict(response.get("params") or {}),
                        )
                    )
        except BaseException:
            await self.close()
            raise

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _list_paginated(self, method: str, key: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        cursor: str | None = None
        seen: set[str] = set()
        first = True
        while True:
            params = {} if first else {"cursor": cursor}
            response = await self._request(method, params)
            for item in response.get(key, []):
                if isinstance(item, dict):
                    items.append(item)

            if "nextCursor" not in response or response["nextCursor"] is None:
                return items
            cursor = str(response["nextCursor"])
            if cursor in seen:
                raise RuntimeError(f"MCP pagination repeated cursor for {method}")
            seen.add(cursor)
            first = False

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._writer is None:
            raise RuntimeError("MCP client is not connected")
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
        self._writer.write(header + data)
        await self._writer.drain()

    async def _read(self) -> dict[str, Any]:
        if self._reader is None:
            raise RuntimeError("MCP client is not connected")
        headers: dict[str, str] = {}
        while True:
            line = await self._reader.readline()
            if not line:
                raise RuntimeError("MCP server closed the pipe")
            if line in {b"\r\n", b"\n"}:
                break
            try:
                key, value = line.decode("utf-8").split(":", 1)
            except ValueError as error:
                raise RuntimeError("Invalid MCP header") from error
            headers[key.lower().strip()] = value.strip()
        raw_length = headers.get("content-length")
        if raw_length is None:
            raise RuntimeError("Invalid MCP message: missing Content-Length")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise RuntimeError("Invalid MCP message: invalid Content-Length") from error
        if length < 0:
            raise RuntimeError("Invalid MCP message: invalid Content-Length")
        try:
            body = await self._reader.readexactly(length)
        except asyncio.IncompleteReadError as error:
            raise RuntimeError("Invalid MCP message: truncated body") from error
        try:
            message = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Invalid MCP message: body was not valid JSON") from error
        if not isinstance(message, dict):
            raise RuntimeError("Invalid MCP message: body must be a JSON object")
        return message

    def _next_id(self) -> int:
        self._message_id += 1
        return self._message_id

    def _build_tool(self, raw_tool: dict[str, Any]) -> Tool:
        tool_name = f"mcp__{_safe_tool_name(self.server.name)}__{_safe_tool_name(raw_tool['name'])}"
        description = str(raw_tool.get("description") or raw_tool["name"])
        input_schema = _mcp_tool_schema(raw_tool.get("inputSchema"))
        annotations = _mcp_tool_annotations(raw_tool.get("annotations"))
        raw_name = raw_tool["name"]

        async def call_tool(**kwargs: Any) -> Any:
            return await self.call_tool(raw_name, kwargs)

        return Tool(
            name=tool_name,
            description=description,
            input_schema=input_schema,
            func=call_tool,
            permission=None
            if annotations.get("readOnlyHint") is True
            else ask_before("calling MCP tool"),
            meta=_mcp_tool_meta(annotations),
        )


def _build_env(env: dict[str, str | None], *, inherit: bool) -> dict[str, str]:
    built = dict(os.environ.items()) if inherit else {}
    for key, value in env.items():
        if value is None:
            built.pop(key, None)
        else:
            built[key] = value
    return built


def _validate_unique_server_names(servers: list[MCPServer]) -> None:
    seen: set[str] = set()
    for server in servers:
        if server.name in seen:
            raise ValueError(f"duplicate MCP server name: {server.name}")
        seen.add(server.name)


def _resource_uri_schema(resources: list[MCPResource]) -> dict[str, object]:
    schema: dict[str, object] = {
        "type": "string",
        "description": "MCP resource URI to read.",
    }
    uris = sorted(resource.uri for resource in resources)
    if uris:
        schema["enum"] = uris
    return schema


def _mcp_tool_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return _empty_tool_schema()
    if schema.get("type") != "object":
        return _empty_tool_schema()
    normalized = dict(schema)
    if not isinstance(normalized.get("properties", {}), dict):
        normalized["properties"] = {}
    if not isinstance(normalized.get("required", []), list):
        normalized.pop("required", None)
    normalized.setdefault("additionalProperties", False)
    return normalized


def _mcp_tool_annotations(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: item
        for key, item in value.items()
        if key in {"readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"}
        and isinstance(item, bool)
    }


def _mcp_tool_meta(annotations: dict[str, Any]) -> ToolMeta:
    read_only = annotations.get("readOnlyHint") is True
    destructive = annotations.get("destructiveHint") is True
    concurrency_safe = (
        read_only
        or (
            annotations.get("idempotentHint") is True
            and annotations.get("openWorldHint") is not True
            and not destructive
        )
    )
    return ToolMeta(
        read_only=read_only,
        destructive=destructive,
        concurrency_safe=concurrency_safe,
    )


def _empty_tool_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }


def _mcp_content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text", ""))
        if "text" in content:
            return str(content["text"])
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts = [_mcp_content_text(item) for item in content]
        return "\n".join(part for part in parts if part).strip()
    return str(content)


@dataclass(slots=True)
class MCPHub:
    servers: list[MCPServer] = field(default_factory=list)
    protocol_version: str = "2024-11-05"
    _clients: dict[str, _MCPClient] = field(default_factory=dict, init=False, repr=False)
    _tools_cache: list[Tool] | None = field(default=None, init=False, repr=False)
    _resources_cache: list[MCPResource] | None = field(default=None, init=False, repr=False)
    _prompts_cache: list[MCPPrompt] | None = field(default=None, init=False, repr=False)
    _resource_clients: dict[str, _MCPClient] = field(default_factory=dict, init=False, repr=False)
    _prompt_clients: dict[str, _MCPClient] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_unique_server_names(self.servers)

    async def __aenter__(self) -> "MCPHub":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def tools(self) -> list[Tool]:
        if self._tools_cache is not None:
            return list(self._tools_cache)

        tools: list[Tool] = []
        for server in self.servers:
            client = self._clients.get(server.name)
            if client is None:
                client = _MCPClient(server, protocol_version=self.protocol_version)
                self._clients[server.name] = client
            tools.extend(await client.list_tools())

        deduped: dict[str, Tool] = {}
        for tool in tools:
            deduped[canonical_tool_name(tool.name)] = tool

        self._tools_cache = list(deduped.values())
        return list(self._tools_cache)

    async def info(self) -> list[MCPInfo]:
        infos: list[MCPInfo] = []
        for client in await self._clients_in_order():
            await client.start()
            infos.append(client.info)
        return infos

    async def resources(self) -> list[MCPResource]:
        if self._resources_cache is not None:
            return list(self._resources_cache)

        self._resource_clients.clear()
        resources: list[MCPResource] = []
        for client in await self._clients_in_order():
            for resource in await client.list_resources():
                resources.append(resource)
                self._resource_clients[resource.uri] = client

        deduped: dict[str, MCPResource] = {}
        for resource in resources:
            deduped[resource.uri] = resource

        self._resources_cache = list(deduped.values())
        return list(self._resources_cache)

    async def read_resource(self, uri: str) -> str:
        client = await self._client_for_resource(uri)
        return await client.read_resource(uri)

    async def prompts(self) -> list[MCPPrompt]:
        if self._prompts_cache is not None:
            return list(self._prompts_cache)

        self._prompt_clients.clear()
        prompts: list[MCPPrompt] = []
        for client in await self._clients_in_order():
            for prompt in await client.list_prompts():
                prompts.append(prompt)
                self._prompt_clients[prompt.name] = client

        deduped: dict[str, MCPPrompt] = {}
        for prompt in prompts:
            deduped[prompt.name] = prompt

        self._prompts_cache = list(deduped.values())
        return list(self._prompts_cache)

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPPromptResult:
        client = await self._client_for_prompt(name)
        return await client.get_prompt(name, arguments)

    async def resource_tool(self) -> Tool | None:
        resources = await self.resources()
        if not resources:
            return None

        hub = self

        async def read_mcp_resource(uri: str) -> str:
            """Read a resource exposed by an MCP server."""
            return await hub.read_resource(uri)

        return Tool(
            name="read_mcp_resource",
            description="Read a resource exposed by an MCP server.",
            input_schema={
                "type": "object",
                "properties": {"uri": _resource_uri_schema(resources)},
                "required": ["uri"],
                "additionalProperties": False,
            },
            func=read_mcp_resource,
            meta=ToolMeta(read_only=True, concurrency_safe=True),
        )

    async def skills(self):
        from .skills import Skill, SkillRegistry

        prompts = await self.prompts()
        skills = []
        for prompt in prompts:
            try:
                body = (await self.get_prompt(prompt.name)).text()
            except MCPError:
                args = ", ".join(prompt.arguments) or "$ARGUMENTS"
                body = (
                    f"Use the MCP prompt named {prompt.name} with arguments: {args}."
                )
            skills.append(
                Skill(
                    name=prompt.name,
                    description=prompt.description or prompt.name,
                    body=body or (prompt.description or prompt.name),
                    allowed_tools=[],
                    arguments=list(prompt.arguments),
                    user_invocable=True,
                    model_invocable=True,
                    metadata={"source": "mcp"},
                )
        )
        return SkillRegistry(skills=skills)

    def refresh(self) -> None:
        """Forget discovered MCP tools, resources, and prompts without closing servers."""
        self._tools_cache = None
        self._resources_cache = None
        self._prompts_cache = None
        self._resource_clients.clear()
        self._prompt_clients.clear()

    async def close(self) -> None:
        for client in self._clients.values():
            await client.close()
        self._clients.clear()
        self.refresh()

    def notices(self) -> list[MCPNotice]:
        notices: list[MCPNotice] = []
        for client in self._clients.values():
            notices.extend(client.notices)
        return notices

    def clear_notices(self) -> None:
        for client in self._clients.values():
            client.notices.clear()

    async def _clients_in_order(self) -> list[_MCPClient]:
        clients: list[_MCPClient] = []
        for server in self.servers:
            client = self._clients.get(server.name)
            if client is None:
                client = _MCPClient(server, protocol_version=self.protocol_version)
                self._clients[server.name] = client
            clients.append(client)
        return clients

    async def _client_for_resource(self, uri: str) -> _MCPClient:
        client = self._resource_clients.get(uri)
        if client is not None:
            return client

        await self.resources()
        client = self._resource_clients.get(uri)
        if client is not None:
            return client
        raise RuntimeError(f"Unknown MCP resource: {uri}")

    async def _client_for_prompt(self, name: str) -> _MCPClient:
        client = self._prompt_clients.get(name)
        if client is not None:
            return client

        await self.prompts()
        client = self._prompt_clients.get(name)
        if client is not None:
            return client
        raise RuntimeError(f"Unknown MCP prompt: {name}")
