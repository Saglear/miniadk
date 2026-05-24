import os
import json
from typing import Any
from urllib.error import HTTPError

from ..core.messages import Message
from ..core.model import ModelResult, ModelStreamEvent, ToolCall, ToolCallDelta
from ..core.tools import Tool
from ._errors import http_error
from ._http import JsonHttpClient
from ._schema import tool_parameters


class AnthropicModel:
    """Anthropic Messages API adapter."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        http_client: JsonHttpClient | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        retries: int | None = None,
        retry_delay: float | None = None,
        opts: dict[str, Any] | None = None,
        anthropic_version: str = "2023-06-01",
    ):
        self.api_key = (
            api_key
            or _env(
                "ANTHROPIC_KEY",
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_AUTH_TOKEN",
                "MINIADK_MODEL_KEY",
            )
        )
        self.base_url = (
            base_url
            or _env("ANTHROPIC_URL", "ANTHROPIC_BASE_URL", "MINIADK_MODEL_URL")
            or "https://api.anthropic.com"
        ).rstrip("/")
        self.model = (
            model
            or _env("ANTHROPIC_MODEL", "MINIADK_MODEL_NAME")
            or "claude-4-5-haiku-latest"
        )
        self.http_client = http_client or JsonHttpClient(
            timeout_seconds=_transport_float(
                timeout, 600, "ANTHROPIC_TIMEOUT", "MINIADK_MODEL_TIMEOUT"
            ),
            retries=_transport_int(
                retries, 0, "ANTHROPIC_RETRIES", "MINIADK_MODEL_RETRIES"
            ),
            retry_delay=_transport_float(
                retry_delay, 0.25, "ANTHROPIC_RETRY_DELAY", "MINIADK_MODEL_RETRY_DELAY"
            ),
        )
        self.max_tokens = (
            _int_env("ANTHROPIC_MAX_TOKENS", "MINIADK_MODEL_MAX_TOKENS")
            if max_tokens is None
            else max_tokens
        ) or _default_max_tokens(self.model)
        self.temperature = (
            _float_env("ANTHROPIC_TEMPERATURE", "MINIADK_MODEL_TEMPERATURE")
            if temperature is None
            else temperature
        )
        self.opts = dict(opts or {})
        self.anthropic_version = anthropic_version

        if not self.api_key:
            raise ValueError(
                "AnthropicModel requires api_key, "
                "ANTHROPIC_KEY/ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN, "
                "or MINIADK_MODEL_KEY"
            )

    @property
    def endpoint_url(self) -> str:
        if self.base_url.endswith("/v1/messages"):
            return self.base_url
        return f"{self.base_url}/v1/messages"

    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool],
    ) -> ModelResult:
        try:
            response = await self.http_client.post_json(
                url=self.endpoint_url,
                payload=self.build_payload(messages, tools),
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": self.anthropic_version,
                },
            )
        except HTTPError as error:
            raise self._http_error(error) from error
        return self.parse_response(response)

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool],
    ):
        content_blocks: dict[int, dict[str, Any]] = {}
        text_parts: list[str] = []
        try:
            async for chunk in self.http_client.post_sse(
                url=self.endpoint_url,
                payload=self.build_payload(messages, tools, stream=True),
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": self.anthropic_version,
                },
            ):
                event_type = chunk.get("type")
                if event_type == "error":
                    error = chunk.get("error") or {}
                    raise RuntimeError(f"Model stream failed: {error.get('message', error)}")
                if event_type == "content_block_start":
                    index = int(chunk.get("index", 0))
                    content_blocks[index] = dict(chunk.get("content_block") or {})
                elif event_type == "content_block_delta":
                    index = int(chunk.get("index", 0))
                    block = content_blocks.setdefault(index, {})
                    delta = chunk.get("delta") or {}
                    delta_type = delta.get("type")
                    if delta_type == "text_delta":
                        text = delta.get("text", "")
                        block["text"] = block.get("text", "") + text
                        text_parts.append(text)
                        if text:
                            yield ModelStreamEvent(delta=text)
                    elif delta_type == "input_json_delta":
                        partial = delta.get("partial_json", "")
                        block["_partial_json"] = block.get("_partial_json", "") + partial
                        if partial:
                            yield ModelStreamEvent(
                                tool_call=ToolCallDelta(
                                    index=index,
                                    id=block.get("id"),
                                    name=block.get("name"),
                                    arguments=partial,
                                )
                            )
                    elif delta_type == "thinking_delta":
                        thinking = delta.get("thinking", "")
                        block["thinking"] = block.get("thinking", "") + thinking
                        if thinking:
                            yield ModelStreamEvent(thinking=thinking)
                    elif delta_type == "signature_delta":
                        block["signature"] = block.get("signature", "") + delta.get("signature", "")
        except HTTPError as error:
            raise self._http_error(error) from error

        blocks = self._finalize_stream_blocks(content_blocks)
        message = self._message_from_blocks(blocks) or "".join(text_parts) or None
        yield ModelStreamEvent(
            result=ModelResult(
                message=message,
                tool_calls=self._tool_calls_from_blocks(blocks),
                content_blocks=blocks or None,
            )
        )

    def build_payload(
        self,
        messages: list[Message],
        tools: list[Tool],
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        system_parts = [message.content for message in messages if message.role == "system"]
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._messages_to_api(messages),
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        payload.update(self.opts)

        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if stream:
            payload["stream"] = True

        if tools:
            payload["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool_parameters(tool),
                }
                for tool in tools
            ]

        return payload

    def _messages_to_api(self, messages: list[Message]) -> list[dict[str, Any]]:
        api_messages: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        def flush_tool_results() -> None:
            nonlocal tool_results
            if tool_results:
                api_messages.append({"role": "user", "content": tool_results})
                tool_results = []

        for message in messages:
            if message.role == "system":
                continue

            if message.role == "tool":
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id or "",
                        "content": message.content,
                    }
                )
                continue

            flush_tool_results()
            api_messages.append(self._message_to_api(message))

        flush_tool_results()
        return api_messages

    def parse_response(self, response: dict[str, Any]) -> ModelResult:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        content_blocks = list(response.get("content", []))

        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                if "id" not in block or "name" not in block:
                    raise RuntimeError("Anthropic tool_use block missing id or name")
                if not isinstance(block.get("input", {}), dict):
                    raise RuntimeError("Anthropic tool_use block input must be an object")
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block.get("input", {}),
                    )
                )

        message = "\n".join(part for part in text_parts if part)
        return ModelResult(
            message=message or None,
            tool_calls=tool_calls,
            content_blocks=content_blocks or None,
        )

    def _message_to_api(self, message: Message) -> dict[str, Any]:
        if message.role == "assistant" and message.tool_calls:
            if message.content_blocks is not None:
                return {"role": "assistant", "content": list(message.content_blocks)}
            content: list[dict[str, Any]] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            content.extend(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
                for call in message.tool_calls
            )
            return {"role": "assistant", "content": content}

        if message.role == "assistant" and message.content_blocks is not None:
            return {"role": "assistant", "content": list(message.content_blocks)}

        return {
            "role": message.role,
            "content": message.content,
        }

    def _finalize_stream_blocks(
        self,
        content_blocks: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        blocks = []
        for index in sorted(content_blocks):
            block = dict(content_blocks[index])
            partial_json = block.pop("_partial_json", None)
            if block.get("type") == "tool_use":
                source = partial_json if partial_json else "{}"
                parsed, repair_note = _parse_tool_input(source)
                if not isinstance(parsed, dict):
                    raise RuntimeError("Anthropic streamed tool_use input must be an object")
                block["input"] = parsed
                if repair_note:
                    # Stash on the block so the runtime / TUI can surface
                    # a friendly "tool input was truncated" notice rather
                    # than mistaking a partial result for a real one.
                    block["_partial_json_repair"] = repair_note
            blocks.append(block)
        return blocks

    def _tool_calls_from_blocks(self, blocks: list[dict[str, Any]]) -> list[ToolCall]:
        tool_calls = []
        for block in blocks:
            if block.get("type") != "tool_use":
                continue
            if "id" not in block or "name" not in block:
                raise RuntimeError("Anthropic streamed tool_use block missing id or name")
            tool_calls.append(
                ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=block.get("input", {}),
                )
            )
        return tool_calls

    @staticmethod
    def _message_from_blocks(blocks: list[dict[str, Any]]) -> str | None:
        text = "\n".join(
            str(block.get("text", ""))
            for block in blocks
            if block.get("type") == "text" and block.get("text")
        )
        return text or None

    @staticmethod
    def _http_error(error: HTTPError) -> RuntimeError:
        return http_error(error)


def _env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _float_env(*names: str) -> float | None:
    value = _env(*names)
    return None if value is None or value == "" else float(value)


def _int_env(*names: str) -> int | None:
    value = _env(*names)
    return None if value is None or value == "" else int(value)


def _transport_float(explicit: float | None, default: float, *env: str) -> float:
    if explicit is not None:
        return explicit
    value = _float_env(*env)
    return default if value is None else value


def _transport_int(explicit: int | None, default: int, *env: str) -> int:
    if explicit is not None:
        return explicit
    value = _int_env(*env)
    return default if value is None else value


# Per-model default ``max_tokens`` — the **response** ceiling, distinct
# from the model's context window. Curated for 2026-05 (current model
# generation only — older families are dropped because they're not in
# active use). Values reflect each provider's published output ceiling
# at the time of writing; if a number is wrong, override via the
# ``ANTHROPIC_MAX_TOKENS`` env var or the ``max_tokens`` constructor arg.
#
# Keys are matched as case-insensitive substrings, so dated aliases like
# ``claude-opus-4-7-20260416`` pick up the right entry without us
# enumerating every variant.
_MAX_TOKENS_TABLE: tuple[tuple[str, int], ...] = (
    # ── Anthropic Claude 4.x ─────────────────────────────────────────
    ("claude-opus-4", 64_000),
    ("claude-sonnet-4", 64_000),
    ("claude-haiku-4", 16_384),
    # ── OpenAI GPT-5.x (used over Anthropic-compat proxies) ──────────
    ("gpt-5-pro", 128_000),
    ("gpt-5-codex", 128_000),
    ("gpt-5.5", 128_000),
    ("gpt-5.4", 128_000),
    ("gpt-5.3", 128_000),
    ("gpt-5.1", 128_000),
    ("gpt-5", 128_000),
    # ── DeepSeek V4 ──────────────────────────────────────────────────
    ("deepseek-v4-pro", 65_536),
    ("deepseek-v4-flash", 32_768),
    ("deepseek-v4", 32_768),
    # ── MiniMax 2.x ──────────────────────────────────────────────────
    ("minimax-2.7", 65_536),
    ("minimax-2.5", 65_536),
    ("minimax2.7", 65_536),
    ("minimax2.5", 65_536),
    # ── Zhipu GLM 5.x ────────────────────────────────────────────────
    ("glm-5.1", 32_768),
    ("glm-5", 32_768),
    # ── Moonshot Kimi K2.x ───────────────────────────────────────────
    ("kimi-2.6", 65_536),
    ("kimi-2.5", 65_536),
    ("kimi2.6", 65_536),
    ("kimi2.5", 65_536),
    # ── Alibaba Qwen 3.x (still widely deployed in 2026) ─────────────
    ("qwen3-coder", 65_536),
    ("qwen3", 65_536),
)

# Fallback when the model name doesn't match any known family. Picked
# at 32k — covers a typical "modern frontier model" output cap without
# being so high that smaller endpoints reject the request. Users on
# something with a higher cap can override via env or constructor arg.
_DEFAULT_MAX_TOKENS_FALLBACK = 32_768


def _default_max_tokens(model: str | None) -> int:
    if not model:
        return _DEFAULT_MAX_TOKENS_FALLBACK
    needle = model.lower()
    for prefix, value in _MAX_TOKENS_TABLE:
        if prefix in needle:
            return value
    return _DEFAULT_MAX_TOKENS_FALLBACK


def _parse_tool_input(source: str) -> tuple[Any, str | None]:
    """Parse a streamed tool_use input JSON.

    Anthropic's streaming protocol assembles a tool's ``input`` from
    ``input_json_delta`` events. Some upstreams (proxies, max_tokens
    truncation) deliver an incomplete JSON document. Rather than
    failing the entire turn we attempt to repair the payload by
    closing dangling strings / objects / arrays so the model can at
    least see *what it tried to call* and recover on the next turn.

    Returns a ``(parsed, repair_note)`` tuple. ``repair_note`` is
    ``None`` when the payload parsed cleanly; otherwise it describes
    the repair so callers can flag the call as suspect.
    """

    try:
        return json.loads(source), None
    except json.JSONDecodeError as error:
        first_error = error

    repaired = _repair_truncated_json(source)
    if repaired is not None:
        try:
            return json.loads(repaired), f"truncated input was auto-repaired (orig len={len(source)})"
        except json.JSONDecodeError:
            pass

    preview = source if len(source) <= 200 else f"{source[:100]}…{source[-100:]}"
    note = (
        f"tool input was not valid JSON and could not be repaired "
        f"(len={len(source)}, error={first_error.msg} at pos {first_error.pos}): {preview!r}"
    )
    return {"_miniadk_invalid_input": source}, note


def _repair_truncated_json(source: str) -> str | None:
    """Best-effort close of an unterminated JSON document.

    Walks the string tracking quote / brace / bracket state and
    appends whatever closers are missing. Handles backslash escapes
    inside strings. Conservative — bails on anything weirder than a
    single trailing-cut document.
    """

    if not source:
        return None
    stack: list[str] = []
    in_string = False
    escape = False
    last_value_terminator = -1
    expecting_value = False
    for i, char in enumerate(source):
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            stack.append("}" if char == "{" else "]")
            expecting_value = char == "{"
            continue
        if char in "}]":
            if not stack:
                return None
            stack.pop()
            expecting_value = False
            continue
        if char == ":":
            expecting_value = True
        elif char == ",":
            expecting_value = True
        elif not char.isspace():
            expecting_value = False

    closers = list(stack)
    if not in_string and not closers:
        return None  # well-formed already, no repair needed

    suffix = ""
    if in_string:
        # We just closed a string — that's a value. Don't add `null`
        # afterwards.
        suffix += '"'
        expecting_value = False
    # If we cut after a `:` or `,` with no value yet, the still-open
    # object/array needs a value. Insert null so the JSON is valid.
    if expecting_value and closers and closers[-1] in {"}", "]"}:
        suffix += "null"
    while closers:
        suffix += closers.pop()
    return source + suffix
