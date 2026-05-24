import json
import os
from typing import Any
from urllib.error import HTTPError

from ..core.messages import Message
from ..core.model import ModelResult, ModelStreamEvent, ToolCall, ToolCallDelta
from ..core.tools import Tool
from ._errors import http_error
from ._http import JsonHttpClient
from ._schema import tool_parameters


class OpenAIModel:
    """OpenAI-compatible chat completions adapter.

    This adapter is intentionally small. It speaks the common
    `/chat/completions` shape used by OpenAI and many OpenAI-compatible
    gateways.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        http_client: JsonHttpClient | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        retries: int | None = None,
        retry_delay: float | None = None,
        opts: dict[str, Any] | None = None,
    ):
        self.api_key = (
            api_key
            or _env("OPENAI_KEY", "OPENAI_API_KEY", "MINIADK_MODEL_KEY")
        )
        self.base_url = (
            base_url
            or _env("OPENAI_URL", "OPENAI_BASE_URL", "MINIADK_MODEL_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = (
            model
            or _env("OPENAI_MODEL", "MINIADK_MODEL_NAME")
            or _default_model_for_url(self.base_url)
        )
        self.http_client = http_client or JsonHttpClient(
            timeout_seconds=_transport_float(
                timeout, 600, "OPENAI_TIMEOUT", "MINIADK_MODEL_TIMEOUT"
            ),
            retries=_transport_int(retries, 0, "OPENAI_RETRIES", "MINIADK_MODEL_RETRIES"),
            retry_delay=_transport_float(
                retry_delay, 0.25, "OPENAI_RETRY_DELAY", "MINIADK_MODEL_RETRY_DELAY"
            ),
        )
        self.temperature = (
            _float_env("OPENAI_TEMPERATURE", "MINIADK_MODEL_TEMPERATURE")
            if temperature is None
            else temperature
        )
        self.max_tokens = (
            _int_env("OPENAI_MAX_TOKENS", "MINIADK_MODEL_MAX_TOKENS")
            if max_tokens is None
            else max_tokens
        )
        self.opts = dict(opts or {})

        if not self.api_key:
            raise ValueError(
                "OpenAIModel requires api_key, OPENAI_KEY/OPENAI_API_KEY, "
                "or MINIADK_MODEL_KEY"
            )

    @property
    def endpoint_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool],
    ) -> ModelResult:
        try:
            response = await self.http_client.post_json(
                url=self.endpoint_url,
                payload=self.build_payload(messages, tools),
                headers={"authorization": f"Bearer {self.api_key}"},
            )
        except HTTPError as error:
            raise self._http_error(error) from error
        return self.parse_response(response)

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool],
    ):
        text_parts: list[str] = []
        tool_call_parts: dict[int, dict[str, Any]] = {}
        try:
            async for chunk in self.http_client.post_sse(
                url=self.endpoint_url,
                payload=self.build_payload(messages, tools, stream=True),
                headers={"authorization": f"Bearer {self.api_key}"},
            ):
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    text_parts.append(content)
                    yield ModelStreamEvent(delta=content)
                for call in delta.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        raise RuntimeError("OpenAI streamed tool call must be an object")
                    index = int(call.get("index", 0))
                    part = tool_call_parts.setdefault(
                        index,
                        {"id": "", "name": "", "arguments": ""},
                    )
                    if call.get("id"):
                        part["id"] = call["id"]
                    function = call.get("function")
                    if function is not None and not isinstance(function, dict):
                        raise RuntimeError("OpenAI streamed tool call missing function")
                    function = function or {}
                    if function.get("name"):
                        part["name"] = function["name"]
                    argument_delta = _stream_tool_arguments(function.get("arguments"))
                    if argument_delta:
                        part["arguments"] += argument_delta
                    yield ModelStreamEvent(
                        tool_call=ToolCallDelta(
                            index=index,
                            id=call.get("id"),
                            name=function.get("name"),
                            arguments=argument_delta,
                        )
                    )
        except HTTPError as error:
            raise self._http_error(error) from error

        yield ModelStreamEvent(
            result=ModelResult(
                message="".join(text_parts) or None,
                tool_calls=self._parse_streamed_tool_calls(tool_call_parts),
            )
        )

    def build_payload(
        self,
        messages: list[Message],
        tools: list[Tool],
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._message_to_api(message) for message in messages],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        payload.update(self.opts)
        if stream:
            payload["stream"] = True

        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool_parameters(tool),
                    },
                }
                for tool in tools
            ]

        return payload

    def parse_response(self, response: dict[str, Any]) -> ModelResult:
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("OpenAI response did not include a message") from error
        tool_calls = []

        for call in message.get("tool_calls") or []:
            function = call.get("function")
            if not isinstance(function, dict):
                raise RuntimeError("OpenAI tool call missing function")
            if not function.get("name"):
                raise RuntimeError("OpenAI tool call missing function name")
            raw_arguments = function.get("arguments") or "{}"
            arguments = _parse_tool_arguments(raw_arguments, function.get("name", ""))
            tool_calls.append(
                ToolCall(
                    id=call.get("id", ""),
                    name=function["name"],
                    arguments=arguments,
                )
            )

        return ModelResult(
            message=message.get("content"),
            tool_calls=tool_calls,
        )

    def _message_to_api(self, message: Message) -> dict[str, Any]:
        if message.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id or "",
                "content": message.content,
            }

        api_message: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }

        if message.tool_calls:
            api_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ]

        return api_message

    def _parse_streamed_tool_calls(
        self,
        tool_call_parts: dict[int, dict[str, Any]],
    ) -> list[ToolCall]:
        tool_calls = []
        for index in sorted(tool_call_parts):
            part = tool_call_parts[index]
            if not part["name"]:
                continue
            arguments = _parse_tool_arguments(part["arguments"] or "{}", part["name"])
            tool_calls.append(
                ToolCall(
                    id=part["id"],
                    name=part["name"],
                    arguments=arguments,
                )
            )
        return tool_calls

    @staticmethod
    def _http_error(error: HTTPError) -> RuntimeError:
        return http_error(error)


def _default_model_for_url(base_url: str) -> str:
    if "deepseek" in base_url.lower():
        return "deepseek-chat"
    return "gpt-4.1-mini"


def _parse_tool_arguments(raw: Any, name: str) -> dict:
    if isinstance(raw, dict):
        return raw
    if raw is None:
        parsed = {}
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError as error:
            raise RuntimeError(f"OpenAI tool call {name or '<unknown>'} arguments are not valid JSON") from error
    else:
        raise RuntimeError(f"OpenAI tool call {name or '<unknown>'} arguments must be a JSON object")
    if not isinstance(parsed, dict):
        raise RuntimeError(f"OpenAI tool call {name or '<unknown>'} arguments must be a JSON object")
    return parsed


def _stream_tool_arguments(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return json.dumps(raw)
    raise RuntimeError("OpenAI streamed tool call arguments must be a JSON object or string")


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
