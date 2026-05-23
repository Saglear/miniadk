from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4

from .messages import Message
from .tools import Tool


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict
    id: str = field(default_factory=lambda: f"call_{uuid4().hex[:8]}")


@dataclass(slots=True)
class ModelResult:
    message: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    content_blocks: list[object] | None = None


@dataclass(slots=True)
class ToolCallDelta:
    index: int = 0
    id: str | None = None
    name: str | None = None
    arguments: str | None = None


@dataclass(slots=True)
class ModelStreamEvent:
    delta: str | None = None
    thinking: str | None = None
    tool_call: ToolCallDelta | None = None
    result: ModelResult | None = None


class Model(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool],
    ) -> ModelResult:
        ...


class StreamingModel(Model, Protocol):
    def stream(
        self,
        messages: list[Message],
        tools: list[Tool],
    ) -> AsyncGenerator[ModelStreamEvent, None]:
        ...


class ScriptedModel:
    def __init__(self, results: list[ModelResult]):
        self._results = list(results)
        self.calls: list[tuple[list[Message], list[Tool]]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool],
    ) -> ModelResult:
        self.calls.append((list(messages), list(tools)))
        if not self._results:
            return ModelResult(message="")
        return self._results.pop(0)
