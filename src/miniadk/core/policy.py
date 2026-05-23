from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from .messages import Message
from .model import ModelResult
from .tools import Tool

StopReason = Literal[
    "completed",
    "max_steps",
    "policy_stop",
    "tool_error",
    "permission_denied",
]


@dataclass(slots=True)
class RunState:
    step: int
    max_steps: int
    messages: list[Message]
    active_tools: list[Tool]
    result: ModelResult | None = None
    run_id: str | None = None


@dataclass(slots=True)
class RunDecision:
    action: Literal["continue", "stop"] = "continue"
    reason: StopReason | str = "completed"
    message: str | None = None
    inject: list[Message] = field(default_factory=list)

    @classmethod
    def continue_with(cls, *messages: Message) -> "RunDecision":
        return cls(action="continue", inject=list(messages))

    @classmethod
    def stop(cls, message: str | None = None, reason: StopReason | str = "completed") -> "RunDecision":
        return cls(action="stop", message=message, reason=reason)


class RunHook(Protocol):
    async def on_stop(self, state: RunState) -> RunDecision | None:
        ...


class RunPolicy(Protocol):
    async def after_model(self, state: RunState) -> RunDecision:
        ...

    async def after_tools(self, state: RunState) -> RunDecision:
        ...


class DefaultRunPolicy:
    async def after_model(self, state: RunState) -> RunDecision:
        result = state.result
        if result is not None and result.message is not None and not result.tool_calls:
            return RunDecision.stop(result.message)
        return RunDecision()

    async def after_tools(self, state: RunState) -> RunDecision:
        return RunDecision()
