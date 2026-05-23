import json
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .messages import Message
from .model import Model, ToolCall


@dataclass(frozen=True, slots=True)
class SessionStats:
    messages: int = 0
    system: int = 0
    user: int = 0
    assistant: int = 0
    tool: int = 0
    tool_calls: int = 0
    chars: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "messages": self.messages,
            "system": self.system,
            "user": self.user,
            "assistant": self.assistant,
            "tool": self.tool,
            "tool_calls": self.tool_calls,
            "chars": self.chars,
        }


@dataclass(slots=True)
class Session:
    messages: list[Message] = field(default_factory=list)

    @property
    def stats(self) -> SessionStats:
        counts = {
            "system": 0,
            "user": 0,
            "assistant": 0,
            "tool": 0,
        }
        tool_calls = 0
        chars = 0
        for message in self.messages:
            if message.role in counts:
                counts[message.role] += 1
            chars += len(message.content)
            tool_calls += len(message.tool_calls or [])
        return SessionStats(
            messages=len(self.messages),
            system=counts["system"],
            user=counts["user"],
            assistant=counts["assistant"],
            tool=counts["tool"],
            tool_calls=tool_calls,
            chars=chars,
        )

    def compact(self, summary: str, *, keep: int = 10) -> None:
        system = (
            self.messages[:1]
            if self.messages and self.messages[0].role == "system"
            else []
        )
        history = self.messages[1:] if system else self.messages
        recent = history[-keep:] if keep > 0 else []
        self.messages[:] = [
            *system,
            Message("system", summary),
            *recent,
        ]

    def trim(self, *, keep: int = 10) -> None:
        system = (
            self.messages[:1]
            if self.messages and self.messages[0].role == "system"
            else []
        )
        history = self.messages[1:] if system else self.messages
        recent = history[-keep:] if keep > 0 else []
        self.messages[:] = [*system, *recent]

    async def summarize(
        self,
        *,
        model: Model,
        keep: int = 10,
        prompt: str | None = None,
    ) -> str:
        summary_prompt = prompt or _SUMMARY_PROMPT
        transcript = self.transcript(exclude_recent=keep)
        if not transcript.strip():
            return ""
        result = await model.complete(
            messages=[
                Message("system", summary_prompt),
                Message("user", transcript),
            ],
            tools=[],
        )
        summary = (result.message or "").strip()
        self.compact(summary, keep=keep)
        return summary

    def transcript(
        self,
        *,
        keep: int | None = None,
        exclude_recent: int = 0,
    ) -> str:
        messages = self.messages
        if exclude_recent > 0:
            messages = messages[:-exclude_recent]
        if keep is not None:
            messages = messages[-keep:]
        lines = []
        for message in messages:
            if message.role == "system" and message is self.messages[0]:
                continue
            label = message.role
            if message.role == "tool" and message.name:
                label = f"tool:{message.name}"
            content = message.content.strip()
            if message.tool_calls:
                calls = ", ".join(call.name for call in message.tool_calls)
                content = f"{content}\n[tool calls: {calls}]" if content else f"[tool calls: {calls}]"
            lines.append(f"{label}: {content}")
        return "\n".join(lines)

    def branch(self, *, keep: int | None = None) -> "Session":
        messages = self.messages
        if keep is not None:
            system = (
                messages[:1]
                if messages and messages[0].role == "system"
                else []
            )
            history = messages[1:] if system else messages
            messages = [*system, *(history[-keep:] if keep > 0 else [])]
        return self.from_dict(
            {"messages": [self._message_to_dict(message) for message in messages]}
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        _write_text_atomic(
            target,
            json.dumps(self.to_dict(), indent=2),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Session":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "messages": [self._message_to_dict(message) for message in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        return cls(
            messages=[
                cls._message_from_dict(message)
                for message in data.get("messages", [])
            ]
        )

    @staticmethod
    def _message_to_dict(message: Message) -> dict[str, Any]:
        return {
            "role": message.role,
            "content": message.content,
            "name": message.name,
            "tool_call_id": message.tool_call_id,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": deepcopy(call.arguments),
                }
                for call in (message.tool_calls or [])
            ] or None,
            "content_blocks": deepcopy(message.content_blocks),
        }

    @staticmethod
    def _message_from_dict(data: dict[str, Any]) -> Message:
        tool_calls = data.get("tool_calls")
        return Message(
            role=data["role"],
            content=data.get("content", ""),
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
            tool_calls=[
                ToolCall(
                    id=call.get("id", ""),
                    name=call["name"],
                    arguments=call.get("arguments", {}),
                )
                for call in tool_calls
            ] if tool_calls else None,
            content_blocks=data.get("content_blocks"),
        )


_SUMMARY_PROMPT = """
Summarize the conversation so far for a future agent turn. Preserve user goals,
important decisions, files or tools mentioned, and unresolved next steps. Keep it
short and factual.
""".strip()


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
