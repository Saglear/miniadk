from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Message:
    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[Any] | None = None
    content_blocks: list[Any] | None = None
