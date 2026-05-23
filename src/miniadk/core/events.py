from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Event:
    type: str
    data: dict[str, Any]

