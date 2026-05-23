from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote

from .core.session import Session
from .core.model import Model


@dataclass(frozen=True, slots=True)
class Compact:
    chars: int | None = 80_000
    keep: int = 10
    prompt: str | None = None


CompactSpec = Compact | int | bool | None


@dataclass(frozen=True, slots=True)
class SessionStore:
    root: Path
    suffix: str = ".json"

    def __init__(self, root: str | Path = ".miniadk/sessions", *, suffix: str = ".json"):
        if not suffix or "/" in suffix or "\\" in suffix:
            raise ValueError("session suffix must be a file suffix")
        object.__setattr__(self, "root", Path(root))
        object.__setattr__(self, "suffix", suffix)

    def path(self, name: str = "main") -> Path:
        return self.root / f"{_encode_name(name)}{self.suffix}"

    def load(self, name: str = "main") -> Session:
        path = self.path(name)
        if not path.exists():
            return Session()
        return Session.load(path)

    def save(self, session: Session, name: str = "main") -> None:
        session.save(self.path(name))

    def exists(self, name: str = "main") -> bool:
        return self.path(name).exists()

    def delete(self, name: str = "main") -> bool:
        path = self.path(name)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True

    def names(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            _decode_name(path.name[: -len(self.suffix)])
            for path in self.root.glob(f"*{self.suffix}")
            if path.is_file()
        )


def sessions(root: str | Path = ".miniadk/sessions") -> SessionStore:
    return SessionStore(root)


async def compact(
    session: Session,
    *,
    model: Model,
    spec: CompactSpec = None,
) -> str:
    config = _compact_config(spec)
    if config is None:
        return ""
    if config.chars is not None and session.stats.chars <= config.chars:
        return ""
    return await session.summarize(
        model=model,
        keep=config.keep,
        prompt=config.prompt,
    )


def _compact_config(spec: CompactSpec) -> Compact | None:
    if spec is None or spec is False:
        return None
    if spec is True:
        return Compact()
    if isinstance(spec, int):
        return Compact(chars=spec)
    if isinstance(spec, Compact):
        return spec
    raise TypeError("compact must be a Compact, int, bool, or None")


def _encode_name(name: str) -> str:
    text = str(name).strip()
    if not text:
        raise ValueError("session name cannot be empty")
    return quote(text, safe="-_.@")


def _decode_name(name: str) -> str:
    return unquote(name)
