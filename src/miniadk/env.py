import os
from pathlib import Path


def load_env(path: str | Path = ".env", *, override: bool = False) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(_strip_inline_comment(value.strip()))

        if not key:
            continue

        if override or key not in os.environ:
            os.environ[key] = value
            loaded[key] = value

    return loaded


def load_env_upwards(
    *,
    start: str | Path = ".",
    filename: str = ".env",
    override: bool = False,
) -> Path | None:
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for directory in [current, *current.parents]:
        candidate = directory / filename
        if candidate.exists():
            load_env(candidate, override=override)
            return candidate
    return None


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
        elif char == "#" and quote is None:
            return value[:index].rstrip()
    return value
