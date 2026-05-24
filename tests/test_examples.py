"""Examples should at least parse + import.

We don't actually run them (most need a real LLM key), but every
example file should compile cleanly so a reader can `python` it
without a syntax/import surprise. Catches the case where an example
references a removed symbol after a refactor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _example_files() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize("path", _example_files(), ids=lambda p: p.name)
def test_example_compiles(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")


def test_examples_have_a_readme() -> None:
    readme = EXAMPLES_DIR / "README.md"
    assert readme.exists(), "examples/README.md is missing — keep the index up to date"
    body = readme.read_text(encoding="utf-8")
    # Every numbered example should appear in the index.
    for example in _example_files():
        assert example.name in body, f"{example.name} missing from examples/README.md"
