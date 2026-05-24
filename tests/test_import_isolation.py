"""Regression: importing :mod:`miniadk` must not pull in any TUI deps.

A consumer who only wants ``Agent`` / ``Runtime`` / ``Tool`` / model
adapters should be able to ``pip install miniadk`` without paying for
Textual, Ink, or React. We enforce this two ways:

1. After ``import miniadk`` in a fresh interpreter, neither ``textual``
   nor any ``miniadk.adapters.tui.*`` submodule should be in
   ``sys.modules``.
2. With ``textual`` deliberately broken, ``import miniadk`` and core
   surface access (``Agent``, ``Runtime``, ``run`` …) should still
   succeed.

The probes run in a fresh ``python -c`` subprocess so test pollution
from sibling tests (which legitimately load Textual) can't mask a
regression here.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_miniadk_does_not_load_tui() -> None:
    result = _run(
        """
        import sys
        import miniadk  # noqa: F401

        bad_textual = [m for m in sys.modules if m == "textual" or m.startswith("textual.")]
        bad_tui = [m for m in sys.modules if m.startswith("miniadk.adapters.tui")]
        if bad_textual or bad_tui:
            print("FAIL", bad_textual[:3], bad_tui[:3])
            raise SystemExit(1)
        print("OK")
        """
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout


def test_core_works_with_textual_blocked() -> None:
    result = _run(
        """
        import sys
        sys.modules["textual"] = None  # poison textual import
        import miniadk
        assert miniadk.Agent is not None
        assert miniadk.Runtime is not None
        assert miniadk.run is not None
        from miniadk import make_read_file
        assert callable(make_read_file)
        print("OK")
        """
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout


def test_run_cli_dispatcher_is_lazy() -> None:
    # Touching run_cli should resolve the dispatcher without loading
    # the Textual app module — only when the dispatcher actually falls
    # through to the Textual backend should textual get imported.
    result = _run(
        """
        import sys
        import miniadk
        _ = miniadk.run_cli  # force lazy resolution
        assert "miniadk.adapters.tui.cli_dispatch" in sys.modules
        assert "miniadk.adapters.tui.app" not in sys.modules
        assert not any(m == "textual" or m.startswith("textual.") for m in sys.modules)
        print("OK")
        """
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout
