"""Default ``run_cli`` re-export.

The implementation lives in :mod:`miniadk.adapters.tui`. This module is a
thin shim so ``from miniadk.adapters.cli import run_cli`` keeps working.
"""

from .tui import run_cli

__all__ = ["run_cli"]
