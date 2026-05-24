#!/usr/bin/env bash
# Wrapper Python's run_cli launches via MINIADK_TUI_BIN.
#
# We point NODE_PATH at this folder's node_modules so that when
# Bun resolves modules from inside the linked `@miniadk/tui`, it
# finds the consumer's react/ink rather than (incorrectly) failing
# or duplicating instances.
set -e
cd "$(dirname "$0")"
export NODE_PATH="$(pwd)/node_modules"
exec bun tui.tsx "$@"
