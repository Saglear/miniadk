/**
 * Default MiniADK TUI entry point.
 *
 * This is the file ``bun build --compile`` consumes to produce the
 * ``miniadk-tui`` binary that ``run_cli`` spawns. It does the absolute
 * minimum: wires the bridge, mounts the default ``<MiniADKApp>``.
 *
 * Custom TUIs should import ``mount`` (and any building blocks they
 * want) from ``@miniadk/tui`` instead — see ``examples/custom_tui/``
 * in the Python repo.
 */

import React from "react";
import { mount, MiniADKApp } from "./lib.js";

mount((bridge) => <MiniADKApp send={bridge.send} subscribe={bridge.subscribe} />);
