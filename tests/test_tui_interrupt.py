"""Tests for the bridge's interrupt path.

Goal: prove that when the TUI sends ``{type:"interrupt"}`` mid-turn,
the in-flight runtime turn is actually cancelled — not just marked
"please stop politely". A long-running tool call must be torn down
within a few hundred ms.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

from miniadk.adapters.tui_ink.bridge import IntroPayload, TUIBridge
from miniadk.core.events import Event


class _FakeProcess:
    """Stand-in for the Bun subprocess. See examples/tui_bridge_smoke.py."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self._inbox: asyncio.Queue[dict | None] = asyncio.Queue()

    async def start(self) -> None: return None
    def write(self, message: dict) -> None: self.sent.append(message)
    async def read_line(self) -> dict | None: return await self._inbox.get()
    async def stop(self) -> None: return None
    async def feed(self, event: dict) -> None: await self._inbox.put(event)


@pytest.mark.asyncio
async def test_interrupt_cancels_in_flight_turn():
    """Send a submit, wait for the turn to start, then send interrupt.

    The turn_runner sleeps for 60s — if cancellation didn't propagate
    we'd hit the test's own timeout instead. We assert the bridge sent
    a ``cancelled`` notice + ``run_end`` with ``cancelled=True``.
    """

    fake = _FakeProcess()

    async def turn_runner(text: str) -> AsyncIterator[Event]:
        # Yield once so the bridge enters the loop, then sleep — this
        # is the cancellation point. asyncio.sleep is the cheapest
        # cooperative cancellation site.
        yield Event(type="thinking_delta", data={"text": "thinking"})
        await asyncio.sleep(60)
        yield Event(type="message", data={"text": "should never arrive", "streamed": False})

    intro = IntroPayload(agent="t", model="ScriptedModel", cwd=".", tool_count=0)
    bridge = TUIBridge(intro=intro, turn_runner=turn_runner)

    async def driver() -> None:
        await fake.feed({"type": "ready", "data": {}})
        await fake.feed({"type": "submit", "data": {"text": "hi"}})
        # Wait until the turn has actually started (run_start was sent).
        for _ in range(50):
            await asyncio.sleep(0.02)
            if any(m["type"] == "run_start" for m in fake.sent):
                break
        await fake.feed({"type": "interrupt", "data": {}})
        # Give the cancellation a moment to land.
        await asyncio.sleep(0.05)
        await fake.feed({"type": "quit", "data": {}})

    driver_task = asyncio.create_task(driver())
    await asyncio.wait_for(bridge.run(process=fake), timeout=5.0)
    await driver_task

    types = [m["type"] for m in fake.sent]
    assert "run_start" in types
    assert "run_end" in types
    cancel_notices = [
        m for m in fake.sent
        if m["type"] == "notice" and m["data"].get("text") == "cancelled"
    ]
    assert cancel_notices, f"expected a 'cancelled' notice, got {fake.sent}"
    run_end = next(m for m in fake.sent if m["type"] == "run_end")
    assert run_end["data"].get("cancelled") is True
