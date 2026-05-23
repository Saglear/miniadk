from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from ..core.agent import Agent
from ..core.middleware import AskUser, Middleware
from ..core.model import Model
from ..core.policy import RunPolicy
from ..core.runtime import Runtime
from ..core.session import Session
from ..core.tools import Tool
from ..sessions import CompactSpec, compact as compact_session
from ..skills import resolve_agent
from . import json as json_adapter
from .._guards import bind_guards
from ..run import merge_tools
from .json import _load_session, _save_session, astream_runtime


async def ws_json(
    ws: Any,
    agent: Agent,
    text: str,
    *,
    model: Model | None = None,
    middleware: list[Middleware] | None = None,
    policy: RunPolicy | None = None,
    session: Session | str | Path | bool | None = None,
    tools: list[Tool] | None = None,
    max_steps: int = 20,
    lifecycle: bool = False,
    trace: bool = False,
    resolve: bool = True,
    compact: CompactSpec = None,
    ask_user: AskUser | None = None,
) -> int:
    middleware = _unwrap_middleware(agent, middleware)
    agent, policy = _unwrap_agent(agent, policy)
    runtime_middleware = bind_guards(middleware, ask_user=ask_user)
    active_agent = await resolve_agent(agent) if resolve else agent
    active_session, session_path = _load_session(session, active_agent)
    active_model = model or json_adapter.default_model()
    if active_session is not None:
        await compact_session(active_session, model=active_model, spec=compact)
    runtime = Runtime(
        agent=active_agent,
        model=active_model,
        middleware=runtime_middleware,
        policy=policy,
        session=active_session,
        max_steps=max_steps,
    )
    count = 0
    stream = astream_runtime(
        runtime,
        text,
        tools=merge_tools(active_agent.tools, tools),
        lifecycle=lifecycle,
        trace=trace,
    )
    try:
        async for event in stream:
            await _send(ws, event)
            count += 1
        await compact_session(runtime.session, model=active_model, spec=compact)
    except BaseException:
        runtime.cancel("websocket_closed")
        await stream.aclose()
        raise
    finally:
        _save_session(runtime.session, session_path)
    return count


def _unwrap_agent(agent, policy: RunPolicy | None) -> tuple[Agent, RunPolicy | None]:
    wrapped_agent = getattr(agent, "agent", None)
    wrapped_policy = getattr(agent, "policy", None)
    if isinstance(wrapped_agent, Agent):
        return wrapped_agent, policy or wrapped_policy
    return agent, policy


def _unwrap_middleware(agent, middleware: list[Middleware] | None) -> list[Middleware] | None:
    wrapped_middleware = getattr(agent, "middleware", None)
    if not wrapped_middleware:
        return middleware
    return [*wrapped_middleware, *(middleware or [])]


async def _send(ws: Any, event: dict[str, Any]) -> None:
    send_json = getattr(ws, "send_json", None)
    if send_json is not None:
        result = send_json(event)
    else:
        send = getattr(ws, "send", None)
        if send is None:
            raise TypeError("websocket must provide send_json or send")
        result = send(json.dumps(event, ensure_ascii=False))

    if inspect.isawaitable(result):
        await result
