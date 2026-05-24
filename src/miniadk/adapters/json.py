from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Generator
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ..core.agent import Agent, resolve_composition
from ..core.events import Event
from ..core.middleware import AskUser, Middleware
from ..core.model import Model
from ..core.policy import RunPolicy
from ..core.runtime import Runtime
from ..core.session import Session
from ..core.tools import Tool
from ..models.factory import model as default_model
from ..sessions import CompactSpec, compact as compact_session, sessions
from ..skills import resolve_agent
from .._guards import bind_guards
from ..run import merge_tools


def event_dict(event: Event) -> dict[str, Any]:
    return {
        "type": event.type,
        "data": _jsonable(event.data),
    }


async def astream_json(
    agent: Agent,
    text: str,
    *,
    model: Model | None = None,
    middleware: list[Middleware] | None = None,
    policy: RunPolicy | None = None,
    session: Session | str | Path | bool | None = None,
    tools: list[Tool] | None = None,
    max_steps: int | None = None,
    lifecycle: bool = False,
    trace: bool = False,
    resolve: bool = True,
    compact: CompactSpec = None,
    ask_user: AskUser | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    middleware = list(middleware) if middleware else None
    agent, middleware, policy = resolve_composition(
        agent, middleware=middleware, policy=policy,
    )
    runtime_middleware = bind_guards(middleware, ask_user=ask_user)
    active_agent = await resolve_agent(agent) if resolve else agent
    active_session, session_path = _load_session(session, active_agent)
    active_model = model or default_model()
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
    try:
        async for event in astream_runtime(
            runtime,
            text,
            tools=merge_tools(active_agent.tools, tools),
            lifecycle=lifecycle,
            trace=trace,
        ):
            yield event
        await compact_session(runtime.session, model=active_model, spec=compact)
    finally:
        _save_session(runtime.session, session_path)


async def astream_runtime(
    runtime: Runtime,
    text: str,
    *,
    tools: list[Tool] | None = None,
    lifecycle: bool = False,
    trace: bool = False,
) -> AsyncGenerator[dict[str, Any], None]:
    stream = runtime.run(text, tools=tools, lifecycle=lifecycle, trace=trace)
    try:
        async for event in stream:
            yield event_dict(event)
    except BaseException:
        runtime.cancel("stream_closed")
        await stream.aclose()
        raise


def jsonl(
    agent: Agent,
    text: str,
    *,
    model: Model | None = None,
    middleware: list[Middleware] | None = None,
    policy: RunPolicy | None = None,
    session: Session | str | Path | bool | None = None,
    tools: list[Tool] | None = None,
    max_steps: int | None = None,
    lifecycle: bool = False,
    trace: bool = False,
    resolve: bool = True,
    compact: CompactSpec = None,
    ask_user: AskUser | None = None,
) -> Generator[str, None, None]:
    events = asyncio.run(
        _collect(
            agent,
            text,
            model=model,
            middleware=middleware,
            policy=policy,
            session=session,
            tools=tools,
            max_steps=max_steps,
            lifecycle=lifecycle,
            trace=trace,
            resolve=resolve,
            compact=compact,
            ask_user=ask_user,
        )
    )
    for event in events:
        yield json.dumps(event, ensure_ascii=False)


async def _collect(
    agent: Agent,
    text: str,
    *,
    model: Model | None,
    middleware: list[Middleware] | None,
    policy: RunPolicy | None,
    session: Session | str | Path | bool | None,
    tools: list[Tool] | None,
    max_steps: int,
    lifecycle: bool,
    trace: bool,
    resolve: bool,
    compact: CompactSpec,
    ask_user: AskUser | None,
) -> list[dict[str, Any]]:
    return [
        event
        async for event in astream_json(
            agent,
            text,
            model=model,
            middleware=middleware,
            policy=policy,
            session=session,
            tools=tools,
            max_steps=max_steps,
            lifecycle=lifecycle,
            trace=trace,
            resolve=resolve,
            compact=compact,
            ask_user=ask_user,
        )
    ]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _load_session(
    session: Session | str | Path | bool | None,
    agent: Agent,
) -> tuple[Session | None, Path | None]:
    if session is None:
        return None, None
    if session is True:
        path = sessions(".miniadk/sessions").path(agent.name)
        if path.exists():
            return Session.load(path), path
        return Session(), path
    if session is False:
        return None, None
    if isinstance(session, Session):
        return session, None
    path = Path(session)
    if path.exists():
        return Session.load(path), path
    return Session(), path


def _save_session(session: Session, path: Path | None) -> None:
    if path is not None:
        session.save(path)
