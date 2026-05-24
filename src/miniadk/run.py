import asyncio
from pathlib import Path

from ._guards import bind_guards
from .core.agent import Agent, resolve_composition
from .core.middleware import AskUser, Middleware
from .core.model import Model
from .core.policy import RunPolicy
from .core.runtime import Runtime
from .core.session import Session
from .core.tools import Tool, canonical_tool_name
from .models.factory import model as default_model
from .sessions import CompactSpec, compact as compact_session, sessions
from .skills import resolve_agent


async def arun(
    agent: Agent,
    text: str,
    *,
    model: Model | None = None,
    middleware: list[Middleware] | None = None,
    policy: RunPolicy | None = None,
    session: Session | str | Path | bool | None = None,
    tools: list[Tool] | None = None,
    max_steps: int | None = None,
    compact: CompactSpec = None,
    ask_user: AskUser | None = None,
    resolve: bool = True,
) -> str:
    # An ``Agent`` may carry its own ``policy`` / ``middleware`` (the
    # standard composition idiom). Caller-provided values still take
    # precedence; agent-level middleware is prepended so guards stay in
    # front of caller-added decorators. ``resolve_composition`` also
    # unwraps wrapper structs like the legacy ``Agentic`` for back-compat.
    middleware = list(middleware) if middleware else None
    agent, middleware, policy = resolve_composition(
        agent, middleware=middleware, policy=policy,
    )
    active_agent = await resolve_agent(agent) if resolve else agent
    runtime_middleware = bind_guards(middleware, ask_user=ask_user)
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
        answer = await runtime.ask(text, tools=merge_tools(active_agent.tools, tools))
        await compact_session(runtime.session, model=active_model, spec=compact)
        return answer
    except asyncio.CancelledError:
        runtime.cancel("cancelled")
        raise
    finally:
        _save_session(runtime.session, session_path)


def run(
    agent: Agent,
    text: str,
    *,
    model: Model | None = None,
    middleware: list[Middleware] | None = None,
    policy: RunPolicy | None = None,
    session: Session | str | Path | bool | None = None,
    tools: list[Tool] | None = None,
    max_steps: int | None = None,
    compact: CompactSpec = None,
    ask_user: AskUser | None = None,
    resolve: bool = True,
) -> str:
    return asyncio.run(
        arun(
            agent,
            text,
            model=model,
            middleware=middleware,
            policy=policy,
            session=session,
            tools=tools,
            max_steps=max_steps,
            compact=compact,
            ask_user=ask_user,
            resolve=resolve,
        )
    )


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


def merge_tools(base: list[Tool], extra: list[Tool] | None) -> list[Tool] | None:
    if extra is None:
        return None
    merged: dict[str, Tool] = {}
    for item in [*base, *extra]:
        merged[canonical_tool_name(item.name)] = item
    return list(merged.values())
