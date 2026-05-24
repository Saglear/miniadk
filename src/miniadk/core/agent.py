from dataclasses import dataclass, field
from typing import Any

from .session import Session
from .tools import Tool


_MISSING = object()


@dataclass(slots=True)
class Agent:
    name: str
    instructions: str
    tools: list[Tool] = field(default_factory=list)
    skills: Any = None
    mcp: Any = None
    # ``policy`` and ``middleware`` make composition (presets, helpers,
    # plugins) part of the agent itself rather than something every
    # adapter / Runtime caller has to thread through. ``Runtime`` and
    # the adapters fall back to these when no explicit override is
    # provided.
    policy: Any = None
    middleware: list[Any] | None = None

    def copy(
        self,
        *,
        name: str | None = None,
        instructions: str | None = None,
        tools: list[Tool] | None = None,
        extra: list[Tool] | None = None,
        skills: Any = _MISSING,
        mcp: Any = _MISSING,
        policy: Any = _MISSING,
        middleware: list[Any] | None | object = _MISSING,
    ) -> "Agent":
        next_tools = list(self.tools if tools is None else tools)
        if extra:
            next_tools.extend(extra)
        return Agent(
            name=self.name if name is None else name,
            instructions=self.instructions if instructions is None else instructions,
            tools=next_tools,
            skills=self.skills if skills is _MISSING else skills,
            mcp=self.mcp if mcp is _MISSING else mcp,
            policy=self.policy if policy is _MISSING else policy,
            middleware=(
                list(self.middleware) if self.middleware is not None else None
            ) if middleware is _MISSING else middleware,
        )


def as_tool(
    agent: Agent,
    *,
    model: Any,
    name: str | None = None,
    description: str | None = None,
    middleware: list[Any] | None = None,
    policy: Any = None,
    session: Session | None = None,
    max_steps: int | None = None,
) -> Tool:
    from .._guards import bind_guards
    from .runtime import Runtime
    from .tools import tool

    @tool
    async def run_agent(prompt: str) -> str:
        """Run an agent and return its final response."""
        runtime = Runtime(
            agent=agent,
            model=model,
            middleware=bind_guards(middleware, ask_user=None),
            policy=policy,
            session=session,
            max_steps=max_steps,
        )
        return await runtime.ask(prompt)

    run_agent.name = name or agent.name
    run_agent.description = description or f"Run the {agent.name} agent."
    return run_agent


def resolve_composition(
    obj: Any,
    *,
    middleware: list[Any] | None = None,
    policy: Any = None,
) -> tuple[Agent, list[Any] | None, Any]:
    """Resolve an ``Agent`` plus its policy/middleware composition.

    Accepts either a bare :class:`Agent` or any object exposing
    ``.agent``, ``.policy``, ``.middleware`` attributes (e.g. the legacy
    :class:`miniadk.Agentic` struct). Caller-provided values take
    precedence; agent-level middleware is prepended so guards stay in
    front of caller decorators.

    Adapters call this once at the start so they can stop hand-rolling
    the same unwrap pattern.
    """
    wrapped_agent = getattr(obj, "agent", None)
    if isinstance(wrapped_agent, Agent):
        # Wrapper-style object — Agentic, custom presets, etc.
        # The wrapper is authoritative for policy/middleware: its
        # contents already include whatever the inner Agent carries
        # (the wrapper builds the Agent with that policy attached), so
        # we MUST NOT also inspect ``wrapped_agent.policy`` /
        # ``.middleware`` below — that would double-count.
        if policy is None:
            policy = getattr(obj, "policy", None)
            if policy is None:
                policy = getattr(wrapped_agent, "policy", None)
        wrapped_middleware = getattr(obj, "middleware", None)
        if wrapped_middleware is None:
            wrapped_middleware = getattr(wrapped_agent, "middleware", None) or []
        if not middleware:
            middleware = list(wrapped_middleware) if wrapped_middleware else None
        elif wrapped_middleware:
            middleware = [*wrapped_middleware, *middleware]
        return wrapped_agent, middleware, policy

    # Bare Agent path: pull policy/middleware off the agent itself.
    agent = obj
    if policy is None and getattr(agent, "policy", None) is not None:
        policy = agent.policy
    agent_middleware = getattr(agent, "middleware", None)
    if agent_middleware:
        if not middleware:
            middleware = list(agent_middleware)
        else:
            middleware = [*agent_middleware, *middleware]
    return agent, middleware, policy
