from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

from .._guards import bind_guards
from ..core.agent import Agent
from ..core.middleware import AskUser, Middleware
from ..core.model import Model
from ..core.policy import RunPolicy
from ..core.runtime import Runtime
from ..core.session import Session
from ..core.tools import Tool, ToolValidation, canonical_tool_name, tool
from ..models.factory import model as build_model


@dataclass(slots=True)
class SpawnResult:
    agent: str
    answer: str
    session: Session | None = None

    def __str__(self) -> str:
        return self.answer


@dataclass(slots=True)
class WorkResult:
    id: str
    agent: str
    status: str
    answer: str | None = None
    error: str | None = None
    session: Session | None = None

    def __str__(self) -> str:
        if self.answer is not None:
            return self.answer
        if self.error is not None:
            return self.error
        return f"{self.id}: {self.status}"


@dataclass(slots=True)
class _WorkItem:
    id: str
    agent: str
    session: Session
    task: asyncio.Task


@dataclass(slots=True)
class Spawn:
    agents: dict[str, Agent]
    models: dict[str, Model] = field(default_factory=dict)
    model: Model | None = None
    tools: dict[str, list[Tool]] = field(default_factory=dict)
    policies: dict[str, RunPolicy] = field(default_factory=dict)
    policy: RunPolicy | None = None
    middlewares: dict[str, list[Middleware]] = field(default_factory=dict)
    middleware: list[Middleware] = field(default_factory=list)
    ask_user: AskUser | None = None
    max_steps: int | None = None
    keep_session: bool = False

    def __call__(
        self,
        *,
        name: str = "spawn_agent",
        description: str | None = None,
    ) -> Tool:
        agents = self.agents
        models = self.models
        fallback_model = self.model
        extra_tools = self.tools
        policies = self.policies
        fallback_policy = self.policy
        middlewares = self.middlewares
        fallback_middleware = self.middleware
        ask_user = self.ask_user
        max_steps = self.max_steps
        keep_session = self.keep_session

        async def validate(agent: str, prompt: str) -> ToolValidation:
            if agent not in agents:
                return ToolValidation.deny(f"Unknown agent: {agent}")
            if not prompt.strip():
                return ToolValidation.deny("prompt is required")
            return ToolValidation.allow()

        @tool(
            validate=validate,
            format=_format_spawn_result,
            schema={
                "agent": _agent_schema(agents),
                "prompt": _prompt_schema(),
            },
        )
        async def spawn_agent(agent: str, prompt: str) -> SpawnResult:
            """Run another agent and return its final answer."""
            child = await _resolve_child_agent(agents[agent])
            session = Session()
            runtime = Runtime(
                agent=child,
                model=models.get(agent) or fallback_model or build_model(),
                middleware=bind_guards(
                    list(middlewares.get(agent, fallback_middleware)),
                    ask_user=ask_user,
                ),
                policy=policies.get(agent) or fallback_policy,
                session=session,
                max_steps=max_steps,
            )
            answer = await runtime.ask(
                prompt,
                tools=_child_tools(child, extra_tools.get(agent)),
            )
            return SpawnResult(
                agent=agent,
                answer=answer,
                session=session if keep_session else None,
            )

        spawn_agent.name = name
        spawn_agent.description = description or _description(agents)
        return spawn_agent


def make_spawn(
    agents: list[Agent] | dict[str, Agent],
    *,
    model: Model | None = None,
    models: dict[str, Model] | None = None,
    tools: dict[str, list[Tool]] | None = None,
    policies: dict[str, RunPolicy] | None = None,
    policy: RunPolicy | None = None,
    middlewares: dict[str, list[Middleware]] | None = None,
    middleware: list[Middleware] | None = None,
    ask_user: AskUser | None = None,
    max_steps: int | None = None,
    keep_session: bool = False,
    name: str = "spawn_agent",
    description: str | None = None,
) -> Tool:
    return Spawn(
        agents=_agent_map(agents),
        models=models or {},
        model=model,
        tools=tools or {},
        policies=policies or {},
        policy=policy,
        middlewares=middlewares or {},
        middleware=middleware or [],
        ask_user=ask_user,
        max_steps=max_steps,
        keep_session=keep_session,
    )(name=name, description=description)


@dataclass(slots=True)
class Work:
    agents: dict[str, Agent]
    models: dict[str, Model] = field(default_factory=dict)
    model: Model | None = None
    tools: dict[str, list[Tool]] = field(default_factory=dict)
    policies: dict[str, RunPolicy] = field(default_factory=dict)
    policy: RunPolicy | None = None
    middlewares: dict[str, list[Middleware]] = field(default_factory=dict)
    middleware: list[Middleware] = field(default_factory=list)
    ask_user: AskUser | None = None
    max_steps: int | None = None
    keep_session: bool = False
    _items: dict[str, _WorkItem] = field(default_factory=dict, init=False, repr=False)

    def tools_list(self) -> list[Tool]:
        return [
            self.start_tool(),
            self.list_tool(),
            self.read_tool(),
            self.cancel_tool(),
        ]

    def start_tool(
        self,
        *,
        name: str = "start_work",
        description: str | None = None,
    ) -> Tool:
        agents = self.agents

        async def validate(agent: str, prompt: str) -> ToolValidation:
            if agent not in agents:
                return ToolValidation.deny(f"Unknown agent: {agent}")
            if not prompt.strip():
                return ToolValidation.deny("prompt is required")
            return ToolValidation.allow()

        @tool(
            validate=validate,
            format=_format_work_result,
            schema={
                "agent": _agent_schema(agents),
                "prompt": _prompt_schema(),
            },
        )
        async def start_work(agent: str, prompt: str) -> WorkResult:
            """Start another agent in the background and return its work id."""
            item = self._start(agent, prompt)
            return WorkResult(id=item.id, agent=agent, status="running")

        start_work.name = name
        start_work.description = description or _work_description(self.agents)
        return start_work

    def list_tool(
        self,
        *,
        name: str = "list_work",
        description: str | None = None,
    ) -> Tool:
        @tool(
            read_only=True,
            concurrency_safe=True,
            format=_format_work_list,
        )
        async def list_work() -> list[WorkResult]:
            """List background agent work."""
            return [self._read(id) for id in sorted(self._items)]

        list_work.name = name
        list_work.description = description or "List background agent work."
        return list_work

    def read_tool(
        self,
        *,
        name: str = "read_work",
        description: str | None = None,
    ) -> Tool:
        async def validate(id: str, wait: float = 0) -> ToolValidation:
            if id not in self._items:
                return ToolValidation.deny(f"Unknown work id: {id}")
            if wait < 0:
                return ToolValidation.deny("wait must be >= 0")
            return ToolValidation.allow()

        @tool(
            validate=validate,
            read_only=True,
            concurrency_safe=True,
            format=_format_work_result,
            schema={
                "id": _work_id_schema(),
                "wait": {"type": "number", "minimum": 0, "default": 0},
            },
        )
        async def read_work(id: str, wait: float = 0) -> WorkResult:
            """Read the status or result of background agent work."""
            if wait > 0:
                await self._wait(id, wait)
            return self._read(id)

        read_work.name = name
        read_work.description = description or "Read background agent work."
        return read_work

    def cancel_tool(
        self,
        *,
        name: str = "cancel_work",
        description: str | None = None,
    ) -> Tool:
        async def validate(id: str) -> ToolValidation:
            if id not in self._items:
                return ToolValidation.deny(f"Unknown work id: {id}")
            return ToolValidation.allow()

        @tool(
            validate=validate,
            destructive=True,
            format=_format_work_result,
            schema={"id": _work_id_schema()},
        )
        async def cancel_work(id: str) -> WorkResult:
            """Cancel background agent work."""
            return await self._cancel(id)

        cancel_work.name = name
        cancel_work.description = description or "Cancel background agent work."
        return cancel_work

    def _start(self, agent: str, prompt: str) -> _WorkItem:
        child = self.agents[agent]
        session = Session()
        task = asyncio.create_task(self._run_child(agent, child, prompt, session))
        task.add_done_callback(_consume_task_error)
        item = _WorkItem(
            id=f"work_{uuid4().hex[:8]}",
            agent=agent,
            session=session,
            task=task,
        )
        self._items[item.id] = item
        return item

    async def _run_child(
        self,
        agent: str,
        child: Agent,
        prompt: str,
        session: Session,
    ) -> str:
        child = await _resolve_child_agent(child)
        runtime = Runtime(
            agent=child,
            model=self.models.get(agent) or self.model or build_model(),
            middleware=bind_guards(
                list(self.middlewares.get(agent, self.middleware)),
                ask_user=self.ask_user,
            ),
            policy=self.policies.get(agent) or self.policy,
            session=session,
            max_steps=self.max_steps,
        )
        return await runtime.ask(prompt, tools=_child_tools(child, self.tools.get(agent)))

    def _read(self, id: str) -> WorkResult:
        item = self._items[id]
        task = item.task
        session = item.session if self.keep_session and task.done() else None
        if task.cancelled():
            return WorkResult(id=id, agent=item.agent, status="cancelled", session=session)
        if not task.done():
            return WorkResult(id=id, agent=item.agent, status="running")
        error = task.exception()
        if error is not None:
            return WorkResult(
                id=id,
                agent=item.agent,
                status="error",
                error=f"{type(error).__name__}: {error}",
                session=session,
            )
        return WorkResult(
            id=id,
            agent=item.agent,
            status="done",
            answer=task.result(),
            session=session,
        )

    async def _cancel(self, id: str) -> WorkResult:
        item = self._items[id]
        if item.task.done():
            return self._read(id)
        item.task.cancel()
        try:
            await item.task
        except asyncio.CancelledError:
            pass
        return self._read(id)

    async def _wait(self, id: str, seconds: float) -> None:
        item = self._items[id]
        if item.task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(item.task), timeout=seconds)
        except asyncio.TimeoutError:
            return


def make_work(
    agents: list[Agent] | dict[str, Agent],
    *,
    model: Model | None = None,
    models: dict[str, Model] | None = None,
    tools: dict[str, list[Tool]] | None = None,
    policies: dict[str, RunPolicy] | None = None,
    policy: RunPolicy | None = None,
    middlewares: dict[str, list[Middleware]] | None = None,
    middleware: list[Middleware] | None = None,
    ask_user: AskUser | None = None,
    max_steps: int | None = None,
    keep_session: bool = False,
) -> list[Tool]:
    return Work(
        agents=_agent_map(agents),
        models=models or {},
        model=model,
        tools=tools or {},
        policies=policies or {},
        policy=policy,
        middlewares=middlewares or {},
        middleware=middleware or [],
        ask_user=ask_user,
        max_steps=max_steps,
        keep_session=keep_session,
    ).tools_list()


def _agent_map(agents: list[Agent] | dict[str, Agent]) -> dict[str, Agent]:
    if isinstance(agents, dict):
        return dict(agents)
    result: dict[str, Agent] = {}
    for agent in agents:
        if agent.name in result:
            raise ValueError(f"duplicate agent name: {agent.name}")
        result[agent.name] = agent
    return result


def _agent_schema(agents: dict[str, Agent]) -> dict[str, object]:
    schema: dict[str, object] = {
        "type": "string",
        "description": "Agent name to run.",
    }
    names = sorted(agents)
    if names:
        schema["enum"] = names
    return schema


def _prompt_schema() -> dict[str, object]:
    return {
        "type": "string",
        "description": "Task prompt for the agent.",
        "minLength": 1,
    }


def _work_id_schema() -> dict[str, object]:
    return {
        "type": "string",
        "description": "Background work id.",
        "minLength": 1,
    }


async def _resolve_child_agent(agent: Agent) -> Agent:
    if agent.skills is None and agent.mcp is None:
        return agent
    from ..skills import resolve_agent

    return await resolve_agent(agent)


def _child_tools(agent: Agent, extra: list[Tool] | None) -> list[Tool] | None:
    if extra is None:
        return None
    merged: dict[str, Tool] = {}
    for item in [*agent.tools, *extra]:
        merged[canonical_tool_name(item.name)] = item
    return list(merged.values())


def _description(agents: dict[str, Agent]) -> str:
    names = ", ".join(sorted(agents))
    if names:
        return f"Run one of these agents: {names}."
    return "Run another agent and return its final answer."


def _work_description(agents: dict[str, Agent]) -> str:
    names = ", ".join(sorted(agents))
    if names:
        return f"Start background work with one of these agents: {names}."
    return "Start background work with another agent."


def _format_spawn_result(item: SpawnResult) -> str:
    if item.answer:
        return f"{item.agent}: {item.answer}"
    return f"{item.agent}: no answer"


def _format_work_list(items: list[WorkResult]) -> str:
    if not items:
        return "no background work"
    return "\n".join(_format_work_item(item) for item in items)


def _format_work_result(item: WorkResult) -> str:
    lines = [_format_work_item(item)]
    if item.answer is not None:
        lines.append(item.answer)
    return "\n".join(lines)


def _format_work_item(item: WorkResult) -> str:
    suffix = ""
    if item.status == "done":
        suffix = ": answer ready" if item.answer is not None else ": done"
    elif item.error is not None:
        suffix = f": {item.error}"
    return f"{item.id} [{item.status}] {item.agent}{suffix}"


def _consume_task_error(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    try:
        task.exception()
    except asyncio.CancelledError:
        pass
