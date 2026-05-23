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

    def copy(
        self,
        *,
        name: str | None = None,
        instructions: str | None = None,
        tools: list[Tool] | None = None,
        extra: list[Tool] | None = None,
        skills: Any = _MISSING,
        mcp: Any = _MISSING,
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
    max_steps: int = 20,
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
