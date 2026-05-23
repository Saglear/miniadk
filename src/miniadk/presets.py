from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

from .agentic import Agentic, TodoStore, agentic
from .core.agent import Agent
from .core.middleware import Guard, Middleware
from .core.model import Model
from .core.policy import RunPolicy
from .core.tools import Tool
from .mcp import MCPHub
from .skills import SkillRegistry
from .stdtools import make_spawn, make_tools, make_work
from .stdtools.kit import DEFAULT_IGNORE
from .stdtools.shell import ReadRule


CODER_INSTRUCTIONS = """
You are a careful local coding agent.

Work in small steps. Read relevant files before editing them. Use search when
you do not know where something lives. Run focused checks after changes when a
check is available. Keep changes scoped to the user's request and summarize what
changed.
""".strip()

CODER_SHELL_ENV = {
    "ANTHROPIC_BASE_URL": None,
    "ANTHROPIC_API_KEY": None,
    "ANTHROPIC_AUTH_TOKEN": None,
    "ANTHROPIC_KEY": None,
    "ANTHROPIC_MAX_TOKENS": None,
    "ANTHROPIC_MODEL": None,
    "ANTHROPIC_RETRIES": None,
    "ANTHROPIC_RETRY_DELAY": None,
    "ANTHROPIC_TEMPERATURE": None,
    "ANTHROPIC_TIMEOUT": None,
    "ANTHROPIC_URL": None,
    "MINIADK_MODEL_KEY": None,
    "MINIADK_MODEL_MAX_TOKENS": None,
    "MINIADK_MODEL_NAME": None,
    "MINIADK_MODEL_PROVIDER": None,
    "MINIADK_MODEL_RETRIES": None,
    "MINIADK_MODEL_RETRY_DELAY": None,
    "MINIADK_MODEL_TEMPERATURE": None,
    "MINIADK_MODEL_TIMEOUT": None,
    "MINIADK_MODEL_URL": None,
    "OPENAI_API_KEY": None,
    "OPENAI_BASE_URL": None,
    "OPENAI_KEY": None,
    "OPENAI_MAX_TOKENS": None,
    "OPENAI_MODEL": None,
    "OPENAI_RETRIES": None,
    "OPENAI_RETRY_DELAY": None,
    "OPENAI_TEMPERATURE": None,
    "OPENAI_TIMEOUT": None,
    "OPENAI_URL": None,
}

CODER_READ_SHELL = (
    "cat *",
    "find *",
    "git diff*",
    "git grep*",
    "git log*",
    "git show*",
    "git status*",
    "ls*",
    "pwd",
    "rg *",
    "sed -n *",
    "tree*",
)


def coder(
    root: str | Path = ".",
    *,
    name: str = "coder",
    instructions: str | None = None,
    tools: list[Tool] | None = None,
    extra: list[Tool] | None = None,
    agents: list[Agent] | dict[str, Agent] | None = None,
    work: bool = False,
    agent_model: Model | None = None,
    agent_models: dict[str, Model] | None = None,
    agent_tools: dict[str, list[Tool]] | None = None,
    agent_policy: RunPolicy | None = None,
    agent_policies: dict[str, RunPolicy] | None = None,
    agent_middleware: list[Middleware] | None = None,
    agent_middlewares: dict[str, list[Middleware]] | None = None,
    agent_steps: int = 20,
    keep_agent_session: bool = False,
    skills: SkillRegistry | str | Path | Sequence[str | Path] | Literal["auto"] | None = "auto",
    mcp: MCPHub | None = None,
    files: bool = True,
    shell: bool = True,
    write: bool = True,
    web: bool = False,
    search_limit: int = 20,
    list_limit: int = 200,
    max_read: int | None = 20000,
    max_search_file: int | None = 1_000_000,
    max_list_files: int | None = 5000,
    max_search_files: int | None = 1000,
    ignore: list[str] | tuple[str, ...] | set[str] | None = DEFAULT_IGNORE,
    max_shell: int | None = 20000,
    timeout: float = 30,
    validate_shell: Callable[[str], bool | str | None] | None = None,
    read_shell: ReadRule = CODER_READ_SHELL,
    shell_env: Mapping[str, str | None] | None = None,
    fetch_timeout: float = 10,
    max_fetch: int | None = 200_000,
    allow_url: Callable[[str], bool | str | None] | None = None,
    todos: TodoStore | None = None,
    middleware: list[Middleware] | None = None,
    guard: bool | Middleware = True,
    max_stop_retries: int = 3,
    chat: bool = True,
) -> Agentic:
    """Build a small but capable coding-agent preset.

    The preset stays outside core. It only assembles ordinary MiniADK pieces:
    standard tools, optional skills, optional MCP, and the agentic todo policy.
    """

    workspace = Path(root)
    base_tools = tools if tools is not None else make_tools(
        root=workspace,
        files=files,
        shell=shell,
        write=write,
        web=web,
        search_limit=search_limit,
        list_limit=list_limit,
        max_read=max_read,
        max_search_file=max_search_file,
        max_list_files=max_list_files,
        max_search_files=max_search_files,
        ignore=ignore,
        max_shell=max_shell,
        timeout=timeout,
        validate_shell=validate_shell,
        read_shell=read_shell,
        shell_env=CODER_SHELL_ENV if shell_env is None else shell_env,
        fetch_timeout=fetch_timeout,
        max_fetch=max_fetch,
        allow_url=allow_url,
    )
    helper_tools = _agent_tools(
        agents,
        work=work,
        model=agent_model,
        models=agent_models,
        tools=agent_tools,
        policy=agent_policy,
        policies=agent_policies,
        middleware=agent_middleware,
        middlewares=agent_middlewares,
        guard=guard,
        max_steps=agent_steps,
        keep_session=keep_agent_session,
    )
    registry = _skills(skills, root=workspace)
    agent = Agent(
        name=name,
        instructions=instructions or CODER_INSTRUCTIONS,
        tools=[*base_tools, *helper_tools, *(extra or [])],
        skills=registry,
        mcp=mcp,
    )
    return agentic(
        agent,
        todos=todos,
        middleware=_middleware(middleware, guard=guard),
        max_stop_retries=max_stop_retries,
        chat=chat,
    )


def _agent_tools(
    agents: list[Agent] | dict[str, Agent] | None,
    *,
    work: bool,
    model: Model | None,
    models: dict[str, Model] | None,
    tools: dict[str, list[Tool]] | None,
    policy: RunPolicy | None,
    policies: dict[str, RunPolicy] | None,
    middleware: list[Middleware] | None,
    middlewares: dict[str, list[Middleware]] | None,
    guard: bool | Middleware,
    max_steps: int,
    keep_session: bool,
) -> list[Tool]:
    if not agents:
        return []
    common = {
        "model": model,
        "models": models,
        "tools": tools,
        "policy": policy,
        "policies": policies,
        "middleware": _middleware(middleware, guard=guard),
        "middlewares": _agent_middlewares(middlewares, guard=guard),
        "max_steps": max_steps,
        "keep_session": keep_session,
    }
    result = [make_spawn(agents, **common)]
    if work:
        result.extend(make_work(agents, **common))
    return result


def _skills(
    skills: SkillRegistry | str | Path | Sequence[str | Path] | Literal["auto"] | None,
    *,
    root: Path,
) -> SkillRegistry | None:
    if skills is None or isinstance(skills, SkillRegistry):
        return skills
    if skills == "auto":
        registry = SkillRegistry.from_paths(root / ".claude" / "skills")
        return registry if registry.all() else None
    if isinstance(skills, Sequence) and not isinstance(skills, (str, bytes, Path)):
        registry = SkillRegistry.from_paths(*skills)
        return registry if registry.all() else None
    return SkillRegistry.from_paths(skills)


def _middleware(
    middleware: list[Middleware] | None,
    *,
    guard: bool | Middleware,
) -> list[Middleware]:
    items = list(middleware or [])
    if guard is True:
        return [Guard("ask"), *items]
    if guard is False:
        return items
    return [guard, *items]


def _agent_middlewares(
    middlewares: dict[str, list[Middleware]] | None,
    *,
    guard: bool | Middleware,
) -> dict[str, list[Middleware]] | None:
    if middlewares is None:
        return None
    return {
        name: _middleware(items, guard=guard)
        for name, items in middlewares.items()
    }
