import asyncio
import os
from collections.abc import Callable
from pathlib import Path

from ..core.agent import Agent
from ..core.messages import Message
from ..core.middleware import Guard, Middleware, PermissionRequest
from ..core.model import Model
from ..core.policy import RunPolicy
from ..core.runtime import Runtime
from ..core.session import Session
from ..core.tools import Tool
from ..models.factory import model as default_model
from ..run import merge_tools
from ..sessions import CompactSpec, compact as compact_session, sessions
from ..skills import parse_slash_command, resolve_agent, tools_for_skill
from .._guards import copy_guard
from ._cli_ui import CLIRenderer, CLIStatus, CLITheme, OutputMode


InputFunc = Callable[[str], str]
OutputFunc = Callable[[str], None]


def run_cli(
    agent,
    *,
    model: Model | None = None,
    input_func: InputFunc = input,
    output_func: OutputFunc = print,
    prompt: str = "> ",
    output_mode: OutputMode = "auto",
    theme: CLITheme | None = None,
    middleware: list[Middleware] | None = None,
    policy: RunPolicy | None = None,
    session: Session | str | Path | bool | None = None,
    tools: list[Tool] | None = None,
    max_steps: int = 20,
    compact: CompactSpec = None,
    compact_keep: int = 10,
) -> None:
    middleware = _unwrap_middleware(agent, middleware)
    agent, policy = _unwrap_agent(agent, policy)
    active_model = model or default_model()
    asyncio.run(
        _run_cli(
            agent=agent,
            model=active_model,
            input_func=input_func,
            output_func=output_func,
            prompt=prompt,
            output_mode=output_mode,
            theme=theme,
            policy=policy,
            middleware=middleware,
            session=session,
            tools=tools,
            max_steps=max_steps,
            compact=compact,
            compact_keep=compact_keep,
        )
    )


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


async def _run_cli(
    agent: Agent,
    model: Model,
    input_func: InputFunc,
    output_func: OutputFunc,
    prompt: str,
    output_mode: OutputMode,
    theme: CLITheme | None,
    policy: RunPolicy | None,
    middleware: list[Middleware] | None,
    session: Session | str | Path | None,
    tools: list[Tool] | None,
    max_steps: int,
    compact: CompactSpec,
    compact_keep: int,
) -> None:
    resolved_agent = await resolve_agent(agent)
    active_tools = merge_tools(resolved_agent.tools, tools)
    active_agent = (
        resolved_agent.copy(tools=active_tools)
        if active_tools is not None
        else resolved_agent
    )
    renderer = CLIRenderer(output_func, mode=output_mode, theme=theme)
    active_session, session_path = _load_session(session, active_agent)
    if active_session is not None:
        await compact_session(active_session, model=model, spec=compact)
    runtime_middleware = _bind_guards(
        middleware,
        input_func=input_func,
        renderer=renderer,
    )
    if not _has_guard(runtime_middleware):
        runtime_middleware.append(Guard("ask", ask_user=_ask_user(input_func, renderer)))

    runtime = Runtime(
        agent=active_agent,
        model=model,
        middleware=runtime_middleware,
        policy=policy,
        session=active_session,
        max_steps=max_steps,
    )
    renderer.intro(CLIStatus.from_agent(active_agent))
    prompt_text = renderer.prompt(prompt)

    while True:
        try:
            user_input = input_func(prompt_text)
        except (EOFError, StopIteration):
            return

        if user_input.strip() in {"/exit", "/quit"}:
            return

        slash = parse_slash_command(user_input)
        if slash is not None:
            command, args = slash
            if await _handle_builtin_command(
                command=command,
                agent=active_agent,
                runtime=runtime,
                renderer=renderer,
                model_label=_model_label(model),
                model=model,
                policy=policy,
                compact=compact,
                compact_keep=compact_keep,
            ):
                _save_session(runtime.session, session_path)
                continue

            if active_agent.skills is not None:
                skill = active_agent.skills.get(command)
                if skill is not None:
                    if not skill.user_invocable:
                        renderer.skill_not_invocable(skill.name)
                        continue
                    filtered_tools = tools_for_skill(active_agent, skill.name)
                    user_input = skill.render(args).text
                    try:
                        await _render_run(
                            runtime,
                            user_input,
                            renderer,
                            tools=filtered_tools,
                        )
                        await compact_session(runtime.session, model=model, spec=compact)
                        _save_session(runtime.session, session_path)
                    except RuntimeError as error:
                        renderer.error(str(error))
                        _save_session(runtime.session, session_path)
                    continue

        try:
            await _render_run(runtime, user_input, renderer)
            await compact_session(runtime.session, model=model, spec=compact)
        except RuntimeError as error:
            renderer.error(str(error))
        finally:
            _save_session(runtime.session, session_path)


async def _render_run(
    runtime: Runtime,
    user_input: str,
    renderer: CLIRenderer,
    *,
    tools=None,
) -> None:
    async for event in runtime.run(user_input, tools=tools):
        renderer.event(event)


def _ask_user(input_func: InputFunc, renderer: CLIRenderer):
    def ask(request: PermissionRequest) -> bool:
        answer = input_func(renderer.permission_prompt(request))
        return answer.strip().lower() in {"y", "yes"}

    return ask


def _bind_guards(
    middleware: list[Middleware] | None,
    *,
    input_func: InputFunc,
    renderer: CLIRenderer,
) -> list[Middleware]:
    items = []
    for item in middleware or []:
        if isinstance(item, Guard) and item.ask_user is None:
            item = copy_guard(item, ask_user=_ask_user(input_func, renderer))
        items.append(item)
    return items


def _has_guard(middleware: list[Middleware]) -> bool:
    return any(isinstance(item, Guard) for item in middleware)


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


async def _handle_builtin_command(
    *,
    command: str,
    agent: Agent,
    runtime: Runtime,
    renderer: CLIRenderer,
    model_label: str,
    model: Model,
    policy: RunPolicy | None,
    compact: CompactSpec,
    compact_keep: int,
) -> bool:
    normalized = command.strip().lower()

    if normalized == "help":
        _render_help(agent, renderer)
        return True
    if normalized == "status":
        _render_status(agent, runtime, renderer, model_label)
        return True
    if normalized == "tools":
        _render_tools(agent, renderer)
        return True
    if normalized == "skills":
        _render_skills(agent, renderer)
        return True
    if normalized == "todos":
        _render_todos(policy, renderer)
        return True
    if normalized == "clear":
        renderer.clear()
        return True
    if normalized == "reset":
        runtime.session.messages[:] = [Message("system", agent.instructions)]
        renderer.notice("session reset")
        return True
    if normalized == "compact":
        await _render_compact(runtime, renderer, model, compact_keep)
        return True
    return False


def _render_help(agent: Agent, renderer: CLIRenderer) -> None:
    builtins = [
        "/help - show this screen",
        "/status - show session state",
        "/tools - list tools",
        "/skills - list skills",
        "/todos - show current todo list",
        "/clear - clear the screen",
        "/reset - clear conversation history",
        "/compact - show a compact transcript snapshot",
        "/exit or /quit - leave the session",
    ]
    renderer.section(f"{agent.name} command center", "adapter-level commands")
    renderer.bullets(builtins, heading="Built-ins")
    _render_skills(agent, renderer, heading="Available skills")


def _render_status(
    agent: Agent,
    runtime: Runtime,
    renderer: CLIRenderer,
    model_label: str,
) -> None:
    skills = agent.skills.all() if agent.skills is not None else []
    stats = runtime.session.stats
    renderer.rows(
        [
            ("agent", agent.name),
            ("model", model_label),
            ("cwd", os.getcwd()),
            ("messages", str(stats.messages)),
            ("tool calls", str(stats.tool_calls)),
            ("chars", str(stats.chars)),
            ("tools", str(len(agent.tools))),
            ("skills", str(len(skills))),
        ],
        heading="Session status",
    )
    if runtime.messages:
        renderer.notice(f"last role: {runtime.messages[-1].role}")


def _render_todos(policy: RunPolicy | None, renderer: CLIRenderer) -> None:
    todo_store = getattr(policy, "todo_store", None)
    if todo_store is None:
        renderer.notice("no todo store")
        return
    renderer.section("Todos")
    renderer.bullets(str(todo_store.summary()).splitlines())


def _render_tools(agent: Agent, renderer: CLIRenderer) -> None:
    rows = [
        (_tool_label(tool), tool.description or "no description")
        for tool in agent.tools
    ]
    renderer.rows(rows, heading="Tools")


def _tool_label(tool) -> str:
    tags = []
    if _tool_flag(tool, "is_read_only"):
        tags.append("read-only")
    if _tool_flag(tool, "is_destructive"):
        tags.append("destructive")
    if _tool_flag(tool, "is_concurrency_safe"):
        tags.append("safe")
    if not tags:
        return tool.name
    return f"{tool.name} [{', '.join(tags)}]"


def _tool_flag(tool, method: str) -> bool:
    try:
        return bool(getattr(tool, method)())
    except Exception:
        return False


def _render_skills(
    agent: Agent,
    renderer: CLIRenderer,
    *,
    heading: str = "Skills",
) -> None:
    if agent.skills is None:
        renderer.notice("no skills loaded")
        return
    rows = []
    for skill in agent.skills.all():
        mode = "user" if skill.user_invocable else "model"
        details = skill.description or skill.name
        rows.append((f"/{skill.name} [{mode}]", details))
    renderer.rows(rows, heading=heading)
    problems = agent.skills.problems()
    if problems:
        renderer.bullets(
            [f"{problem.skill}: {problem.message}" for problem in problems],
            heading="Skill problems",
        )


async def _render_compact(
    runtime: Runtime,
    renderer: CLIRenderer,
    model: Model,
    compact_keep: int,
) -> None:
    summary = await runtime.session.summarize(model=model, keep=compact_keep)
    if summary:
        renderer.assistant(summary)
        return
    lines = _compact_transcript(runtime.messages)
    if not lines:
        renderer.notice("nothing to compact")
        return
    renderer.assistant("\n".join(lines))


def _compact_transcript(messages: list[Message]) -> list[str]:
    compacted: list[str] = []
    for message in messages[-10:]:
        if message.role == "system":
            continue
        label = message.role
        content = message.content.strip().replace("\n", " ")
        if message.role == "assistant" and message.tool_calls:
            tool_names = ", ".join(call.name for call in message.tool_calls[:3])
            if content:
                content = f"{content} [{tool_names}]"
            else:
                content = f"tool calls: {tool_names}"
        if message.role == "tool" and message.name:
            label = f"tool:{message.name}"
        if not content:
            content = "(empty)"
        compacted.append(f"{label}: {content[:160]}")
    return compacted


def _model_label(model: Model) -> str:
    explicit = getattr(model, "model", None)
    if explicit:
        return str(explicit)
    return model.__class__.__name__
