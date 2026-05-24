from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, TypedDict

from .core.agent import Agent
from .core.messages import Message
from .core.middleware import Middleware
from .core.policy import DefaultRunPolicy, RunDecision, RunHook, RunState
from .core.tools import tool


AGENTIC_INSTRUCTIONS = """
For multi-step work, keep a todo list. Mark exactly one item in_progress before
working on it. Mark it completed only after it is fully done. If tests or checks
fail, keep the relevant item in_progress, fix the issue, and check again before
giving a final answer. Mark an item blocked only when you cannot make further
progress without user input or an external change.
""".strip()


CHAT_INSTRUCTIONS = """
If the user's latest message is only a greeting, thanks, goodbye, or short
social chat, answer directly. Do not inspect files, search the workspace, run
shell commands, call MCP tools, or spawn agents unless the user asks for project
work.
""".strip()


@dataclass(slots=True)
class TodoStore:
    items: list[dict[str, Any]] = field(default_factory=list)

    def replace(self, items: list[dict[str, Any]]) -> None:
        self.items = [dict(item) for item in items]

    def open_items(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.items
            if str(item.get("status", "pending")) not in {"completed", "blocked"}
        ]

    def summary(self) -> str:
        if not self.items:
            return "no todos"
        lines = []
        for index, item in enumerate(self.items, 1):
            status = item.get("status", "pending")
            content = item.get("content", "")
            lines.append(f"{index}. [{status}] {content}")
        return "\n".join(lines)


@dataclass(slots=True)
class Agentic:
    agent: Agent
    policy: AgenticPolicy
    todos: TodoStore
    middleware: list[Middleware] = field(default_factory=list)


class TodoItem(TypedDict):
    content: str
    status: NotRequired[Literal["pending", "in_progress", "completed", "blocked"]]


def make_todo_tool(store: TodoStore | None = None):
    todo_store = store or TodoStore()

    def validate_todos(todos: list[TodoItem]) -> bool | str:
        result = _normalize_todos(todos)
        if isinstance(result, str):
            return result
        return True

    @tool(validate=validate_todos, schema={"todos": _todo_list_schema()})
    def todo_write(todos: list[TodoItem]) -> str:
        """Update the current task checklist."""
        normalized = _normalize_todos(todos)
        if isinstance(normalized, str):
            return normalized
        todo_store.replace(normalized)
        open_count = len(todo_store.open_items())
        return (
            f"todo list updated: {len(todo_store.items)} items, "
            f"{open_count} not completed"
        )

    return todo_write


def _normalize_todos(todos: list[TodoItem]) -> list[dict[str, str]] | str:
    normalized = []
    if not todos:
        return "todo list needs at least one item"
    for index, item in enumerate(todos, 1):
        if isinstance(item, dict):
            content = str(item.get("content", "")).strip()
            if not content:
                return f"todo {index} needs content"
            status = str(item.get("status", "pending"))
            if status not in {"pending", "in_progress", "completed", "blocked"}:
                return (
                    f"todo {index} has invalid status: {status}. "
                    "Use pending, in_progress, completed, or blocked."
                )
            normalized.append({"content": content, "status": status})
        else:
            content = str(item).strip()
            if not content:
                return f"todo {index} needs content"
            normalized.append({"content": content, "status": "pending"})
    in_progress = [
        str(index)
        for index, item in enumerate(normalized, 1)
        if item["status"] == "in_progress"
    ]
    if len(in_progress) > 1:
        return "only one todo can be in_progress: " + ", ".join(in_progress)
    return normalized


def _todo_list_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "minLength": 1},
                "status": {
                    "enum": ["pending", "in_progress", "completed", "blocked"],
                    "type": "string",
                },
            },
            "additionalProperties": False,
            "required": ["content"],
        },
    }


def make_todo_read(store: TodoStore | None = None):
    todo_store = store or TodoStore()

    @tool(read_only=True, concurrency_safe=True)
    def todo_read() -> str:
        """Read the current task checklist."""
        return todo_store.summary()

    return todo_read


def agentic(
    agent: Agent,
    *,
    todos: TodoStore | None = None,
    middleware: list[Middleware] | None = None,
    max_stop_retries: int = 3,
    chat: bool = False,
) -> "Agentic":
    """Compose ``agent`` with the standard agentic loop policy.

    Returns an :class:`Agentic` struct whose ``.agent`` is a regular
    :class:`miniadk.Agent` already carrying ``policy=AgenticPolicy(...)``
    and ``middleware=[...]``. Because the policy/middleware live on the
    Agent itself, every adapter (run, run_cli, astream_json, ws_json…)
    picks them up through :func:`resolve_composition` with no special
    case for this preset.

    The struct is the convenience handle — ``kit.todos`` lets the
    caller poke at the shared todo store; ``kit.policy`` is just the
    policy instance for explicit access. The struct is intentionally
    not a :class:`miniadk.Agent` subclass: presets stay thin, and
    adapters must not learn about preset-specific shapes.
    """
    todo_store = todos or TodoStore()
    todo_tool = make_todo_tool(todo_store)
    todo_read = make_todo_read(todo_store)
    tools = [
        tool
        for tool in agent.tools
        if tool.name not in {todo_tool.name, todo_read.name}
    ]
    instructions = with_agentic_instructions(agent.instructions)
    if chat:
        instructions = with_chat_instructions(instructions)
    policy = AgenticPolicy(
        todo_store=todo_store,
        max_stop_retries=max_stop_retries,
        chat=chat,
    )
    composed_agent = Agent(
        name=agent.name,
        instructions=instructions,
        tools=[*tools, todo_read, todo_tool],
        skills=agent.skills,
        mcp=agent.mcp,
        policy=policy,
        middleware=list(middleware) if middleware else None,
    )
    return Agentic(
        agent=composed_agent,
        policy=policy,
        todos=todo_store,
        middleware=list(middleware or []),
    )


class AgenticPolicy(DefaultRunPolicy):
    def __init__(
        self,
        *,
        todo_store: TodoStore | None = None,
        stop_hooks: list[RunHook] | None = None,
        max_stop_retries: int = 3,
        chat: bool = False,
    ):
        self.todo_store = todo_store
        self.stop_hooks = stop_hooks or []
        self.max_stop_retries = max_stop_retries
        self.chat = chat
        self._stop_retries = 0

    async def after_model(self, state: RunState) -> RunDecision:
        chat_decision = self._stop_low_intent_tool_use(state)
        if chat_decision is not None:
            return chat_decision

        decision = await super().after_model(state)
        if decision.action != "stop":
            return decision

        retry = self._continue_for_open_todos(state)
        if retry is not None:
            return retry

        for hook in self.stop_hooks:
            hook_decision = await hook.on_stop(state)
            if hook_decision is not None and hook_decision.action == "continue":
                return self._retry(hook_decision)

        return decision

    async def after_tools(self, state: RunState) -> RunDecision:
        self._stop_retries = 0
        return RunDecision()

    def _continue_for_open_todos(self, state: RunState) -> RunDecision | None:
        if self.todo_store is None:
            return None
        if not self.todo_store.items and _needs_todo_plan(_last_user_text(state.messages)):
            message = (
                "This looks like multi-step project work. Start by writing a "
                "short todo list with todo_write, then continue with the first "
                "item. Do not give a final answer until the work is complete or "
                "blocked."
            )
            return self._retry(RunDecision.continue_with(Message("user", message)))
        open_items = self.todo_store.open_items()
        if not open_items:
            return None
        message = (
            "You are not done yet. The todo list still has unfinished items:\n\n"
            f"{self.todo_store.summary()}\n\n"
            "Continue working on the next unfinished item. Do not write a final "
            "answer until the list is complete or you are blocked."
        )
        return self._retry(RunDecision.continue_with(Message("user", message)))

    def _retry(self, decision: RunDecision) -> RunDecision:
        if self._stop_retries >= self.max_stop_retries:
            return RunDecision.stop(
                "Stopped after repeated policy continuations.",
                reason="policy_stop",
            )
        self._stop_retries += 1
        return decision

    def _stop_low_intent_tool_use(self, state: RunState) -> RunDecision | None:
        if not self.chat or state.result is None or not state.result.tool_calls:
            return None
        user_text = _last_user_text(state.messages)
        if not _is_low_intent_chat(user_text):
            return None

        answer = _chat_answer(user_text)
        skipped = [
            Message(
                "tool",
                "Skipped by chat policy because the user did not ask for project work.",
                name=call.name,
                tool_call_id=call.id,
            )
            for call in state.result.tool_calls
        ]
        return RunDecision(
            action="stop",
            reason="policy_stop",
            message=answer,
            inject=[*skipped, Message("assistant", answer)],
        )


def with_agentic_instructions(instructions: str) -> str:
    return f"{instructions.rstrip()}\n\n{AGENTIC_INSTRUCTIONS}"


def with_chat_instructions(instructions: str) -> str:
    return f"{instructions.rstrip()}\n\n{CHAT_INSTRUCTIONS}"


_LOW_INTENT_EXACT = {
    "hi",
    "hello",
    "hey",
    "yo",
    "thanks",
    "thank you",
    "thx",
    "bye",
    "goodbye",
    "ok",
    "okay",
}

_GREETING_WORDS = {"hi", "hello", "hey", "yo"}
_THANKS_WORDS = {"thanks", "thank", "thx"}
_BYE_WORDS = {"bye", "goodbye"}
_WORK_WORDS = {
    "add",
    "build",
    "bug",
    "change",
    "code",
    "commit",
    "create",
    "debug",
    "edit",
    "error",
    "file",
    "fix",
    "git",
    "implement",
    "inspect",
    "install",
    "package",
    "project",
    "read",
    "repo",
    "review",
    "run",
    "search",
    "shell",
    "test",
    "tests",
    "traceback",
    "write",
}

_PLAN_WORDS = {
    "and",
    "then",
    "after",
    "before",
    "multiple",
    "multi",
    "several",
    "all",
    "complete",
    "finish",
}

_CJK_LOW_INTENT_EXACT = {
    "你好",
    "您好",
    "嗨",
    "谢谢",
    "感谢",
    "拜拜",
    "再见",
}

_CJK_GREETING = ("你好", "您好", "嗨")
_CJK_THANKS = ("谢谢", "感谢")
_CJK_BYE = ("拜拜", "再见")
_CJK_WORK = (
    "代码",
    "文件",
    "项目",
    "仓库",
    "读取",
    "搜索",
    "运行",
    "测试",
    "修复",
    "实现",
    "修改",
    "编辑",
    "提交",
    "报错",
)

_CJK_PLAN = (
    "然后",
    "并且",
    "以及",
    "所有",
    "全部",
    "完整",
    "完成",
    "多个",
    "几",
)


def _last_user_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _is_low_intent_chat(text: str) -> bool:
    normalized = _normalize(text)
    words = normalized.split()
    if any(word in _WORK_WORDS for word in words):
        return False

    compact = _compact_text(text)
    if _is_cjk_low_intent_chat(compact):
        return True

    if not normalized:
        return False
    if normalized in _LOW_INTENT_EXACT:
        return True
    if len(words) <= 6 and words[0] in _GREETING_WORDS:
        return True
    if len(words) <= 5 and any(word in _THANKS_WORDS for word in words):
        return True
    if len(words) <= 4 and any(word in _BYE_WORDS for word in words):
        return True
    return False


def _needs_todo_plan(text: str) -> bool:
    normalized = _normalize(text)
    words = normalized.split()
    compact = _compact_text(text)
    if _is_low_intent_chat(text):
        return False
    if any(token in compact for token in _CJK_WORK):
        return any(token in compact for token in _CJK_PLAN) or len(compact) >= 18
    work_count = sum(1 for word in words if word in _WORK_WORDS)
    if work_count >= 2:
        return True
    if work_count == 1 and any(word in _PLAN_WORDS for word in words):
        return True
    return False


def _chat_answer(text: str) -> str:
    compact = _compact_text(text)
    if any(token in compact for token in _CJK_THANKS):
        return "不客气。接下来你想处理什么？"
    if any(token in compact for token in _CJK_BYE):
        return "再见。"
    if any(token in compact for token in _CJK_GREETING):
        return "你好。你想处理什么？"

    words = set(_normalize(text).split())
    if words & _THANKS_WORDS:
        return "You're welcome. What would you like to work on next?"
    if words & _BYE_WORDS:
        return "Goodbye."
    return "Hello. What would you like to work on?"


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def _compact_text(text: str) -> str:
    return "".join(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))


def _is_cjk_low_intent_chat(text: str) -> bool:
    if not text or any(token in text for token in _CJK_WORK):
        return False
    if text in _CJK_LOW_INTENT_EXACT:
        return True
    if len(text) <= 12 and text.startswith(_CJK_GREETING):
        return True
    if len(text) <= 10 and any(token in text for token in _CJK_THANKS):
        return True
    if len(text) <= 8 and any(token in text for token in _CJK_BYE):
        return True
    return False
