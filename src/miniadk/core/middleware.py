import inspect
from contextvars import ContextVar, Token
from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Awaitable, Callable, Literal, Protocol

from .policy import RunState
from .tools import Tool, canonical_tool_name


@dataclass(frozen=True, slots=True)
class AskBefore:
    reason: str


def ask_before(reason: str) -> AskBefore:
    return AskBefore(reason=reason)


@dataclass(slots=True)
class PermissionDecision:
    behavior: Literal["allow", "deny", "ask"]
    message: str | None = None


@dataclass(frozen=True, slots=True)
class PermissionRequest:
    tool: Tool
    arguments: dict
    reason: str


class Middleware(Protocol):
    async def before_model_call(self, state: RunState) -> None:
        ...

    async def after_model_call(self, state: RunState) -> None:
        ...

    async def on_model_error(self, state: RunState, error: Exception) -> None:
        ...

    async def before_tool_call(
        self,
        tool: Tool,
        arguments: dict,
    ) -> PermissionDecision:
        ...

    async def after_tool_call(
        self,
        tool: Tool,
        arguments: dict,
        result: Any,
        text: str,
    ) -> None:
        ...

    async def on_tool_error(
        self,
        tool: Tool,
        arguments: dict,
        error: Exception,
    ) -> None:
        ...


AskUser = Callable[[PermissionRequest], bool | Awaitable[bool]]
GuardMode = Literal["trust", "read", "ask"]
RememberMode = Literal["never", "tool", "reason"]
GuardRule = str | Iterable[str] | Callable[[PermissionRequest], bool]

_current_ask_user: ContextVar[AskUser | None] = ContextVar(
    "miniadk_ask_user",
    default=None,
)


def current_ask_user() -> AskUser | None:
    return _current_ask_user.get()


def middleware_ask_user(middleware: list[Middleware] | None) -> AskUser | None:
    for item in middleware or []:
        ask_user = getattr(item, "ask_user", None)
        if ask_user is not None:
            return ask_user
    return None


def push_ask_user(ask_user: AskUser | None) -> Token | None:
    if ask_user is None:
        return None
    return _current_ask_user.set(ask_user)


def pop_ask_user(token: Token | None) -> None:
    if token is not None:
        _current_ask_user.reset(token)


async def ask_allowed(ask_user: AskUser, request: PermissionRequest) -> bool:
    allowed = ask_user(request)
    if inspect.isawaitable(allowed):
        allowed = await allowed
    return bool(allowed)


class AskBeforeMiddleware:
    def __init__(self, ask_user: AskUser | None = None):
        self.ask_user = ask_user

    async def before_tool_call(
        self,
        tool: Tool,
        arguments: dict,
    ) -> PermissionDecision:
        if not isinstance(tool.permission, AskBefore):
            return PermissionDecision("allow")
        if tool.is_read_only(**arguments):
            return PermissionDecision("allow")

        if self.ask_user is None:
            return PermissionDecision(
                "ask",
                f"Permission denied for {tool.name}: {tool.permission.reason}",
            )

        return PermissionDecision(
            "ask",
            f"Permission denied for {tool.name}: {tool.permission.reason}",
        )


class Guard:
    def __init__(
        self,
        mode: GuardMode = "ask",
        *,
        ask_user: AskUser | None = None,
        remember: bool | RememberMode = "never",
        allow: GuardRule | None = None,
        deny: GuardRule | None = None,
    ):
        self.mode = mode
        self.ask_user = ask_user
        self.remember = _remember_mode(remember)
        self.allow = allow
        self.deny = deny
        self._allowed: set[tuple[str, str | None]] = set()

    def reason(self, tool: Tool, arguments: dict) -> str | None:
        request = self._request(tool, arguments)
        if _rule_matches(self.deny, request):
            return "blocked by guard policy"
        if _rule_matches(self.allow, request):
            return None
        if self.mode == "trust" or tool.is_read_only(**arguments):
            return None
        reason = self._reason(tool, arguments)
        if reason is None or self.mode == "read":
            return reason
        key = self._remember_key(tool, reason)
        if key is not None and key in self._allowed:
            return None
        return reason

    async def before_tool_call(
        self,
        tool: Tool,
        arguments: dict,
    ) -> PermissionDecision:
        request = self._request(tool, arguments)
        if _rule_matches(self.deny, request):
            return PermissionDecision(
                "deny",
                f"Permission denied for {tool.name}: blocked by guard policy",
            )
        if _rule_matches(self.allow, request):
            return PermissionDecision("allow")
        if self.mode == "trust":
            return PermissionDecision("allow")
        if tool.is_read_only(**arguments):
            return PermissionDecision("allow")

        reason = self._reason(tool, arguments)
        if reason is None:
            return PermissionDecision("allow")

        if self.mode == "read":
            return PermissionDecision(
                "deny",
                f"Permission denied for {tool.name}: {reason}",
            )

        key = self._remember_key(tool, reason)
        if key is not None and key in self._allowed:
            return PermissionDecision("allow")

        if self.ask_user is None:
            return PermissionDecision(
                "ask",
                f"Permission denied for {tool.name}: {reason}",
            )

        return PermissionDecision("ask", f"Permission denied for {tool.name}: {reason}")

    def remember_allow(self, tool: str, *, reason: str | None = None) -> None:
        self._allowed.add((tool, reason if self.remember == "reason" else None))

    def allow_request(self, request: PermissionRequest) -> None:
        reason = self._reason(request.tool, request.arguments)
        if reason is None:
            return
        key = self._remember_key(request.tool, reason)
        if key is not None:
            self._allowed.add(key)

    def clear(self) -> None:
        self._allowed.clear()

    def _remember_key(self, tool: Tool, reason: str) -> tuple[str, str | None] | None:
        if self.remember == "never":
            return None
        if self.remember == "reason":
            return (tool.name, reason)
        return (tool.name, None)

    @staticmethod
    def _reason(tool: Tool, arguments: dict) -> str | None:
        if isinstance(tool.permission, AskBefore):
            return tool.permission.reason
        if tool.is_destructive(**arguments):
            return "destructive tool use"
        return None

    @classmethod
    def _request(cls, tool: Tool, arguments: dict) -> PermissionRequest:
        return PermissionRequest(
            tool=tool,
            arguments=arguments,
            reason=cls._reason(tool, arguments) or "tool use",
        )


def _remember_mode(value: bool | RememberMode) -> RememberMode:
    if value is True:
        return "tool"
    if value is False:
        return "never"
    if value not in {"never", "tool", "reason"}:
        raise ValueError("remember must be one of: never, tool, reason")
    return value


def _rule_matches(rule: GuardRule | None, request: PermissionRequest) -> bool:
    if rule is None:
        return False
    if callable(rule):
        return bool(rule(request))
    if isinstance(rule, str):
        return _rule_string_matches(rule, request)
    return any(_rule_string_matches(str(name), request) for name in rule)


def _rule_string_matches(rule: str, request: PermissionRequest) -> bool:
    tool_name, _, pattern = rule.partition(":")
    if canonical_tool_name(tool_name) != canonical_tool_name(request.tool.name):
        return False
    if not pattern:
        return True
    pattern = pattern.strip()
    return any(fnmatch(str(value), pattern) for value in request.arguments.values())
