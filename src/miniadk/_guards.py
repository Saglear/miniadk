from __future__ import annotations

from .core.middleware import AskUser, Guard, Middleware, current_ask_user


def bind_guards(
    middleware: list[Middleware] | None,
    *,
    ask_user: AskUser | None,
) -> list[Middleware] | None:
    ask_user = ask_user or current_ask_user()
    if ask_user is None:
        return middleware
    items = []
    for item in middleware or []:
        if isinstance(item, Guard) and item.ask_user is None:
            item = copy_guard(item, ask_user=ask_user)
        items.append(item)
    return items


def copy_guard(guard: Guard, *, ask_user: AskUser) -> Guard:
    copied = Guard(
        guard.mode,
        ask_user=ask_user,
        remember=guard.remember,
        allow=guard.allow,
        deny=guard.deny,
    )
    copied._allowed = set(guard._allowed)
    return copied
