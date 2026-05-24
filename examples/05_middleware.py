"""05 — Middleware: audit and gate tool calls.

Middleware sees every model + tool event. Implement only the hooks
you care about — there's no ABC to subclass, just match the
:class:`Middleware` protocol.

Here we log every tool invocation and forbid one tool by name. The
``Agent.middleware`` field is the canonical home for this kind of
composition; no special preset needed.

Run:

    uv run python examples/05_middleware.py
"""

from miniadk import Agent, PermissionDecision, load_env_upwards, run, tool

load_env_upwards()


@tool
def ping() -> str:
    """Return 'pong'."""
    return "pong"


@tool
def secret() -> str:
    """A tool we don't want the model to call."""
    return "shh"


class AuditMiddleware:
    async def before_tool_call(self, tool, arguments):
        print(f"[audit] {tool.name}({arguments})")
        if tool.name == "secret":
            return PermissionDecision("deny", message="this tool is off-limits")
        return PermissionDecision("allow")

    async def after_tool_call(self, tool, arguments, result, text):
        print(f"[audit] {tool.name} → {text!r}")


agent = Agent(
    name="audited",
    instructions="Use any of your tools. If a tool is denied, just say so.",
    tools=[ping, secret],
    middleware=[AuditMiddleware()],
)

print(run(agent, "Call both tools and report what you saw."))
