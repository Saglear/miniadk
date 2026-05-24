"""09 — MCP client: borrow tools from another process.

The Model Context Protocol (MCP) lets an agent call tools that live
in a separate server process — write a fetch server in Node, a search
server in Rust, the agent doesn't care.

We connect to a hypothetical ``my-mcp-server`` over stdio, harvest
its tools, and pass them to the agent. The hub starts the subprocess
when needed and tears it down on exit.

Adjust the ``command`` / ``args`` to point at a real server you have.

Run:

    uv run python examples/09_mcp_client.py
"""

import asyncio

from miniadk import Agent, MCPServer, load_env_upwards, model
from miniadk.core import Runtime
from miniadk.mcp import MCPHub

load_env_upwards()


async def main() -> None:
    hub = MCPHub(
        servers=[
            MCPServer(
                name="files",
                command="python",
                args=["-m", "mcp_server_filesystem", "--root", "."],
            ),
        ],
    )
    async with hub:
        tools = await hub.tools()
        print(f"discovered {len(tools)} MCP tool(s):", [t.name for t in tools])

        agent = Agent(
            name="mcp-user",
            instructions="Use the MCP tools to answer.",
            tools=tools,
        )
        rt = Runtime(agent=agent, model=model())
        print(await rt.ask("List the project's top-level files."))

asyncio.run(main())
