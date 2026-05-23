# Safety Boundaries

MiniADK is an agent development kit. It provides safety primitives and safe
defaults, but the product that uses MiniADK still owns its final policy.

The core runtime stays small:

- models choose messages and tool calls
- tools run ordinary Python functions
- middleware decides whether sensitive tools may run
- adapters render events and ask users for decisions

Do not hide product policy inside `src/miniadk/core/`.

## Permissions

Sensitive tools should declare intent with permission metadata:

```python
from miniadk import ask_before, tool


@tool(permission=ask_before("writing files"))
def save(path: str, text: str) -> str:
    """Write text."""
    ...
```

`Runtime` emits permission events. CLI and web products decide how to ask the
user, whether to allow, and how to display denials.

Use `Guard` for common modes:

- `Guard("trust")`: allow all tool calls.
- `Guard("read")`: allow read-only tools and deny risky tools.
- `Guard("ask")`: ask before non-read-only or destructive tools.

Products can add small allow/deny rules without changing tools:

```python
Guard(
    "ask",
    allow=["read_file", "search_text"],
    deny="shell:rm *",
)
```

`deny` rules run before `allow` rules. Use them for product-specific hard stops,
such as blocking selected shell commands, while still asking for other risky
tools. String rules can be either a tool name, such as `"shell"`, or a short
`"tool:pattern"` rule matched against stringified argument values.

`Guard("ask", remember=True)` can remember an allowed tool decision for the
current middleware instance. Use this for product flows where the user explicitly
trusts repeated use of a tool during one session. The default is not to remember
decisions. Products that expose remembered permissions should give users a clear
way to clear or scope that trust.

## File Tools

Standard file tools live in `miniadk.stdtools`, not core.

File safety defaults:

- paths are resolved inside the configured root
- path traversal outside root is rejected
- list/search traversal skips symlinks that resolve outside root
- write and edit tools ask before running
- write and edit tools validate inputs before permission prompts and execution
- write and edit tools replace files atomically after preparing content
- multi-edit validates all replacements before writing
- search tools validate regex patterns before execution
- read/list/glob/search tools are marked read-only
- `read_file` can read a line range with `offset` and `limit`, and can include line numbers with `numbers=True`
- empty file and empty line-range reads are surfaced clearly in model-facing text
- `read_file` rejects obvious binary files and reports invalid UTF-8 text clearly
- `search_text` skips obvious binary files
- generated directories can be ignored with `ignore=...`
- large reads, broad list/glob scans, large search files, and broad search scans can be limited

Example:

```python
from miniadk import Agent, make_tools

agent = Agent(
    "repo",
    "Help with this workspace.",
    tools=make_tools(root=".", write=True, shell=False),
)
```

For product agents, prefer a narrow root such as the current workspace. Do not
point tools at a user's home directory unless the product really needs that
access.

## Shell Tools

Shell access is powerful and should remain explicit.

Shell safety defaults:

- shell tools ask before running
- empty commands and invalid working directories fail before execution
- commands have a timeout
- timeouts terminate the shell process group
- task cancellation terminates the shell process group
- output can be capped with `max_shell`
- commands can be validated before execution
- environment variables can be overridden or removed
- `coder()` removes common model API keys from shell tools unless `shell_env` is explicitly provided
- `coder()` can pass `validate_shell`, scan limits, and ignore rules to its standard tools without replacing the preset
- `coder(agents=[...])` adds child-agent tools only when requested; `work=True` adds background work tools for starting, listing, reading, and cancelling work

Example:

```python
from miniadk import make_tools

tools = make_tools(
    root=".",
    validate_shell=lambda command: "rm is disabled" if command.startswith("rm ") else True,
    max_shell=20000,
    shell_env={"OPENAI_API_KEY": None, "ANTHROPIC_API_KEY": None},
)
```

`shell=True` is intentionally kept in the standard tool because coding agents
need normal shell behavior. Treat it as sensitive. Production products should
combine validation, permission prompts, narrow working directories, output
limits, process-group timeouts, and environment filtering. This is still not a
sandbox.

## URL Fetch Tools

Network reads are opt-in standard tools, not core runtime behavior.

`make_fetch_url()` and `make_tools(web=True)` provide a read-only `fetch_url`
tool for HTTP(S) text:

- only `http://` and `https://` URLs are accepted
- requests have a timeout
- responses can be capped with `max_fetch`
- truncated or non-2xx responses keep status context in model-facing text
- products can pass `allow_url` to restrict hosts or URL shapes
- `coder(web=True)` enables URL fetch for the preset; default `coder()` does not

This is not a browser automation or web-search product. Products that fetch
untrusted URLs should apply their own host allowlist, content policy, and
network sandbox.

## MCP

MCP integrations stay outside core and resolve into ordinary tools, resources,
prompts, and skills.

MCP safety defaults:

- MCP subprocesses are closed through `MCPHub.close()` or `async with MCPHub(...)`
- duplicate MCP server names are rejected so tool/resource ownership stays unambiguous
- server stderr is drained to avoid deadlocks
- initialize failures clean up the process
- malformed server framing raises stable runtime errors and startup cleanup still runs
- MCP `isError` tool results become MiniADK tool errors instead of successful text
- MCP tools are ordinary MiniADK tools after discovery
- server info and advertised capabilities can be inspected with `MCPHub.info()`
- resource and prompt discovery skips servers that do not advertise those capabilities
- discovery caches can be refreshed with `MCPHub.refresh()`
- server notifications can be inspected with `MCPHub.notices()` and cleared with `MCPHub.clear_notices()`
- subprocess environment inheritance can be disabled with `MCPServer(inherit_env=False)`
- inherited environment values can be removed with `MCPServer(env={"NAME": None})`

Recommended product practice:

- use explicit server definitions
- keep server environment variables minimal
- inspect `await hub.info()` when product behavior depends on server capabilities
- call `hub.refresh()` after changing server capabilities or configuration
- call `await hub.close()` in `finally`, or use `async with MCPHub(...) as hub`
- treat unknown third-party MCP servers like untrusted code
- place product-specific allowlists outside MiniADK core

## Model Providers

Provider adapters live in `miniadk.models`.

Provider safety boundaries:

- runtime depends on the `Model` protocol, not a vendor SDK
- provider payload shape stays in provider adapters
- HTTP transport wraps network and invalid JSON errors
- provider HTTP errors include status and reason, but not response bodies
- `model()` can load the nearest `.env` but does not override existing environment variables
- real API keys should come from environment variables or explicit constructor args

Do not print `.env` contents, raw API keys, or full provider error bodies into
logs or generated docs.

## Sessions And Compaction

Sessions keep conversation messages. Long-running products should decide how to
store, compact, or expire session state.

MiniADK provides:

- `Session.save(path)`
- `Session.load(path)`
- `Session.trim(keep=...)`
- `Session.compact(summary, keep=...)`
- `Session.summarize(model=..., keep=...)`
- `Session.stats`
- `sessions("dir")` for simple named session files
- `run_cli(..., session=True)` for `.miniadk/sessions/<agent>.json`
- `run_cli(..., session="path.json")`

Session files are written through a temporary file and atomic replace so failed
saves do not leave a partially written session file.
`SessionStore` encodes names into one file name so a user-provided session name
cannot become a path traversal through the store root.

The runtime does not silently delete conversation history. Products should make
trimming or compaction visible and test their own persistence policy. CLI
`/compact` defaults to preview mode; products opt into model-backed compaction
with `run_cli(..., compact=True)`. Passing `session=True` or a session path to
`run_cli()` persists normal turns and compacted session state to disk.

## Adapter Boundary

Adapters own UX policy:

- CLI prompts
- web rendering
- websocket message shape
- slash commands
- status display

Adapters should consume runtime events. They should not require runtime core to
know about terminals, browsers, sockets, or product workflows.

## Product Checklist

Before shipping a MiniADK-based agent product:

- choose a narrow file root
- decide whether shell is enabled
- validate or deny high-risk shell commands
- filter secrets from shell environments
- use `Guard("ask")` or stricter middleware for risky tools
- keep remembered permissions session-scoped and clearable
- close MCP hubs
- keep API keys in environment variables or secret stores
- run product-specific tests for permission prompts
- keep product instructions outside MiniADK core
