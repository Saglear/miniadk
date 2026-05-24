"""07 — A practical repo assistant.

A read-only agent over the current working directory: it can list
files, search text, fetch URLs, and open small files. No write/shell
permissions, so you can hand it the keys without worrying.

Run:

    uv run python examples/07_repo_assistant.py
"""

from pathlib import Path

from miniadk import Agent, load_env_upwards, make_tools, run_cli

load_env_upwards()

tools = make_tools(
    root=Path.cwd(),
    files=True,    # read_file, list_files, glob_files, search_text
    shell=False,   # leave shell off for read-only safety
    write=False,   # no file mutation
    web=True,      # fetch_url for docs lookups
)

run_cli(
    Agent(
        name="repo-buddy",
        instructions=(
            "You answer questions about the user's current repository. "
            "Use list_files / search_text / read_file before guessing. "
            "Cite paths and line numbers when you reference code."
        ),
        tools=tools,
    ),
)
