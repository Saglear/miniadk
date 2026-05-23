# MiniADK

[English](README.md)

MiniADK 是一个小型 Python Agent Development Kit，用来用简洁、可读的 API
构建可以调用工具的智能体。

它提供构建智能体产品所需要的基础部件：

- `Agent`：指令和能力
- `Model`：大模型适配器
- `Tool`：Python 函数工具封装
- `Runtime`：智能体运行循环
- `Event`：给 CLI、Web、UI 使用的事件流
- `Session`：会话状态

MiniADK 是开发套件，不是一个已经固定好的智能体产品。核心保持小而稳定，
模型适配器、工具、策略、skills、MCP、CLI 适配器和 preset 都围绕核心组合。

## 安装

从源码安装：

```bash
git clone https://github.com/Saglear/miniadk.git
cd miniadk
uv sync --extra dev
```

运行测试：

```bash
uv run --extra dev pytest -q
```

## 快速开始

```py
from miniadk import Agent, model, run_cli, tool


@tool
def add(left: int, right: int) -> int:
    """Add two numbers."""
    return left + right


agent = Agent(
    "calc",
    "Use tools when they help.",
    tools=[add],
)

run_cli(agent, model=model())
```

运行：

```bash
uv run python calc.py
```

## 单次调用

```py
from miniadk import Agent, model, run, tool


@tool
def greet(name: str) -> str:
    """Return a greeting."""
    return f"hello {name}"


agent = Agent("hello", "Use tools when useful.", tools=[greet])
answer = run(agent, "Greet Ada", model=model())
print(answer)
```

## 模型

`model()` 会从环境变量读取配置，并返回可用的模型适配器。

OpenAI 兼容接口：

```txt
OPENAI_KEY=...
OPENAI_URL=...
OPENAI_BASE_URL=...
OPENAI_MODEL=...
```

Anthropic 接口：

```txt
ANTHROPIC_KEY=...
ANTHROPIC_URL=...
ANTHROPIC_BASE_URL=...
ANTHROPIC_MODEL=...
```

如果同时配置了多个提供方，可以明确指定默认值：

```txt
MINIADK_MODEL_PROVIDER=openai
```

或：

```txt
MINIADK_MODEL_PROVIDER=anthropic
```

## 工具

带类型标注的 Python 函数可以直接变成工具：

```py
from pathlib import Path

from miniadk import tool


@tool
def read_note(path: str) -> str:
    """Read a UTF-8 note."""
    return Path(path).read_text(encoding="utf-8")
```

MiniADK 会用函数名、docstring 和类型标注生成工具 schema。同步函数和异步函数
都支持。

## 内置工具

`miniadk.stdtools` 提供可复用工具：

```py
from pathlib import Path

from miniadk import Agent, model, run_cli
from miniadk.stdtools import make_list_files, make_read_file, make_search_text

root = Path.cwd()
agent = Agent(
    "repo",
    "Help inspect this repository.",
    tools=[
        make_list_files(root=root),
        make_read_file(root=root),
        make_search_text(root=root),
    ],
)

run_cli(agent, model=model())
```

文件和 shell 工具的路径检查、权限提示、输出限制和超时控制都放在运行核心之外。

## Skills 和 MCP

Skills 和 MCP 是业务层集成。它们会在运行循环开始前解析成普通指令和普通工具。

```py
from miniadk import Agent, MCPHub, MCPServer, SkillRegistry, model, run_cli

agent = Agent(
    "assistant",
    "Use the configured project capabilities.",
    skills=SkillRegistry.from_paths(".miniadk/skills"),
    mcp=MCPHub([
        MCPServer(name="docs", command="uvx", args=["some-mcp-server"]),
    ]),
)

run_cli(agent, model=model())
```

## 核心结构

运行循环保持直接：

```txt
用户消息
  -> 模型调用
  -> 可选工具调用
  -> 工具结果
  -> 模型回复
  -> 事件
```

核心概念是：

```txt
Message  - 智能体看到的内容
Model    - 智能体如何请求大模型
Tool     - 智能体可以做什么
Agent    - 指令和能力
Runtime  - 连接所有部件的循环
Event    - 适配器和 UI 观察到的事件
Session  - 持久化会话状态
```

## 包结构

```txt
src/miniadk/core/       原子运行时类型和循环
src/miniadk/models/     模型提供方适配器
src/miniadk/stdtools/   文件、shell、web、agent 等可复用工具
src/miniadk/adapters/   CLI、JSON、Web、WebSocket 适配器
src/miniadk/skills.py   skill 加载和调用辅助
src/miniadk/mcp.py      MCP 集成
src/miniadk/presets.py  可选的高层组装辅助
```

常用 API 从 `miniadk` 导入。需要更细控制时再使用子模块。

## 示例

```bash
uv run --extra dev python examples/smoke_llm.py
uv run --extra dev python examples/scripted_tiny_product.py
uv run --extra dev python examples/coder_preset.py
uv run --extra dev python examples/compact_coder.py
uv run --extra dev python examples/repo_cli.py
uv run --extra dev python examples/cli_interaction_lab.py
```

## 开发

```bash
uv sync --extra dev
uv run --extra dev pytest -q
uv build
```

## 许可证

MIT
