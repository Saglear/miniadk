# MiniADK

[English](README.md)

一个小巧的 Python Agent Development Kit。用紧凑、可读的 API 构建会用工具的
agent。核心保持精简；providers、tools、policies、skills、MCP、CLI 适配器、
预设都在外围，按需取用。

MiniADK 是开发工具，**不是成品 agent**。我们提供积木，你写智慧。

## 安装

```bash
pip install miniadk
```

核心就这一行命令。默认终端 UI（基于 Ink，TypeScript 写的）会在第一次调用
`run_cli` 时按需下载，缓存在 `~/.cache/miniadk/tui/`。如果不想走网络：

```bash
pip install miniadk[tui-textual]   # 纯 Python 后备 TUI
```

然后设置 `MINIADK_TUI_NO_FETCH=1`，或给 `run_cli` 传 `backend="textual"`。

开发：

```bash
git clone https://github.com/Saglear/miniadk.git
cd miniadk
uv sync --extra dev
uv run --extra dev pytest -q
```

## 三段 Quickstart

### 1. 无 UI agent（≤ 15 行）

```py
from miniadk import Agent, run, tool

@tool
def add(a: int, b: int) -> int:
    "返回 a + b。"
    return a + b

agent = Agent("calc", "用工具来辅助回答。", tools=[add])
print(run(agent, "17 + 25 等于多少？"))
```

### 2. 默认终端 UI（≤ 10 行）

```py
from miniadk import Agent, make_tools, run_cli

run_cli(Agent(
    "repo",
    "回答关于当前仓库的问题。",
    tools=make_tools(write=False, shell=False),
))
```

### 3. 自定义 React/Ink UI（~30 行，见 `examples/custom_tui/`）

```tsx
import { mount, BridgeProvider, useBridgeSend, useBridgeEvents,
         Markdown } from "@miniadk/tui";

function App() {
  const send = useBridgeSend();
  const [items, setItems] = useState([]);
  useBridgeEvents("message", (e) => setItems(p => [...p, e.data.text]));
  return /* …你的布局… */;
}

mount((bridge) => <BridgeProvider bridge={bridge}><App/></BridgeProvider>);
```

通过 `MINIADK_TUI_BIN` 让 Python 找到你的二进制，`run_cli` 负责其余部分。

## 模型

`model()` 从环境变量读取配置：

```txt
ANTHROPIC_API_KEY=...     # 或 ANTHROPIC_AUTH_TOKEN
ANTHROPIC_BASE_URL=...    # 可选，用于代理 / Anthropic 兼容服务
ANTHROPIC_MODEL=claude-opus-4-7
```

```txt
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
OPENAI_MODEL=gpt-5-pro
```

两者都设置时用 `MINIADK_MODEL_PROVIDER=anthropic`（或 `openai`）显式指定。

## 工具

任何带类型注解 + docstring 的 Python 函数都能成为工具：

```py
from miniadk import tool

@tool
def now_utc() -> str:
    "返回当前 UTC 时间，ISO-8601 格式。"
    from datetime import datetime, UTC
    return datetime.now(UTC).isoformat(timespec="seconds")
```

同步、异步皆可。装饰器读函数名、docstring、类型注解生成 JSON Schema。

`make_tools` 返回一组开箱即用的工具：

```py
from miniadk import make_tools

tools = make_tools(
    files=True,    # read_file, list_files, glob_files, search_text
    shell=False,   # subprocess.run 包装
    write=False,   # 修改类文件操作
    web=True,      # fetch_url
)
```

每个工具自行管理路径检查、权限提示、限制——runtime 保持纯净。

## 组合

`Agent` 自己带 `policy` 与 `middleware`。adapter 永远不需要学习预设的形状，
只在入口调一次 `resolve_composition(agent)`。

```py
from miniadk import Agent, RunDecision

class StopAfterThreeTools:
    def __init__(self): self.rounds = 0
    async def after_model(self, state):
        r = state.result
        if r and r.message and not r.tool_calls:
            return RunDecision.stop(r.message)
        return RunDecision()
    async def after_tools(self, state):
        self.rounds += 1
        return RunDecision.stop("达到上限") if self.rounds >= 3 else RunDecision()

agent = Agent("bounded", "简短回答。", policy=StopAfterThreeTools())
```

middleware 同理（`before_tool_call` 拦截，`after_tool_call` 记录）。
参考 `examples/05_middleware.py` 与 `examples/08_custom_policy.py`。

## Skills 与 MCP

Skills 是用户用 `/name` 触发的指令书（prompt + 允许的工具）。
MCP 服务器是外部工具源，通过 stdio 连接。两者都在进入 runtime loop 之前
解析为普通 `Tool`。

```py
from miniadk import Agent, MCPServer, run_cli, skill
from miniadk.mcp import MCPHub
from miniadk.skills import SkillRegistry

run_cli(Agent(
    "assistant",
    "使用现有能力回答。",
    skills=SkillRegistry.from_skills(
        skill("review", "读取 $path 并总结。", tools=["read_file"], args=["path"]),
    ),
    mcp=MCPHub([MCPServer(name="docs", command="uvx", args=["some-mcp-server"])]),
))
```

## Examples

教学梯度 + 实用模板，索引见 [`examples/README.md`](examples/README.md)。要点：

- `01–05` — Agent、工具、流式、会话、middleware（概念）。
- `06–07` — 默认 CLI；只读仓库助手。
- `08` — 写你自己的 `RunPolicy`（这就是怎么"实现 ReAct"的，而不是 import
  一个预设）。
- `09–10` — MCP 客户端；slash-skill 路由。
- `custom_tui/` — 用你自己的 React 组件完整替换 TUI，复用 bridge。

```bash
uv run python examples/01_hello_agent.py
uv run python examples/06_run_cli.py
```

## 架构

两层分层：

```
adapters/      tui_ink   tui_textual   json   web   ws    (按需加载)
core/          Agent  Tool  Model  Runtime  Session  Event  RunPolicy
```

`import miniadk` 只加载核心——不会拉起 Textual、Ink、React。
TUI 依赖在属性访问时才解析。详见
[`docs/architecture.md`](docs/architecture.md)（分层与扩展点）与
[`docs/tui-protocol.md`](docs/tui-protocol.md)（Python ↔ Ink 子进程的 JSON
协议）。

## MiniADK 不提供什么

按设计——这些应该由应用层提供，而不是框架：

- 不提供 agent loop 预设动物园（ReAct、Plan-and-Execute、Tree-of-Thought、
  reflection）。用 `RunPolicy` + middleware 自己组合。
- 不提供 prompt 模板库。字符串就够用。
- 不提供 retrieval / 向量库集成。需要时把它包成一个 `Tool`。

`agentic()` 是我们唯一带主张的预设，把它当成可读可抄的示例，不是规范。

## License

MIT
