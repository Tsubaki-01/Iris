# 编写并接入工具

工具把模型的一个意图变成宿主中的一次明确操作。本页先把一个普通 Python 函数变成可调用工具，再接入 YAML；最后说明较大的结果、文件产物和延迟发现该怎样处理。前提是已经完成[快速开始](../getting-started/quickstart.md)，能够在项目环境中执行 `uv run`。

## 先验证函数到工具的完整路径

在仓库根目录保存 `tool_demo.py`：

```python
"""无需模型服务的工具调用示例。"""

import asyncio
from pathlib import Path

from iris.message import ToolUseBlock
from iris.tools import ToolExecutionContext, ToolExecutor, ToolRegistry, tool


@tool(group="math")
def add(left: int, right: int) -> int:
    """计算两个整数的和。

    Args:
        left: 左侧整数。
        right: 右侧整数。

    Returns:
        两个整数之和。
    """
    return left + right


async def main() -> None:
    """通过 schema 校验和执行器调用工具。"""
    registry = ToolRegistry()
    registered = registry.register_function(add)
    print(registered.input_schema)
    result = await ToolExecutor(registry).execute_one(
        ToolUseBlock(id="sum-1", name="add", input={"left": 2, "right": 3}),
        ToolExecutionContext(workspace_root=Path.cwd()),
    )
    print(result.model_content)


if __name__ == "__main__":
    asyncio.run(main())
```

在仓库根目录运行：

```shell
uv run python tool_demo.py
```

成功时先看到含 `left`、`right` 两个必填整数的 JSON Schema，最后一行是 `5`。这一步执行了真实 Python 工具，但没有调用模型。装饰器保留原函数，`add(2, 3)` 仍然是普通 Python 调用；注册才会创建工具适配对象。

函数参数的类型注解决定 schema，Google Style docstring 的 `Args:` 补充参数含义。没有默认值的参数必填；默认值会进入 schema。连接客户端、固定数据源等由宿主提供的参数，可以用 `preset_kwargs` 绑定，它们不暴露给模型。不要把运行时凭据设计成让模型填写的工具参数。

## 把同一个函数交给 Agent

继续使用上面的 `tool_demo.py`。在同一目录保存 `agent.yaml`：

```yaml
name: calculator
model: deepseek/deepseek-flash
system: 对于整数加法，调用 add 并根据真实结果回答。
tools:
  python:
    functions:
      - tool_demo:add
```

配置好模型凭据后，从该目录执行 `uv run iris chat agent.yaml`，输入“用工具计算 2 加 3”。可观察的成功标志是运行中出现 `add` 调用，工具结果为 `5`，随后 Agent 给出回答。Python 引用按 `module:symbol` 导入，因此该模块必须能被当前 Python 环境导入；它不是相对 YAML 的任意 `.py` 文件路径。

需要一次注册多种工具或给工具类注入服务时，使用 registrar。它的唯一参数是 `ToolRegistry`，在 YAML 中放入 `tools.python.registrars`。例如可将以下函数加入 `tool_demo.py`，并用 `tool_demo:register_tools` 替换 `functions` 声明：

```python
def register_tools(registry: ToolRegistry) -> None:
    """一次装配本应用的工具。"""
    registry.register_function(add)
```

同一个工具不要同时通过 functions 和 registrar 注册；名称与别名在一个 registry 中必须唯一。

## 何时需要工具类

函数适合“参数进去、结果出来”的业务操作。需要读取本次 `workspace_root`、session 身份、取消信号，或返回多模态块与产物引用时，继承 `BaseTool` 更直接。

类工具声明 `ToolDefinition`，在 `validate_input` 把模型参数解析成输入模型，在 `arun(params, context)` 使用已经解析的值并返回 `ToolResult`。`BaseTool.validate_input` 默认原样返回字典；仅写 `input_schema` 不会自动产生校验逻辑。可复用 `schema_from_pydantic_model`，让同一个输入模型同时定义校验与 schema，避免两份契约漂移。完整签名和结果字段见[工具参考](../reference/tools.md)。

同步函数默认在当前线程执行。阻塞 I/O 可显式选择 `CallableExecutionMode.THREAD`；异步函数直接 `await`，不能使用 thread 模式。线程执行不会把 Python 线程变成可强制终止的进程，因而不适合把任意长期任务伪装成可立即取消的调用。要运行独立程序，使用[命令与 Python 工具](commands.md)。

## 让结果适合模型与宿主

函数可返回字符串、JSON 可表示的值或 `ToolResult`。普通值转成模型可读文本；`ToolResult` 可明确提供文本/图片块、结构化 `data`、错误和 `artifact`。宿主数据与模型正文用途不同：不要认为写入 `data` 就一定会成为模型可见内容。

长文本会按工具定义的 `max_result_chars` 保存到工作区 `.iris/tool-results/`，返回预览和已保存正文路径。模型需要全文时应调用 `read_file` 按页继续读取。默认阈值与预览规则在参考页统一维护。

已经生成的报告文件可以配置 `file.publish`，由模型调用 `publish_artifact(file_path=...)`。它复制一个发布时的文件快照，源文件后续变化不影响副本；宿主读取结果中的 `artifact.path`、`mime_type` 等信息展示或下载。它不上传云端，也不等于把图片像素送给模型。完整的生成与发布流程见 [Python 配方](commands.md#直接执行完整-python-代码)，图片输入见[媒体指南](media.md)。

## 工具多起来以后

少量常用工具默认直接暴露。大量专业工具可以在 `@tool(deferred=True)` 中声明延迟工具，并启用：

```yaml
context_policy:
  enabled: true
  deferred_tools: true
```

模型先通过 `tool_search` 提交独立意图，例如 `{"queries": ["查找客户订单", "查看库存"]}`，再在后续请求中调用实际获得完整 schema 的工具。搜索命中和本步可调用是两个步骤：上下文预算仍可能影响 schema 是否装入。本地搜索默认无需外部服务，可选 Decision 接点见[扩展指南](extensions.md)。精确的范围、去重、空结果和披露规则集中在[工具发现参考](../reference/tools.md#工具发现)。

选择内置能力时，先查[内置工具目录](../reference/tools.md#内置工具目录)，再按需接入[外部检索与 MCP](mcp.md)、[Skill 与子 Agent](skills-subagents.md)。希望理解执行顺序、并发和结果如何进入下一次模型请求，继续读[工具生命周期设计](../design/tool-execution.md)。

源码入口：[函数适配与结果模型](../../src/iris/tools/base.py)、[YAML 工具装配](../../src/iris/agents/config/tools.py)、[执行器](../../src/iris/tools/executor.py)。
