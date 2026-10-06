# 使用 Hook、Middleware 与 Decision

扩展的第一步是选择需要介入的阶段。只需在固定事件上执行逻辑，选 Hook；需要包装一次普通工具调用，选 ToolMiddleware；只消费已经提交的运行事实，选 RunEventObserver。Decision 则是工具发现等特定判断接点的可选后端，不是通用执行拦截器。

## 给工具结果追加解释

下面的示例无需模型服务。在仓库根目录保存 `extension_demo.py`：

```python
"""给真实工具结果追加 Hook 反馈。"""

import asyncio
from pathlib import Path

from iris.hooks import (
    HookEvent,
    HookHandler,
    HookRegistration,
    HookResult,
    ToolAfterEvent,
    ToolAfterResult,
)
from iris.hooks.dispatcher import HookDispatcher
from iris.message import ToolUseBlock
from iris.tools import ToolExecutionContext, ToolExecutor, ToolRegistry


def create_feedback(*, text: str) -> HookHandler:
    """同步工厂返回一个异步事件处理器。"""
    async def handle(event: HookEvent) -> HookResult:
        if isinstance(event, ToolAfterEvent):
            return ToolAfterResult(feedback=text)
        return None

    return handle


def square(value: int) -> int:
    """返回整数的平方。"""
    return value * value


async def main() -> None:
    """执行一次工具并打印追加反馈后的正文。"""
    registry = ToolRegistry()
    registry.register_function(square)
    dispatcher = HookDispatcher([
        HookRegistration(
            name="explain-result",
            event="tool.after",
            tool_names=("square",),
            handler=create_feedback(text="请向用户解释这个数的计算方式。"),
        ),
    ])
    result = await ToolExecutor(registry, hook_dispatcher=dispatcher).execute_one(
        ToolUseBlock(id="square-1", name="square", input={"value": 4}),
        ToolExecutionContext(workspace_root=Path.cwd()),
    )
    print(result.model_content)


if __name__ == "__main__":
    asyncio.run(main())
```

执行 `uv run python extension_demo.py`，成功输出包含 `16` 和 `[Hook feedback]` 后的解释要求。真实工具结果保留，反馈作为后续模型可见内容追加。

要在正常 Agent 装配时使用该工厂，把以下片段合并到已有 YAML，并确保模块可被当前 Python 环境导入：

```yaml
hooks:
  - name: explain-result
    event: tool.after
    tools: [square]
    handler:
      type: python
      factory: extension_demo:create_feedback
      options:
        text: 请向用户解释这个数的计算方式。
tools:
  python:
    functions: [extension_demo:square]
```

`tools` 过滤填写模型实际调用名，比如 `write_file`，而不是 YAML builtin 键 `file.write`。工厂按 `factory(**options)` 同步执行一次，返回的 handler 在匹配事件时异步执行。

## 包装一次调用

Middleware 适合计时、缓存命中返回或替换模型可见结果。以下类和工厂可加入 `extension_demo.py`：

```python
from time import perf_counter
from iris.tools import ToolCall, ToolMiddleware, ToolNext, ToolResult


class TimingMiddleware(ToolMiddleware):
    """为结果增加一次包装链的耗时。"""

    async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
        """调用一次下游，返回包含计时信息的新结果。"""
        started = perf_counter()
        result = await call_next()
        return result.model_copy(update={
            "stats": {**result.stats, "wrapped_seconds": perf_counter() - started},
        })


def create_timing() -> ToolMiddleware:
    """构造无外部资源的包装器。"""
    return TimingMiddleware()
```

在 YAML 中声明：

```yaml
middleware:
  tools:
    - factory: extension_demo:create_timing
```

宿主查看工具结果的 `stats.wrapped_seconds` 即可观察效果。修改应返回新 `ToolResult`，不要原地改下游对象；`ToolCall.arguments` 是调用快照，修改它不会改真实参数。第一个注册项最外层，`call_next()` 在当前包装器里最多调用一次，也可以完全不调用而直接返回替代结果。它不是任意重试引擎。

## 事件的作用范围

Hook 支持 `run.started`、`run.finished`、`tool.before` 和 `tool.after`。`tool.before` 可返回 `ToolBeforeResult(deny_reason=...)` 拒绝当前调用；`tool.after` 可追加反馈。run 事件只接受 `None` 返回值。等待人工输入并不是 run.finished；恢复原 run 也不是重新 run.started。

普通工具路径先做输入及权限判定，再执行 before Hook、Middleware、body、after Hook、最终结果处理。Middleware 短路而没有运行 body 时，不会触发 after Hook。permission WAITING、`ask_question` 和外层 `subagent` 属于运行控制路径，不进入普通工具包装链；child 内部自己的普通工具照常使用 child 扩展。

也可使用 `handler.type: command` 运行固定脚本。脚本从 stdin 接收事件 JSON，stdout 返回一个 JSON 对象，诊断写 stderr；空结果用 `{}`。该脚本借用当前 Agent 的命令环境和独立 Hook 期限，不需要暴露 `exec_command` 给模型。具体输出字段和错误行为见[扩展参考](../reference/tools.md#hook-与-middleware)。

如果目的是把已完成提交的事件送到自己的面板或日志服务，实现 `RunEventObserver.on_event` 并通过 `AgentRunner.from_config_path(..., observers=[...])` 注入。observer 失败不回滚运行结果，也不能阻止工具执行。完整例子与观测服务见[观察运行](observability.md)。

## 为延迟工具发现启用 Decision

默认 `tool_search` 在本地元数据上检索。需要使用 TypeSafe/Jev 选择候选时，在 Agent YAML 所在目录创建 `decision.yaml`：

```yaml
tools:
  discovery: true
```

Agent 配置加入：

```yaml
decision:
  path: decision.yaml
context_policy:
  enabled: true
  deferred_tools: true
```

同时配置 `IRIS_PROVIDER_API_KEYS__TYPESAFE`，并按[工具指南](tools.md)注册至少一个 `deferred=True` 工具。模型仍使用同一个 `tool_search` 输入，结果仍是逐意图至多一个工具或 null。Decision 只从当前允许的 deferred 目录选择；eager 工具、Skill 和子 Agent 不进入它的候选目录。

这是一次额外网络判断，失败返回工具错误，不静默切回本地检索。结果 `metadata.decision` 含 feature、provider、model、question_count 和独立 usage，便于宿主确认实际采用了哪个接点。它不替换主模型，不改变工具执行权限，也不保证实际任务效果提升。精确配置与 SDK 注入见[Decision 参考](../reference/tools.md#decision)。

后续阅读：[扩展边界设计](../design/extensions.md)、[工具生命周期](../design/tool-execution.md)。源码入口：[扩展装配](../../src/iris/runtime/_extensions.py)、[Hooks](../../src/iris/hooks/models.py)、[包装链](../../src/iris/tools/_middleware_chain.py)、[Decision 配置](../../src/iris/decision/config.py)。
