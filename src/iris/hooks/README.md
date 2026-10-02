# Hooks 核心

`iris.hooks` 定义四种事件、Python 处理器注册和工具反馈结果。`HookDispatcher` 按固定顺序派发事件，区分普通处理器失败与调用取消，并把已取得的反馈和控制事实交回执行 owner。

当前已实现核心模型与独立派发器。`AgentConfig` 尚未开放 `hooks` YAML；RuntimeEnvironment 可保存派发器依赖，但工具与 logical run 的自动触发尚未接通。命令脚本适配器、SDK 装配入口由后续阶段接入。

## 独立派发示例

公共模型从 `iris.hooks` 导入；框架装配代码从 `iris.hooks.dispatcher` 导入派发器。

```python
import asyncio

from iris.hooks import HookEvent, HookRegistration, ToolAfterEvent, ToolAfterResult
from iris.hooks.dispatcher import HookDispatcher
from iris.message import TextBlock
from iris.tools import ToolResult


async def feedback(event: HookEvent) -> ToolAfterResult:
    return ToolAfterResult(feedback="请在回答中说明验证结果。")


async def main() -> None:
    dispatcher = HookDispatcher([
        HookRegistration(
            event="tool.after",
            name="verification-reminder",
            handler=feedback,
            tool_names=["exec_command"],
        ),
    ])
    event = ToolAfterEvent(
        agent_id="agent",
        session_id="session",
        workspace=".",
        call_id="call-1",
        tool_name="exec_command",
        arguments={"command": "echo hello"},
        result=ToolResult(
            tool_use_id="call-1",
            tool_name="exec_command",
            content=[TextBlock(text="hello")],
        ),
        body_status="success",
    )
    outcome = await dispatcher.dispatch(event)
    assert outcome.feedback == ("请在回答中说明验证结果。",)


asyncio.run(main())
```

`tool_names` 精确匹配事件里的实际调用名，省略表示所有普通工具。它不是 builtin 配置键：`exec.command` 加载的工具名为 `exec_command`。

## 事件与返回值

| 事件 | 特有输入 | Python 返回值 |
| --- | --- | --- |
| `run.started` | `run: RunSnapshot`、与 `Run.request.input` 一致的 `str \| list[DataBlock]` 完整输入；脚本 JSON 中块列表逐块序列化，保留图片引用 | `None` |
| `run.finished` | 新终态提交产生的 `result: RunResult` | `None` |
| `tool.before` | `call_id`、`tool_name`、`arguments` | `None` 或 `ToolBeforeResult(deny_reason=...)` |
| `tool.after` | 工具调用字段、`result: ToolResult`、真实 `body_status` | `None` 或 `ToolAfterResult(feedback=...)` |

所有事件还包含 agent/session/run/activation 身份、host workspace 和 UTC 时间。低层独立工具事件允许 run/activation 身份为空。事件是 frozen dataclass；每个处理器收到原事件的独立深快照，修改嵌套字段不会影响执行数据或后续处理器。输入仍以只读使用。

结果模型在公开构造边界约束非空文本，禁止额外字段。Python 返回 `{}`、错误事件的结果模型或其他值属于协议错误。脚本的空 JSON 对象需要由后续命令 adapter 转换为 `None`，Python 处理器不使用该约定。

## 顺序、错误和取消

同一事件按注册顺序串行执行，不同调用可以同时派发。`tool.before` 的第一次拒绝或普通失败停止余项，分别返回 `HOOK_REJECTED` 或 `HOOK_ERROR`；其他事件记录普通失败并继续。`tool.after` 的有效反馈按顺序累积，不作为下一处理器的输入，也不改写工具正文。

Python 处理器必须可等待，默认期限为 10 秒。派发器使用协作式 `asyncio.timeout`，处理器吞掉超时取消后返回的迟到结果也会被丢弃。同步回调不会自动搬到线程。命令注册使用独立私有类型，其期限由命令后端拥有，不再套 Python 超时。

执行 owner 可以通过私有 `cancellation` 参数传入 Run 信号。派发器中断当前处理器并等待必要清理；重复取消不反复打断收口。控制返回保留之前取得的反馈，不继续执行后续处理器。`run.finished` 的 owner 应传 `cancellation=None`，避免使用已终态 Run 的旧信号；非 `COMPLETED` 的结束事件只运行 Python 处理器。

[`_dispatch_types.py`](_dispatch_types.py) 定义仅在进程内使用的交接类型。`HookControl.origin/error` 保留原控制来源，`unknown_error` 独立保存收口过程中发现的真实结果未知，`stop_slot` 引用唯一命令事实槽。后续执行 owner 必须同时消费这些事实，再按执行阶段确定结算方式；它们不进入事件 JSON、工具 metadata 或 checkpoint。

## 维护与验证

[`models.py`](models.py) 只在类型检查时引用 lifecycle 和 ToolResult；[`__init__.py`](__init__.py) 不加载派发器，避免形成 lifecycle/tools/hooks 的循环导入。`event_to_dict` 从可信字段生成脚本输入投影，不重新验证已解析领域模型。

核心测试位于 `tests/hooks/test_models.py` 和 `tests/hooks/test_dispatcher.py`，覆盖事件序列化、注册约束、精确匹配、输入隔离、部分反馈、处理器期限及取消与 unknown/cleanup 的交接。工具结果反馈的持久化和 artifact 投影由 [`tools`](../tools/README.md) 的结果模型负责。
