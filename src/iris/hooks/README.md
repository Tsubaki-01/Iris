# Hooks

`iris.hooks` 定义四种事件、Python 处理器注册和工具反馈结果。`HookDispatcher` 按固定顺序派发事件，区分普通处理器失败与调用取消，并把已取得的反馈和控制事实交回执行 owner。

通过 `AgentConfig.hooks` 配置 Python 或 Native/Docker 命令处理器；`middleware.tools` 配置普通工具包装链。`AgentRunner` 和 `RuntimeFactory` 的 `from_config()` / `from_config_path()` 都接受 `hooks=` 与 `tool_middlewares=`，固定追加在 YAML 项之后。同一次装配只构造一份处理器和 Middleware 实例，`RuntimeEnvironment`、工具执行器与 harness 复用同一派发器。

## 从配置开始

在可导入的 `my_extensions.py` 中定义同步工厂。工厂只构造对象；Hook 返回异步 callable，Middleware 返回实现唯一 `wrap_tool_call()` 方法的实例：

```python
import logging

from iris.hooks import HookEvent, HookHandler, ToolAfterResult
from iris.tools import ToolCall, ToolMiddleware, ToolNext, ToolResult


def create_feedback(*, text: str) -> HookHandler:
    async def feedback(event: HookEvent) -> ToolAfterResult:
        return ToolAfterResult(feedback=text)

    return feedback


class LoggingMiddleware(ToolMiddleware):
    async def wrap_tool_call(self, call: ToolCall, call_next: ToolNext) -> ToolResult:
        result = await call_next()
        logging.getLogger(__name__).info("工具 %s，错误=%s", call.tool_name, result.is_error)
        return result


def create_logging() -> ToolMiddleware:
    return LoggingMiddleware()
```

在 `agent.yaml` 声明扩展，模块需要可被运行 Iris 的 Python 环境导入：

```yaml
name: local-agent
model: openai/gpt-4o-mini
system: 你是一个本地助手。
tools:
  builtin: [exec.command]
permissions:
  workspace: .
  execute: allow
hooks:
  - name: feedback
    event: tool.after
    tools: [exec_command]
    timeout_seconds: 10
    handler:
      type: python
      factory: my_extensions:create_feedback
      options:
        text: 请说明验证结果。
middleware:
  tools:
    - factory: my_extensions:create_logging
      options: {}
```

YAML 只在加载边界校验结构。装配时按唯一 `factory(**options)` 协议调用一次，不预跑 handler，不探测旧签名；`options` 由用户工厂解释。导入、构造或返回对象类型错误为 `IrisConfigError`。同 Agent 的 session 复用实例；不同工具可以并发执行，临时调用状态应放局部变量。没有扩展声明时不创建派发器。

SDK 可以继续追加处理器与 Middleware，默认参数都是空不可变序列：

```python
from iris.harness import AgentRunRequest, AgentRunner
from iris.hooks import HookEvent, HookRegistration


async def observe_finish(event: HookEvent) -> None:
    print(event.event, event.run_id)


runner = AgentRunner.from_config_path(
    "agent.yaml",
    hooks=[HookRegistration(event="run.finished", name="print-finish", handler=observe_finish)],
    tool_middlewares=[],
)
try:
    result = await runner.start(AgentRunRequest(input="请执行 echo hello", session_id="default"))
    print(result.assistant_message)
finally:
    await runner.aclose()
```

上述真实模型示例需要对应 provider 配置。`RuntimeFactory` 也可装配 Run 处理器后交给 `AgentRunner(runtime)`，但直接 `AgentRuntime.execute()` 只派发工具事件。child 只使用自己的 YAML，父级 SDK 追加项不会隐式继承；父侧 `subagent` 委派不进入普通工具链。独立 `ToolExecutor` 可以注入 Python 工具处理器，run ID 可空；没有 harness/命令 binding 时不支持配置式脚本 Run 作用域，也不能接收无人消费的延迟控制。

Middleware 首个注册项最外层，`call_next()` 零参、至多调用一次；`call` 和返回的下游结果只读，要改结果应返回新 `ToolResult`。短路结果不会触发 `tool.after`，参数不能改写，不能重试真实 body。完整边界见 [工具说明](../tools/README.md)。Hooks 尽力执行，不补发，也不是资源释放或可靠投递保证。

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

`run.started.input` 保留字符串、图文混排或纯图片输入，与已接纳请求的块顺序一致。
`tool.after.result.content` 是 Middleware 返回的完整文字/图片块列表；需要文字视图时读取
`model_content`，需要完整模型内容时读取 `model_blocks`。图片包含 `original`、`model` 文件引用
与可选 `name`，事件 JSON 保留这些字段，不内嵌图片 bytes 或 provider 的 base64 编码。
Hook 反馈仍只能返回文字，并在正文末尾追加；结构化错误替换正文文字时也保留图片。

结果模型在公开构造边界约束非空文本，禁止额外字段。Python 返回 `{}`、错误事件的结果模型或其他值属于协议错误。命令 adapter 将脚本的空 JSON 对象转换为 `None`，Python 处理器不使用该约定。

## 命令 Hook

框架装配代码使用 [`command.py`](command.py) 的 `CommandHookAdapter(binding=..., workspace=..., command=..., timeout_seconds=10)`，再将该 callable 放入私有 `CommandHookRegistration.handler`。它持有当前 Agent 的命令 binding 与 host workspace，直接调用 `CommandService.execute`；不经过工具注册表或 `exec.command`，因此不会递归触发工具 Hook。

每次执行生成独立 `hook_*` call ID，原工具 call ID 只保留在事件数据中。事件通过显式 JSON 投影编码为 UTF-8 bytes，作为一次性 stdin 传入。Native 使用二进制临时文件句柄，Docker 上传独立输入文件供 helper 打开；读取到末尾后得到 EOF，不提供交互式输入。

脚本只向 stdout 输出一个 JSON object。下面的 Python 脚本接受任意事件，在工具完成时追加反馈：

```python
import json
import sys

event = json.loads(sys.stdin.buffer.read().decode("utf-8"))
result = {"feedback": "请说明验证结果。"} if event["event"] == "tool.after" else {}
sys.stdout.buffer.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
```

`tool.before` 可以输出 `{"deny_reason": "本次调用的拒绝原因"}`；`tool.after` 可以输出 `{"feedback": "追加反馈"}`；四种事件都接受 `{}`。空 stdout、`null`、多个 JSON、额外字段或错误事件的结果字段都是协议错误。stderr 作为日志记录，不作为反馈。stdout 字节被截断时禁止解析预览；只有 stderr 达到 byte limit 时，完整 stdout JSON 仍可使用。任何流出现 drain timeout、stream error 或 stream closed 都按采集不完整处理。

命令直接采用 Hook 的期限，不与 `command.timeout_seconds` 取最小值，也不额外套 Python timeout。外层 Run 或工具期限仍能取消调用。Python 扩展在宿主执行；Docker 模式的命令脚本与依赖需要已存在于该环境，不自动安装。

事件里的 `workspace` 和图片路径始终使用宿主坐标。Native 脚本可以直接读取图片引用；Docker
脚本读取 workspace 内图片时，先求图片路径相对事件 `workspace` 的路径，再拼到容器
`/workspace`。在 Windows 宿主上可使用 `PureWindowsPath` 解析事件路径，在 POSIX 宿主上使用
`PurePosixPath`；不要用容器本地 `Path` 直接解析 Windows 路径。工作区之外的路径仍取决于
已有容器挂载，不由 Hook 额外复制或挂载。

适配器先处理控制事实，再决定是否解析 JSON。服务即使消费了 `CancelledError` 并返回成功，也不能清除当前 task/Run 的取消。`EXITED + 0` 同时带 stop receipt 表示该调用参与了环境停止；没有直接取消来源时按环境中断交回，合法 JSON 也不能使其继续执行后续处理器。普通超时或非零退出若带 receipt，先等待原停止操作排空，再报告普通处理器失败；清理失败或等待期间取消则保留收据交回 owner。unknown/cleanup 使用独立控制字段与唯一命令槽，不写入 JSON 或模型反馈。

## 顺序、错误和取消

同一事件按注册顺序串行执行，不同调用可以同时派发。`tool.before` 的第一次拒绝或普通失败停止余项，分别返回 `HOOK_REJECTED` 或 `HOOK_ERROR`；其他事件记录普通失败并继续。`tool.after` 的有效反馈按顺序累积，不作为下一处理器的输入，也不改写工具正文。

工具执行器在权限刷新、熔断检查及 durable claim 后派发 before，拒绝时跳过 Middleware、body 和 after。拒绝结果保持 `is_error=True`，这两个 Hook 错误码不触发 `ToolErrorPolicy.STOP`。after 只跟随实际 body 的已知结果；Middleware 短路、前检失败、body 取消或未收口的命令不触发 after。body 的普通失败被 Middleware 恢复后，after 仍可看到真实的 `body_status="error"`。

反馈与工具原结果一起完成一次 artifact 处理，再由 Runtime 原子提交结果、消息和 checkpoint。后置处理被取消或清理失败时，先保存可提交的已知结果及已有反馈，再交接控制；未知脚本结果不能把已知工具 body 改记为 unknown。恢复直接复用已提交内容，不补发工具 Hook。父侧 subagent 委派使用专用路径，child 自己的普通工具使用自身派发器。

图片不占工具文字截断额度；截断保留图片顺序，外置正文保存图片引用和完整反馈。
压缩仅在摘要输入副本中将图片转为文字引用，持久化工具结果保留原图片块，仍可通过
`context_read` 回读引用，再由 `file.read` 读取模型版图片。

Python 处理器必须可等待，默认期限为 10 秒。派发器使用协作式 `asyncio.timeout`，处理器吞掉超时取消后返回的迟到结果也会被丢弃。同步回调不会自动搬到线程。命令注册使用独立私有类型，其期限由命令后端拥有，不再套 Python 超时。

执行 owner 可以通过私有 `cancellation` 参数传入 Run 信号。派发器中断当前处理器并等待必要清理；重复取消不反复打断收口。控制返回保留之前取得的反馈，不继续执行后续处理器。`run.finished` 的 owner 应传 `cancellation=None`，避免使用已终态 Run 的旧信号；非 `COMPLETED` 的结束事件只运行 Python 处理器。

[`_dispatch_types.py`](_dispatch_types.py) 定义仅在进程内使用的交接类型。`HookControl.origin/error` 保留原控制来源，`unknown_error` 独立保存收口过程中发现的真实结果未知，`stop_slot` 引用唯一命令事实槽。后续执行 owner 必须同时消费这些事实，再按执行阶段确定结算方式；它们不进入事件 JSON、工具 metadata 或 checkpoint。

## Run 生命周期与完成等待

`run.started` 在普通、Goal 或 child 的新 Run 已获准、deadline/signal/active task 已建立之后执行，早于首个模型请求。未获准或准备失败不会派发；resume、recover 不补发。普通处理器失败记录后继续；真实 unknown、取消或清理失败由 harness 映射到既有结算路径，不构造虚假工具结果或 engine cursor。

`run.finished` 只由实际提交新终态的 producer 触发，包括 recovery FINALIZE。所有停止原因都可运行 Python 处理器，命令仅在 `COMPLETED` 时运行；读取旧终态不会重放。完成事件没有旧 Run 的取消信号或 deadline，处理器仍使用自己的期限。

root 的 [`HookLifecycle`](../harness/_hooks.py) 在同步终态提交前登记实际 owner task，在原 command settlement 移除后发布结果启动处理器。child 和为后台 deadline 重建的 child 借用同一 owner，但使用各自的处理器列表。正常收口的 owner 结束后释放自己的登记与 RunResult 引用；资源失败的有限 owner 在 root 准入锁定期间保留，使外层 SDK 取消仍能取得该 Run 的清理错误。完成通知用于唤醒等待者；没有适用 finished 处理器时不登记，也不改变原 observer 并发行为。

终态事实可以先被观察到，而同 session 的新 Run 仍须等完成通知。直接 SDK 新准入会报告尚未完成；SessionManager 普通 submit 在准入锁外等待，显式 follow-up 保持原 FIFO。取消普通等待者不取消 owner。实际 SDK 驱动者在终态后被取消时，会明确取消 owner、排空当前脚本，再传播调用者取消；`close(cancel_run=True)` 也会取消适用 owner，包括关闭过程中刚产生的新 finished 任务。默认 detach 保持 owner 运行，root close 仍能找到并等待后台任务。

finished 脚本 unknown/cleanup 只处理附加动作：有停止收据则等待原操作排空，无收据则停止对应命令 scope 并等待排空；不再次提交终态、不改 RunResult，也不重跑脚本。资源未收口时报告 `IrisCommandCleanupError`，并锁存 root 的新 Run 准入错误、唤醒所有 session。已有 cancel/resume/recover 和资源关闭仍可使用；错误不会自动解除，宿主关闭资源后重建 root。资源错误优先于同时发生的 SDK 取消，并保留取消异常链。

## 维护与验证

[`models.py`](models.py) 只在类型检查时引用 lifecycle 和 ToolResult；[`__init__.py`](__init__.py) 不加载派发器，避免形成 lifecycle/tools/hooks 的循环导入。`event_to_dict` 从可信字段生成脚本输入投影，不重新验证已解析领域模型。

核心测试位于 `tests/hooks/test_models.py` 和 `tests/hooks/test_dispatcher.py`，覆盖事件序列化、注册约束、精确匹配、输入隔离、部分反馈、处理器期限及取消与 unknown/cleanup 的交接。工具结果反馈的持久化和 artifact 投影由 [`tools`](../tools/README.md) 的结果模型负责。

使用与设计：[扩展用法](../../../docs/cookbook/extensions.md) · [扩展点职责](../../../docs/design/extensions.md)。
