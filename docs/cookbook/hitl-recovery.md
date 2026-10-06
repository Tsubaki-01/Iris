# 等待人工回答并恢复运行

当 Agent 需要确认工具操作或询问用户时，`start()` 可以返回 `waiting` 结果。保存的 Interaction 和 Checkpoint 允许宿主稍后、甚至在新进程中提交回答。这里的继续操作是 `resume()`；意外退出后接手遗留 active Run 则使用 `recover()`。

前提是已完成[源码安装](../getting-started/quickstart.md)。本页先给一个不联网、实际使用 SQLite 的例子，再给仓库现有脚本的真实模型使用入口。

## 完整示例：保存问题，重建 Runner，再回答

保存为 `hitl_demo.py`。演示 provider 第一次返回 `ask_question` 工具调用，第二个 Runner 使用另一个 provider 返回固定结束文字；全部运行、交互和存储行为由 Iris 实现。数据库保留在脚本打印的临时目录中，方便检查；退出 Python 进程后可删除这个目录。

```python
"""用 SQLite 演示 HITL 等待和新的 Runner 继续。"""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from iris.agents import AgentConfig, ModelConfig, PermissionsConfig, ToolsConfig
from iris.harness import AgentRunner, AgentRunRequest
from iris.hitl import QuestionInteractionResponse
from iris.message import LLMRequest, LLMResponse, TextBlock, ToolUseBlock
from iris.store import SQLiteStore


class DemoProvider:
    """返回预先给定的完整模型响应。"""

    def __init__(self, response: LLMResponse) -> None:
        self.response = response

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        """给短演示提供固定预算估算。"""
        return 1

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """无需网络即可驱动真实 Runtime。"""
        return self.response


async def main() -> None:
    """先持久化等待事项，再重新装配并提交答案。"""
    with TemporaryDirectory(delete=False) as workspace:
        print("演示数据目录:", workspace)
        config = AgentConfig(
            name="hitl-demo",
            system="询问报告读者，获得回答后完成任务。",
            model=ModelConfig(provider="demo", name="demo"),
            permissions=PermissionsConfig(workspace=workspace),
            tools=ToolsConfig(builtin=["human.ask"]),
        )
        database = Path(workspace) / "lifecycle.db"
        first = AgentRunner.from_config(
            config,
            store=SQLiteStore(database),
            provider=DemoProvider(LLMResponse(
                provider="demo",
                model="demo",
                content=[ToolUseBlock(
                    id="question-1",
                    name="ask_question",
                    input={"question": "报告写给谁？", "options": ["研发", "业务"]},
                )],
                finish_reason="tool_calls",
            )),
        )
        try:
            waiting = await first.start(
                AgentRunRequest(input="准备报告", session_id="report", run_id="report-run")
            )
            print("首次返回:", waiting.run.phase)
        finally:
            await first.aclose()

        reopened = AgentRunner.from_config(
            config,
            store=SQLiteStore(database),
            provider=DemoProvider(LLMResponse(
                provider="demo",
                model="demo",
                content=[TextBlock(text="将面向业务读者编写报告。")],
                finish_reason="stop",
            )),
        )
        try:
            saved = reopened.get_result("report-run")
            assert saved is not None and saved.pending_interaction is not None
            interaction = saved.pending_interaction
            print("已保存问题:", interaction.request.prompt)
            result = await reopened.resume(
                saved.run.run_id,
                interaction_id=interaction.interaction_id,
                response=QuestionInteractionResponse(answer="业务"),
            )
            print("恢复结果:", result.run.stop_reason)
            print("累计模型步:", result.run.usage.model_steps_committed)
            print("已提交消息条数:", len(reopened.get_session("report").messages))
        finally:
            await reopened.aclose()


asyncio.run(main())
```

在仓库根目录运行：

```shell
uv run python hitl_demo.py
```

成功标志是第一次返回 `waiting`，重新打开数据库能读到原问题，回答后同一 `report-run` 变为 `completed`，累计两个已提交模型步。这个例子重建了 Store 与 Runner；它演示的不是操作系统进程崩溃，也没有调用真实模型。

## 把等待事项交给实际 UI

检查 `result.pending_interaction.request.prompt.kind`：

| kind | 展示内容 | 提交的响应 |
| --- | --- | --- |
| `question` | `question` 和可选 `options` | `QuestionInteractionResponse(answer=text)` |
| `permission` | `reason` 与 `request.tool_call` 的工具名、参数 | `PermissionInteractionResponse(decision="approve")` 或 `"reject"` |

响应类型从 `iris.hitl` 导入。宿主必须使用当前返回的 `interaction_id`，不要把普通“同意”聊天文本直接当作批准。多次等待时，重复读取新的 `RunResult`，逐个处理当前 Interaction。

如果通过 SessionManager 启动了当前 Run，使用 `manager.resume(interaction_id=..., response=...)` 等待结果，或 `manager.admit_resume(...)` 只等待响应被接纳，让 UI 继续通过事件观察。后者返回 `ResumeReceipt`，不是最终运行结果。

拒绝工具调用会产生 `USER_REJECTED` 工具错误；默认把错误交回模型，模型可以解释或改换方案。需要立即结束错误工具所在 Run 时，创建 Run 时设置 `RuntimeExecutionOptions(tool_error_policy="stop")`。

## 用仓库脚本接入真实模型

仓库提供 [examples/lifecycle](../../examples/lifecycle/) 中的 start、status、events、resume、cancel、recover 脚本。这些是 `uv run python -m ...` 示例模块，**不是 `iris` CLI 的子命令**。

先按[配置参考](../reference/configuration.md)配置示例所需 provider 凭据。默认示例为 `deepseek/deepseek-flash`，协议沿用当前模型配置默认值；实际服务使用的协议、模型和地址可通过自己的 YAML 显式配置，并给各脚本统一传 `--config path/to/agent.yaml`。

在仓库根目录运行一次新任务，`--run-id` 每次使用新值：

```shell
uv run python -m examples.lifecycle.start --env-file .env --session-id interview --run-id interview-1 --input "请先调用 ask_question 询问报告面向研发还是业务，获得回答后再总结。"
uv run python -m examples.lifecycle.status --run-id interview-1
uv run python -m examples.lifecycle.events --run-id interview-1
```

模型是否按请求提问要看实际结果；当输出为 waiting 时，将下面的 `INTERACTION_ID` 替换为返回的 `pending_interaction.interaction_id`，再继续：

```shell
uv run python -m examples.lifecycle.resume --env-file .env --run-id interview-1 --interaction-id INTERACTION_ID --answer "面向研发"
```

权限问题改用 `--decision approve` 或 `--decision reject`，不能同时传 `--answer`。status 和 events 使用只读 provider，不调用模型。默认示例数据库位于 `examples/lifecycle/.iris/lifecycle.db`，因为路径相对示例 YAML 所在目录解析；后续命令必须读取同一个数据库。

## 区分 resume、recover 和重新开始

| 当前情况 | 操作 |
| --- | --- |
| Run 正在等人工答案 | `resume()`，携带准确 Interaction ID 和 typed response |
| 原执行者已退出，数据库仍是 active | 读取 `current_activation_id` 后显式 `recover()` |
| Run 已 terminal，需要新一轮 | 同会话 `start()`；或从历史截点 fork 后 start |
| 用户要求停止 | `cancel()`；使用 Manager 时调用 `interrupt()` 或关闭流程 |

接手遗留 active Run 的宿主代码核心是先读取 `snapshot = runner.get_run(run_id)`，再调用 `await runner.recover(run_id, expected_activation_id=snapshot.current_activation_id)`。原执行者仍在正常运行时不要把 recover 当作轮询接口；当前 Runner 仍持有 live Activation 时会拒绝接管。

仓库命令形式是：

```shell
uv run python -m examples.lifecycle.recover --env-file .env --run-id $runId --activation-id $activationId
```

这里 `$runId`、`$activationId` 必须来自实际遗留 Run 的状态输出。恢复可能继续执行，也可能返回 `outcome_unknown`：当工具已开始但结果未提交，框架不会重放它。检查 `result.error` 和 `runner.list_tool_calls(run_id)`，结合实际工具效果决定下一步。模型输出已完整提交、只差结算时，恢复可以直接完成，不再调用模型。

## 取消与正常失败怎么处理

`await runner.cancel(run_id, reason="用户停止", settlement_timeout=10)` 请求取消并等待终态；超时抛出 `IrisRunObservationTimeoutError`，只说明这次等待到期。取消请求仍然保留，宿主可重新读取结果，不应据此宣称取消失败或任务已完成。

进程退出前，Manager 宿主使用 `await manager.close(cancel_run=True)`，之后 `await runner.aclose()`。直接使用 Runner 的宿主还要等待原 `start/resume/recover` 任务退出；`cancel()` 得到持久终态不等于原任务的观察者回调都已结束。

对正常执行失败，先看 `result.run.stop_reason` 和 `result.error`。部分流式文本不是成功结果；参数/状态冲突或持久化异常也可能直接抛出。若启用了显式 command 执行，资源清理失败有独立的 `IrisCommandCleanupError`，不能把它当作已经完成取消；详见[命令配方](commands.md)。

下一步：[生命周期设计](../design/lifecycle.md)、[HITL 与输入准入](../design/human-interaction.md)、[SDK 参数与错误](../reference/runtime.md)。
