# 管理会话、运行中输入与历史分支

顺序执行多轮对话时，重复使用同一个 `session_id` 调用 `runner.start()` 即可；每次调用创建一个新 Run，并读取该 Session 的已提交历史。需要在 Run 执行期间接收新输入时，再使用 `SessionManager`。

本页先用一个不联网的完整示例展示输入的实际顺序，再说明如何持久化、浏览历史和创建分支。前提是已完成[源码安装](../getting-started/quickstart.md)，在仓库根目录使用 `uv run`。

## 跑通 steer、follow-up 和分支

把以下内容保存为 `session_demo.py`。示例 provider 故意等待一个信号，让当前 Run 保持执行中，确保两个排队输入具有确定的演示顺序；它不是语言模型，不用于验证生成效果。脚本使用真实 Runner、Runtime、SessionManager 和内存 Store，临时工作区会在退出时删除。

```python
"""演示会话输入准入、历史查询和分支，无需 API key。"""

import asyncio
from tempfile import TemporaryDirectory

from iris.agents import AgentConfig, ModelConfig, PermissionsConfig
from iris.harness import (
    AgentRunner,
    AgentRunRequest,
    RunEvent,
    RunEventKind,
    SessionHistory,
    SessionManager,
    SubmissionEvent,
)
from iris.message import LLMRequest, LLMResponse, TextBlock


class DemoProvider:
    """等待放行后返回固定文本，让队列行为可观察。"""

    def __init__(self) -> None:
        self.release = asyncio.Event()

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        """仅为这个短示例提供确定的预算估算。"""
        return 1

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """返回完整的示例响应。"""
        await self.release.wait()
        return LLMResponse(
            provider="demo",
            model="demo",
            content=[TextBlock(text="这一模型步已完成。")],
            finish_reason="stop",
        )


async def main() -> None:
    """执行两轮、查看截点，再创建独立分支。"""
    with TemporaryDirectory() as workspace:
        provider = DemoProvider()
        runner = AgentRunner.from_config(
            AgentConfig(
                name="session-demo",
                system="按输入完成任务。",
                model=ModelConfig(provider="demo", name="demo"),
                permissions=PermissionsConfig(workspace=workspace),
            ),
            provider=provider,
        )
        manager = SessionManager(runner, "main")
        events = manager.events()
        try:
            first = await manager.submit("先分析报告")
            steer = await manager.submit("重点看成本", mode="steer")
            follow_up = await manager.submit("再写一份摘要", mode="follow_up")
            print(first.state, steer.state, follow_up.state)
            provider.release.set()

            async for event in events:
                if isinstance(event, SubmissionEvent):
                    print(event.mode, event.state)
                if (
                    isinstance(event, RunEvent)
                    and event.run_id == follow_up.run_id
                    and event.kind is RunEventKind.RUN_TERMINAL
                ):
                    break

            history = SessionHistory(runner.store)
            page = history.list_fork_points("main", limit=10)
            print("分支点数量:", len(page.items))
            preview = history.get_at_run(first.run_id)
            print("第一轮截点:", [message.text for message in preview.messages])
            branch = history.fork(first.run_id)
            result = await runner.start(
                AgentRunRequest(input="换一个角度分析", session_id=branch.session_id)
            )
            print("分支结果:", result.run.stop_reason)
            print("分支直接来源:", branch.forked_from_run_id)
        finally:
            await manager.close(cancel_run=True)
            await events.aclose()
            await runner.aclose()


asyncio.run(main())
```

在项目根目录运行：

```shell
uv run python session_demo.py
```

应看到初始回执 `delivered pending pending`，随后两个排队输入分别产生 `pending`、`delivered` 事件。主会话有两个分支点：steer 没有创建新 Run，follow-up 创建了第二个 Run。第一轮截点包含“先分析报告”“重点看成本”及两次模型响应，不包含“再写一份摘要”。分支以新的 Session ID 继续，原会话保持原状。

接入真实模型时，用配置创建 `AgentRunner.from_config_path("agent.yaml")`，移除演示 provider 的放行信号；配置和全局凭据初始化见 [Python SDK 入门](../getting-started/python-sdk.md)。用户输入应由实际 UI 驱动，而不是通过固定 sleep 猜测 Run 是否执行中。

## 选择输入模式

| 需求 | 调用 | 含义 |
| --- | --- | --- |
| 明确从空闲状态开始 | `await manager.submit(text)` | 创建一个新 Run |
| 按当前状态自动选择 | `await manager.submit(text, mode="auto")` | 空闲时创建，忙碌时作为 steer |
| 修正当前任务 | `await manager.submit(text, mode="steer")` | 排队进入当前 Run，沿用原预算 |
| 下一轮再做 | `await manager.submit(text, mode="follow_up", options=options)` | 当前 Run 终态后创建新 Run |
| 停止当前任务 | `await manager.interrupt(reason="用户停止")` | 持久化取消请求；保留 follow-up 等待结算 |

显式 `steer`、`follow_up` 要求当前处于忙碌状态；默认 `mode=None` 要求空闲。`auto` 适合普通聊天输入：状态判断在 Manager 的锁内完成，不需要宿主先查一次状态再决定。`auto` 在忙碌分支会忽略只用于新 Run 的 `options`；显式 steer 则不允许传 options。

运行中的输入不立即改写已经发出的模型请求。steer 在当前模型响应或工具批次可以完整提交的边界加入历史。回执表示接纳，最终投递与失败看 `SubmissionEvent`；模型任务是否完成看 `RunResult`。

如果当前 Run 在等待人工回答，steer 可以排队，follow-up 也可以排队，但二者都不会代替回答。使用[人工交互配方](hitl-recovery.md)中的 typed response 继续。

## 让会话跨进程保留

Web 宿主在当前进程内刷新时，用 `manager.snapshot()` 重建排队正文和可用操作。
不要只依据 durable WAITING 点亮继续按钮：旧任务可能仍在投递收尾，应以快照中
`allowed_commands` 是否包含 `resume` 为准。控制变化由 critical live 事件通知，
字段见[运行参考](../reference/runtime.md#只读控制快照)。

以下是 `agent.yaml` 中的配置片段，追加到已可运行的 Agent 配置：

```yaml
session:
  backend: sqlite
  path: .iris/session.db
```

`path` 相对 **agent.yaml 所在目录**解析。默认 `backend: none` 仍有内存 Store，但退出进程后不保留状态。SDK 也可以显式注入 `SQLiteStore(path)`；注入的 Store 优先于 YAML 的 session 配置。

新进程打开相同 SQLite 数据库后，可用 `runner.get_session(session_id)` 读取当前已提交消息。若该会话还有非终态 Run，不能直接创建另一个 Run：先处理其 waiting 或遗留 active 状态。`runner.store.load_session_lane(session_id)` 可取得当前非终态 Run ID，没有时返回 `None`。

SessionManager 自己的排队输入和投递回执不持久化。新建 Manager 不会重建原有队列，也不自动附着数据库中的旧 Run。用户选择继续旧任务时，显式调用 `await manager.restore(run_id, expected_activation_id=observed_activation_id)`；WAITING 仅附着待回答交互，ACTIVE 的 fence 和恢复由 Runner 裁决。之后的 steer、回答及取消继续经过同一 Manager。

## 查历史与创建分支

`SessionHistory(runner.store)` 借用与 Runner 相同的 Store。三个操作分别用于选择位置、预览内容和创建新 Session：

- `list_fork_points(session_id, after=None, limit=50)`：按 `(created_at, run_id)` 升序列出终态顶层 Run；下一页传上次的 `next_cursor`。
- `get_at_run(source_run_id)`：读取该 Run 结束时的完整已提交历史前缀，而不是只读这轮新增消息。
- `fork(source_run_id)`：自动生成新 Session ID 并复制历史，不调用模型。随后向 `branch.session_id` 提交新输入。

failed、cancelled 等终态也可以作为分支点，内容仅包括当时已提交消息；未完成的流式片段不会因此变成历史。active、waiting 和子 Agent Run 不能作为来源。原会话之后是否又开展了任务，不影响既有截点。

分支继承当时的摘要，但不继承旧 Run 的预算、Checkpoint、Interaction 或固定上下文窗口。图片消息继承稳定文件引用，不重新复制图片缓存；需要继续保留这些资源。Goal、Todo 等其他会话能力不会因复制消息而自动成为原任务的执行续体。

## 正确关闭宿主

`await manager.close()` 默认只停止输入准入和观察，当前 Run 仍可由 Runner 推进。如果准备退出整个程序，使用 `await manager.close(cancel_run=True)` 等待取消及其持有的任务收尾，再调用 `await runner.aclose()`。

默认混合事件流只允许一个消费者，且有容量限制。长期运行的应用要持续消费；想给多个浏览器客户端订阅，使用[流式宿主接入](streaming.md)的 broker/gateway，而不是重复调用 `manager.events()`。

进一步了解：[输入准入设计](../design/human-interaction.md)、[生命周期与历史分支](../design/lifecycle.md)、[精确 SDK 参考](../reference/runtime.md)。
