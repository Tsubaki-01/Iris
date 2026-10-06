# 把运行过程接入流式宿主

如果只需要终端交互，`iris chat` 已经展示文字增量。自己的 Web 或桌面应用可以把 Runner 的 live facts 接到 `LiveStreamBroker`，再通过 gateway 消费事件或适配为 SSE/WebSocket。

Iris 提供嵌入式接口，不会启动 HTTP 服务器。你的宿主负责路由、连接和界面，Runner 继续负责运行状态。

## 先跑通进程内订阅

沿用[快速开始](../getting-started/quickstart.md)的 `agent.yaml` 和凭据，在仓库根目录保存 `stream_agent.py`：

```python
"""通过同一个 broker 接收模型增量和已提交的运行事件。"""

import asyncio

from iris import init_config
from iris.harness import AgentRunner, SessionManager
from iris.lifecycle import AgentRunOptions, RuntimeExecutionOptions
from iris.streaming import (
    LiveEnvelope,
    LiveStreamBroker,
    StreamingGateway,
    SubmitAccepted,
    SubmitCommand,
    SubscribeCommand,
)


async def main() -> None:
    """订阅先于任务提交，结束后按所有权关闭组件。"""
    init_config()
    broker = LiveStreamBroker(
        replay_capacity_per_scope=256,
        subscription_capacity=256,
    )
    runner = AgentRunner.from_config_path("agent.yaml", live_publisher=broker)
    manager = SessionManager(
        runner,
        "stream-demo",
        submission_publisher=broker,
        observation_mode="broker_only",
    )
    gateway = StreamingGateway(
        runner=runner,
        manager=manager,
        broker=broker,
        session_id="stream-demo",
        durable_page_size=64,
    )
    subscription = gateway.subscribe(
        SubscribeCommand(request_id="watch", scope="session", scope_id="stream-demo")
    )
    try:
        receipt = await gateway.handle(
            SubmitCommand(
                request_id="first-input",
                input="用三句话介绍异步迭代器。",
                options=AgentRunOptions(
                    runtime=RuntimeExecutionOptions(include_tools=False)
                ),
            )
        )
        if not isinstance(receipt, SubmitAccepted):
            print(receipt.model_dump_json())
            return
        run_id = receipt.receipt.run_id
        async for item in subscription:
            print(item.model_dump_json())
            if isinstance(item, LiveEnvelope) and item.run_id == run_id:
                if item.kind == "run.terminal":
                    break
        result = runner.get_result(run_id)
        if result is not None and result.assistant_message is not None:
            print("最终回答：", result.assistant_message.text)
    finally:
        await subscription.aclose()
        await manager.close(cancel_run=True)
        await runner.aclose()
        broker.close()


if __name__ == "__main__":
    asyncio.run(main())
```

```powershell
uv run python stream_agent.py
```

成功时会看到多条 JSON 事件和 `run.terminal`，随后从 Runner 读取最终回答。若所用 provider 支持流式，过程中还有 `model.block.delta`；没有文字增量不等于任务没有完成。

实际界面应按 `run_id`、payload 中的 `model_stream_id`、`block_id` 和 `channel` 定位展示区域，使用 payload 的 `snapshot` 替换该区域全文。Broker 可以把同一通道尚未交付的多次 delta 合并成最新快照；因此仅把每条 `delta` 追加起来会漏字。普通 partial 的合并可能导致 live sequence 跳号，并不一定伴随 `ReplayGap`。

本例关闭工具，专注展示订阅流程。启用工具后，还需要处理人工等待和恢复命令，见[HITL 指南](hitl-recovery.md)。

两个 publisher 接口接到同一个 broker：Runner 发布运行/模型事实，SessionManager 发布提交状态。选择 `broker_only` 后不再消费 `manager.events()`；若你需要本地 mixed 事件流，应采用[会话配方](sessions.md)的模式。

## 把订阅交给网络层

SSE 使用 `SSEAdapter(heartbeat_interval_s=15).stream(subscription)`，得到 `AsyncIterator[bytes]`。宿主把它作为 `text/event-stream` 响应体；每条 live envelope 的 `id` 编码 live cursor，空闲时产生不占序号的 heartbeat。发送输入仍需单独路由到 gateway 的 typed command。

WebSocket 使用 `WebSocketAdapter(gateway=gateway).serve(receive, send)`。`receive` 异步返回下一帧字符串、bytes 或断线标记 `None`；`send` 异步发送字符串。适配器负责单个连接内命令回执和事件的发送顺序。

WebSocket 首个有效命令必须是 `subscribe`、`sync` 或 `snapshot`；若先同步，完成后仍需订阅才能提交输入。下面是一条订阅帧，session 必须与宿主已绑定的 gateway 一致：

```json
{"kind":"subscribe","request_id":"watch","scope":"session","scope_id":"stream-demo"}
```

连接断开只结束这条观察通道，不自动取消正在执行的 Run。关闭整个应用时，宿主再按其任务策略调用 manager/runner 的关闭接口。

## 断线后怎样恢复界面

live ring 有容量上限，也只在当前 broker 进程中存在。收到 `replay.gap` 时，不应继续把后续文本当成无缝补齐的完整历史。

1. 保存自己已知的 Run ID 和每个 Run 的 durable sequence。
2. 使用 `SyncCommand` 或 `gateway.durable_sync(...)` 分页补读已提交事件。
3. 需要最终正文、状态和工具记录时，使用 `SnapshotCommand` 或 `durable_snapshot(...)`。
4. 重新建立 live 订阅，继续显示实时变化。

live cursor 与 durable cursor 分别维护。模型增量未持久化为完整模型结果前，不能保证在进程重启后逐字重放。详细数据模型和分页规则见[流式参考](../reference/streaming-observability.md)。

理解这条链路的职责：[流式输出、事件与观测设计](../design/streaming-observability.md)。
