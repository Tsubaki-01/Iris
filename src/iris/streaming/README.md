# `iris.streaming`

`iris.streaming` 提供 host-embedded 的 live observation plane：进程内 broker 负责有界 replay、
顺序和 fan-out；gateway 绑定一个已授权的 exact session，并把 typed command 路由到
`SessionManager`；SSE 与 WebSocket adapter 只处理 framing 和连接内 task 生命周期。

本包不创建 server、route、认证、授权、TLS、CORS、tenant registry 或 artifact download URL。
Host 必须在每次连接和命令进入 Iris 前完成这些工作。Cursor 只表示 observation 位置，不授予
任何权限。

## Host 组合

Host 应把同一个 broker 注入 runner、manager 和 gateway：

```python
from iris.harness import AgentRunner, SessionManager
from iris.streaming import LiveStreamBroker, StreamingGateway

broker = LiveStreamBroker(
    replay_capacity_per_scope=256,
    subscription_capacity=64,
)
runner = AgentRunner.from_config_path(
    "agent.yaml",
    live_publisher=broker,
)
manager = SessionManager(
    runner,
    "default",
    submission_publisher=broker,
)
gateway = StreamingGateway(
    runner=runner,
    manager=manager,
    broker=broker,
    session_id="default",
    durable_page_size=64,
)
```

`LiveStreamBroker` 必须在同一个 event loop/thread 中使用。`replay_capacity_per_scope` 限制每个
run/session ring，`subscription_capacity` 限制单 consumer future-live backlog。Partial 可以
合并或丢弃；critical event 无法入队时，subscription 产生 `ReplayGap` 和
`SubscriptionTerminal`，客户端应重连并执行 durable sync。slow-consumer 转换只从 active
subscription 的 offer 路径进入，并只产生一组 gap/terminal。

## Gateway 与命令

`StreamingGateway` 绑定构造时的 exact runner、manager、broker 和 session。Session scope 必须
等于 bound session；run scope 和 durable cursor 中的 run 必须由 runner 证明属于该 session。
Gateway 从不调用 `SessionManager.events()`，也不直接使用 store mutation 或 recovery。

```python
from iris.streaming import (
    DurableRunCursor,
    SubmitCommand,
    SubscribeCommand,
)

subscription = gateway.subscribe(
    SubscribeCommand(
        request_id="subscribe-1",
        scope="session",
        scope_id="default",
        durable_cursors=(
            DurableRunCursor(run_id="run-known", after_sequence=0),
        ),
    )
)
receipt = await gateway.handle(
    SubmitCommand(request_id="submit-1", input="继续分析", mode="steer")
)
```

`subscribe()` 不创建 network task。若请求携带 caller-known durable cursors，
`GatewaySubscription` 先产生一个 `DurableSyncItem`，再委托 broker live stream。
`durable_sync()` 只读取显式给出的 run，并按输入顺序返回有限 event page；它不会发现 session 下
的其他 run，也不会自动 recover。`next_cursor` 只表示该 run 的 durable event high-water。

`handle()` 支持：

- `SubmitCommand` → `SessionManager.submit()`；
- `ResumeCommand` → `SessionManager.resume()`；
- `CancelCommand` → `SessionManager.interrupt()`；
- `SyncCommand` → read-only durable sync。

`request_id` 只用于关联 receipt，不提供幂等或去重。预期 `IrisError` 会变成稳定的
`CommandRejected`；unexpected error 只返回通用拒绝并记录不含 raw frame/payload 的 warning。

## Disclosure policy

默认 `allow_thinking=False`、`allow_tool_arguments=False`：gateway subscription 不发送 thinking
block/delta 或 tool-arguments partial，并从 live tool facts 删除 tool name；durable tool call 与
waiting interaction 中的 arguments 投影为空字典。只有 host 已完成 tenant/session 授权且确实需要
这些字段时，才应显式启用对应选项。

Durable sync 始终删除 assistant/tool-result metadata、run/tool error details、artifact 本地路径、
tool result data/stats/metadata 与 pending interaction 的 workspace path；这些字段不受参数 opt-in
放行。当前 durable wire shape 无法在不携带 path 的情况下表达 artifact，因此 gateway 返回
`artifact=None`，由 host 另行实现授权下载接口。

Phase 03 projection 已在更早边界删除 raw provider chunk/header/key/traceback，以及 artifact path、
bytes 和任意 metadata。Gateway filtering 只消费该 trusted allowlist，不重新解析 payload。

## SSE adapter

`SSEAdapter` 返回 `AsyncIterator[bytes]`，不创建 HTTP route：

```python
from iris.streaming import SSEAdapter

adapter = SSEAdapter(heartbeat_interval_s=15)
async for frame in adapter.stream(subscription):
    await host_send_bytes(frame)
```

`LiveEnvelope` frame 带 `id`，其值是 compact JSON `LiveCursor`；`ReplayGap`、
`SubscriptionTerminal` 和 `DurableSyncItem` 没有伪造的 id。Heartbeat 固定为
`: heartbeat\n\n`，不进入 broker/ring，也不占 live sequence。Heartbeat timeout 复用同一个 pending
`anext()`，不会取消或替换它。Iterator 结束或调用方断开时只关闭 subscription。

Host 可用 `encode_live_cursor()` / `decode_live_cursor()` 处理 SSE cursor header；decode 是 raw
boundary validation，失败应由 host 映射为无敏感细节的请求错误。

## WebSocket adapter

`WebSocketAdapter.serve(receive, send)` 接收 framework 提供的 callback。Raw frame 必须是 UTF-8
JSON typed command；解析失败返回 `INVALID_COMMAND`。首个有效命令只能是 `subscribe` 或
`sync`，sync-first 后仍需 subscribe 才能执行 mutation command。

连接内只有 sender task 调用 `send()`。Receive task 只顺序处理命令并把 receipt 写入
capacity-1 queue；queue 饱和时连接按 backpressure error path 清理，不创建第二个 writer。
第二个 subscribe 会被拒绝。`receive() -> None`、receive/send exception 或 task cancellation 都会
drain/cancel child tasks 并关闭 subscription。

断线始终只是 observation loss：SSE/WS cleanup 不调用 manager interrupt、runner cancel 或
recover。只有客户端显式发送 `CancelCommand`，gateway 才会请求 durable cancellation。

## Restart 与恢复

Broker epoch、live cursor、subscriptions 和 partial 都是 process-local 状态，不持久化。进程重启后
旧 cursor 会产生 `ReplayGap(reason="epoch_changed")`；客户端必须丢弃不完整 partial，并用自己
已知的 per-run `DurableRunCursor` 请求 sync。Durable authority 始终是 runner/store，live replay
不能替代 durable event/result/tool snapshot。

## 验证

```bash
uv run pytest tests/streaming
uv run ruff check src/iris/streaming tests/streaming
uv run mypy src/iris/streaming
```
