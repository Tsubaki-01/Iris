[English](README.en.md)

# `iris.streaming`

`iris.streaming` 提供 host-embedded 的 live observation plane：进程内 broker 负责有界 replay、
顺序和 fan-out；gateway 绑定一个已授权的 exact session，并把 typed command 路由到
`SessionManager`；SSE 与 WebSocket adapter 只处理 framing 和连接内 task 生命周期。

本包不创建 server、route、认证、授权、TLS、CORS、tenant registry 或 artifact download URL。
Host 必须在每次连接和命令进入 Iris 前完成这些工作。Cursor 只表示 observation 位置，不授予
任何权限。

随 Iris 一同安装，要求 Python 3.12 或更高版本，不需要额外的 transport 依赖。

## Host 组合

Host 应把同一个 broker 注入 runner、manager 和 gateway：

```python
from iris.harness import AgentRunner, SessionManager
from iris.streaming import LiveStreamBroker, StreamingGateway

broker = LiveStreamBroker(
    replay_capacity_per_scope=256,
    subscription_capacity=64,
    max_replay_scopes=256,
)
runner = AgentRunner.from_config_path(
    "agent.yaml",
    live_publisher=broker,
)
manager = SessionManager(
    runner,
    "default",
    submission_publisher=broker,
    observation_mode="broker_only",
)
gateway = StreamingGateway(
    runner=runner,
    manager=manager,
    broker=broker,
    session_id="default",
    durable_page_size=64,
)
```

仅通过 gateway/broker 观察时使用 `observation_mode="broker_only"`：manager 要求提供
`submission_publisher`，不创建 mixed event buffer，也不允许调用 `events()`。完成的 run 不会
因无人消费 mixed stream 而占用下一轮 admission 容量。需要本地 mixed stream 的 host 可使用默认
`observation_mode="mixed"`，并持续消费 `manager.events()`。

`LiveStreamBroker` 必须在同一个 event loop/thread 中使用。`replay_capacity_per_scope` 限制每个
run/session ring；`max_replay_scopes`（默认 256）限制全局 ring 数，publish 或 subscribe 会更新
scope 的 LRU 顺序。淘汰 ring 不关闭活跃订阅，其 published sequence 会保留到订阅关闭且无 ring；
其余旧 scope 的 ring 和 counter 一起回收。因此 counter 数不超过 replay scope 数加活跃 scope 数。
重建已回收的 scope 时序号会跳过旧值，旧 cursor 明确得到 `unknown_cursor` 或 `cursor_expired`
gap；不要假设新 scope 从 1 开始。`close()` 清理 replay/counter 并给活跃订阅发送终态。

`subscription_capacity` 限制单 consumer future-live backlog。Partial 可以
合并或丢弃；critical event 无法入队时，subscription 产生 `ReplayGap` 和
`SubscriptionTerminal`，客户端应重连并执行 durable sync。slow-consumer 转换只从 active
subscription 的 offer 路径进入，并只产生一组 gap/terminal。

## Gateway 与命令

自动压缩沿现有 live plane 发布 `context.compaction.started`、
`context.compaction.completed` 和 `context.compaction.failed`，携带原有 run/session/activation
identity 与 `step_index`。三者是 critical 状态，摘要正文和摘要模型增量不会进入 live payload。
完成状态晚于 durable `context.compacted` 提交、早于主 `model.step.started`；失败状态的具体
原因由最终 run 结果说明。CLI 只显示 live 短状态，忽略 durable event，避免重复完成通知。

`StreamingGateway` 绑定构造时的 exact runner、manager、broker 和 session。Session scope 必须
等于 bound session；run scope 和 durable cursor 中的 run 必须由 runner 证明属于该 session。
Gateway 从不调用 `SessionManager.events()`，也不直接使用 store mutation 或 recovery。

```python
from iris.streaming import (
    DurableRunCursor,
    SnapshotCommand,
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
snapshot_receipt = await gateway.handle(
    SnapshotCommand(request_id="snapshot-1", run_ids=("run-known",))
)
```

`subscribe()` 不创建 network task。若请求携带 caller-known durable cursors，
`GatewaySubscription` 先产生一个 `DurableSyncItem(kind="sync.page")`，再委托 broker live stream。
`durable_sync(cursors)` 按输入顺序返回 `DurableSync.runs` 中的 `DurableRunPage`；每页只有
`run_id`、有限 `events` 和 `next_cursor`。它通过窄 run control 检查 session 归属，不读取完整
run、result、tool calls 或 session history。`next_cursor` 表示还有下一页，为 `None` 时已读完
本次查询可见的事件；它不表示 run 已结束。

`durable_snapshot(run_ids)` 显式返回 `DurableSnapshot`，其 `runs` 按输入顺序包含
`DurableRunSnapshot(run, result, tool_calls)`。Active run 的 `result` 可以为 `None`。
事件分页与状态快照各自读取当前 durable 事实，不承诺两次调用之间的原子快照。两种读取均只接受
caller 已知且属于 bound session 的 run，不发现其他 run，也不自动 recover。

`handle()` 支持：

- `SubmitCommand` → `SessionManager.submit()`；
- `ResumeCommand` → `SessionManager.admit_resume()`，返回 `ResumeAccepted.receipt`
  中的 `run_id` 和 `interaction_id`，不会等待恢复后的模型或工具执行结束；
- `CancelCommand` → `SessionManager.interrupt()`；
- `SyncCommand` → 只读有限事件页，返回 `SyncAccepted.sync`；
- `SnapshotCommand` → 按需完整状态，返回 `SnapshotAccepted.snapshot`。

`request_id` 只用于关联 receipt，不提供幂等或去重。预期 `IrisError` 会变成稳定的
`CommandRejected`；unexpected error 只返回通用拒绝并记录不含 raw frame/payload 的 warning。
Resume 的最终结果通过 live terminal/interaction 事件后显式请求 snapshot 获取。直接调用 SDK 的
`SessionManager.resume()` 仍等待完整 `RunResult`。

## Disclosure policy

默认 `allow_thinking=False`、`allow_tool_arguments=False`：gateway subscription 不发送 thinking
block/delta 或 tool-arguments partial，并从 live tool facts 删除 tool name；durable tool call 与
waiting interaction 中的 arguments 投影为空字典。只有 host 已完成 tenant/session 授权且确实需要
这些字段时，才应显式启用对应选项。

Durable snapshot 始终删除 assistant/tool-result metadata、run/tool error details、artifact 本地路径、
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
JSON typed command；解析失败返回 `INVALID_COMMAND`。首个有效命令可以是 `subscribe`、`sync`
或 `snapshot`；订阅前可以继续只读分页或取快照，执行 mutation command 前必须 subscribe。

连接内只有 sender task 调用 `send()`。Receive task 在路由命令前预留一个 receipt 容量，随后
把确认写入 capacity-1 queue；容量直到 sender 完成发送才释放。慢 sender 会暂停后续命令的
admission，因此流水发送多个命令不会先改变 manager 状态再因 queue 已满丢弃确认。
Resume 只等待 admission，后续 sync、snapshot、steer、cancel 或 disconnect 可在运行完成前继续处理。
第二个 subscribe 会被拒绝。`receive() -> None`、receive/send exception 或 task cancellation 都会
drain/cancel child tasks 并关闭 subscription。

断线始终只是 observation loss：SSE/WS cleanup 不调用 manager interrupt、runner cancel 或
recover。只有客户端显式发送 `CancelCommand`，gateway 才会请求 durable cancellation。

## Restart 与恢复

Broker epoch、live cursor、subscriptions 和 partial 都是 process-local 状态，不持久化。进程重启后
旧 cursor 会产生 `ReplayGap(reason="epoch_changed")`；客户端必须丢弃不完整 partial，并用自己
已知的 per-run `DurableRunCursor` 请求 sync；需要恢复完整状态时，另行请求 snapshot。
Durable authority 始终是 runner/store，live replay
不能替代 durable event/result/tool snapshot。

## 开发与验证

`models.py` 定义公开 wire 模型，`projection.py` 把 runner/runtime facts 投影为 live payload；
`broker.py` 管理顺序、回放与订阅，`gateway.py` 组合命令、事件分页和状态快照，`sse.py` /
`websocket.py` 负责传输 framing。包级公开入口由 `__init__.py` 导出。

修改回放/容量时补 `tests/streaming/test_broker.py`，修改命令契约时同步
`test_gateway.py` 和 `test_models.py`，修改连接生命周期时补 `test_transports.py` 与
`test_system.py`。系统测试使用真实 runner/manager/provider adapter 与脚本化模型流，无外部调用；
broker-only 场景以 `max_tracked_durable_runs=1` 验证连续运行接纳。

```powershell
$env:UV_CACHE_DIR = "$PWD/tmp/uv-cache"
uv sync --dev
uv run pytest tests/streaming -p no:cacheprovider --basetemp="$PWD/tmp/pytest-streaming"
uv run ruff check src/iris/streaming tests/streaming
uv run mypy src/iris/streaming
```
