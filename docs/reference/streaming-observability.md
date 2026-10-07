# 流式宿主与观测参考

流式接线见[Cookbook](../cookbook/streaming.md)，采集步骤见[观测指南](../cookbook/observability.md)。模型 provider 自身的流式事件定义在[消息参考](media.md#模型流式事件)；这里描述宿主层聚合后的接口。

## Broker 与订阅

scope 包括 `run`（精确 Run）、`session`（精确会话）、`session_tree`（根会话及 child）和
`resource`（独立维护资源）。root 事实同时投影到自身 session_tree；child publisher 在执行前
固定持久 lineage，首先发布 critical `subagent.linked`。后续 envelope 保留 child 原来的
run_id/session_id/activation_id，并附 `RunLineage`，不会改写成父身份或逐 token 查询 Store。

`StreamingGateway` 可订阅 bound session 的 session_tree；已知 child 的 sync/snapshot
通过持久父关系核对根会话。resource 订阅由宿主使用对应资源的 broker，不能伪装成会话。
重连可用 `runner.list_child_runs()` / `get_run_lineage()` 补读关系，人工回答仍经父 proxy。
开启 live publisher 时 child provider 也须满足与 root 相同的 streaming provider 契约。

从 `iris.streaming` 导入：

```text
LiveStreamBroker(*, replay_capacity_per_scope: int,
    subscription_capacity: int, max_replay_scopes: int = 256)
broker.publish(fact: LiveFact) -> None
broker.subscribe(request: LiveSubscriptionRequest) -> LiveSubscription
broker.current_epoch() -> str
broker.close() -> None
```

容量均为正整数，前两项必须显式提供。broker 属于单一 event loop/thread，不是跨线程的消息队列。订阅是异步迭代器，用 `await subscription.aclose()` 结束观察；`broker.close()` 结束其全部订阅。

`LiveSubscriptionRequest(scope, scope_id, cursor=None)` 的 scope 为 `run` 或 `session`。`LiveCursor` 包含 `stream_epoch`、`scope`、`scope_id`、`after_live_sequence`，必须与订阅范围一致。没有 cursor 时从当前订阅点开始，不自动回放所有历史。

## Gateway

```text
StreamingGateway(*, runner, manager, broker, session_id,
    durable_page_size: int, allow_thinking=False, allow_tool_arguments=False)
gateway.subscribe(request: SubscribeCommand) -> GatewaySubscription
await gateway.handle(command: GatewayCommand) -> CommandReceipt
gateway.durable_sync(cursors: Sequence[DurableRunCursor]) -> DurableSync
gateway.durable_snapshot(run_ids: Sequence[str]) -> DurableSnapshot
```

manager 必须绑定同一个 session；使用者应传入同一 Runner 的 manager 和 publisher 链路。Gateway 不创建 Runner、不发现宿主的全部会话。需要同步的 Run ID 由调用方保存，已知 Run 必须属于这个 session。

默认不向 gateway 消费者展示 thinking 和工具参数通道。更改这两项开关只改变展示投影，不改变模型执行。

### 命令与回执

所有命令包含非空 `request_id` 和固定 `kind`：

| kind / 类 | 输入字段 | 成功回执 |
| --- | --- | --- |
| `subscribe` / `SubscribeCommand` | scope、scope_id、cursor=None、durable_cursors=() | WebSocket 返回 `SubscribeAccepted`；Python 调用 `subscribe()` 返回迭代器 |
| `submit` / `SubmitCommand` | input 非空文本、mode=None、options=None | `SubmitAccepted(receipt=SubmitReceipt)` |
| `resume` / `ResumeCommand` | interaction_id、typed response | `ResumeAccepted(receipt=ResumeReceipt)` |
| `cancel` / `CancelCommand` | reason=None | `CancelAccepted(run=RunSnapshot \| None)` |
| `sync` / `SyncCommand` | cursors=() | `SyncAccepted(sync=DurableSync)` |
| `snapshot` / `SnapshotCommand` | run_ids=() | `SnapshotAccepted(snapshot=DurableSnapshot)` |

Gateway `submit` 当前接收文本；图片可通过 Runner/SessionManager 的 typed SDK 提交，不能向网关的字符串字段塞入图片块。失败回执为 `CommandRejected`，包含 code、message 和可选的 request_id/command_kind。命令回执的判别字段是 `event`，流事件的判别字段是 `kind`。

submit/resume 接受不表示任务完成。cancel 在只暂停 Goal 续跑意图而没有当前 Run 时可返回 `run=None`。

### 事件与缺口

`ContextPreparation` 是原始 publisher 接收的完整上下文准备快照；其网络事件
`context.preparation` 仅包含索引与计量摘要，可按 step 合并。Host recorder 应在投影前保存
原始事实；字段及计量含义见[Context 参考](context.md#观察真实准备过程)。

模型 span 包含 `iris.context.preparation_id`、`iris.configuration.snapshot_id`；typed
stream 还记录实际 `iris.model_stream.id`。一次步骤中的多个摘要调用共享 preparation_id，
仍各有自己的 span 身份。complete-only 调用没有虚构 stream ID，完整输入只在原采集策略
允许时记录，span 结束前不承诺完整请求已经导出。

`session.control.changed` 是 session scope 的 critical 事件，payload.snapshot 为 Manager
刚发布的完整 `SessionControlSnapshot`。容量不足沿既有 slow-consumer gap/terminal 结束
该订阅；重新订阅并读取 `manager.snapshot()` 可恢复 pending/ready。它不进入 durable
sequence，也不随模型 token 重复发布。

`LiveEnvelope` 的字段：stream_epoch、scope、scope_id、live_sequence、kind、可选 run_id/session_id/activation_id、可选 durable_sequence 和 JSON payload。

常见 kind 包括：`model.response.started`、`model.block.delta`、`model.response.completed/failed/cancelled`、`run.started`、`run.terminal`、`interaction.suspended/resolved`、`submission.pending/delivered/failed`、`goal.changed`。最终正文应从持久结果获取，不从一个完成事件的元数据猜出。

`model.block.delta` 的 payload 带有 `model_stream_id`、`block_id`、`channel`、`delta` 和 `snapshot`。Broker 可能合并同一通道的待交付 partial，容量压力下也可能移除普通 partial；消费者必须按流/块/通道用 snapshot 更新显示。live sequence 连续性不是完整文本的保证，普通 partial 的合并或移除不一定触发 ReplayGap。provider 原始 typed stream 与经过 broker 的聚合流应分别理解。

另有三类控制/同步数据：

| 类型 | 语义 |
| --- | --- |
| `ReplayGap`，kind `replay.gap` | reason 为 epoch_changed、cursor_expired、unknown_cursor 或 slow_consumer；需要重新同步 |
| `SubscriptionTerminal`，kind `subscription.terminal` | reason 为 slow_consumer 或 broker_closed；结束此订阅 |
| `DurableSyncItem`，kind `sync.page` | 带有 durable cursors 的 gateway 订阅可先交付持久事件页，不占 live 序号 |

`DurableRunCursor(run_id, after_sequence)` 的 sequence 与 live 无关。`DurableRunPage` 包含 events 和可选 next_cursor；非空 next_cursor 表示还有下一页，不等于未来新事件的永久订阅。客户端也应记住所读取的最后 durable sequence。

`DurableRunSnapshot` 包含 run、可选 result 和 tool_calls。快照按需请求，普通每条增量不会自动附带整段历史或全部状态。

## SSE 与 WebSocket

`SSEAdapter(*, heartbeat_interval_s)` 要求正有限间隔，`stream(subscription)` 返回 bytes 异步迭代器。live 事件的 SSE id 由 `encode_live_cursor(cursor)` 生成；收到 Last-Event-ID 后可用 `decode_live_cursor(value)` 解析单行 JSON。heartbeat 为 SSE 注释，不增加 live sequence。

`WebSocketAdapter(*, gateway).serve(receive, send)` 接受两个异步回调；receive 返回 `str | bytes | None`，send 接收 str。首个有效命令为 subscribe/sync/snapshot；先执行 sync/snapshot 后，仍需 subscribe 才能提交或继续任务。重复订阅会返回命令拒绝。

适配器不创建 HTTP 路由。断线关闭 subscription，不自动取消 Run。SSE 提前停止迭代时，应由宿主关闭其异步流，使适配器的收尾逻辑执行。

## 观测配置

Agent YAML 的 `observability`：

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| enabled | false | 构建时固定的采集开关 |
| capture_content | false | 是否记录支持的输入输出正文 |
| max_content_chars | 65536 | 正整数，正文长度上限 |

进程 `Config.observability` 的 `ObservabilityExportConfig`：

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| traces_endpoint | None | OTLP HTTP/protobuf traces 的完整 endpoint |
| headers | {} | 导出请求头 |
| service_name | iris | OTel resource 的 service.name |
| timeout_seconds | 5.0 | 正有限导出请求时限 |

对应环境变量例如 `IRIS_OBSERVABILITY__TRACES_ENDPOINT`、`IRIS_OBSERVABILITY__SERVICE_NAME`。启用时必须提供 endpoint 或借用的 tracer provider；自行导出需要 observability extra。

## 观测 SDK 与所有权

配置模型从 `iris.observability` 导入；服务从 `iris.observability.service` 导入：

```text
Observability.from_config(capture_config, export_config,
    *, tracer_provider=None) -> Observability
await observability.aclose() -> None
```

Runner 配置装配时自行创建的 Observability 随其关闭；外部注入服务归宿主所有。`aclose` 只排空和关闭该服务自行创建的 SDK，不关闭借用的 tracer provider，也不替换全局 provider。

自定义集成可用 `scope(name, *, kind, attributes)` 记录同步作用域，`bind(attributes, *, replace=False)` 关联上下文，`detached()` 为延后工作分离旧关联。需要更细控制时有 start_span/use_span/end_span、attributes、event、error 等标准 span 辅助方法。内置运行代码已在 owner 处记录业务事实，普通宿主无需重复包裹每个内部函数。

`iris.observability.provider.observe_provider(provider, observability)` 包装标准 provider 的调用与流式行为。使用配置构造的 Runner 时已有装配，不需要自行重复包装。

依据：[wire models](../../src/iris/streaming/models.py)、[gateway](../../src/iris/streaming/gateway.py)、[SSE](../../src/iris/streaming/sse.py)、[WebSocket](../../src/iris/streaming/websocket.py)、[观测配置](../../src/iris/observability/config.py)、[服务](../../src/iris/observability/service.py)。
