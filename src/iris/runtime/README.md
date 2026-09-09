[English](README.en.md)

# `iris.runtime`

`iris.runtime` 是 Agent lifecycle 的低层 inner engine。它从一个已持久化的
`RuntimeCursor` 开始，通过调用方提供的 `RuntimeCommitPort` 推进 provider 与工具循环，直到
completed、waiting、budget、cancel、deadline、failed 或 outcome unknown。它不创建 logical
run、不选择 store，也不拥有 cancellation/recovery 的公开编排。

完整运行请使用 `iris.harness.AgentRunner`。只有实现自定义 lifecycle owner 时才直接调用
`AgentRuntime.execute()`。

## 依赖方向

```text
AgentRunner -> AgentRuntime.execute -> RuntimeCommitPort
     |                                  |
     +------------ LifecycleStore <-----+
```

- `RuntimeFactory` 只装配 context、provider、tools、workspace 与可选 memory service；
- `RuntimeEnvironment` 只保存 engine live dependencies，没有 session/lifecycle store 或
  interaction service；
- runtime 不 import harness，也不直接写 SQLite；
- exact session、checkpoint、tool claim/result 与 interaction 写入由 commit port 提供。
- commit、reservation 与 claim DTO 使用不可变 dataclass，仅传递进程内已验证事实；
  包装提交事实不会再次扫描 cursor。持久化 cursor 的完整解析仍由 load/recovery 边界负责。
- 可选的 `RuntimeSteeringPort` 只向当前 activation 的安全边界提供瞬时输入，不拥有 queue 或
  persistence。

## 低层调用契约

```python
result = await runtime.execute(
    activation,
    commits=commit_port,
    cancellation=cancellation_signal,
    steering=steering_port,  # 可选；省略时保持原行为
    stream_sink=stream_sink,  # 可选；启用同进程 live event
)
```

`RuntimeActivationInput` 包含 run/activation/session identity、`start | resume | recover` kind、
固定的 `RuntimeExecutionOptions` 和 JSON-safe cursor。`RuntimeActivationResult` 只返回 engine
事实；调用方必须从 durable store 重载最终 `RunResult`。

`start` 与 `before_model / step 0` 的初始 `recover` activation 携带当前用户 input；`resume` 和
非初始 `recover` 不携带。engine 只在该字段存在时将其注入 provider request 一次，后续恢复依赖
committed session history。

cursor 位置只有：

- `before_model`：可预留下一次 provider step；
- `tool_batch`：provider response 已提交，按 `next_tool_index` 推进 exact tool calls；
- `outcome_ready`：assistant outcome 已提交，只差 lifecycle terminal settlement。

无工具的 provider response 会以 `CheckpointResumability.OUTCOME_READY` 提交。工具 effect 前
必须 durable claim，result 后必须 durable commit；claim 后无法证明结果时返回
`TOOL_OUTCOME_UNKNOWN`，不得重放 effect。

模型返回工具批次后，runtime 先提交 assistant、整批 prepared facts 与初始 tool cursor，
再按原始顺序推进。普通调用的结果提交后，才在当前 `next_tool_index` 对应的人工 gate
原子提交 waiting checkpoint 与 interaction；恢复不会重复已经提交的前缀调用。
人工响应绑定该 durable subject。等待期间动态权限变为 ALLOW 时，批准可继续执行；变为
DENY 时，批准仍返回权限拒绝结果。用户主动拒绝保持 `USER_REJECTED`，执行前仍刷新权限。

工具路径通过 `ToolBridge.preflight()` 形成计划，再调用带执行守卫的 `execute_prepared()`；
工具结果统一经 `ToolResult.to_msg()` 投影为 history 消息。

## 可选 live streaming

`stream_sink=None` 精确保留 complete-only 路径：runtime 继续调用
`RuntimeProvider.complete()`，请求的 `stream` 强制为 `False`，不受 `request_options` 覆盖。
传入同步 `RuntimeEventSink` 时，
runtime 通过独立的 `StreamingRuntimeProvider` structural capability 检测 `stream()`；capability
缺失会以 `PROVIDER_STREAM_ERROR/provider` 失败，不回退到 `complete()`，也不伪造 token。

streaming 路径只复制当前可信请求并把 `stream` 设为 `True`，然后在 provider async iterator 上
direct-pull。Runtime 先发布 `model.step.started`，再把每条 `ModelStreamEvent` 包装为
`model.event` 同步交给 sink。partial、usage 和 provider terminal 都只是 live facts；只有
`response.completed` 携带的完整 `LLMResponse` 会进入既有 `to_msg()`、steering、tool preflight
与 `RuntimeModelStepCommit` 路径。failed/cancelled terminal 或合法终态前 EOF 不提交 assistant、
history、checkpoint 或工具调用。Runtime 在 terminal、EOF、失败或取消退出时显式关闭 typed
provider iterator；清理失败只记录 warning。Provider completed 也不代表 durable commit 已成功。

工具 live event 保持既有 effect gate：完整 `ToolUseBlock` 才发布 `tool.preparing` 并进入
preflight；`tool.started` 只在 permission refresh、activation fence 与 durable claim 成功后、
middleware/body 前发布；`tool.completed` 只在 ordered `commit_tool_result()` 成功后携带完整
`ToolResult` 发布。并发工具 body 可以乱序结束，但 completed event 仍按 model ordinal。
Runtime 不 await sink、不创建 queue，也不捕获自定义 sink 的异常；publisher 隔离由后续
harness-owned sink 负责。

## Runtime steering

自定义 lifecycle owner 可以为一次 `execute()` 调用传入 activation-scoped
`RuntimeSteeringPort`。`claim(run_id, activation_id)` 每个安全边界最多返回一条
`SteeringInput`；该 frozen model 只包含非空 `submission_id` 与 `Role.USER` message。Runtime
不创建 queue，也不把 claim 状态写入 cursor、checkpoint 或 store。

Runtime 只在两个位置 claim：

- 无工具 assistant response 已生成、`commit_model_step` 之前；成功时将 assistant 与 steer
  user message 放入同一 delta，cursor 进入下一 `before_model`，resumability 为 `SAFE`；
- 同批次 final ordered tool result 已知、最终 `commit_tool_result` 之前；成功时将 tool result
  与 steer user message 放入同一 delta，沿用下一 `before_model` cursor。

中间工具结果、provider/tool effect 执行中、HITL waiting、`outcome_ready`、cancellation、deadline
和 STOP terminal error 都不会 claim。Claim 返回后到同步 commit 与 `acknowledge()` / `fail()`
之间没有 `await`：commit 成功才 acknowledge，commit 异常则 fail `commit_failed` 并原样传播；
callback 自身的异常只记录日志，不会覆盖 durable 结果。传入 `None` 或 claim 返回 `None` 时，
既有 cursor、message delta、resumability 与 outcome 语义不变。

## 有界工具并发

在 `RETURN_TO_MODEL` 策略下，runtime 会把连续的“只读且声明为并发安全”调用组成内部窗口，
每个窗口最多 8 条。8 是私有实现上限，不是 YAML、`RuntimeExecutionOptions` 或环境变量配置；
本次能力没有改变 public config、schema、model 或导出。

窗口只覆盖连续候选。STOP、HITL、preflight result、WRITE/EXECUTE/NETWORK/MCP/AGENT，以及任一
不安全或分类失败的调用都是串行屏障，后序调用不能跨过屏障启动。整个 batch 复用首次生成的
typed tool plan；每个 child 在 body 前只刷新 permission 并提交 exact durable claim，不重复
schema validation。body 可以乱序结束，但 result message、cursor、
session history、checkpoint 和 committed event 只按原始 ordinal 的连续前缀推进。多个
`TOOL_CALL_CLAIMED` telemetry event 的先后顺序不是契约。

control interruption 只提交首个异常/空洞之前的已知 `ToolResult`；后序内存结果不会跳洞。
任何未提交的 durable claim 都会让取消、deadline 或程序中断最终 fail closed 为
`OUTCOME_UNKNOWN`。父 task 或基础设施退出前，runtime 会 cancel 并 drain 自己创建的 children。
协作式取消使用 `iris.exceptions.IrisCancellationRequestedError`；runtime 将它转换为 activation
outcome，而不是普通工具错误。

并发文件读取共享同一个 `ReadFileState` identity；worker 只返回不可变 observation，由 event
loop 合并。窗口 settle 后的 checkpoint snapshot 包含合并记录，后续串行 write barrier 可以
继续执行 stale-read 检查。checkpoint 中的 raw dict 只在 `ToolBridge.restore_read_state()`
恢复边界解析一次；runtime 内部始终传递 typed state，snapshot 直接序列化该对象。
同步 callable 默认 inline；显式 `CallableExecutionMode.THREAD` 才把
阻塞 body 放入 worker。线程无法安全强停，取消或 timeout 只停止等待并丢弃晚到返回；claim 已
存在时 runtime 以 `OUTCOME_UNKNOWN` 收口，晚到结果不能推进 history、cursor 或 checkpoint。
thread placement 不承诺 CPU 加速。NETWORK/MCP 并发或 write 并发未来必须另行设计 effect、
retry、timeout、冲突与 crash reconciliation 协议，不能直接放宽当前 classifier；本轮也没有
引入 delta/merge/lock/hash 模型。

## 显式 Memory 注入

`RuntimeExecutionOptions.memory_query` 和 `memory_results` 是显式 opt-in 的动态 memory 输入。
每个 logical run 只在第一次 `before_model` step 注入一次；同一用户输入后续因工具循环或
HITL resume 产生的 provider 请求不会再次附加这条动态 memory。新的用户输入会创建新的
`start` activation，因此可以重新注入一次。`context.yaml` 中声明的静态 memory slot 不受此
规则影响。`memory_results` 只处理调用方提供的本地快照；`memory_query` 才会 await
`MemoryService.abuild_context()`。配置构造的 SQLite service 会在单个 worker job 中完成建连、
查询、物化和关闭，runtime 不消费取消后的迟到结果。

## Factory

`ToolBridge` 的内部 Sub Agent adapters 分别提供 raw-only continuation prepare 与专用执行，
只在 bridge 从现有 run/call ID 构造 `SubagentParentCall`。它们不读取 store、不新增 context
identity，也不改变普通 file read state。Runtime 在两处整批 preflight 前点查 link 或存储的
outer response，使用 raw-only prepare 保留模型顺序；subagent 作为现有并发规则中的串行边界。
Linked WAITING/terminal 分别调用 commit port 的 rebind/finalize，不创建 parent effect claim。
ACTIVE 与 WAITING continuation 共用结果 cursor 投影，WAITING 只委托最终结果归一化。

内部 `_assembly.py` 统一装配 context、skills、provider 与工具。ROOT 接收已加载的路由/port
bundle 才注册 `subagent`；CHILD 始终排除它。唯一 boundary resolver 选择父子 workspace 的
较窄目录，不相交时报配置错误，并组合实际 parent policy 与 child 默认 policy。
Public `RuntimeFactory.from_config*()` 保持普通参数；配置 `tools.subagent` 时要求使用
`AgentRunner.from_config*()`，因为委派需要完整 lifecycle owner。

```python
from iris.runtime import RuntimeFactory

runtime = RuntimeFactory.from_config_path("agent.yaml", provider=provider)
```

Factory 不读取或创建 lifecycle database。`agent.yaml` 的 `session` 配置由 harness composition
解释；直接调用 Factory 时该字段不会产生持久化副作用。

Factory 创建 `ProviderClient` 时，将合并全局配置后的有效 provider、LiteLLM provider、endpoint
和 headers 投影到 `RuntimeEnvironment.provider_fingerprint`，供 harness 比较恢复环境；API key
不参与。显式注入的 provider 由 host 负责声明版本：在创建 runner 前设置
`runtime.environment.provider_fingerprint = {"version": "my-provider-v2"}`。默认空字典不推断
自定义 provider 的内部实现，未实际使用的原始路由配置也不计入指纹。

Factory 会先解析 `permissions.workspace`，再构建基础 context 和用户声明的工具。若
`skills.enabled: true`，它以该 workspace 做一次项目级发现快照：非空结果会追加
`available_skills` system slot，并在创建 `ToolRegistryView` / `ToolExecutor` 前注册共享同一
registry 的 `load_skill`。关闭 Skill 或发现结果为空时会精确绕过 catalog 和 loader，不改变
原有 context/tool 形状；每个 factory/runtime 实例内不自动刷新快照。

`RuntimeEnvironment.skill_registry` 保存这个共享 registry，供 harness 将启动发现的 Skill
内容版本纳入恢复指纹；不再次读取 Skill 文件。`load_skill` 完整读取时会核对发现版本，文件
变化后返回 `SKILL_VERSION_CHANGED`，需要重新创建 runtime 并开始新 run。

`skills.root` 越出 workspace、`skills.require` 缺失，或 `load_skill` 与用户工具名称/别名冲突，
都会在装配阶段转为 `IrisConfigError` 并 fail closed。完整契约见
[`iris.skill`](../skill/README.md)。

## 公开接口

包级导出包括 `AgentRuntime`、`RuntimeFactory`、`RuntimeEnvironment`、
`StreamingRuntimeProvider`、`streaming_provider_for()`、`RuntimeEventSink`、
`RuntimeStreamEvent`、provider/assembler/tool bridge、`RuntimeSteeringPort`、`SteeringInput`，以及
activation/commit-port contracts。不存在 complete-run options/status/result、
`run_turn()`、`run_loop()`、`resume()` 或旧 checkpoint helper。

## 验证

```bash
uv run pytest tests/runtime
uv run ruff check src/iris/runtime tests/runtime
uv run mypy src/iris/runtime
```
