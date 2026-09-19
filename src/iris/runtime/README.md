[English](README.en.md)

# `iris.runtime`

`iris.runtime` 是 Agent lifecycle 的低层 inner engine。它从一个已持久化的
`RuntimeCursor` 开始，通过调用方提供的 `RuntimeCommitPort` 推进 provider 与工具循环，直到
completed、waiting、budget、cancel、deadline、failed 或 outcome unknown。它不创建 logical
run、不选择 store，也不拥有 cancellation/recovery 的公开编排。

完整运行请使用 `iris.harness.AgentRunner`。只有实现自定义 lifecycle owner 时才直接调用
`AgentRuntime.execute()`。

MCPTool 经普通工具链进入串行执行。`IrisMCPOutcomeUnknownError` 使用已有
`_unknown_tool_outcome` 结算未确定的 claim，不新增 stop reason 或持久化协议；受信只读的
SDK 错误仍是普通 ToolResult，遵守 ToolErrorPolicy。MCP 不另建取消 watcher。

shared assembly 同步读取 `AgentConfig.mcp` 声明，将 `MCPManager` 绑定到原 registry，构造时
不连接。`RuntimeEnvironment.aprepare()` / `aclose()` 直接委托该 manager；低层调用者必须
在 execute 前准备并在所有执行结束后关闭。环境不关闭外部注入的 provider、memory 或 store。
root runner 自动管理准备时机，多 run 复用同一固定目录与连接；child 由 harness 在 admission
前准备，并在 WAITING/结束后关闭独立资源，恢复时重建。

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

所有 activation 都携带原始 `run_input` 与创建 run 时的 `initial_session_message_count`。
engine 在 `before_input` 准备动态 memory、BCI 和用户输入，通过 `commit_run_input()`
原子归档后进入 `before_model`，不消耗模型 reservation 或增加 step index。
BCI 只在输入阶段构建；后续步骤及已提交输入的 resume/recover 使用历史，不重复追加或渲染。
checkpoint 使用版本 2；旧版在恢复边界拒绝，不推断旧 `before_model/step0` 的输入状态。

`RuntimeCommitPort.record_compaction_usage(TokenUsage)` 独立保存每份摘要响应用量；
`commit_compaction(RuntimeCompactionCommit)` 按选择区间时的 session revision 原子替换摘要投影。
后者推进 session/checkpoint revision，保持原文、执行 cursor 和 pending 主模型 reservation。
每次 `before_model` 在主步骤 reservation 获准后检查完整输入，压缩不额外消耗主步骤预算。

### 历史投影与摘要构造

内部 `compaction.py` 在完整原文上定位本 run 的原始输入、最新已归档 steer 与已注入 BCI。
历史投影依次放入摘要消息、已覆盖锚点、未覆盖原文；assembler 将固定 system、静态 memory
放在历史之前。BCI 的 `context_kind=before_current_input` 标记与动态 memory 明确区分；
动态快照按普通历史压缩，不加入强制保护集合。摘要只在投影时包装一层 `<summary>`，不追加回原文。
切点保持 assistant 的整批 tool calls/results 完整，近期原文是软目标，大组放不下时可以仅留
较小的最近组，或将 suffix 留空。当前 run 已完成的工具步骤也可压缩。

`_compaction_summary.py` 把全部文本块、调用参数、工具结果及必要 error/artifact 引用按顺序
序列化；大块按字符覆盖范围分片，调用是否完成与结果文字是否读完分别标识。每一批都用
当前工作摘要重新计算完整输入，不丢弃尚未处理的片段。

摘要指令来自独立 Jinja2 文件，默认使用 [`prompts/compaction.j2`](../prompts/compaction.j2)，
要求七栏 Markdown、正文跟随对话主要语言。`compaction.prompt` 可以替换指令与输出格式；
旧摘要与本批历史仍由框架提供。每次压缩操作通过既有模板渲染器取得一次指令，供全部分块
计量和请求共用；Jinja 复用编译缓存并按 mtime 检测更新，文件修改在下次压缩操作生效。没有标题 parser 或
格式修复循环。路径配置见 [agents 说明](../agents/README.md#compactionconfig)。

摘要请求复用有效主模型选项，覆盖为非流式、无工具/response schema、输出上限 S，并设置
`num_retries=0`。候选只存在于内存，全部分块完成后才由外层提交。摘要消费只接受完整非空
文本；`IrisContextCompactionError` 使用 `context` 来源及 `CONTEXT_COMPACTION_*` 错误码。

完整输入达到可用预算 B 的 80% 时选择新增前缀；没有新增前缀且输入不超过 B 时直接继续。
有新增前缀时，先保存每份返回响应的 `RunUsage.compaction`，再检查摘要是否完整有效。
全部分块完成后，完整主请求须不超过 80% 且比压缩前更小，才能原子提交投影。摘要不进入主
response 的 message delta，也不增加主步骤 reservation。

整次操作共用默认 300 秒额度，分块与重试不重置时钟；每次请求同时受剩余 run deadline 和
更短的 request timeout 限制。只对当前失败分块的连接、超时或限流错误重试一次。压缩后重新
读取 run 剩余时间，不延长原始 deadline。取消沿用原语义，排队 steer 留到既有主响应/工具边界。

压缩一旦开始，失败就结束当前 run；保留原文、上次已提交摘要和已经记录的摘要用量。
摘要投影已提交但主响应尚未提交时，恢复使用新摘要与同一个 pending reservation；WAITING
先继续原工具流程，`outcome_ready` 只结算。主 provider 实际超窗不会触发额外压缩重试。

cursor 位置只有：

- `before_input`：准备并原子归档本轮输入，完成后进入 `before_model`；
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
生成 tool fact 时直接复用已有人工请求的调用指纹；没有人工请求时才按精确参数和 workspace 计算。

工具路径通过 `ToolBridge.preflight()` 形成计划，再调用带执行守卫的 `execute_prepared()`；
工具结果统一经 `ToolResult.to_msg()` 投影为 history 消息。
普通工具与 child 续接复用 executor 的最终结果处理，统一保留完整输出引用并裁剪模型正文。

## 可选 live streaming

`RuntimeProvider` 必须同时实现 `complete()` 和同步 `estimate_input_tokens(request)`；后者
计量应用模型选项及工具 schema 后的完整请求。自定义 provider 与测试替身直接满足同一契约。
`RuntimeEnvironment.agent_config.compaction` 携带压缩配置，无需独立环境字段。
摘要始终直接使用 `complete()`，不会向 host 发布摘要正文或摘要模型 stream 事件。

真实处理新前缀时发布 `context.compaction.started`；投影提交成功后发布
`context.compaction.completed`，然后才发布主 `model.step.started`。未完成则发布
`context.compaction.failed`，具体错误仍由最终 run 结果解释。三种状态复用现有 identity，
没有独立 payload 模型；durable `context.compacted` 保留在事件历史中。

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
`OUTCOME_UNKNOWN`，包括只读调用。父 task 或基础设施退出前，runtime 会 cancel 并 drain
自己创建的 children。
协作式取消使用 `iris.exceptions.IrisCancellationRequestedError`；runtime 将它转换为 activation
outcome，而不是普通工具错误。

普通 async callable、自定义异步 `BaseTool` 和 THREAD callable 共用 `ToolExecutor` 的 body
取消桥：signal 触发 body task 取消并等待其清理，已完成或响应取消后正常返回的结果仍进入既有
后处理和提交路径。外层取消、timeout 或 sibling cancellation 已打断 executor 时只 drain 并
传播原取消，不消费清理期间的返回值。`before_call` / `after_call` 不由这条 body 取消桥中断；
慢 middleware、压住 `CancelledError` 的协程及 INLINE 阻塞仍可能延迟退出。

并发文件读取共享同一个 `ReadFileState` identity；worker 只返回不可变 observation，由 event
loop 合并。窗口 settle 后的 checkpoint snapshot 包含合并记录，后续串行 write barrier 可以
继续执行 stale-read 检查。checkpoint 中的 raw dict 只在 `ToolBridge.restore_read_state()`
恢复边界解析一次；runtime 内部始终传递 typed state，snapshot 直接序列化该对象。
同步 callable 默认 inline；显式 `CallableExecutionMode.THREAD` 才把
阻塞 body 放入 worker。线程无法安全强停，取消或 timeout 只停止 async waiter；claim 未结算时
runtime 以 `OUTCOME_UNKNOWN` 收口，晚到结果不能推进 history、cursor、checkpoint 或 events。
thread placement 不承诺 CPU 加速。NETWORK/MCP 并发或 write 并发未来必须另行设计 effect、
retry、timeout、冲突与 crash reconciliation 协议，不能直接放宽当前 classifier；本轮也没有
引入 delta/merge/lock/hash 模型。

## Memory 召回与历史快照

配置启用或宿主注入 memory service 后，默认每个新用户 logical run 在 `before_input` 用当前
输入自动召回一次。`memory.recall_mode=manual` 关闭自动召回；显式 `memory_results`（包括
空列表）或 `memory_query` 优先于自动查询，两个显式字段互斥。每个片段一条 `sender=context`
历史消息，metadata 保存 `context_kind=memory`、`namespace`、`item_id` 和 `truncated`。
工具循环、同 run 的 steer、HITL resume 与
已提交输入后的 recover 不再查询，继续重放历史中的快照；普通压缩仍可把原文替换为摘要。
输入提交前中断则可在恢复时重新准备。新的 run 可指定新的查询或结果。
`context.yaml` 中的静态 memory slot 保持固定位置，不复制进历史。
`memory_results` 只处理调用方提供的本地快照；显式 `memory_query` 调用
`MemoryService.abuild_context()`，自动路径调用 `arecall()` 并形成预算内片段。只有自动路径
按当前历史投影中相同 item_id、相同渲染原文跳过重复追加；不把摘要/工具输出当原文，也不为
跳过的条目补查或添加压缩保护。自动配置的词项预算不传给显式查询和工具。
自动读取失败以带 run_id 的 WARNING 提示并继续；渲染、显式输入和服务初始化错误正常报告。
配置构造的 SQLite service 会在单个 worker job 中完成建连、
查询、物化和关闭，runtime 不消费取消后的迟到结果。

## Factory

`ToolBridge` 的内部 Sub Agent adapters 分别提供 raw-only continuation prepare 与专用执行，
只在 bridge 从现有 run/call ID 构造 `SubagentParentCall`。它们不读取 store、不新增 context
identity，也不改变普通 file read state。Runtime 在两处整批 preflight 前点查 link 或存储的
outer response，使用 raw-only prepare 保留模型顺序；subagent 作为现有并发规则中的串行边界。
Linked WAITING/terminal 分别调用 commit port 的 rebind/finalize，不创建 parent effect claim。
ACTIVE 与 WAITING continuation 共用结果 cursor 投影，WAITING 只委托最终结果归一化。
Special branch 的 timeout 由 harness 按 child admission absolute 时间管理；runtime 不套
普通工具 timeout。Child await 后先检查 parent cancellation/deadline，再 rebind/finalize。
只有 fresh dispatch 发布 `tool.started`，linked recovery 保留原 logical start。

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

Factory 会先解析 `permissions.workspace`，再构建基础 context 和用户声明的工具。若
`skills.enabled: true`，它以该 workspace 做一次项目级发现快照：非空结果会追加
`available_skills` system slot，并在创建 `ToolRegistryView` / `ToolExecutor` 前注册共享同一
registry 的 `load_skill`。关闭 Skill 或发现结果为空时会精确绕过 catalog 和 loader，不改变
原有 context/tool 形状；每个 factory/runtime 实例内不自动刷新快照。

`RuntimeEnvironment.skill_registry` 保存这个共享目录快照。`load_skill` 每次读取登记路径的当前文本，
返回含 frontmatter 的前 1000 行，不重新解析 frontmatter。文件编辑无需新 run；名称、描述与路径的
目录更新仍需重新构造 runtime，因此旧 catalog 描述可以与返回文件中的新描述并存。

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
