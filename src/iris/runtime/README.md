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

装配先解析 provider，再由 memory 配置工厂统一处理 `memory.enabled`。关闭时不接入服务，
即使传入 `memory_service` 也不挂载；开启时优先复用注入对象，否则将同一 provider、
`model.name`、`memory.overview` 与 `memory.generation` 配置绑定到新建的 SQLite service。
有效服务自动提供 Search/Fetch，写工具仍显式配置。构造过程不调用模型；宿主可以显式调用
`refresh_overview()`，开启自动 generation 的 root runner 也会在空闲时生成。注入服务保留自己的
生成依赖。开关在构建时确定，改配置后重建 Agent 并使用新会话。

`RuntimeEnvironment.execution_scope` 明确保留 ROOT/CHILD，自动维护由 root harness 独占。
Runtime 只在 `_compact_request()` 已经选出实际压缩范围后，通过可选
`RuntimeMemoryCapturePort.request_capture(run_id, through_count)` 通知已提交原文范围。
普通请求与没有可压缩前缀的早退不会提示；该同步端口不等待 IO 或记忆模型，不改变
`RuntimeCommitPort` 的持久运行事实职责。后台 capture/flush/dream 和关闭均由 harness 管理，
当前压缩无需等待新记忆，成功后仍只采用当时已经发布的概览。

## 依赖方向

```text
AgentRunner -> AgentRuntime.execute -> RuntimeCommitPort
     |                                  |
     +------------ LifecycleStore <-----+
```

- `RuntimeFactory` 装配 context、provider、tools、workspace，以及可选 memory service 和宿主 context source；
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
engine 在 `before_input` 准备首次窗口、BCI 和用户输入，通过 `commit_run_input()`
原子保存窗口与输入后进入 `before_model`，不消耗模型 reservation 或增加 step index。
BCI 只在输入阶段构建；后续步骤及已提交输入的 resume/recover 使用历史，不重复追加或渲染。
checkpoint 使用版本 3，cursor 必须提供 `visible_tool_names`；旧版在恢复边界拒绝，不推断
旧 cursor 的输入状态或当时可见工具集合。

`RuntimeCommitPort.record_compaction_usage(TokenUsage)` 独立保存每份摘要响应用量；
`commit_compaction(RuntimeCompactionCommit)` 按选择区间时的 session revision 原子替换摘要投影与窗口。
后者推进 session/checkpoint revision，保持原文、执行 cursor 和 pending 主模型 reservation。
每次 `before_model` 检查完整输入，实际 LLM 摘要在主步骤 reservation 获准后进行，不额外消耗主步骤预算。

### 宿主动态上下文与选材

`RuntimeFactory.from_config()` 与 `from_config_path()` 接收可选 `context_source=`，协议与示例见
[`iris.context`](../context/README.md#宿主动态快照)。source 存在时，每次 `before_model` 在
model-step reservation 获准并检查取消和 deadline 后采集一次；未获准的步骤不采集。scope
提供 session/run 身份、step index、workspace 和原始 `run_input`。下一步骤及恢复到
`before_model` 时重新采集，child 不继承 parent source。

采集受当前 run 剩余 deadline 限制，额度到期返回 `DEADLINE_EXCEEDED`；source 自身的普通
异常（包括自行抛出的 `TimeoutError`）以 `IrisContextError` 结束本次 activation，不沿用旧值。
协作式取消与 task cancellation 沿既有控制路径结算。`context_policy.enabled=false` 时注入
source 会在装配时报 `IrisConfigError`。

完整快照渲染为 history 后的一条 `runtime_snapshot` user 消息。未注入 source 不加消息；
空快照明确表示当前无已提供状态。它不改写稳定 system、原始任务、已归档 BCI 或 durable
history，也不进入摘要原料。需要后续精确回读的内容应先由宿主通过工具结果或文件保存。

低于压力线时保留全部贡献。达到压力线后，先尝试精确重复正文折叠，再依次移除宿主显式
标为 `required=False` 的贡献，再撤下可选 deferred schema，最后短化旧工具正文；仍不足时
进入原有 LLM compaction。
required 条目保持，priority 较低者先移除，相同 priority 时后返回者先移除。每次变动重算
完整请求，包含静态上下文、历史、动态消息、模型选项与工具 schema。模型不能按正文语义
自行决定删除哪些宿主约束。

可选条目只在本步骤初次请求选材一次；候选摘要切点、摘要重试和成功压缩后的最终请求共用
已选集合，不因腾出空间重新补入。下一步骤重新采集后可重新选择。即使 `include_tools=false`
或 effective `tool_choice` 不允许回读，动态快照、optional 选材和 LLM compaction 仍然工作；
工具正文裁剪则遵循下节的回读条件。

### 按需工具 schema

`context_policy.deferred_tools` 默认关闭。开启后 shared assembly 自动注册 `tool_search`，
将 MCP 目录标记 deferred；Python 工具保留作者的声明。MCP 仍完整 prepare，原有 eager
工具、`context_read/search` 与 `load_skill` 不因此隐藏。搜索继续使用本地 BM25-like 排名，
参数与结果见 [tools 说明](../tools/README.md#deferred-discovery--tool_search)。

[`_tool_context.py`](./_tool_context.py) 从当前 session 原始已提交消息中的
`metadata.extra.context_revealed_tools` 派生候选，不解析搜索正文或摘要。只有成功且已提交的
搜索结果产生披露；同批次的后续工具调用也不能使用刚搜到的名称。下一次 `before_model`
才选择其完整 schema，不截断参数定义。候选必须仍存在于当前 registry 且符合静态 base
view；deny 优先，只有宿主原始 allow 可越过组过滤，搜索不会改变共享 `allow`。

eager 工具和 host base allow 为必需集合。模型配置与 request_options 合并后的 effective
`tool_choice` 若强制某个 function，该工具也必需，即使原本 deferred 或尚未搜索。别名解析为
canonical name；目标不存在或被 base view 排除时报告 `IrisConfigError`，不发送缺少目标
schema 的强制请求。`include_tools=false` 或 effective `tool_choice="none"` 不发送工具 schema。

可选候选按最新一次搜索排名优先，再按最近成功使用、最近发现和 canonical name 排序。
当前搜索批次之后尚无已提交 assistant 主响应时，保护该批次每次成功搜索的首项，保证
“搜索后使用”的机会；后继主响应包括最终文字答复都会消费保护，未提交响应和 summary 不会。
其余候选在宿主 optional 贡献之后按逆序撤下，每次重算完整请求。必需项和受保护首项仍然
过大时进入既有压缩/容量错误路径，不撤下它们以回避预算。

优先级决定成员，最终 schema 保持 registry 顺序。首次选材后的 schema 集合在本步骤的
预算规划、摘要重试与压缩后请求中保持；移除 schema 不删除历史发现事实，下一步骤可重选，
再次搜索也可提升其顺序。回读型正文裁剪仍只在最终请求可调用 `context_read` 时执行。

最终请求的 canonical names 作为 immutable `RuntimeCursor.visible_tool_names` 与 assistant
调用同事务保存。新响应 preflight、批次继续、HITL 和恢复使用同一集合，并保留当前 base
目录过滤与执行前 permission refresh。未披露名称返回 `TOOL_NOT_ALLOWED`；其它 session 的
搜索不能改变正在等待的批次。部分推进保留集合，离开 `tool_batch` 时清空；恢复批次不重新
采集 source 或运行选材。Fork 只继承复制历史前缀中的发现，child 使用自己的配置、历史和搜索。

### 历史投影与摘要构造

内部 `compaction.py` 在完整原文上定位本 run 的原始输入、最新已归档 steer 与已注入 BCI。
历史投影依次放入摘要消息、已覆盖锚点、未覆盖原文；assembler 将固定 system、静态 memory
放在历史之前。BCI 用 `context_kind=before_current_input` 标记；Search/Fetch 结果属于普通工具
历史，不加入强制保护集合。摘要只在投影时包装一层 `<summary>`，不追加回原文。
切点保持 assistant 的整批 tool calls/results 完整，近期原文是软目标，大组放不下时可以仅留
较小的最近组，或将 suffix 留空。当前 run 已完成的工具步骤也可压缩。

默认启用的 `context_policy` 在投影前给已外置工具结果附加 `result:<message_index>:<block_index>`
回读引用，下标来自原始历史，summary 插入后不会重新编号。该视图使用 copy-on-write，不把
取回提示写回 durable message。引用和 `context_read/search` schema 一并进入完整请求计量。

[`_context_projection.py`](./_context_projection.py) 的 `project_context_request()` 在完整请求
达到既有 80% trigger 时，先折叠精确重复工具正文，再选择可选动态贡献与 deferred schema，
最后从旧到新短化 observation 结果；正文替换只有让完整 `LLMRequest` 的 token 估算实际下降才采用。
低于 trigger 即停止，已足够时不调用摘要模型。单次大结果仍由 artifact
处理；这里同时处理多轮中型输出累积造成的压力。

- 资格来自执行时保存的 `metadata.extra.context_retention="observation"`，不是当前工具目录。
  错误、未闭合调用、keep 结果、用户和 assistant 文字、summary 与任务锚点不被短化。
- 默认保留最近两个已闭合工具批次；一批是 assistant 的完整 calls 与对应 results，普通消息
  不算工具批次。该保护只用于确定性裁剪，不增加 LLM 摘要的硬保留条件。
- 精确判等使用已保存的规范工具名、key 排序后的 JSON 参数及完整内联正文。保留最新代表和
  近期保护组；较早副本换成明确的代表 ref 与本次原文 ref。不同结果、artifact 预览或文件路径
  不作为相等证据，也不为判等读取大文件。每次工具调用照常执行，所有 call/result 均保留。
- 旧结果预览默认 512 字符，分配给正文 head 384 / tail 128；说明和稳定 ref 额外计入请求。
  已 offload 的长预览也可短化，但实际保留的去重代表不会再被短化。

正文裁剪要求最终 schema 中可见 `context_read`，且模型配置与本次 request_options 合并后的
effective `tool_choice` 允许调用它。`include_tools=false`、`tool_choice="none"`、隐藏回读
schema 或强制另一具体 function 时跳过裁剪；强制 `context_read` 本身仍可裁剪。原有 LLM
compaction 不因此关闭。

主请求、候选摘要切点和压缩后的最终请求使用同一个投影入口，每个候选重新确定可见正文
中的重复代表。投影只 copy-on-write 修改派生正文，不写原始消息、session revision、checkpoint
或裁剪列表；restart/fork 从各自历史重算。摘要原料仍是未裁剪的已归档原文，大结果只保留其
已有预览与 ref，不自动扫描全部 artifact。

`_compaction_summary.py` 把全部文本块、调用参数、工具结果及必要 error/artifact 引用按顺序
序列化；大块按字符覆盖范围分片，调用是否完成与结果文字是否读完分别标识。每一批都用
当前工作摘要重新计算完整输入，不丢弃尚未处理的片段。
记录 header 包含原始 `message:<index>` / `result:<message_index>:<block_index>`，摘要指令要求
保留后续仍需使用的精确引用；回读通过 harness 提供的 context access port 访问保存材料，
不让 runtime 直接读取生命周期数据库或重新执行历史工具。
完整记录前缀采用指数探测与二分细化，只有下一条正文需要时才做字符切分，避免逐条重算
全部已接受前缀。每个返回批次都经过完整请求计量；分批不承诺最大装填率，也不跨批缓存工作摘要。
切点规划复用相同空后缀的估算，不改变完整消息组和近期原文保留规则。

摘要指令来自独立 Jinja2 文件，默认使用 [`prompts/compaction.j2`](../prompts/compaction.j2)，
要求七栏 Markdown、正文跟随对话主要语言。`compaction.prompt` 可以替换指令与输出格式；
旧摘要与本批历史仍由框架提供，user 消息的包装文案来自
[`compaction_input.j2`](../prompts/compaction_input.j2)。每次压缩操作直接通过
`RuntimeEnvironment.prompt_renderer` 取得一次摘要指令并去除首尾空白，供全部分块
计量和请求共用；Jinja 复用编译缓存并按 mtime 检测更新，文件修改在下次压缩操作生效。没有标题 parser 或
格式修复循环。路径配置见 [agents 说明](../agents/README.md#compactionconfig)。

`prompt_renderer` 是环境持有的共享 `iris.utils.TemplateRenderer`，默认关闭自动转义；
摘要输入中的 JSON、引号和 `<>&` 按原文保留。模板读取或渲染失败在 runtime 边界转换为
`IrisContextError`，继续使用 `context` 错误来源。

摘要请求复用有效主模型选项，覆盖为非流式、无工具/response schema、输出上限 S，并设置
`num_retries=0`。候选只存在于内存，全部分块完成后才由外层提交。摘要消费只接受完整非空
文本；`IrisContextCompactionError` 使用 `context` 来源及 `CONTEXT_COMPACTION_*` 错误码。

确定性选材与正文减载后，完整输入仍达到可用预算 B 的 80% 时选择新增摘要前缀；没有新增前缀且
输入不超过 B 时直接继续。
有新增前缀时，先保存每份返回响应的 `RunUsage.compaction`，再检查摘要是否完整有效。
全部分块完成后，完整主请求须不超过 80% 且比压缩前更小，才能原子提交投影。摘要不进入主
response 的 message delta，也不增加主步骤 reservation。

整次操作共用默认 300 秒额度，分块与重试不重置时钟；每次请求同时受剩余 run deadline 和
更短的 request timeout 限制。只对当前失败分块的连接、超时或限流错误重试一次。压缩后重新
读取 run 剩余时间，不延长原始 deadline。取消沿用原语义，排队 steer 留到既有主响应/工具边界。

压缩一旦开始，失败就结束当前 run；保留原文、上次已提交摘要与窗口和已经记录的摘要用量。
摘要投影已提交但主响应尚未提交时，恢复使用新摘要、已提交窗口与同一个 pending reservation；WAITING
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

`iris.providers.CompletionProvider` 必须同时实现 `complete()` 和同步
`estimate_input_tokens(request)`；后者
计量应用模型选项及工具 schema 后的完整请求。自定义 provider 与测试替身直接满足同一契约。
`RuntimeEnvironment.agent_config.compaction` 携带压缩配置，无需独立环境字段。
摘要始终直接使用 `complete()`，不会向 host 发布摘要正文或摘要模型 stream 事件。

真实处理新前缀时发布 `context.compaction.started`；投影提交成功后发布
`context.compaction.completed`，然后才发布主 `model.step.started`。未完成则发布
`context.compaction.failed`，具体错误仍由最终 run 结果解释。三种状态复用现有 identity，
没有独立 payload 模型；durable `context.compacted` 保留在事件历史中。

`stream_sink=None` 精确保留 complete-only 路径：runtime 继续调用
`CompletionProvider.complete()`，请求的 `stream` 强制为 `False`，不受 `request_options` 覆盖。
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
后处理和提交路径。外层取消、timeout 或 sibling cancellation 到达时也先 drain，按 ordinal
提交已收回的确定结果；随后传播外层取消或结算期限，不把延后返回误判为正常成功。
`asyncio.timeout().expired()` 保留单次 timeout 事实；无确定结果时仍沿用 unknown 语义。
`before_call` / `after_call` 不由这条 body 取消桥中断；
慢 middleware、压住 `CancelledError` 的协程及 INLINE 阻塞仍可能延迟退出。

并发文件读取共享同一个 `ReadFileState` identity；worker 只返回不可变 observation，由 event
loop 合并。窗口 settle 后的 checkpoint snapshot 包含合并记录，后续串行 write barrier 可以
继续执行 stale-read 检查。checkpoint 中的 raw dict 只在 `ToolBridge.restore_read_state()`
恢复边界解析一次；runtime 内部始终传递 typed state，snapshot 直接序列化该对象。
内置 write/edit 使用独立记录快照在线程完成本地操作，再由 loop 合并该文件观测；artifact
归一化与落盘同样在线程执行。这些有限本地 IO 会收回实际结果后再结束等待，不强停线程。
同步 callable 默认 inline；显式 `CallableExecutionMode.THREAD` 才把
阻塞 body 放入 worker。线程无法安全强停，取消或 timeout 只停止 async waiter；claim 未结算时
runtime 以 `OUTCOME_UNKNOWN` 收口，晚到结果不能推进 history、cursor、checkpoint 或 events。
thread placement 不承诺 CPU 加速。NETWORK/MCP 并发或 write 并发未来必须另行设计 effect、
retry、timeout、冲突与 crash reconciliation 协议，不能直接放宽当前 classifier；本轮也没有
引入 delta/merge/lock/hash 模型。

## Memory 概览窗口与自主读取

有效 memory service 存在且 session 的 `context_window` 尚为 `None` 时，Runtime 按 `read_namespaces` 的配置顺序
调用一次 `MemoryService.aload_overviews()`。用包含本次 BCI/user 的完整待发送请求选择窗口，
将概览、输入与 checkpoint 同次提交，再向 provider 发送。显式空窗口表示已经初始化。
Service 存在时，普通新 run、工具循环、steer、HITL 与输入提交后的 recovery 复用已提交文本；只有新 session
或成功压缩才采用新版概览。Fork 的目标窗口为 `None`，首输入重新采用。

`RuntimeEnvironment.memory_service is None` 时，统一请求构造入口传入空 `system_addendum`，
即使 session 保存着旧概览也不追加到 system。普通请求与恢复不因此读取或改写窗口，也不额外
推进 session revision；静态 memory、BCI 和包含既有工具结果的普通历史继续保留。成功压缩沿
原事务一起提交新摘要与空窗口；失败保留旧摘要和窗口，后续无 Service 请求仍屏蔽该概览。

`full` 包含核心事实与知识范围，`navigation` 仅包含知识范围。全部 namespace、状态警告、
实际工具指引和包装共享 `floor(compaction.input_budget_tokens * memory.overview.system_budget_ratio)`
额度，默认比例为 2%。Provider 对同一完整请求有无概览的估算差额就是开销；原有 system、
静态 memory、历史和工具 schema 不重复计费。Full 超专用额度或可降级的 system/请求容量时，
整体尝试 navigation；知识范围仍超额则报告容量错误，不截断 namespace 或增加第三种降级。
普通历史的整体容量继续由原有压缩流程处理。
full/base 的专用额度差额使用同形、未裁剪的历史计算；压缩后采用新窗口时，两者也携带同一份
已选动态快照与 schema，避免把历史正文释放量计入概览成本。选定窗口后，实际主请求再经过
上述投影与完整计量；成功压缩采用的新窗口也走同一路径，不重新选择动态贡献或工具集合。

概览指引、标题和正文包装由 [`memory_context.j2`](../prompts/memory_context.j2) 管理，
通过同一 `RuntimeEnvironment.prompt_renderer` 渲染。Python 提供概览与实际可用工具数据，
窗口预算仍以实际渲染后准备发送的请求计算。

概览通过 `ContextBuilder.build(system_addendum=...)` 放在 system 模板结果之后，计入 system
字符上限，不写入消息历史。`context.yaml` 中的静态 memory slot 保持原位置。成功压缩把新摘要、
实际新窗口、checkpoint 和事件一起提交，随后的主请求立即使用该窗口；失败或取消保留旧窗口。

模型指引以当前概览为长期记忆范围：未提及的主题默认没有，不搜索这些主题；已覆盖且相关时
按需读取。缺概览或无 mirror 时仍正常聊天，但本窗口不使用长期记忆。程序不执行数据库主题
拦截；覆盖主题下的 Search/Fetch 可读取当前最新条目。工具说明仅列实际启用的
`memory_search`/`memory_fetch`，尊重 `include_tools`；仅 Fetch 时按已知 ID 读取。窗口中的
说明保持稳定，执行时仍使用当前工具注册表和权限。Search/Fetch 结果沿普通工具历史与压缩处理。

窗口指引要求模型按正文核对事实的适用对象和条件：提及某个对象不代表事实属于它；片段足够
时停止，仅为缺失的必要信息补查。Search 的参数语义由工具说明提供，支持普通 query 的 OR
匹配与可选 `required_terms` 必要词组；`has_more` 不要求取全候选。Fetch 用于缺少的正文或
来源元数据，也可重新核对已知 ID 的当前值；没有新增自动查询或固定搜索次数限制。

配置构造的 SQLite service 在单个 worker job 内读取全部 namespace 的发布物；runtime
不消费取消后的迟到结果。生成仅由宿主显式调用 `refresh_overview()`，运行时采用过程不生成。

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
配置 `tools.subagent` 时要求使用 `AgentRunner.from_config*()`，而非低层
`RuntimeFactory.from_config*()`，因为委派需要完整 lifecycle owner。

Factory 接收可选 `context_access: ContextAccessPort`（定义于
[`iris.tools.context_access`](../tools/context_access.py)）。默认 `context_policy.enabled=true`
要求提供它；缺少时装配抛出 `IrisConfigError`。完整 `AgentRunner` 自动提供此依赖；直接使用
低层工厂时由宿主实现，或显式配置 `context_policy.enabled: false` 关闭回读工具。下例的
`provider` 与 `access` 均由宿主提供：

```python
from iris.runtime import RuntimeFactory

runtime = RuntimeFactory.from_config_path(
    "agent.yaml", provider=provider, context_access=access
)
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
`RuntimeStreamEvent`、assembler/tool bridge、`RuntimeSteeringPort`、`SteeringInput`，以及
activation/commit-port contracts。不存在 complete-run options/status/result、
`run_turn()`、`run_loop()`、`resume()` 或旧 checkpoint helper。
共同非流式协议 `CompletionProvider` 从 `iris.providers` 导入。

## 验证

正文减载的定向用例见 `tests/runtime/test_context_projection.py` 与
`tests/harness/test_context_pruning.py`，分别检查纯请求投影以及实际调用、原文保存和回读。
动态采集与选材见 `tests/runtime/test_context_source.py`、`tests/runtime/test_context_selection.py`
及 `tests/harness/test_context_source_integration.py`，覆盖步骤与恢复、预算顺序和真实 runner 接线。
按需 schema 见 `tests/runtime/test_deferred_selection.py` 与
`tests/harness/test_deferred_tool_context.py`，覆盖发现排序、强制工具、session 隔离和批次恢复。

```bash
uv run pytest tests/runtime
uv run ruff check src/iris/runtime tests/runtime
uv run mypy src/iris/runtime
```
