# 运行、会话与持久化 SDK

本页查询完整运行入口、状态模型、人工交互、会话输入准入、历史分支和 Store 扩展契约。首次接入见 [Python SDK](../getting-started/python-sdk.md)；任务示例见[管理会话](../cookbook/sessions.md)与[人工输入和恢复](../cookbook/hitl-recovery.md)。

普通宿主从 `iris.harness` 导入 `AgentRunner`、`SessionManager`、`SessionHistory` 和运行请求/结果类型。持久化模型、命令与协议从 `iris.lifecycle` 导入；人工交互从 `iris.hitl` 导入；具体 Store 从 `iris.store` 导入。

## AgentRunner 的构建

`AgentRunner.from_config_path(path: str | Path, *, ...) -> AgentRunner` 读取 YAML 后装配。`AgentRunner.from_config(config: AgentConfig, *, config_path: Path | None = None, ...) -> AgentRunner` 使用已解析配置。二者共有的关键字参数如下；这是接口参数表，表中的省略记号仅表示共用本表，不是可执行 Python。

| 参数 | 类型 | 默认值与用途 |
| --- | --- | --- |
| `provider` | `CompletionProvider \| None` | `None`；默认按配置构建 provider，可注入自定义实现 |
| `permission_policy` | `PermissionPolicy \| None` | `None`；默认按配置构建权限策略 |
| `child_provider_factory` | `ChildProviderFactory \| None` | `None`；控制子 Agent provider 构造 |
| `memory_service` | `MemoryService \| None` | `None`；按 memory 配置决定是否挂载 |
| `prompt_source` | `PromptSource \| None` | `None`；默认初始化工作区 prompt 来源 |
| `observability` | `Observability \| None` | `None`；观测服务，见[观测参考](streaming-observability.md) |
| `decision_client` | `DecisionEvaluator \| None` | `None`；可选 Decision 评估边界 |
| `context_source` | `ContextSource \| None` | `None`；动态 context 来源 |
| `hooks` | `Sequence[HookRegistration]` | `()`；程序注册 Hook |
| `tool_middlewares` | `Sequence[ToolMiddleware]` | `()`；程序注册工具 Middleware |
| `store` | `LifecycleStore \| None` | `None`；显式传入优先，否则按 session 配置选择 |
| `observers` | `Sequence[RunEventObserver]` | `()`；已提交运行事件的观察者 |
| `observer_event_timeout_s` | `float` | `30.0`；单个 observer 处理单条事件的期限，要求有限正数 |
| `clock` | `Clock \| None` | `None`；默认 UTC 时间；扩展实现提供 `now() -> datetime` |
| `api_key` | `str \| None` | `None`；provider 构造的显式凭据 |
| `live_publisher` | `LivePublisher \| None` | `None`；同进程实时事实发布入口 |

`config_path` 提供配置相对路径基准；未提供时使用当前工作目录。`session.backend="none"` 构建 `InMemoryLifecycleStore`；`sqlite` 构建 `SQLiteStore`，省略数据库路径时使用 `.iris/session.db`。YAML 中数据库相对路径以配置文件目录解析。

低层组合入口为：

```text
AgentRunner(
    *,
    runtime: AgentRuntime,
    store: LifecycleStore,
    observers: Sequence[RunEventObserver] = (),
    observer_event_timeout_s: float = 30.0,
    clock: Clock | None = None,
    interaction_service: HumanInteractionService | None = None,
    live_publisher: LivePublisher | None = None,
)
```

以上是签名展示。自定义内核组合才需要直接构造 `AgentRuntime`；普通宿主优先使用两个工厂方法。[源码：Runner 构造与装配](../../src/iris/harness/runner.py)

### 资源生命周期

| 调用 | 行为 |
| --- | --- |
| `await runner.aprepare() -> None` | 准备 MCP、command 等资源；正常 start 会调用它。准备失败会关闭相关资源，此后需新建 Runner |
| `await runner.aclose() -> None` | 关闭 Runner 自有运行环境资源；仍有 active Activation 时抛出 `IrisRunStateError` |
| `runner.bind_maintenance(coordinator, *, memory=None, evolution=None) -> None` | 首次准备/运行前借用宿主的维护协调器；类型分别为 `MaintenanceCoordinator`、`MemoryMaintenanceBinding`、`ProjectEvolutionBinding` |

自动记忆生成与项目经验维护启用时，宿主必须在首次使用前完成对应维护绑定。协调器的独立生命周期与具体装配见[长期能力参考](memory-goals.md)。SessionManager 的关闭不代替 Runner 关闭；Runner 也不替宿主结束事件循环。

## 请求与固定运行选项

这些类型可从 `iris.harness` 或 `iris.lifecycle` 导入，是禁止未知字段的冻结 Pydantic 模型。

| 模型/字段 | 类型、默认值 | 规则 |
| --- | --- | --- |
| `AgentRunRequest.input` | `str \| list[DataBlock]`，必填 | 字符串去除首尾空白且不能为空；数据块必须至少有非空文字或图片 |
| `.session_id` | `str = "default"` | 非空会话身份 |
| `.run_id` | `str \| None = None` | 非空指定 ID，或由 Runner 生成 `run_...`；重复 ID 不是自动新建请求 |
| `.metadata` | `dict[str, Any] = {}` | 必须可严格 JSON 序列化 |
| `AgentRunOptions.limits` | `RunLimits()` | 整个 Run 的限额，resume/recover 不重置 |
| `.runtime` | `RuntimeExecutionOptions()` | 整个 Run 固定的执行选项 |
| `RunLimits.max_model_steps` | `int = 20` | 正数；模型调用前预留，包含恢复共享的逻辑模型步 |
| `.deadline_at` | `datetime \| None = None` | 带时区的绝对截止时间，规范化为 UTC |
| `.interaction_timeout_seconds` | `float \| None = None` | 正数；人工等待期限，None 不设该期限 |
| `RuntimeExecutionOptions.include_tools` | `bool = True` | 是否向本次执行开放配置的工具 |
| `.request_options` | `dict[str, Any] = {}` | JSON-safe 请求覆盖项；`tool_choice`、`response_format`、`provider_options` 按对应消息契约解析 |
| `.tool_timeout_seconds` | `float \| None = None` | 正数；工具执行超时选项 |
| `.tool_error_policy` | `ToolErrorPolicy = RETURN_TO_MODEL` | `return_to_model` 将工具错误交给模型；`stop` 结束本轮为 failed |

图片先用 `await runner.import_image(source: Path | bytes, *, session_id: str, name: str | None = None) -> ImageBlock` 导入。这个操作不创建 Run，也不占用会话运行位置；相对 source 路径以工作区解析。媒体完整契约见[媒体参考](media.md)。

## 启动、继续、取消与恢复

| 精确调用签名 | 返回和适用状态 |
| --- | --- |
| `await start(request: AgentRunRequest, *, options: AgentRunOptions \| None = None) -> RunResult` | 创建新 Run，推进到 waiting 或 terminal 才返回 |
| `await resume(run_id: str, *, interaction_id: str, response: HumanInteractionResponse) -> RunResult` | 提交当前 waiting Run 的对应回答，创建新 Activation，推进到下次 waiting 或 terminal |
| `request_cancel(run_id: str, *, reason: str \| None = None) -> RunSnapshot` | 同步保存首次取消请求；None 使用 `"cancel requested"`；重复请求不改写首次原因 |
| `await cancel(run_id: str, *, reason: str \| None = None, settlement_timeout: float \| None = None) -> RunResult` | 请求取消并观察 durable terminal；超时必须为正，None 无限等待 |
| `await recover(run_id: str, *, expected_activation_id: str \| None = None) -> RunResult` | active 接管必须给出准确 Activation ID；terminal 返回原结果；普通 waiting 应走 resume |

同一 Session 只容纳一个非终态 Run。start 的新输入不会绕过既有 active/waiting Run。`resume()` 沿用已保存的请求和运行选项；完整配置环境由当前 Runner 装配。

active 恢复检查当前 Activation 和版本，并要求当前 Runner 没有该 Run 的 live Activation。可继续的检查点会进入新执行段；已完成输出只差结算时直接终态；存在未结算工具 claim 时以 `outcome_unknown` 结束，不重放工具。waiting 已取消或已到期时可以由控制入口结算；已保存回答的子 Agent 代理另有可恢复继续路径。

`cancel()` 等待的是持久化终态，不保证原执行协程与 observer 已全部退出。直接使用 Runner 时还要等待原任务；Manager 宿主可使用 `close(cancel_run=True)`。

### 状态与结果

| `RunPhase` | 状态含义 |
| --- | --- |
| `active` | 有当前 Activation，正在推进或留下待接管的执行事实；`get_result()` 返回 None |
| `waiting` | 有 `pending_interaction_id`，没有当前 Activation；存在可读取的等待结果 |
| `terminal` | 有 `stop_reason`、`finished_at`，没有当前 Activation 和 pending Interaction |

`RunStopReason` 值为 `completed`、`failed`、`cancelled`、`deadline_exceeded`、`interaction_expired`、`budget_exhausted`、`outcome_unknown`。只有 terminal 才有 stop reason；waiting 不是失败也不是完成。

`RunResult` 字段：

| 字段 | 类型与含义 |
| --- | --- |
| `run` | `RunSnapshot`：当前 waiting/terminal 的持久快照 |
| `assistant_message` | `Msg \| None`：已提交的助手消息；有文本不等于 completed |
| `pending_interaction` | `HumanInteraction \| None`：waiting 时必有，terminal 时为空 |
| `error` | `RunErrorInfo \| None`：failed/outcome_unknown 必有，completed 为空 |

`RunSnapshot` 包含 `run_id`、`session_id`、`agent_id`、`phase`、`stop_reason`、`revision`、`current_activation_id`、`pending_interaction_id`、`cancellation_requested_at`、`cancellation_reason`、`limits`、`usage`、`checkpoint_sequence`、`last_event_sequence`、`created_at`、`started_at`、`updated_at`、`finished_at`。

`RunUsage` 包含 `model_steps_reserved`、`model_steps_committed`、`tool_calls_committed`、`input_tokens`、`output_tokens`、`total_tokens`，以及单独的 `compaction: TokenUsage`。`TokenUsage` 包含三种 token 计数，均默认 0。已预留不等于已收到完整响应；恢复会复用尚未提交的逻辑步骤。

`RunErrorInfo` 字段为 `code: str`、`message: str`、`source`、`details: dict[str, Any] = {}`。source 可取 `config/context/provider/tool/memory/session/runtime/lifecycle/persistence`。完整模型定义见[Lifecycle models](../../src/iris/lifecycle/models.py)。

### 查询

以下方法均属于 `AgentRunner`；除 Todo 外为同步只读。

| 方法 | 结果 |
| --- | --- |
| `get_run(run_id: str) -> RunSnapshot` | 运行快照；不存在抛 `IrisRunNotFoundError` |
| `get_run_control(run_id: str) -> RunControlSnapshot` | 轻量控制快照，避免加载请求和模型输出 |
| `get_result(run_id: str) -> RunResult \| None` | active 为 None，waiting/terminal 为结果；不存在抛错 |
| `get_session(session_id: str) -> SessionSnapshot` | 当前会话全部已提交消息与投影；不存在时为空快照 |
| `list_tool_calls(run_id: str) -> list[RunToolCallRecord]` | 现有 Run 的工具调用记录 |
| `list_events(run_id: str, after_sequence: int = 0, *, limit: int \| None = None) -> list[RunEvent]` | 读取序号严格大于游标的事件；None 不限制数量 |
| `await get_todo(session_id: str) -> TodoSnapshot` | 按需读取文件；Todo 未启用时抛 `IrisTodoError`，见[长期能力参考](memory-goals.md) |

`RunControlSnapshot` 只包含运行与会话 ID、phase、revision、当前 Activation、取消时间/原因、最后事件序号和更新时间。

`SessionSnapshot` 包含 `session_id`、`revision=0`、`messages=[]`、`compaction=None`、`context_window=None`、`forked_from_run_id=None`。摘要是模型视图的投影，原始消息仍在 `messages`；大量历史分页使用下面的 Store 查询，而不是先加载全部再截取。

## 人工交互类型

从 `iris.hitl` 导入：

| 类型 | 字段/取值 |
| --- | --- |
| `PermissionPrompt` | `kind="permission"`，`reason: str` |
| `QuestionPrompt` | `kind="question"`，`question: str`，`options: list[str] = []` |
| `PermissionInteractionResponse` | `kind="permission"`，`decision: Literal["approve", "reject"]` |
| `QuestionInteractionResponse` | `kind="question"`，`answer: str`，非空文本 |
| `HumanInteractionPrompt` | 上述两种 prompt 的判别联合 |
| `HumanInteractionResponse` | 上述两种 response 的判别联合 |
| `InteractionStatus` | `pending`、`resolved`、`closed` |

`HumanInteraction` 包含 `interaction_id`（默认生成）、`session_id`、`run_id`、`step_index`、`tool_call_id`、`status=pending`、`request`、`response=None`、`version=1`，以及 `created_at`、`expires_at`、`resolved_at`、`closed_at`、`close_reason`。`request` 是 `HumanInteractionRequest(tool_call, prompt, subagent_origin=None)`。

`ToolCallSnapshot` 保存 `tool_call_id`、`tool_name`、`arguments`、`workspace_root`、`fingerprint`；这些由执行系统构建，普通宿主应展示和回复收到的 Interaction，不重新构造批准对象。子 Agent 代理的 `SubagentProxyOrigin` 包含 `child_run_id`、`child_interaction_id`、`agent_selector` 和可选 `expiry_owner`。

响应 kind 必须匹配当前请求，并绑定准确 run/interaction。已经保存的回答不可被不同回答覆盖；部分同值重试会返回现有结果或继续未完的结算，不应把这一规则当成通用的“重复 start 自动去重”。[实现：HITL 模型](../../src/iris/hitl/models.py)、[响应校验和投影](../../src/iris/hitl/service.py)

## SessionManager：同进程输入准入

构造签名：

```text
SessionManager(
    runner: AgentRunner,
    session_id: str,
    *,
    max_pending_steer: int = 64,
    max_pending_follow_up: int = 64,
    max_buffered_submission_events: int = 256,
    max_tracked_durable_runs: int = 64,
    submission_publisher: LivePublisher | None = None,
    observation_mode: Literal["mixed", "broker_only"] = "mixed",
)
```

所有容量必须为正。Manager 只绑定此 Runner 与 Session，不扫描或接管 Store 中已有 active/waiting Run。队列和 submission 状态不持久化。

| 方法/属性 | 契约 |
| --- | --- |
| `await submit(input: str \| list[DataBlock], *, mode: Literal["steer", "follow_up", "auto"] \| None = None, options: AgentRunOptions \| None = None) -> SubmitReceipt` | 空闲要求 None；忙碌要求显式 steer/follow_up；auto 在锁内自动选空闲或 steer |
| `await resume(*, interaction_id: str, response: HumanInteractionResponse) -> RunResult` | 继续 Manager 当前 waiting Run，等待完整结果 |
| `await admit_resume(*, interaction_id: str, response: HumanInteractionResponse) -> ResumeReceipt` | 只等待继续执行被接纳或立即结算，之后从事件观察 |
| `await interrupt(*, reason: str \| None = None) -> RunSnapshot \| None` | 请求取消当前 Run，使其未投递 steer 失败；follow-up 保留。仅停止 Goal 意图时可返回 None；普通无当前 Run 时抛错 |
| `events() -> AsyncIterator[SessionEvent]` | mixed 模式唯一消费者，不重放 Manager 创建前的事件；close 后排空并结束 |
| `await close(*, cancel_run: bool = False, reason: str \| None = None) -> None` | 默认 detach；True 关闭准入、取消当前任务并等待自身管理的任务收尾 |
| `goal: GoalSession \| None` | Goal 启用时的会话控制入口，见[Goal 参考](memory-goals.md) |

显式 steer 不接收 options；auto 在忙碌分支忽略 options。follow-up 有独立的未来 Run ID，在当前 Run terminal 前没有真正创建 Run。waiting 仍属于忙碌。

### 回执、投递事件与容量

`SubmitReceipt` 字段为 `submission_id: str`、`run_id: str`、`mode: Literal["steer", "follow_up"] | None`、`state: Literal["pending", "delivered"]`。空闲提交返回 `mode=None, state="delivered"`，表示 Run 创建已提交；忙碌提交返回 pending，不能据此推断输入已进模型历史。

`SubmissionEvent` 有相同 ID 和 mode，`state` 为 `pending/delivered/failed`；failed 才有 `reason`，可取：

| reason | 含义 |
| --- | --- |
| `target_terminal` | 当前 Run 已终态，steer 未送达 |
| `target_cancelling` | 当前 Run 取消中，steer 不再接纳 |
| `session_closed` | Manager 关闭，待投递输入被放弃 |
| `commit_failed` | steer 在执行边界提交失败 |
| `start_failed` | follow-up 创建 Run 失败 |

`ResumeReceipt` 只含 `run_id`、`interaction_id`。`SessionEvent = RunEvent | SubmissionEvent | GoalChanged`：只有 RunEvent 使用持久序号；SubmissionEvent 只在进程内存在，GoalChanged 是目标观察投影。

mixed 模式保留有界 submission 缓冲和 durable 事件水位，按需从 Store 补读，不把无限量持久事件复制进内存。队列或相关缓冲容量不足会拒绝新输入。宿主应持续消费。`broker_only` 必须提供 `submission_publisher`，并禁用 `events()`；多客户端集成见[流式参考](streaming-observability.md)。

## SessionHistory：终态截点与分支

`SessionHistory(store: LifecycleStore)` 借用 Store，不接管其资源生命周期。

| 方法 | 结果和规则 |
| --- | --- |
| `list_fork_points(session_id: str, *, after: ForkPointCursor \| None = None, limit: int = 50) -> ForkPointPage` | 正数 limit；按 `(created_at, run_id)` 升序分页；不存在会话或无合格点时返回空页 |
| `get_at_run(source_run_id: str) -> RunHistorySnapshot` | 读取终态截点的全部原文前缀，不包括之后的轮次 |
| `fork(source_run_id: str) -> SessionSnapshot` | 自动生成 `session_...`，原子复制历史，每次成功调用创建不同分支；尚未创建 Run |

仅 **terminal 顶层 Run** 合格，所有 stop reason 均可。子 Agent Run 被排除；有子任务的父 Run 仍可合格。来源不存在抛 `IrisRunNotFoundError`，来源未终态或不是顶层抛 `IrisRunStateError`。

| 数据类型 | 字段 |
| --- | --- |
| `ForkPointCursor` | `created_at: datetime`、`run_id: str` |
| `ForkPoint` | `run_id/session_id/agent_id/input`、`stop_reason`、`created_at/finished_at`、`message_count` |
| `ForkPointPage` | `items: tuple[ForkPoint, ...]`、`next_cursor: ForkPointCursor \| None` |
| `RunHistorySnapshot` | `point: ForkPoint`、`messages: tuple[Msg, ...]`；不包含当前会话 CAS revision |

`ForkPoint.input` 是展示文本；纯图片输入会投影图片名称。`message_count` 是 Session 累计消息截点，不是该 Run 新增消息数。下一页使用上次返回的 next_cursor；None 表示没有后页。

新 Session 的 revision 为 0，`forked_from_run_id` 记录直接来源，继承截点消息及该 Run 结束时的摘要。context_window 为空，首次新输入重新建立；旧 Run 的 Activation、工具执行事实、Interaction、预算不复制。消息里的图片引用保留，不复制图片文件。来源 Session 后续继续运行不影响分支点。

[实现：SessionHistory](../../src/iris/harness/session_history.py)、[历史查询数据类型](../../src/iris/lifecycle/history.py)、[共享来源资格规则](../../src/iris/store/_session_history.py)

## RunEvent 与工具调用事实

`RunEvent` 在相应 aggregate mutation 的同一事务中追加。字段为 `run_id`、`session_id`、`sequence`、`kind`、`occurred_at`、`activation_id=None`、`step_index=None`、`correlation_id=None`、`payload={}`；sequence 从 1 开始，属于**单个 Run**。

`RunEventKind`：`run.started`、`activation.started`、`model_step.reserved`、`model_step.committed`、`context.compacted`、`tool_call.claimed`、`tool_call.committed`、`tool_call.outcome_unknown`、`interaction.suspended`、`interaction.resolved`、`run.cancellation_requested`、`activation.abandoned`、`run.terminal`。

`RunEventObserver` 协议只有 `async on_event(event: RunEvent) -> None`。Runner 在结算后投递已提交事件；同一 observer 保序，不同 observer 可并行。单事件超时和普通回调异常记录后继续，不改变已经提交的结果。实时 token/tool 展示使用 live publisher，不靠 observer 回调；两者区别见[流式与观测参考](streaming-observability.md)。

`RunToolCallRecord` 保存 run/step/ordinal、工具调用 ID、工具名、参数、fingerprint、关联 Interaction、phase、claim Activation、result、version 和时间。phase 为 `prepared/claimed/committed/outcome_unknown`，committed 才有确定工具结果；claimed 的记录不证明工具成功。

## 自定义 LifecycleStore

从 `iris.store` 导入 `InMemoryLifecycleStore()` 或 `SQLiteStore(path: str | Path)`。SQLite 构造会按需创建父目录和当前 schema，打开已有库时要求 schema 精确匹配；不自动迁移旧版本。当前 SQLiteStore 没有公开 `close()` 方法。

扩展实现应满足 `iris.lifecycle.LifecycleStore` 的完整同步协议，不能只实现消息读写。Runner 直接在运行路径调用它；远程慢 I/O 并不会因 Protocol 而自动异步化。`source_id: str` 是持久来源身份，SQLite 重开保持不变，进程内实现具有自身身份。

### 读协议

除下面特别说明，所有方法均为同步调用：

| 方法签名 | 返回/语义 |
| --- | --- |
| `load_run(run_id: str)` | `RunRecord \| None` |
| `load_run_control(run_id: str)` | `RunControlSnapshot \| None` |
| `load_session(session_id: str)` | `SessionSnapshot`，不存在时空快照 |
| `load_session_header(session_id: str)` | `SessionHeader(session_id, revision, message_count, context_window)` |
| `load_session_revision(session_id: str)` | `int`，不存在返回 0，不加载消息 |
| `load_run_context(run_id: str, *, include_tool_discovery: bool)` | `SessionContextSnapshot`：同版本 header、compaction、raw_tail、保护位置与可选工具发现状态 |
| `read_session_messages(session_id: str, *, start: int, limit: int)` | `SessionMessagePage(items, next_index, total_count)`；items 为 `(绝对索引, Msg)` 元组，start 从 0 开始 |
| `load_run_message_slice(run_id: str, after_count: int = 0, *, limit: int = 128)` | `RunMessageSlice`，本 Run 已提交的有限消息区间；计数使用 Session 累计位置 |
| `load_session_lane(session_id: str)` | `str \| None`，当前非终态 Run ID |
| `load_interaction(interaction_id: str)` | `HumanInteraction \| None` |
| `load_checkpoint(run_id: str)` | `RunCheckpoint \| None` |
| `load_tool_call(run_id: str, tool_call_id: str)` | `RunToolCallRecord \| None` |
| `list_tool_calls(run_id: str, *, step_index: int \| None = None)` | `list[RunToolCallRecord]`，按 step/ordinal 排序 |
| `load_result(run_id: str)` | `RunResult \| None` |
| `list_events(run_id: str, after_sequence: int = 0, *, limit: int \| None = None)` | `list[RunEvent]`，严格大于游标 |
| `load_subagent_link(parent_run_id: str, parent_tool_call_id: str)` | `SubagentRunLink \| None` |
| `list_fork_points(session_id: str, *, after: ForkPointCursor \| None = None, limit: int = 50)` | `ForkPointPage` |
| `load_session_at_run(source_run_id: str)` | `RunHistorySnapshot` |

`RunMessageSlice` 含 source/run/session ID、initial/start/end/terminal 消息计数、outcome 与 messages；读取上界为本 Run 终态截点，尚未终态时为当前已提交范围。消息分页、上下文读取及其 revision 必须来自一致读取快照。

### 写协议与命令

所有写方法只接受一个相应命令对象；命令类型从 `iris.lifecycle` 导入。除标注外返回 `RunCommit`：

| 方法 | 命令类型 | 职责 |
| --- | --- | --- |
| `create_run` | `CreateRun` | 原子创建 Run、Activation、Checkpoint 与会话运行占位 |
| `commit_run_input` | `CommitRunInput` | 提交输入及首次上下文窗口 |
| `reserve_model_step` | `ReserveModelStep` | 返回 `ModelStepReservationResult`，先预留模型步数 |
| `commit_model_step` | `CommitModelStep` | 完整响应、usage、工具准备事实和下一检查点 |
| `record_compaction_usage` | `RecordCompactionUsage` | 记录压缩调用用量 |
| `commit_compaction` | `CommitCompaction` | 原子更新摘要、窗口与检查点 |
| `claim_tool_call` | `ClaimToolCall` | 工具体执行前登记 claim |
| `commit_tool_result` | `CommitToolResult` | 结果、消息增量、工具状态与检查点 |
| `suspend_run` | `SuspendRun` | 保存 Interaction 并进入 waiting |
| `resolve_interaction` | `ResolveInteraction` | 保存准确 Interaction 的回答 |
| `resume_waiting_run` | `ResumeWaitingRun` | 从 waiting 绑定新 Activation |
| `request_cancellation` | `RequestCancellation` | 首次取消事实，不等于终态 |
| `finish_run` | `FinishRun` | 终态、结果、消息截点与运行位置释放 |
| `recover_active_run` | `RecoverActiveRun` | 按精确 Activation 身份与版本接管/结算 |
| `admit_child_run` | `AdmitChildRun` | 原子创建子 Run 与父工具关联，返回 `SubagentRunLink` |
| `rebind_subagent_proxy` | `RebindSubagentProxy` | 更新父侧子任务等待代理 |
| `finalize_subagent_result` | `FinalizeSubagentResult` | 提交子结果为父工具结果并推进父检查点 |
| `fork_session` | `ForkSession(source_run_id, target_session_id, now)` | 原子创建独立历史分支，返回 `SessionSnapshot` |

命令携带相关 Run revision、Session revision、Activation ID、Checkpoint sequence 或工具/Interaction version，Store 在同一 mutation 内检查受影响状态并原子写入。`RunCommit` 是提交收据，不是新的持久化 owner；它包含更新后的 `run`、可选 `session_revision/checkpoint/interaction/result` 与本次 `events`。

`RunCheckpoint` 当前 `checkpoint_version=4`，包含 run/activation ID、sequence、engine_cursor、session_revision、预留和已提交模型步数、resumability。它是恢复载荷版本，不是 SQLite schema 版本。`RunRecord` 还保留初始/终态消息计数与终态摘要，支持稳定的历史截点。

具体命令字段与约束以[Store Protocol 和命令定义](../../src/iris/lifecycle/store.py)为唯一完整定义；实现者可复用[两种 Store 共用的契约测试](../../tests/store/test_lifecycle_store_contract.py)。启用 Goal 时，注入的 Store 还需满足 [GoalStore](../../src/iris/goal/store.py) 协议，不能用只有基础 LifecycleStore 的实现承诺 Goal 支持。

## 面向内核实现者的 Runtime 接口

`iris.runtime` 的 `AgentRuntime(environment: RuntimeEnvironment)` 只执行一次 Activation，入口为：

```text
await execute(
    activation: RuntimeActivationInput,
    *,
    commits: RuntimeCommitPort,
    cancellation: CancellationSignal,
    steering: RuntimeSteeringPort | None = None,
    stream_sink: RuntimeEventSink | None = None,
) -> RuntimeActivationResult
```

`RuntimeActivationInput` 绑定 `run_id/activation_id/session_id`、`kind="start"|"resume"|"recover"`、`run_input`、`initial_session_message_count`、`cursor`、`options`，以及可选 `interaction_projection`。后者为已解析的 `ToolResult` 或 `RuntimeApprovedToolCall`，不由 Runtime 自行向 UI 获取回答。

`RuntimeCursor.position` 为 `before_input/before_model/tool_batch/outcome_ready`，配合 `step_index`、`visible_tool_names`、`todo_reminder_step`、`next_tool_index`、工具调用/结果前缀、助手消息和读取状态表达准确位置。Runtime 通过 `RuntimeCommitPort` 提交事实后才推进位置；Store 到这个端口的适配由 harness 的 [StoreRuntimeCommitPort](../../src/iris/harness/_commit_port.py) 实现。

`RuntimeActivationResult` 包含 outcome、cursor、可选 assistant_message、suspension、error 与失败调用用量。outcome 可为 `completed/suspended/budget_exhausted/cancelled/deadline_exceeded/failed/outcome_unknown`；它描述执行段结束原因，必须经过 Runner 结算才能作为 durable `RunResult` 对宿主发布。

`RuntimeFactory.from_config_path()` / `from_config()` 只组装内核依赖。与 Runner 工厂相比，它们接受 `context_access: ContextAccessPort | None`，不接管 Store；context policy 启用时要求可用的回读端口。子 Agent 或 Goal 配置要求使用 Runner 工厂，不能通过低层 Factory 单独构建完整运行。这些接口适合修改内核、构造测试替身或承担完整 harness 职责的实现者；常规应用直接使用 AgentRunner。

完整低层类型与端口见 [Runtime models](../../src/iris/runtime/models.py)、[commit port](../../src/iris/runtime/commit.py)、[factory](../../src/iris/runtime/factory.py)。

## 异常与执行失败的区别

| 错误 | 正常使用中如何理解 |
| --- | --- |
| `IrisRunNotFoundError` | 查询/操作的 Run 或关联记录不存在 |
| `IrisRunStateError` | 操作不适用于当前状态，例如 active 使用 resume、关闭仍活跃的 Runner |
| `IrisRunConflictError` | 会话已占用、身份或版本变化、Activation 接管冲突 |
| `IrisRunRecoveryError` | 恢复载荷缺失或无法按当前契约解释 |
| `IrisRunPersistenceError` | 读取/提交持久化失败，不能据异常推定状态已落库 |
| `IrisLifecycleSchemaError` | SQLite schema 不符合当前实现 |
| `IrisRunObservationTimeoutError` | cancel 等待终态超时；不改变已经保存的运行事实 |
| `HITLResponseMismatchError` / `HITLConflictError` | 回答类型/身份不匹配，或试图用不同回答覆盖已保存回答 |
| `IrisCommandCleanupError` | 命令资源清理未完成，见[命令配方](../cookbook/commands.md) |

受控的模型、工具或上下文执行失败通常返回 `stop_reason="failed"` 和结构化 `RunErrorInfo`，不会全部以 Python 异常逃逸。工具默认错误策略还允许模型继续处理错误。区分“接口拒绝本次操作”“本次观察失败”和“Run 已经终态失败”，再决定宿主展示或后续动作。

实现与设计导航：[Runner](../../src/iris/harness/runner.py)、[SessionManager](../../src/iris/harness/session_manager.py)、[生命周期设计](../design/lifecycle.md)、[人工交互设计](../design/human-interaction.md)。
