[English](README.en.md)

# `iris.lifecycle`

公开 Sub Agent 契约以 `SubagentRunLink(parent_run_id, parent_tool_call_id, child_run_id)`
关联独立运行；parent 工具保持 PREPARED。Store 增加 `AdmitChildRun`、`RebindSubagentProxy`、
`FinalizeSubagentResult` 与 exact link point read。Rebind 返回完整 WAITING `RunCommit`；
WAITING finalize 在同一 commit 返回绑定 fresh RESUME activation 的 ACTIVE run/checkpoint，
调用方无需追加普通 resume mutation。Child usage 保持 run-local。
四个类型直接从 `iris.lifecycle` 导入。Rebind 只更新 proxy binding 与 checkpoint sequence，
保持 history/usage/cursor；Finalize 才提交 parent result/message/usage/cursor。三条 mutation
沿用既有 revision、activation fence 和 store 事务，不增加新的并发字段。

`iris.lifecycle` 是 logical run 的纯数据与同步 store contract。它定义不可变 run/session/
activation/checkpoint/tool-call/event/result 边界模型、JSON-safe validation、投影函数和 CAS
commands，但不拥有运行控制流或具体数据库。持久化模型使用 Pydantic 校验 raw/load 数据；
同进程 command 使用 frozen slots dataclass，只携带调用方已经类型化的事实。

## 依赖边界

```text
harness -> lifecycle <- store
runtime  -> lifecycle
```

Lifecycle 不 import `iris.harness`、`iris.runtime` 或 `iris.store`。`AgentRunner` 是 owner，
`InMemoryLifecycleStore`/`SQLiteStore` 是实现，`AgentRuntime` 只消费 options/error contracts。

## Aggregate 不变量

- 一个 session 同时最多一个 non-terminal run lane；
- active run 恰好一个 current activation fence；waiting run 恰好一个 open interaction；
- model step 先 reserve 再 commit，最多一个未提交 reservation；
- tool effect 先 claim，再 commit result；unresolved claim 不得重放；
- terminal run 没有 current activation、open interaction 或 lane；
- `RunRecord.terminal_session_message_count` 在首次 terminal settlement 时记录包含工具闭合消息的
  session 累计消息数，此后保持不变；terminal 必须为非负整数，non-terminal 必须为 `None`；
- terminal run 的 durable history 中，每个 `tool_use` 都恰好有一个匹配的 result；tool-call phase
  继续区分已提交结果、结果未知与从未开始，不能用合成 closer 抹去副作用知识；
- run、checkpoint、session revision 与 usage counters 必须交叉一致；
- mutation events 与 aggregate facts 同事务追加，sequence 单调递增。

## Checkpoint v2

`RunCheckpoint.checkpoint_version` 固定为 `2`，加载时拒绝旧版本，不迁移历史 checkpoint。
新 run 从 `before_input` 开始；输入组保存后进入 `before_model`，恢复位置明确区分输入是否已归档。

`RunCheckpoint.resumability` 只有：

- `safe`：可以从 cursor 重新进入 engine；
- `outcome_ready`：assistant outcome 已提交，只补 terminal；
- `blocked_unknown`：effect 结果不可安全解释，禁止自动执行。

checkpoint 只接受当前 payload 形状，也不保存 provider client、task、lock、signal 或 callback。

## Store contract

`LifecycleStore` 提供 create/begin/reserve/model commit/tool claim/tool result/suspend/resolve/
cancellation/finish/recover commands，以及 run/session/lane/interaction/checkpoint/tool/result/event
reads。
`CommitRunInput` / `commit_run_input()` 在既有 run/session CAS 和 activation fence 下原子追加
BCI、用户输入并初始化上下文窗口，推进 checkpoint sequence 到 `before_model`。只改变 cursor
位置，不改变 step index、usage 或 model reservation，也不追加模型事件。重复旧 command 冲突。
`RunCommit.session_revision` 只在 mutation 改变原文、摘要投影或上下文窗口时返回提交后的 revision，不携带完整
`SessionSnapshot`；需要 history 时显式调用 `load_session()`。Store 不承诺历史 command 原样
重交成功；已有相同回答、取消和 child admission 的业务状态幂等按各 mutation 契约保留。
Run 状态 mutation 按各自契约携带 expected revision/fence；stale writer 必须 conflict，而不是覆盖新事实。
`ResolveInteraction` 携带 run ID、当前 interaction ID、expected run revision、expected interaction
version、typed response 和时间，不再接收 `expected_fingerprint`。待回答写入检查 revision/version；
WAITING 已保存相同回答时返回当前事实与空 events，不同回答冲突。工具执行的参数/workspace 指纹仍保留。
Store 只校验当前 mutation 影响的 phase、counter、identity 与 fence，再应用 typed delta；不会为了
更新单个字段而把整个已验证 aggregate `model_dump()` 后重新 `model_validate()`。SQLite row 与
checkpoint recovery 仍是完整验证边界，JSON-safe 约束仍由 durable model/encoder 保证。
`SessionSnapshot` 公开 `session_id`、CAS `revision`、完整 `messages`、可空 `compaction`、
`context_window` 和直接来源
`forked_from_run_id`；后续追加保留来源。revision 每次非空 message delta 只推进一次，与消息条数
无关。终态截点记录在 `RunRecord`，不能用 session revision 替代；持久化 ordinal 不进入公共模型。

### 固定上下文窗口

`SessionSnapshot.context_window: SessionContextWindow | None` 为 `None` 时尚未初始化；
`SessionContextWindow()` 则表示已初始化、没有 memory 文本。窗口保存实际采用的 `memory_overview`、
`mode`（`full` 为核心事实与知识范围，`navigation` 为仅知识范围）和 `sources` tuple。
每个 `MemoryOverviewSource` 包含
`namespace`、`path`、可空 `source_revision`；来源只解释历史快照，不赋予当前工具读取权限。

`CommitRunInput.initial_context_window` 首次必须显式传窗口，之后必须为 `None`，以保留已采用文本。
输入组、窗口、session revision 和 checkpoint 同事务提交；空消息初始化仍推进一次 revision，
消息与窗口一起变化也只推进一次。
后续 run、HITL 和恢复复用该窗口；只有成功的 `CommitCompaction.context_window` 会替换它。
取消、失败或 CAS 冲突均保留旧窗口。Checkpoint v2 通过 session revision 绑定窗口，不复制其正文。
`RuntimeExecutionOptions` 不再接受 memory 查询、结果快照或字符预算，读取与选择由 runtime 装配负责。

### 摘要状态与用量

`SessionCompaction(summary, covered_message_count)` 保存完整 Markdown 正文和覆盖的原文前缀
长度 `[0,c)`；消息原文继续保留。`RunRecord.initial_session_message_count` 在创建事务内记录
当前 run 的起点，首次终态同时冻结 `terminal_compaction` 与消息截点，之后的 session 压缩不改变它。
`RunCheckpoint` 不复制摘要，恢复通过 session revision 绑定该投影。

`RunUsage.compaction: TokenUsage` 独立保存摘要的 input/output/total；原有 token 字段仅记主模型。
全部用量逐字段相加推导，child 消耗仍属于 child。Provider 的 total 值原样保存，不假定它必然等于
input+output。

- `RecordCompactionUsage` / `record_compaction_usage()`：每份 response 返回后保存用量；只推进
  run revision 和更新时间，返回空 events，不改变 checkpoint、session 或主步骤。
- `CommitCompaction` / `commit_compaction()`：在 SAFE/before_model 且已有一个 pending 主步骤时，
  原子替换摘要与必传的 `context_window`、推进 session/run revision 和 checkpoint sequence，
  并追加 `context.compacted`。
  Cursor、原文、主 reservation 与 usage 保持不变。事件仅含覆盖长度和前后输入估算。

两者沿用 active fence 和 CAS，旧 revision 重交会冲突；不承诺跨进程外部模型调用的 exactly-once 计费。
`load_session_lane()` 只是 lane owner 的只读发现入口，不承担恢复、修补或 ownership transfer。
`load_tool_call(run_id, tool_call_id)` 按 exact composite identity 返回单条 tool fact；
`list_tool_calls(run_id, step_index=...)` 将有序工具读取限定为一个模型步，省略时返回整 run；
`load_run_control(run_id)` 只返回 `RunControlSnapshot` 的 session 归属与 fence/cancellation 字段。
gateway 可以据此确认 run 属于所请求的 session，无需加载完整 run。上述读取都不
替代 mutation CAS，也不改变同步 store boundary。

## 会话历史契约

`history.py` 定义四个 frozen slots dataclass，均从 `iris.lifecycle` 导出：

- `ForkPointCursor(created_at, run_id)`：分支点分页位置；
- `ForkPoint`：run/session/agent identity、原始 input、stop reason、创建与结束时间及 `message_count`；
- `ForkPointPage(items, next_cursor)`：分支点 tuple 和下一页游标；
- `RunHistorySnapshot(point, messages)`：指定 run 末尾的历史预览，messages 为独立消息对象的 tuple，
  不携带当前 session 的 CAS revision。

`LifecycleStore` 提供三个同步方法：

| 方法 | 返回值 | 契约 |
| --- | --- | --- |
| `list_fork_points(session_id, *, after=None, limit=50)` | `ForkPointPage` | 按 `(created_at, run_id)` 升序，`after` 接受 `ForkPointCursor`，`limit > 0` |
| `load_session_at_run(source_run_id)` | `RunHistorySnapshot` | 读取 terminal 消息截点内的完整已提交前缀 |
| `fork_session(command)` | `SessionSnapshot` | 接受 `ForkSession` command，原子创建新 session |

来源必须是 terminal 顶层 run；全部 `RunStopReason` 均可使用。有 inbound `SubagentRunLink` 的
child 被排除，拥有 outgoing child link 的 parent 仍可使用。来源 session 正在运行后续轮次时也可
fork，截点不随新消息增长。来源资格、分页参数与目标 identity 由 store 负责检查。

`ForkSession` 是从 `iris.lifecycle` 导出的 keyword-only frozen slots command，携带
`source_run_id`、`target_session_id` 和 `now`。
新 session 的 `revision=0`，`forked_from_run_id` 指向直接来源；继承消息不占用新 revision，
后续非空追加从 1 开始且保留来源。Fork 同时继承来源 run 冻结的 `terminal_compaction`，
不读取来源 session 的最新摘要。目标 `context_window=None`，第一条输入选择新的窗口。
Fork 不创建 run、activation、checkpoint、
tool execution fact、interaction、event 或 lane，也不恢复源执行位置。

## 公开接口

`CreateRun` 的 request/checkpoint identity 及 `RecoverActiveRun` 的 disposition/activation 组合
由公开 command 构造时校验；store 不重复检查这些已确定的 optional 关系，只检查当前 durable
事实及 mutation 所需的 CAS、identity 与 fence。

`iris.lifecycle` 可直接导入所有契约模型、enums、commands、`LifecycleStore` 和
`snapshot_run()`/`project_result()`，包括最小只读投影 `RunControlSnapshot`。完整运行入口只在
`iris.harness`。

## 验证

```bash
uv run pytest tests/store tests/harness
uv run ruff check src/iris/lifecycle
uv run mypy src/iris/lifecycle
```
