[English](README.en.md)

# `iris.store`

`iris.store` 提供 `iris.lifecycle.LifecycleStore` 的两个同步实现：进程内的
`InMemoryLifecycleStore` 和基于 Python 标准库 `sqlite3` 的 `SQLiteStore`。它们共享同一个
logical-run aggregate 契约，统一管理 session revision、run、activation、checkpoint、tool
call、interaction 和 event。

本包只负责具体存储；领域模型和 command/read 协议定义在 `iris.lifecycle`。它不调用
provider、不执行工具，也不承担 `iris.memory` 的长期记忆。运行要求 Python 3.12+。

## 快速入门

```python
from iris.store import SQLiteStore

store = SQLiteStore(".iris/lifecycle.db")
session = store.load_session("default")
print(session.revision, session.messages)
```

`SQLiteStore(path)` 只接受不存在/零字节的数据库，或者精确匹配 lifecycle schema v8 的
文件。新数据库会创建父目录和完整 schema；旧 schema、缺表/多表、索引或版本差异都会在
任何写入前抛出 `IrisLifecycleSchemaError`。旧数据库不受支持，应为新 store 选择新的数据库路径；
constructor 不重置或修改原文件。

## 实现架构

Sub Agent 新增且只新增三条 mutation：`admit_child_run()` 同事务创建普通 child run 与
`subagent_run_links` 三字段 link；重入 exact parent key 返回原 child。`rebind_subagent_proxy()`
保持 parent 工具 PREPARED、history/usage/cursor 不变，返回完整 WAITING snapshot。
`finalize_subagent_result()` 要求 child 已 terminal，提交一次 parent tool result/message/usage；
WAITING 模式同时关闭 proxy、建立 fresh RESUME activation，并返回其 checkpoint。
无回答的 PENDING proxy 仅在 child-owned 到期且 child 已结算后允许 finalize。
两个实现共用 `_subagent.py` 的操作边界检查；SQLite link 没有额外状态列或显式 index。

`InMemoryLifecycleStore` 和 `SQLiteStore` 是两个独立、平级的 protocol 实现。内存实现以一把
`RLock` 管理进程内 facts，并对输入/输出做深拷贝隔离；内部追加只复制 list 容器与新 delta，
不重复复制 store-owned 旧消息。数据随进程退出丢失。SQLite 实现不导入或调用内存实现。

`SQLiteStore` 每次操作打开独立连接并启用 foreign keys。公共 read 使用 targeted query，只
读取目标 run、session、lane owner、interaction、checkpoint、tool calls 或 events；跨多条查询的
read 在同一个 deferred transaction 中取得一致 snapshot，且不执行写入。
SQLite row decode 已创建独立对象，public read 直接返回解析结果，不再做统一 deepcopy；
内存实现继续通过 deepcopy 隔离 store-owned facts。
exact tool call 读取复用现有 `(run_id, tool_call_id)` 主键；run control 读取只选择
`RunControlSnapshot` 所需的 session 归属与控制字段，不解码 request/options/usage/message/error
JSON。内存实现以
per-run call-ID 索引列举目标 run，权威事实仍保存在原有 tuple-key dict。

Mutation 使用 `BEGIN IMMEDIATE`，只加载当前 command 校验和变更所需的 rows。run、session、
checkpoint、tool call 与 interaction 更新分别使用 revision、sequence 或 version CAS
predicates；lane、activation、interaction、tool facts 与 run 在同一事务中增量写入，events
保持 append-only。Active history mutation 由 active precondition 在同一事务内只读取一次 lane
fence，随后 history precondition 只检查 session revision。任一 SQL 失败都会触发完整 transaction
rollback，不暴露半更新状态。
两个 store 共用 lifecycle typed transition helper：mutation 先检查受影响的 phase/fence/delta，
再对已验证模型应用 `model_copy(update=...)`。完整 `model_validate()` 只用于 SQLite row decode 等
load/recovery 边界；durable JSON 投影使用 store 私有 serializer。
schema v8 的 `sessions` 保存 revision、message count、更新时间、可空的 `forked_from_run_id`
及 `compaction_json` 摘要投影、`context_window_json` 固定窗口；后续追加保留直接来源、摘要和窗口。
消息按连续 ordinal 追加到
`session_messages`。非空 delta 只序列化并插入本次消息，同时以 revision + message count 双条件
CAS 推进 metadata；完整 `SessionSnapshot` 读取仍按 ordinal 重建并校验 `1..message_count`。
Mutation 的 `RunCommit` 只携带发生变化的 `session_revision`，不为生成回执重读完整 history。

创建 run 在同一事务内记录 `initial_session_message_count`。首次 terminal settlement 在
`RunRecord.terminal_session_message_count` 记录 session 累计消息数，包括本次工具闭合消息，
并在 `terminal_compaction` 冻结当时的摘要；没有摘要时为 `None`。后续读取保持该快照。
创建时 deadline、预算耗尽、waiting 取消、普通
finish、`OUTCOME_UNKNOWN` recovery 和 `FINALIZE` recovery 均写入该字段。没有 checkpoint 或
没有 closer 时也记录实际消息数，0 合法。SQLite 使用已读取的 session metadata 计数，无需加载完整历史。

Store 不缓存完整 command，也不承诺历史写入原样重交成功。每次 mutation 都按当前状态、
revision/CAS 和 activation fence 执行；旧写入通常抛出冲突或状态错误，不重复追加事实。
已有业务状态幂等仍保留：WAITING 已回答 interaction 的相同 response、同 activation 的相同
未结算取消请求、同 parent/tool 的 child admission。调用方通过现有 read/recovery 接口确认结果。
`resolve_interaction` 先匹配当前等待的 interaction identity 和 response kind，再对 PENDING
写入检查 run revision 与 interaction version；RESOLVED 的同回答直接返回当前事实。

`agent_runs.usage_json` 是 run usage 的唯一存储，不再并存三个重复的标量计数列。首次读取 row
时由既有 `RunUsage` 解析校验非负计数及 committed/reserved 关系。当前数据库为 schema v8，
run 与 checkpoint 不再保存环境总指纹；不迁移或读取旧 schema。

schema v8 包含：

- `lifecycle_schema`、`sessions`、`session_messages`、`agent_runs`、`session_run_lanes`；
- `run_activations`、`run_checkpoints`、`run_tool_calls`；
- `subagent_run_links(parent_run_id, parent_tool_call_id, child_run_id)`；
- `run_interactions`、`run_events`；
- partial unique index `one_open_interaction_per_run`。
- terminal partial index `terminal_runs_by_session(session_id, created_at, run_id)`。

`agent_runs.terminal_session_message_count` 的 SQL 约束要求 terminal 非空、non-terminal 为空且计数
非负；`sessions.forked_from_run_id` 可空并引用来源 `agent_runs.run_id`。

`session_messages` 的 `(session_id, ordinal)` composite primary key 已覆盖有序读取，不额外增加
index。session revision 随非空原文 delta 或摘要投影提交推进，不等于 message count。

SQLite 连接/序列化/腐坏 row 错误映射为带 `path` 和 `operation` context 的
`IrisRunPersistenceError`；预期 facts 已变化或数据库约束竞争使用 lifecycle conflict/state
错误。

## 公开接口

### Run 输入归档

`commit_run_input(CommitRunInput)` 在同一锁或 SQLite 事务内追加 BCI/user 输入组、初始化窗口，并把 checkpoint
从 `before_input` 推进到 `before_model`。它沿用 run revision、session revision、activation fence
和 checkpoint sequence；不消耗模型 reservation，不改变步骤索引、usage 或 event sequence。
旧 command 重交会冲突；SQL 失败整体回滚，恢复不会看到部分输入或单独更新的窗口。
Checkpoint payload 版本维持 `2`；lifecycle schema 升为 `8`，拒绝旧库且不执行迁移。

`SessionSnapshot.context_window=None` 表示未初始化；显式 `SessionContextWindow()` 表示已初始化且
没有 memory 文本。首输入的 `initial_context_window` 必须传实际采用窗口，后续输入必须为 `None`。
窗口保存概览正文、full/navigation 模式与 namespace/path/revision 来源，独立于消息历史；
full 包含核心事实与知识范围，navigation 仅包含知识范围。
后续 run、HITL 和恢复持续读取已提交值。窗口初始化和输入共用一次 revision 推进，checkpoint
通过该 revision 关联窗口。空消息初始化仍推进一次，消息与窗口同时变化也只推进一次。
来源只是历史元数据，工具读取范围仍由当前 runner 配置决定。

### 摘要用量与历史投影

`record_compaction_usage(RecordCompactionUsage)` 在当前 activation fence 和 run CAS 下累加
`RunUsage.compaction`，只推进 run revision 和更新时间。主调用 token 字段、主步骤预算、session、
checkpoint 和 event sequence 不变，回执 `events=()`。两组 token 消耗按字段相加即可取得总量；
child run 的用量不复制到 parent。

`commit_compaction(CommitCompaction)` 在 `SAFE/before_model` 且已有一个 pending 主步骤时，
原子替换 `SessionSnapshot.compaction` 与必传的 `context_window`、推进 session/run revision 和 checkpoint sequence，
追加一个 `context.compacted` 事件。原文、cursor、主步骤 reservation 和 usage 保持不变。
它检查选区时的 session revision、activation fence、取消事实及覆盖范围前移，工具切点和
token 额度由 runtime 负责。事件只含覆盖条数和前后输入估算，不含摘要正文。
取消、候选失败或 CAS 冲突时保留旧窗口；SQLite 后续写入失败时，摘要和窗口一起回滚。

摘要已提交、主响应尚未提交时，恢复仍使用该摘要、窗口与原 pending reservation。两条 mutation
按当前 revision 提交，旧 command 重交会冲突；不承诺跨进程外部模型调用只计费一次。

### Store 与查询

`iris.store` 顶层导出：

- `InMemoryLifecycleStore`：用于测试和单进程运行；
- `SQLiteStore`：只接受 schema v8 的持久化 `LifecycleStore` 实现。

两者实现 `iris.lifecycle.LifecycleStore` 的 create/begin/reserve/commit/claim/suspend/resolve/
finish/recover/cancel commands 及 run/session/lane/checkpoint/tool/interaction/event/result reads。
应通过 `iris.lifecycle` 构造 command 和模型，不依赖 `iris.store` 中的下划线模块。

`load_tool_call()` 的 composite key 不存在时返回 `None`，即使 run 不存在；
`load_run_control()` 与 `load_run()` 一样在 run 不存在时返回 `None`。`list_tool_calls()` 仍在 run
不存在时抛出 `IrisRunNotFoundError`，并保持 `(step_index, ordinal)` 排序。这些定向 read 没有增加
额外索引或连接池，schema identity 为 lifecycle v8。
`list_tool_calls(run_id, step_index=...)` 只返回指定模型步的工具事实；SQLite 在同一连接中将
条件下推到 SQL。prepared batch 使用该限定查询，HITL resume 使用 exact tool-call read。

`list_events(run_id, after_sequence=0, limit=None)` 始终按 sequence 返回；`limit` 如提供必须是正
整数。内存实现先定位游标再复制有限 slice，SQLite 实现把 `LIMIT` 下推到查询，避免分页 consumer
在每轮读取中物化全部剩余 events。

`load_session_lane(session_id)` 只读返回当前 non-terminal lane owner 的 `run_id`，无占用时返回
`None`。它不修复、恢复或接管 run；host 仍需读取 run/interaction，并用精确 activation fence 调用
`recover()`，或用精确 interaction identity 调用 `resume()`。

取消请求、waiting settlement、activation abandon/rebind、outcome-ready finalize 与 unresolved
claim -> outcome unknown 都在 aggregate transaction 内完成。runtime 不包含旧 schema reader、
dual write 或 compatibility adapter；不兼容文件直接拒绝。

同一 active activation 可以在提交任何 result 前持有多个 exact durable claims；每条 claim 仍绑定
step、ordinal、call ID、fingerprint 和 version。durable cancellation 先提交时，store 拒绝新的
claim 且不追加 claim event；claim 先提交时，该调用只能提交明确 result，或在 terminal/
recovery transaction 中与其他 unresolved claims 一起原子关闭为 outcome unknown，绝不重放。

effect 前的预检失败与 `CIRCUIT_OPEN` 熔断结果允许直接从 `PREPARED` 提交，不产生 claim
event；两个 store 使用 `_tool_results.py` 的同一分类规则。Subagent admission 前的
`SUBAGENT_CONFIG_ERROR` / `SUBAGENT_WORKSPACE_DISJOINT` / `SUBAGENT_PREPARE_ERROR`
也属此类；普通工具执行仍必须先 claim。

终态工具消息与 Runtime 提交共用 `ToolResult.to_msg()`，直接投影已归一化元数据。

任何 terminal mutation 都在同一 aggregate transaction 内闭合仍为 `PREPARED` 或 `CLAIMED` 的
tool history。`CLAIMED` fact 转为 `OUTCOME_UNKNOWN`，并追加既有的
`TOOL_CALL_OUTCOME_UNKNOWN` event；`PREPARED` fact 保持不变且不追加 outcome event。两者都会向
session history 追加一个模型可见的合成 error result：前者使用 `TOOL_OUTCOME_UNKNOWN`，后者使用
`TOOL_NOT_STARTED`。这些 closer 不是工具真实返回值，不计 usage，也不产生
`TOOL_CALL_COMMITTED` event。session、run 与 checkpoint 的 session revision 随 closer 在同一事务
推进；SQLite 任一写入失败会整体回滚。

tool body 可以乱序完成，但 session message、checkpoint、cursor 与
`TOOL_CALL_COMMITTED` event 只随 committed ordinal prefix 推进。所有 event sequence 都严格单调，
correlation identity 精确；多个 `TOOL_CALL_CLAIMED` telemetry event 的 ordinal 顺序不是契约。
固定内部窗口 8 属于 runtime，不写入 store，也没有改变 lifecycle schema v8、config、command、
model 或公开导出。future NETWORK/MCP/write concurrency 需要新的 durable effect/recovery 协议，
不能从当前多 claim 支持推导出来。

## 会话历史查询与分支

两个 store 提供同一组同步入口，返回类型和 command 均从 `iris.lifecycle` 导入：

| 方法 | 返回值 |
| --- | --- |
| `list_fork_points(session_id, *, after=None, limit=50)` | `ForkPointPage` |
| `load_session_at_run(source_run_id)` | `RunHistorySnapshot` |
| `fork_session(command)` | `SessionSnapshot` |

分支点只包含 terminal 顶层 run，接受全部停止原因：`completed`、`failed`、`cancelled`、
`deadline_exceeded`、`interaction_expired`、`budget_exhausted` 和 `outcome_unknown`。
有 inbound `SubagentRunLink` 的 child 不合格；调用过 child 的顶层 parent 仍合格，只复制
parent 的 session history，不复制 child transcript 或 link。

列表按 `(created_at, run_id)` 升序，使用上一页的 `next_cursor` 作为 `after`；没有更多结果时
`next_cursor=None`。不存在或没有合格 run 的 session 返回空页。Store 检查 `limit > 0`，先过滤
child 再分页；SQLite 在 SQL 中筛选并读取最多 `limit + 1` 个 run，不读取消息。分页不承诺
admission 顺序或跨页固定快照，刷新时从首页开始。

预览返回独立的 `RunHistorySnapshot(point, messages)`，不提供 session CAS revision。其
`point.message_count` 来自选定 run 的终态截点，messages 仅包含该前缀，不含后续轮次。
SQLite 在同一只读事务中查询来源与 `ordinal <= count` 的消息；`_sqlite_messages.py` 为前缀
和完整历史读取共用解码器。创建时 deadline 已过且没有提交 input 的 run 可以预览空历史，
其 `ForkPoint.input` 仍保留原始请求。

`ForkSession` command 携带 `source_run_id`、全新的 `target_session_id` 和 `now`。
新 session 从 `revision=0` 开始，`forked_from_run_id` 记录直接来源；后续追加或摘要提交推进 revision，
并保留来源。摘要来自来源 run 冻结的 `terminal_compaction`，不会取来源 session 后来的摘要；
目标窗口始终为 `None`，第一条输入取得新的概览；预览仍返回原文前缀。消息的 ID、内容块、
工具引用与 metadata 原样保留，返回值与 store 内部消息隔离。Fork 不执行 provider 或工具，
不复制运行控制事实，也不占用 lane；来源 session 忙于后续 run 时仍可复制旧截点。

`_session_history.py` 共享来源检查与结果投影。内存实现持同一 `RLock` 复制前缀并一次写入目标。
SQLite 在 `BEGIN IMMEDIATE` 事务内检查来源、创建带来源字段的 session、通过 `INSERT ... SELECT`
复制消息，再完整读回目标并 commit。失败会整体回滚，不留下空目标或部分消息；该操作不要求
source 当前 session revision 或空闲 lane。当前 schema v8 的无迁移规则保持不变。

来源不存在时，预览和 fork 抛 `IrisRunNotFoundError`；来源非 terminal 或为 child，以及
非正数 list limit，使用 `IrisRunStateError`。目标已存在（包括空 session）抛
`IrisRunConflictError`，不覆盖或重试；SQLite 读写/解析失败使用 `IrisRunPersistenceError`。

## 维护与验证

| 修改内容 | 主要位置 | 对应测试 |
| --- | --- | --- |
| aggregate 语义与 CAS | `in_memory.py` | `tests/store/test_lifecycle_store_contract.py` |
| 历史列表、预览与 fork | `_session_history.py`、`_sqlite_messages.py`、两个 store | `tests/store/test_lifecycle_store_contract.py`、`tests/store/test_lifecycle_sqlite_faults.py` |
| 当前 schema 创建与精确校验 | `_sqlite_schema.py`、`sqlite.py` | `tests/store/test_lifecycle_sqlite_schema.py` |
| SQLite transaction 与故障回滚 | `sqlite.py` | `tests/store/test_lifecycle_sqlite_faults.py` |
| 公开导出 | `__init__.py` | `tests/store/test_lifecycle_store_contract.py` |

```bash
uv run pytest tests/store/test_lifecycle_store_contract.py tests/store/test_lifecycle_sqlite_schema.py tests/store/test_lifecycle_sqlite_faults.py
uv run ruff check src/iris/store tests/store
uv run mypy src/iris/store
```
