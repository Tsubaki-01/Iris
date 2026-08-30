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

`SQLiteStore(path)` 只接受不存在/零字节的数据库，或者精确匹配 lifecycle schema v2 的
文件。新数据库会创建父目录和完整 schema；旧 schema、缺表/多表、索引或版本差异都会在
任何写入前抛出 `IrisLifecycleSchemaError`。旧数据库不受支持，调用方需要替换后重新创建；
constructor 不重置或修改原文件。

## 实现架构

`InMemoryLifecycleStore` 和 `SQLiteStore` 是两个独立、平级的 protocol 实现。内存实现以一把
`RLock` 管理进程内 facts，并对输入/输出做深拷贝隔离；内部追加只复制 list 容器与新 delta，
不重复复制 store-owned 旧消息。数据随进程退出丢失。SQLite 实现不导入或调用内存实现。

`SQLiteStore` 每次操作打开独立连接并启用 foreign keys。公共 read 使用 targeted query，只
读取目标 run、session、lane owner、interaction、checkpoint、tool calls 或 events；跨多条查询的
read 在同一个 deferred transaction 中取得一致 snapshot，且不执行写入。
exact tool call 读取复用现有 `(run_id, tool_call_id)` 主键；run control 读取只选择
`RunControlSnapshot` 所需的八列，不解码 request/options/usage/message/error JSON。内存实现以
per-run call-ID 索引列举目标 run，权威事实仍保存在原有 tuple-key dict。

Mutation 使用 `BEGIN IMMEDIATE`，只加载当前 command 校验和变更所需的 rows。run、session、
checkpoint、tool call 与 interaction 更新分别使用 revision、sequence 或 version CAS
predicates；lane、activation、interaction、tool facts 与 run 在同一事务中增量写入，events
保持 append-only。任一 SQL 失败都会触发完整 transaction rollback，不暴露半更新状态。
两个 store 共用 lifecycle typed transition helper：mutation 先检查受影响的 phase/fence/delta，
再对已验证模型应用 `model_copy(update=...)`。完整 `model_validate()` 只用于 SQLite row decode 等
load/recovery 边界。
schema v2 的 `sessions` 只保存 revision、message count 与更新时间；消息按连续 ordinal 追加到
`session_messages`。非空 delta 只序列化并插入本次消息，同时以 revision + message count 双条件
CAS 推进 metadata；完整 `SessionSnapshot` 读取仍按 ordinal 重建并校验 `1..message_count`。

schema v2 包含：

- `lifecycle_schema`、`sessions`、`session_messages`、`agent_runs`、`session_run_lanes`；
- `run_activations`、`run_checkpoints`、`run_tool_calls`；
- `run_interactions`、`run_events`；
- partial unique index `one_open_interaction_per_run`。

`session_messages` 的 `(session_id, ordinal)` composite primary key 已覆盖有序读取，不额外增加
index。session revision 表示非空 delta 的提交次数，不等于 message count。

SQLite 连接/序列化/腐坏 row 错误映射为带 `path` 和 `operation` context 的
`IrisRunPersistenceError`；预期 facts 已变化或数据库约束竞争使用 lifecycle conflict/state
错误。

## 公开接口

`iris.store` 顶层导出：

- `InMemoryLifecycleStore`：用于测试和单进程运行；
- `SQLiteStore`：只接受 schema v2 的持久化 `LifecycleStore` 实现。

两者实现 `iris.lifecycle.LifecycleStore` 的 create/begin/reserve/commit/claim/suspend/resolve/
finish/recover/cancel commands 及 run/session/lane/checkpoint/tool/interaction/event/result reads。
应通过 `iris.lifecycle` 构造 command 和模型，不依赖 `iris.store` 中的下划线模块。

`load_tool_call()` 的 composite key 不存在时返回 `None`，即使 run 不存在；
`load_run_control()` 与 `load_run()` 一样在 run 不存在时返回 `None`。`list_tool_calls()` 仍在 run
不存在时抛出 `IrisRunNotFoundError`，并保持 `(step_index, ordinal)` 排序。这些定向 read 没有增加
额外索引或连接池，schema identity 为 lifecycle v2。

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
固定内部窗口 8 属于 runtime，不写入 store，也没有改变 lifecycle schema v2、config、command、
model 或公开导出。future NETWORK/MCP/write concurrency 需要新的 durable effect/recovery 协议，
不能从当前多 claim 支持推导出来。

## 维护与验证

| 修改内容 | 主要位置 | 对应测试 |
| --- | --- | --- |
| aggregate 语义与 CAS | `in_memory.py` | `tests/store/test_lifecycle_store_contract.py` |
| 当前 schema 创建与精确校验 | `_sqlite_schema.py`、`sqlite.py` | `tests/store/test_lifecycle_sqlite_schema.py` |
| SQLite transaction 与故障回滚 | `sqlite.py` | `tests/store/test_lifecycle_sqlite_faults.py` |
| 公开导出 | `__init__.py` | `tests/store/test_lifecycle_store_contract.py` |

```bash
uv run pytest tests/store/test_lifecycle_store_contract.py tests/store/test_lifecycle_sqlite_schema.py tests/store/test_lifecycle_sqlite_faults.py
uv run ruff check src/iris/store tests/store
uv run mypy src/iris/store
```
