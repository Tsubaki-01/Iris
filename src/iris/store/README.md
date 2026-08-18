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

`SQLiteStore(path)` 只接受不存在/零字节的数据库，或者精确匹配 lifecycle schema v1 的
文件。新数据库会创建父目录和完整 schema；旧 schema、缺表/多表、索引或版本差异都会在
任何写入前抛出 `IrisLifecycleSchemaError`，不迁移、重置或修改原文件。

## 实现架构

`InMemoryLifecycleStore` 和 `SQLiteStore` 是两个独立、平级的 protocol 实现。内存实现以一把
`RLock` 管理进程内 facts，并对输入/输出做深拷贝隔离；数据随进程退出丢失。SQLite 实现
不导入或调用内存实现。

`SQLiteStore` 每次操作打开独立连接并启用 foreign keys。公共 read 使用 targeted query，只
读取目标 run、session、lane owner、interaction、checkpoint、tool calls 或 events；跨多条查询的
read 在同一个 deferred transaction 中取得一致 snapshot，且不执行写入。

Mutation 使用 `BEGIN IMMEDIATE`，只加载当前 command 校验和变更所需的 rows。run、session、
checkpoint、tool call 与 interaction 更新分别使用 revision、sequence 或 version CAS
predicates；lane、activation、interaction、tool facts 与 run 在同一事务中增量写入，events
保持 append-only。任一 SQL 失败都会触发完整 transaction rollback，不暴露半更新状态。
schema v1 的 `sessions.messages_json` 仍是单行 JSON 数组，因此追加消息会重写当前 session
row，但不会读取或改写其他 aggregate。

schema v1 只包含：

- `lifecycle_schema`、`sessions`、`agent_runs`、`session_run_lanes`；
- `run_activations`、`run_checkpoints`、`run_tool_calls`；
- `run_interactions`、`run_events`；
- partial unique index `one_open_interaction_per_run`。

SQLite 连接/序列化/腐坏 row 错误映射为带 `path` 和 `operation` context 的
`IrisRunPersistenceError`；预期 facts 已变化或数据库约束竞争使用 lifecycle conflict/state
错误。

## 公开接口

`iris.store` 顶层只导出：

- `InMemoryLifecycleStore`：用于测试和单进程运行；
- `SQLiteStore`：持久化 `LifecycleStore` 实现。

两者实现 `iris.lifecycle.LifecycleStore` 的 create/begin/reserve/commit/claim/suspend/resolve/
finish/recover/cancel commands 及 run/session/lane/checkpoint/tool/interaction/event/result reads。
应通过 `iris.lifecycle` 构造 command 和模型，不依赖 `iris.store` 中的下划线模块。

`load_session_lane(session_id)` 只读返回当前 non-terminal lane owner 的 `run_id`，无占用时返回
`None`。它不修复、恢复或接管 run；host 仍需读取 run/interaction，并用精确 activation fence 调用
`recover()`，或用精确 interaction identity 调用 `resume()`。

取消请求、waiting settlement、activation abandon/rebind、outcome-ready finalize 与 unresolved
claim -> outcome unknown 都在 aggregate transaction 内完成。不存在旧 schema reader、migration、
dual write 或 compatibility adapter。

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
固定内部窗口 8 属于 runtime，不写入 store，也没有改变 lifecycle schema v1、config、command、
model 或公开导出。future NETWORK/MCP/write concurrency 需要新的 durable effect/recovery 协议，
不能从当前多 claim 支持推导出来。

## 维护与验证

| 修改内容 | 主要位置 | 对应测试 |
| --- | --- | --- |
| aggregate 语义与 CAS | `in_memory.py` | `tests/store/test_lifecycle_store_contract.py` |
| schema 与兼容性校验 | `sqlite.py` | `tests/store/test_lifecycle_sqlite_schema.py` |
| SQLite transaction 与故障回滚 | `sqlite.py` | `tests/store/test_lifecycle_sqlite_faults.py` |
| 公开导出 | `__init__.py` | `tests/store/test_lifecycle_store_contract.py` |

```bash
uv run pytest tests/store/test_lifecycle_store_contract.py tests/store/test_lifecycle_sqlite_schema.py tests/store/test_lifecycle_sqlite_faults.py
uv run ruff check src/iris/store tests/store
uv run mypy src/iris/store
```
