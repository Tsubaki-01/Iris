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

`InMemoryLifecycleStore` 是语义参考实现。它在一把 `RLock` 下校验 CAS、activation fence、
session lane、effect claim 和 interaction 绑定，并对输入/输出做深拷贝隔离。数据随进程退出
丢失。

`SQLiteStore` 在每次操作中打开独立连接并启用 foreign keys。读取是无写入的纯操作；
变更使用 `BEGIN IMMEDIATE`，把 durable rows 还原到进程内语义实现、执行一个 command，
再在同一事务中写回全部 aggregate facts。任一 SQL 失败都会整体回滚，不会暴露
半更新状态。

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
finish/recover/cancel commands 及 run/session/checkpoint/tool/interaction/event/result reads。应通过
`iris.lifecycle` 构造 command 和模型，不依赖 `iris.store` 中的下划线模块。

取消请求、waiting settlement、activation abandon/rebind、outcome-ready finalize 与 unresolved
claim -> outcome unknown 都在 aggregate transaction 内完成。不存在旧 schema reader、migration、
dual write 或 compatibility adapter。

## 维护与验证

| 修改内容 | 主要位置 | 对应测试 |
| --- | --- | --- |
| aggregate 语义与 CAS | `in_memory.py` | `tests/store/test_lifecycle_store_contract.py` |
| schema、row projection 和事务 | `sqlite.py` | `test_lifecycle_sqlite_schema.py`, `test_lifecycle_sqlite_faults.py` |
| 公开导出 | `__init__.py` | store contract/import tests |

```bash
uv run pytest tests/store/test_lifecycle_store_contract.py tests/store/test_lifecycle_sqlite_schema.py tests/store/test_lifecycle_sqlite_faults.py
uv run ruff check src/iris/store tests/store
uv run mypy src/iris/store
```
