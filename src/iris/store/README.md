[English](README.en.md)

# `iris.store`

`iris.store` 提供 Iris 的具体持久化实现。当前公开基于标准库 `sqlite3` 的
`SQLiteStore`，以及实现 `iris.lifecycle.LifecycleStore` 的
`InMemoryLifecycleStore`。旧 session/HITL 存储协议仍分别由
`iris.session.SessionStore` 与 `iris.hitl.InteractionStore` 定义。

## 快速入门

```python
from iris.store import SQLiteStore

store = SQLiteStore(".iris/session.db")
store.save_messages("default", [{"role": "user", "content": "hello"}])
messages = store.load_messages("default")
```

## InMemoryLifecycleStore

`InMemoryLifecycleStore()` 是 logical run aggregate 的进程内 reference implementation。
它在一把 `RLock` 下原子处理 run、session lane、activation、checkpoint、tool call、
interaction、result 和 event sequence，并对 command 输入与 read/commit 返回值做深拷贝隔离。

它实现 `iris.lifecycle.LifecycleStore` 的全部同步 command/read 方法，不调用 provider、
不执行工具，也不访问文件系统或网络。数据随进程退出而丢失；当前 `SQLiteStore` 尚未实现
该 lifecycle protocol，持久化 hard cutover 属于后续阶段。

## SQLiteStore

`SQLiteStore(path)` 使用同一个 SQLite 文件实现 `SessionStore` 与
`InteractionStore`：

- `sessions` 保存 messages、run metadata 和 tool events JSON；
- `human_interactions` 独立保存 HITL request、response、checkpoint 和恢复状态；
- interaction 状态更新使用 version compare-and-set；
- 部分唯一索引保证每个 session 最多存在一个 active interaction。

构造时会创建父目录，并用完整有序 `PRAGMA table_info` signature 识别 HITL schema：无表时
创建 current v2，精确 v2 时补齐索引。精确 v1 与其他未知 schema 都会拒绝初始化，
不删除表、不迁移 interaction JSON，也不清理 session recovery marker。

除精确 v1/v2 外的缺列、额外列、列顺序、类型、nullability、default 或 primary-key 差异均
视为未知 schema，初始化抛出 `IrisSessionError` 且不会修改原表。

`append_tool_event(session_id, event)` 从 event 内读取 `event_id` 幂等追加：相同 payload 是
no-op，相同 event ID 的不同 payload 会抛出 `IrisSessionError`。

## 边界

本包只承载具体持久化实现，不定义 lifecycle/session/HITL 领域协议，不做长期记忆、ORM、
连接池或跨进程写入协调。`iris.memory` 的持久化实现仍由 memory 包自行管理。

## 维护与验证

HITL schema 判断是 fail-closed 契约。修改 `human_interactions` 表、索引或反序列化时，
必须覆盖精确 signature、旧/未知 schema 拒绝以及“不修改原库”的测试。

```bash
uv run pytest tests/store tests/hitl/test_store_contract.py tests/harness/test_lifecycle_transitions.py
uv run ruff check src/iris/store tests/store
```
