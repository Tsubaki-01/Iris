# iris.store

`iris.store` 提供 Iris 的具体持久化实现。当前仅包含基于标准库 `sqlite3` 的
`SQLiteStore`；存储协议仍分别由 `iris.session.SessionStore` 与
`iris.hitl.InteractionStore` 定义。

## 快速入门

```python
from iris.store import SQLiteStore

store = SQLiteStore(".iris/session.db")
store.save_messages("default", [{"role": "user", "content": "hello"}])
messages = store.load_messages("default")
```

## SQLiteStore

`SQLiteStore(path)` 使用同一个 SQLite 文件实现 `SessionStore` 与
`InteractionStore`：

- `sessions` 保存 messages、run metadata 和 tool events JSON；
- `human_interactions` 独立保存 HITL request、response、checkpoint 和恢复状态；
- interaction 状态更新使用 version compare-and-set；
- 部分唯一索引保证每个 session 最多存在一个 active interaction。

构造时会创建父目录，并通过 `CREATE TABLE/INDEX IF NOT EXISTS` 初始化 schema。已有
`.iris/session.db` 无需迁移。

`append_tool_event()` 按 `(session_id, event_id)` 幂等追加事件：相同 payload 是
no-op，相同 event ID 的不同 payload 会抛出 `IrisSessionError`。

## 边界

本包只承载具体持久化实现，不定义 session/HITL 领域协议，不做长期记忆、ORM、连接池或
跨进程写入协调。`iris.memory` 的持久化实现仍由 memory 包自行管理。
