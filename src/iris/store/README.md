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

构造时会创建父目录，并用完整有序 `PRAGMA table_info` signature 识别 HITL schema：无表时
创建 v2，精确 v2 时补齐索引，精确 v1 时在单事务内删除并重建 `human_interactions`、清除
`latest_run.waiting_human` / `interaction_id` marker。v1 pending interaction 不迁移、不备份，
普通 messages、非 HITL run metadata 与 tool events 保留；成功升级后不支持降级。

除精确 v1/v2 外的缺列、额外列、列顺序、类型、nullability、default 或 primary-key 差异均
视为未知 schema，初始化抛出 `IrisSessionError` 且不会 drop 原表。v1 重建与 marker 清理共享
事务，提交前失败会整体回滚。

`append_tool_event(session_id, event)` 从 event 内读取 `event_id` 幂等追加：相同 payload 是
no-op，相同 event ID 的不同 payload 会抛出 `IrisSessionError`。

## 边界

本包只承载具体持久化实现，不定义 session/HITL 领域协议，不做长期记忆、ORM、连接池或
跨进程写入协调。`iris.memory` 的持久化实现仍由 memory 包自行管理。
