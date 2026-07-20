# iris.session

`iris.session` 提供轻量 session 存储协议和进程内实现，用于保存消息、run metadata 与
工具事件。具体 SQLite 实现位于 `iris.store`。

本模块不是长期记忆系统，也不做向量检索、embedding、Redis 缓存或 ORM 映射。

## 快速入门

```python
from iris.session import InMemorySessionStore

store = InMemorySessionStore()
store.save_messages("default", [{"role": "user", "content": "hello"}])
messages = store.load_messages("default")
```

需要跨进程持久化时使用：

```python
from iris.store import SQLiteStore

store = SQLiteStore(".iris/session.db")
```

## SessionStore

`SessionStore` 是协议接口，定义：

- `save_messages(session_id, messages)`
- `load_messages(session_id)`
- `save_run_metadata(session_id, metadata)`
- `load_run_metadata(session_id)`
- `append_tool_event(session_id, event_id, event)`
- `load_tool_events(session_id)`

## InMemorySessionStore

`InMemorySessionStore()` 使用进程内字典保存 session 数据，适合测试、无持久化运行和调用方
明确不需要跨进程恢复历史的场景。进程退出后数据丢失。

## 与 agent 配置的关系

`session.backend: none` 使用 `InMemorySessionStore`；`session.backend: sqlite` 由
`RuntimeFactory` 创建 `iris.store.SQLiteStore`。SQLite 默认路径为 `.iris/session.db`。

## 错误处理

Session store 边界使用 `IrisSessionError`。runtime 将其归一化为
`RuntimeErrorInfo(source="session", code="SESSION_ERROR")`。

## 边界

本模块不负责具体 SQLite schema、长期记忆、检索增强或跨进程并发写入协调。
