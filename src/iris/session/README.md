[English](README.en.md)

# `iris.session`

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
- `append_tool_event(session_id, event)`；`event["event_id"]` 是唯一 ID 来源
- `load_tool_events(session_id)`

## InMemorySessionStore

`InMemorySessionStore()` 使用进程内字典保存 session 数据，适合测试、无持久化运行和调用方
明确不需要跨进程恢复历史的场景。实现位于 `src/iris/session/in_memory.py`，进程退出后数据
丢失。

工具事件追加由 `src/iris/session/_tool_events.py` 统一校验：新 `event_id` 追加；同 ID、同
canonical payload 为 no-op；同 ID、不同 payload 抛出 `IrisSessionError`。in-memory 与
SQLite backend 共用这套语义，不维护第二份 event ID 索引。

## 与 agent 配置的关系

`session.backend: none` 使用 `InMemorySessionStore`；`session.backend: sqlite` 由
`RuntimeFactory` 创建 `iris.store.SQLiteStore`。SQLite 默认路径为 `.iris/session.db`。

## 错误处理

Session store 边界使用 `IrisSessionError`。runtime 将其归一化为
`RuntimeErrorInfo(source="session", code="SESSION_ERROR")`。

## 边界

本模块不负责具体 SQLite schema、长期记忆、检索增强或跨进程并发写入协调。

## 维护与验证

修改 `SessionStore` 协议时要同步更新 `iris.store.SQLiteStore`；修改 event 幂等语义时应同时
覆盖内存与 SQLite backend。

```bash
uv run pytest tests/session tests/store
uv run ruff check src/iris/session src/iris/store tests/session tests/store
```
