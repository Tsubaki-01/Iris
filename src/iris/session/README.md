# iris.session

`iris.session` 提供轻量 session 存储接口。当前实现包括进程内的
`InMemorySessionStore` 和标准库 `sqlite3` 版本的 `SQLiteSessionStore`，用于保存消息、
run metadata 和工具事件。

本模块不是长期记忆系统，也不做向量检索、embedding、Redis 缓存或 ORM 映射。

## 快速入门

```python
from iris.session import InMemorySessionStore, SQLiteSessionStore

memory_store = InMemorySessionStore()
memory_store.save_messages("default", [{"role": "user", "content": "hello"}])

store = SQLiteSessionStore(".iris/session.db")

store.save_messages(
    "default",
    [{"role": "user", "content": "hello"}],
)
messages = store.load_messages("default")

store.save_run_metadata("default", {"model": "openai/gpt-4o-mini"})
metadata = store.load_run_metadata("default")

store.append_tool_event("default", {"tool": "file.read", "ok": True})
events = store.load_tool_events("default")
```

## 核心定义

### `SessionStore`

`SessionStore` 是协议接口，定义调用方需要的持久化能力：

- `save_messages(session_id, messages)`
- `load_messages(session_id)`
- `save_run_metadata(session_id, metadata)`
- `load_run_metadata(session_id)`
- `append_tool_event(session_id, event)`
- `load_tool_events(session_id)`

### `InMemorySessionStore`

`InMemorySessionStore()` 使用进程内字典保存 session 数据。它适合测试、无持久化运行，
以及调用方明确不需要跨进程恢复历史的场景。该实现不会写入本地文件。

### `SQLiteSessionStore`

`SQLiteSessionStore(path)` 使用一个本地 SQLite 文件保存 session 数据。构造时会创建父目录并初始化表结构。

存储策略很直接：

- `messages` 存为 JSON 字符串。
- `run_metadata` 存为 JSON 字符串。
- `tool_events` 存为 JSON 字符串数组。

读取时再把 JSON 字符串转换回 Python 对象。不存在的 session 会返回空值：

- messages: `[]`
- run metadata: `{}`
- tool events: `[]`

## 与 agent 配置的关系

`iris.agents.SessionConfig` 可以声明是否启用 SQLite：

```yaml
session:
  backend: sqlite
```

如果没有显式提供 `path`，配置层默认使用 `.iris/session.db`。实际创建
`SQLiteSessionStore` 的时机由后续 agent loop 或调用方决定。

## 错误处理

Session store 边界统一使用 `iris.exceptions.IrisSessionError`。SQLite 初始化、读写失败、
损坏 JSON 解析失败和不可 JSON 序列化的数据都会包装为该异常。`IrisSessionError`
自身声明 runtime 映射，runtime 会读取异常实例并归一化为
`RuntimeErrorInfo(source="session", code="SESSION_ERROR")`。

## 边界

本模块只提供可选的轻量持久化。它不保证跨进程并发写入语义，不提供检索增强记忆，也不负责决定哪些内容应该写入 session。
