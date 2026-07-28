[中文](README.md)

# `iris.session`

`iris.session` defines the lightweight session storage protocol and its in-process implementation.
It stores messages, run metadata, and tool events. The SQLite implementation lives in `iris.store`.
This package is not long-term memory and does not provide retrieval, embeddings, Redis, or an ORM.

## Quick start

```python
from iris.session import InMemorySessionStore

store = InMemorySessionStore()
store.save_messages("default", [{"role": "user", "content": "hello"}])
messages = store.load_messages("default")
```

For cross-process persistence:

```python
from iris.store import SQLiteStore

store = SQLiteStore(".iris/session.db")
```

## Public API and behavior

`iris.session` exports only `SessionStore` and `InMemorySessionStore`.

`SessionStore` defines `save_messages()`, `load_messages()`, `save_run_metadata()`,
`load_run_metadata()`, `append_tool_event()`, and `load_tool_events()`.

Tool-event identity comes exclusively from `event["event_id"]`. A new ID is appended; the same ID
with the same canonical JSON payload is a no-op; the same ID with a different payload raises
`IrisSessionError`. In-memory and SQLite backends share this implementation.

`session.backend: none` selects `InMemorySessionStore`; `sqlite` makes `RuntimeFactory` create
`iris.store.SQLiteStore`, defaulting to `.iris/session.db`. In-memory data is lost when the process
exits and cannot provide durable HITL recovery.

Runtime normalizes `IrisSessionError` to `RuntimeErrorInfo(source="session",
code="SESSION_ERROR")`.

## Maintenance

Changes to `SessionStore` must be mirrored by `SQLiteStore`; event-idempotency changes need tests for
both backends.

```bash
uv run pytest tests/session tests/store
uv run ruff check src/iris/session src/iris/store tests/session tests/store
```
