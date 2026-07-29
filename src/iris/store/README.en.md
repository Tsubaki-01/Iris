[中文](README.md)

# `iris.store`

`iris.store` contains concrete persistence backends. It publicly exports `SQLiteStore`, which uses
the standard library `sqlite3` module, and `InMemoryLifecycleStore`, which implements
`iris.lifecycle.LifecycleStore`. The legacy session/HITL protocols remain in
`iris.session.SessionStore` and `iris.hitl.InteractionStore`.

## Quick start

```python
from iris.store import SQLiteStore

store = SQLiteStore(".iris/session.db")
store.save_messages("default", [{"role": "user", "content": "hello"}])
messages = store.load_messages("default")
```

## InMemoryLifecycleStore

`InMemoryLifecycleStore()` is the process-local reference implementation for the logical-run
aggregate. Under one `RLock`, it atomically updates the run, session lane, activation, checkpoint,
tool calls, interaction, result, and event sequence. Command inputs and read/commit return values
are deep-copy isolated.

It implements every synchronous command/read method in `iris.lifecycle.LifecycleStore`. It never
calls a provider, executes a tool, or accesses the filesystem or network. State is lost when the
process exits. `SQLiteStore` does not yet implement this lifecycle protocol; the persistent hard
cutover belongs to a later phase.

## Storage model

- `sessions` stores messages, run metadata, and tool-event JSON.
- `human_interactions` separately stores HITL requests, responses, checkpoints, and resume state.
- interaction transitions use version compare-and-set.
- a partial unique index permits at most one active interaction per session.

Construction creates the parent directory and checks the complete ordered `PRAGMA table_info`
signature. No table creates the current v2 schema; exact v2 restores missing indexes. Exact v1 and
all unknown shapes fail closed without dropping tables, migrating interaction JSON, or clearing
session recovery markers.

Tool events use the shared `event_id` idempotency rules documented by `iris.session`.

This package does not define lifecycle/session/HITL protocols and does not implement long-term
memory, an ORM, a connection pool, or cross-process write coordination. `iris.memory` owns its
separate memory database.

## Maintenance

Schema changes must preserve tests for exact signatures, rejection of old/unknown schemas, and the
guarantee that rejected databases are not modified.

```bash
uv run pytest tests/store tests/hitl/test_store_contract.py tests/harness/test_lifecycle_transitions.py
uv run ruff check src/iris/store tests/store
```
