[中文](README.md)

# `iris.store`

`iris.store` contains concrete persistence backends. Its only public export is `SQLiteStore`, which
implements both `iris.session.SessionStore` and `iris.hitl.InteractionStore` with the standard
library `sqlite3` module.

## Quick start

```python
from iris.store import SQLiteStore

store = SQLiteStore(".iris/session.db")
store.save_messages("default", [{"role": "user", "content": "hello"}])
messages = store.load_messages("default")
```

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

This package does not define the session/HITL protocols and does not implement long-term memory,
an ORM, a connection pool, or cross-process write coordination. `iris.memory` owns its separate
memory database.

## Maintenance

Schema changes must preserve tests for exact signatures, rejection of old/unknown schemas, and the
guarantee that rejected databases are not modified.

```bash
uv run pytest tests/store tests/hitl/test_store_contract.py
uv run ruff check src/iris/store tests/store
```
