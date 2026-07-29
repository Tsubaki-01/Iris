[中文](README.md)

# `iris.store`

`iris.store` provides two synchronous implementations of `iris.lifecycle.LifecycleStore`: the
process-local `InMemoryLifecycleStore` and the standard-library `sqlite3`-based `SQLiteStore`. Both
implement the same logical-run aggregate contract for session revisions, runs, activations,
checkpoints, tool calls, interactions, and events.

This package owns concrete storage only. Domain models and command/read protocols live in
`iris.lifecycle`. It does not call providers, execute tools, or own the long-term memory managed by
`iris.memory`. Iris requires Python 3.12 or newer.

## Quick start

```python
from iris.store import SQLiteStore

store = SQLiteStore(".iris/lifecycle.db")
session = store.load_session("default")
print(session.revision, session.messages)
```

`SQLiteStore(path)` accepts only an absent/zero-byte database or an exact lifecycle schema v1
database. A new database gets its parent directory and complete schema. An old schema, missing or
extra objects, index differences, or an unknown version raises `IrisLifecycleSchemaError` before
any write; the store never migrates, resets, or changes that file.

## Architecture

`InMemoryLifecycleStore` is the semantic reference implementation. Under one `RLock`, it enforces
CAS, activation fences, session lanes, effect claims, and exact interaction binding. Inputs and
outputs are deep-copy isolated. All state disappears with the process.

`SQLiteStore` opens a scoped connection and enables foreign keys for every operation. Reads are
write-free. Mutations use `BEGIN IMMEDIATE`, hydrate durable rows into the in-memory semantic
implementation, run one command, and replace all aggregate facts in the same transaction. Any SQL
failure rolls the complete transaction back, so readers never observe half-committed facts.

Schema v1 contains only:

- `lifecycle_schema`, `sessions`, `agent_runs`, and `session_run_lanes`;
- `run_activations`, `run_checkpoints`, and `run_tool_calls`;
- `run_interactions` and `run_events`;
- the partial unique index `one_open_interaction_per_run`.

Connection, serialization, and corrupt-row failures map to `IrisRunPersistenceError` with `path`
and `operation` context. Stale expected facts and database constraint races use lifecycle
conflict/state errors.

## Public API

The `iris.store` package exports only:

- `InMemoryLifecycleStore` for tests and process-local execution;
- `SQLiteStore` as the durable `LifecycleStore` implementation.

Both implement the `iris.lifecycle.LifecycleStore` create/begin/reserve/commit/claim/suspend/
resolve/finish/recover/cancel commands and run/session/checkpoint/tool/interaction/event/result
reads. Construct commands and models through `iris.lifecycle`; do not depend on underscored
`iris.store` modules.

The Phase 4 `_legacy_sqlite.py` module exists only for branch-local characterization of the old
runtime. It is not public, does not dual-write with the lifecycle store, and is removed in Phase 5.

## Maintenance and verification

| Change | Main location | Tests |
| --- | --- | --- |
| Aggregate semantics and CAS | `in_memory.py` | `tests/store/test_lifecycle_store_contract.py` |
| Schema, row projection, transactions | `sqlite.py` | schema and fault-injection store tests |
| Public exports | `__init__.py` | store contract/import tests |

```bash
uv run pytest tests/store/test_lifecycle_store_contract.py tests/store/test_lifecycle_sqlite_schema.py tests/store/test_lifecycle_sqlite_faults.py
uv run ruff check src/iris/store tests/store
uv run mypy src/iris/store
```
