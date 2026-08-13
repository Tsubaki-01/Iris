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

`InMemoryLifecycleStore` and `SQLiteStore` are independent, peer protocol implementations. The
in-memory implementation protects process-local facts with one `RLock` and deep-copy isolates
inputs and outputs; its state disappears with the process. The SQLite implementation neither
imports nor calls the in-memory implementation.

`SQLiteStore` opens a scoped connection and enables foreign keys for every operation. Public reads
use targeted reads for the requested run, session, interaction, checkpoint, tool calls, or events.
Reads that need multiple queries use one deferred transaction for a consistent snapshot and remain
write-free.

Mutations use `BEGIN IMMEDIATE` and load only the rows required to validate and apply the current
command. Run, session, checkpoint, tool-call, and interaction updates use revision, sequence, or
version CAS predicates. Lane, activation, interaction, tool, and run changes use incremental writes
in the same transaction, while events remain append-only. Any SQL failure causes a complete
transaction rollback, so readers never observe half-committed facts. Schema v1 still stores
`sessions.messages_json` as one JSON array, so appending messages rewrites the current session row
but does not read or modify other aggregates.

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

Cancellation requests, waiting settlement, activation abandon/rebind, outcome-ready finalization,
and unresolved-claim-to-unknown transitions are aggregate transactions. There is no old-schema
reader, migration, dual write, or compatibility adapter.

One active activation may hold multiple exact durable claims before any result is committed. Every
claim remains bound to its step, ordinal, call ID, fingerprint, and version. If durable cancellation
commits first, the store rejects a new claim without appending a claim event. If a claim commits
first, that call can only commit a proven result or be closed atomically with every other unresolved
claim as outcome unknown during terminal settlement or recovery; it is never replayed.

Tool bodies may finish out of order, while session messages, checkpoints, cursors, and
`TOOL_CALL_COMMITTED` events advance only with the committed ordinal prefix. Every event sequence is
strictly monotonic with exact correlation identity. The ordinal order of multiple
`TOOL_CALL_CLAIMED` telemetry events is not contractual. The fixed internal window bound of 8
belongs to runtime and is not persisted; lifecycle schema v1, config, commands, models, and public
exports remain unchanged. Future NETWORK/MCP/write concurrency requires a new durable effect and
recovery protocol and cannot be inferred from current multiple-claim support.

## Maintenance and verification

| Change | Main location | Tests |
| --- | --- | --- |
| Aggregate semantics and CAS | `in_memory.py` | `tests/store/test_lifecycle_store_contract.py` |
| Schema and compatibility validation | `sqlite.py` | `tests/store/test_lifecycle_sqlite_schema.py` |
| SQLite transactions and fault rollback | `sqlite.py` | `tests/store/test_lifecycle_sqlite_faults.py` |
| Public exports | `__init__.py` | `tests/store/test_lifecycle_store_contract.py` |

```bash
uv run pytest tests/store/test_lifecycle_store_contract.py tests/store/test_lifecycle_sqlite_schema.py tests/store/test_lifecycle_sqlite_faults.py
uv run ruff check src/iris/store tests/store
uv run mypy src/iris/store
```
