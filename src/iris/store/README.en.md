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

Sub Agent adds exactly three mutations. `admit_child_run()` atomically creates an ordinary child
and its three-field `subagent_run_links` row; reentry returns the original child for the exact
parent key. `rebind_subagent_proxy()` keeps the parent tool PREPARED and preserves history, usage,
and cursor, returning complete WAITING facts. `finalize_subagent_result()` requires a terminal
child and commits one parent result/message/usage update. Its WAITING mode also closes the proxy,
creates a fresh RESUME activation, and returns the corresponding checkpoint. An unanswered
PENDING proxy can finalize only after child-owned expiry and child settlement. Both stores share
the operation checks in `_subagent.py`; the link has no additional state columns or explicit index.

```python
from iris.store import SQLiteStore

store = SQLiteStore(".iris/lifecycle.db")
session = store.load_session("default")
print(session.revision, session.messages)
```

`SQLiteStore(path)` accepts only an absent/zero-byte database or an exact lifecycle schema v5
database. A new database gets its parent directory and complete schema. An old schema, missing or
extra objects, index differences, or an unknown version raises `IrisLifecycleSchemaError` before
any write. Old databases are unsupported and must be replaced before creating a new store; the
constructor never resets or changes that file.

## Architecture

`InMemoryLifecycleStore` and `SQLiteStore` are independent, peer protocol implementations. The
in-memory implementation protects process-local facts with one `RLock` and deep-copy isolates
inputs and outputs. Internal append copies only the list container and the new delta instead of
copying store-owned old messages again. Its state disappears with the process. The SQLite
implementation neither imports nor calls the in-memory implementation.

`SQLiteStore` opens a scoped connection and enables foreign keys for every operation. Public reads
use targeted reads for the requested run, session, lane owner, interaction, checkpoint, tool calls,
or events.
Reads that need multiple queries use one deferred transaction for a consistent snapshot and remain
write-free.
SQLite row decoding already creates independent objects, so public reads return those results
without another blanket deepcopy. The in-memory implementation still deep-copies store-owned facts.
Exact tool-call reads reuse the existing `(run_id, tool_call_id)` primary key. Run-control reads
select only the session identity and control columns in `RunControlSnapshot` and do not decode
request, options, usage,
message, or error JSON. The in-memory implementation lists one run through a per-run call-ID index;
the existing tuple-key dictionary remains authoritative.

Mutations use `BEGIN IMMEDIATE` and load only the rows required to validate and apply the current
command. Run, session, checkpoint, tool-call, and interaction updates use revision, sequence, or
version CAS predicates. Lane, activation, interaction, tool, and run changes use incremental writes
in the same transaction, while events remain append-only. Any SQL failure causes a complete
transaction rollback, so readers never observe half-committed facts. For an active history
mutation, the active precondition reads the lane fence once in the transaction and the following
history precondition checks only the session revision. Both stores share lifecycle typed-transition
helpers: a mutation checks the affected phase, fence, and delta, then applies
`model_copy(update=...)` to the validated model. Full `model_validate()` is reserved for
load/recovery boundaries such as SQLite row decoding, while one private store serializer projects
replay keys and durable commands to JSON values. Schema v5 keeps revision,
message count, update time, and nullable `forked_from_run_id` in `sessions`; later appends preserve
the direct source. Messages append under contiguous ordinals in
`session_messages`. A non-empty delta serializes and inserts only its own messages while advancing
metadata with a revision-and-message-count CAS. Full `SessionSnapshot` reads still rebuild and
validate exact ordinals `1..message_count`.
Mutation `RunCommit` receipts carry only a changed `session_revision`; generating a receipt does not
reread full history.

The first terminal settlement records the cumulative session message count in
`RunRecord.terminal_session_message_count`, including tool closers from that settlement. Later
replays preserve the cutoff. Creation-time deadlines, budget exhaustion, waiting cancellation,
ordinary finish, `OUTCOME_UNKNOWN` recovery, and `FINALIZE` recovery all record it. Runs without a
checkpoint or closer still record the actual count, including zero. SQLite uses already loaded
session metadata instead of loading full history to count messages.

Exact-retry cache values contain only a run ID, a flag for returning the session revision, and an
interaction ID. Hits reload current authoritative facts with empty events. Each mutation encodes
its complete canonical command key once. The cache retains its existing process-local lifetime,
without TTL/LRU eviction; complete command keys still grow with the number of mutations.

`agent_runs.usage_json` is the sole stored run usage; the three duplicate scalar counter columns are
removed. Existing `RunUsage` parsing validates nonnegative counters and committed/reserved relations
when rows are first loaded. The current database is schema v5; older schemas are not migrated or read.

Schema v5 contains:

- `lifecycle_schema`, `sessions`, `session_messages`, `agent_runs`, and `session_run_lanes`;
- `run_activations`, `run_checkpoints`, and `run_tool_calls`;
- `run_interactions` and `run_events`;
- the partial unique index `one_open_interaction_per_run`.
- the terminal partial index `terminal_runs_by_session(session_id, created_at, run_id)`.

SQL constraints require `agent_runs.terminal_session_message_count` to be nonnegative and present
exactly when the run is terminal. Nullable `sessions.forked_from_run_id` references the source
`agent_runs.run_id`.

The `(session_id, ordinal)` composite primary key already supports ordered message reads, so no
extra index is added. Session revision counts non-empty delta commits; it is not the message count.

Connection, serialization, and corrupt-row failures map to `IrisRunPersistenceError` with `path`
and `operation` context. Stale expected facts and database constraint races use lifecycle
conflict/state errors.

## Public API

The `iris.store` package exports:

- `InMemoryLifecycleStore` for tests and process-local execution;
- `SQLiteStore` as the schema-v5-only durable `LifecycleStore` implementation.

Both implement the `iris.lifecycle.LifecycleStore` create/begin/reserve/commit/claim/suspend/
resolve/finish/recover/cancel commands and run/session/lane/checkpoint/tool/interaction/event/result
reads. Construct commands and models through `iris.lifecycle`; do not depend on underscored
`iris.store` modules.

`load_tool_call()` returns `None` for an absent composite key even when the run is absent, and
`load_run_control()` follows `load_run()` by returning `None` for an absent run.
`list_tool_calls()` still raises `IrisRunNotFoundError` for an absent run and preserves
`(step_index, ordinal)` ordering. These targeted reads add no extra index or connection pool; the
schema identity is lifecycle v5.
`list_tool_calls(run_id, step_index=...)` returns only the specified model step. SQLite applies the
filter in SQL on one connection. Prepared batches use this bounded read, while HITL resume uses an
exact tool-call read.

`list_events(run_id, after_sequence=0, limit=None)` always preserves sequence order; when provided,
`limit` must be a positive integer. The in-memory store locates the cursor before copying a bounded
slice, while SQLite pushes `LIMIT` into the query so paged consumers do not materialize all
remaining events on every read.

`load_session_lane(session_id)` is a pure read that returns the current non-terminal lane owner's
`run_id`, or `None` when the lane is free. It does not repair, recover, or adopt a run. A host still
loads the run/interaction and calls `recover()` with the exact activation fence or `resume()` with
the exact interaction identity.

Cancellation requests, waiting settlement, activation abandon/rebind, outcome-ready finalization,
and unresolved-claim-to-unknown transitions are aggregate transactions. Runtime has no old-schema
reader, dual write, or compatibility adapter; incompatible files are rejected directly.

One active activation may hold multiple exact durable claims before any result is committed. Every
claim remains bound to its step, ordinal, call ID, fingerprint, and version. If durable cancellation
commits first, the store rejects a new claim without appending a claim event. If a claim commits
first, that call can only commit a proven result or be closed atomically with every other unresolved
claim as outcome unknown during terminal settlement or recovery; it is never replayed.

Preflight failures and `CIRCUIT_OPEN` short-circuit results can commit directly from `PREPARED`
without a claim event. Both stores use the same classification in `_tool_results.py`; actual tool
execution still requires a claim first. Pre-admission subagent `SUBAGENT_CONFIG_ERROR` and
`SUBAGENT_WORKSPACE_DISJOINT` also belong to this claimless classification.

Terminal tool messages and Runtime commits share `ToolResult.to_msg()`, directly projecting
already normalized metadata.

Every terminal mutation closes tool history that is still `PREPARED` or `CLAIMED` in the same
aggregate transaction. A `CLAIMED` fact becomes `OUTCOME_UNKNOWN` and emits the existing
`TOOL_CALL_OUTCOME_UNKNOWN` event. A `PREPARED` fact remains unchanged and emits no outcome event.
Both append a model-visible synthetic error result to session history: `TOOL_OUTCOME_UNKNOWN` for
the former and `TOOL_NOT_STARTED` for the latter. These closers are not real tool results, consume
no usage, and emit no `TOOL_CALL_COMMITTED` event. Session, run, and checkpoint session revisions
advance with the closer in the same transaction; any SQLite write failure rolls the whole change
back.

Tool bodies may finish out of order, while session messages, checkpoints, cursors, and
`TOOL_CALL_COMMITTED` events advance only with the committed ordinal prefix. Every event sequence is
strictly monotonic with exact correlation identity. The ordinal order of multiple
`TOOL_CALL_CLAIMED` telemetry events is not contractual. The fixed internal window bound of 8
belongs to runtime and is not persisted; lifecycle schema v5, config, commands, models, and public
exports remain unchanged. Future NETWORK/MCP/write concurrency requires a new durable effect and
recovery protocol and cannot be inferred from current multiple-claim support.

## Maintenance and verification

| Change | Main location | Tests |
| --- | --- | --- |
| Aggregate semantics and CAS | `in_memory.py` | `tests/store/test_lifecycle_store_contract.py` |
| Current schema creation and exact validation | `_sqlite_schema.py`, `sqlite.py` | `tests/store/test_lifecycle_sqlite_schema.py` |
| SQLite transactions and fault rollback | `sqlite.py` | `tests/store/test_lifecycle_sqlite_faults.py` |
| Public exports | `__init__.py` | `tests/store/test_lifecycle_store_contract.py` |

```bash
uv run pytest tests/store/test_lifecycle_store_contract.py tests/store/test_lifecycle_sqlite_schema.py tests/store/test_lifecycle_sqlite_faults.py
uv run ruff check src/iris/store tests/store
uv run mypy src/iris/store
```
