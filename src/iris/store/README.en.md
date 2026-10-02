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

`SQLiteStore(path)` accepts only an absent/zero-byte database or an exact lifecycle schema v11
database. A new database gets its parent directory and complete schema. An old schema, missing or
extra objects, index differences, or an unknown version raises `IrisLifecycleSchemaError` before
any write. Old databases are unsupported; choose a new database path for the new store. The
constructor never resets or changes that file.
Schema 11 is required even with Goal disabled; there is no legacy reader or automatic migration.

Both implementations expose a read-only `source_id`. SQLite saves a generated UUID in
`lifecycle_schema.source_id` at creation and preserves it across reopenings; each InMemory instance
gets its own UUID. `load_run_message_slice()` returns run boundaries, terminal outcome, and the
run's bounded page of committed messages from one read snapshot, with `limit=128` by default.
SQLite reads the run, counts, and raw message rows by ordinal range in one read transaction, then
decodes messages after releasing the transaction and shared lock. InMemory copies at most `limit`
messages under the same lock. `end_message_count` is the page end for the next read, while
`terminal_message_count` retains the full cutoff. Both exclude earlier turns,
inherited fork history, and later runs. See the [lifecycle contract](../lifecycle/README.en.md#store-contract)
for message counts and the returned model.

## Architecture

Both backends also implement `iris.goal.store.GoalStore`, using `goals` and `goal_runs` in the
same transaction domain. A session has at most one current goal. Creating a goal before the first
chat creates an empty session at revision 0; goal controls never change history revisions.
`admit_goal_run` atomically creates a normal run, its goal binding, and its spent round. Failed
admission leaves no partial facts; an admitted attempt is not refunded. The in-memory backend
builds all candidates before publishing, while SQLite reuses run creation on the same connection.
Clearing a goal retains its records; forks do not copy goal selection or bindings. Reads do not
start or recover work. See the [Goal package](../goal/README.md) for the domain interface.
settle_goal_run reads the Run outcome, report, and ordering evidence in one transaction, updating
the goal and binding together without a separate report table. Reads expose settlement_pending
when the run is terminal but its goal is unsettled; explicit reconciliation repairs that window.
Repeated settlement does not mutate the goal, and a valid final-round completion precedes the round limit.

`InMemoryLifecycleStore` and `SQLiteStore` are independent, peer protocol implementations. The
in-memory implementation protects process-local facts with one `RLock` and deep-copy isolates
inputs and outputs. Internal append copies only the list container and the new delta instead of
copying store-owned old messages again. Its state disappears with the process. The SQLite
implementation neither imports nor calls the in-memory implementation.

The private `_MemorySession(snapshot, read_state)` installs original history and derived state as one
unit. Append builds a candidate and publishes it only after the mutation's checks pass. An empty
delta reuses the internal aggregate; public full reads remain isolated copies, while context reads
copy only their returned tail and anchors. Non-empty appends still copy the old list's references,
so write cost is not independent of total history length.

`SQLiteStore` opens a scoped connection and enables foreign keys for every operation. Public reads
use targeted reads for the requested run, session, lane owner, interaction, checkpoint, tool calls,
or events.
Reads that need multiple queries use one deferred transaction for a consistent snapshot and remain
write-free.
SQLite row decoding already creates independent objects, so public reads return those results
without another blanket deepcopy. The in-memory implementation still deep-copies store-owned facts.
Like paged Capture, complete `load_session()` reads metadata and raw message rows in one transaction,
then decodes messages after releasing the transaction and instance lock. Session reads inside mutations
such as fork remain in their original transaction. This shortens lock ownership; the synchronous store
call itself still executes on its caller's thread.
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
load/recovery boundaries such as SQLite row decoding; a private store serializer projects durable
values to JSON. Schema v11 keeps revision,
message count, update time, nullable `forked_from_run_id`, the `compaction_json` projection, and the
fixed `context_window_json` in `sessions`; later appends preserve the source, summary, and window.
Messages append under contiguous ordinals in
`session_messages`. A non-empty delta serializes and inserts only its own messages while advancing
metadata with a revision-and-message-count CAS. Full `SessionSnapshot` reads still rebuild and
validate exact ordinals `1..message_count`.
Mutation `RunCommit` receipts carry only a changed `session_revision`; generating a receipt does not
reread full history.

`sessions.tool_discovery_json` stores the discovery projection, and `last_ordinary_user_index` stores
the latest ordinary input's absolute position. Both stores share `_session_projection.py` and fold
only the new delta. All eight message-append paths publish original messages and derived state at
their existing transaction/install point. SQLite reads the small projection once per non-empty delta
and encodes discovery JSON only when it changes. Empty deltas, windows, summaries, and control
operations do not load it. Fork folds its returned cutoff prefix once rather than copying the
parent's current projection; its one-time prefix copy and decoding costs remain.

Run creation records `initial_session_message_count` within its transaction. The first terminal
settlement records the cumulative session message count in `RunRecord.terminal_session_message_count`,
including tool closers, and freezes the current summary in `terminal_compaction` (`None` without a
summary). Later reads preserve this snapshot. Ordinary `FinishRun` and normal FINALIZE recovery
record these terminal fields. Runs without a closer still record the actual count, including zero.
SQLite uses already loaded
session metadata instead of loading full history to count messages.

An expired creation still acquires an ACTIVE run, initial checkpoint, activation, and lane.
Model-step admission returns `ModelStepReservationResult(granted, commit)`; refusal returns current
facts without changing revisions, usage, or events. Cancellation requests only record intent and
leave WAITING interactions open. UNKNOWN recovery first acquires a new RECOVER fence through CAS,
increments the checkpoint sequence, and marks it `BLOCKED_UNKNOWN`, preserving claims, usage, and
committed history. The asynchronous owner performs cleanup before submitting `FinishRun`. Stores
hold no transaction across that wait and do not release the lane early.

Stores do not cache complete commands or promise successful resubmission of historical writes.
Each mutation uses current state, revision/CAS, and activation fences; stale writes normally fail
with a conflict or state error without duplicating facts. Existing state-based idempotence remains
for the same response to a resolved WAITING interaction, the same unsettled cancellation request
within an activation, and child admission under the same parent/tool key. Callers use the existing
read/recovery interfaces to inspect outcomes.
`resolve_interaction` first matches the current waiting interaction identity and response kind.
PENDING writes check run revision and interaction version; a matching RESOLVED answer returns current facts.

`agent_runs.usage_json` is the sole stored run usage; the three duplicate scalar counter columns are
removed. Existing `RunUsage` parsing validates nonnegative counters and committed/reserved relations
when rows are first loaded. The current database is schema v11. Runs and checkpoints no longer store
an environment fingerprint; older schemas are not migrated or read.

Schema v11 contains:

- `lifecycle_schema`, `sessions`, `session_messages`, `agent_runs`, and `session_run_lanes`;
- `run_activations`, `run_checkpoints`, and `run_tool_calls`;
- `run_interactions` and `run_events`;
- the partial unique index `one_open_interaction_per_run`.
- the terminal partial index `terminal_runs_by_session(session_id, created_at, run_id)`.

SQL constraints require `agent_runs.terminal_session_message_count` to be nonnegative and present
exactly when the run is terminal. Nullable `sessions.forked_from_run_id` references the source
`agent_runs.run_id`.

The `(session_id, ordinal)` composite primary key already supports ordered message reads, so no
extra index is added. Session revision advances for a non-empty raw-message delta or a summary
projection commit; it is not the message count.

Connection, serialization, and corrupt-row failures map to `IrisRunPersistenceError` with `path`
and `operation` context. Stale expected facts and database constraint races use lifecycle
conflict/state errors.

## Public API

### Run input archival

`commit_run_input(CommitRunInput)` appends the BCI/user input group, initializes the window, and advances the checkpoint
from `before_input` to `before_model` within one lock or SQLite transaction. It reuses run and
session revisions, the activation fence, and checkpoint sequence. It consumes no model reservation
and leaves the step index, usage, and event sequence unchanged. Old commands conflict; SQL failures
roll back the entire group, so recovery cannot observe partial input or a separately updated window.
Checkpoint payload version is `4`, including the runtime cursor's required `visible_tool_names`
and `todo_reminder_step`. Todo file content is not stored, and there is no Todo table.
Lifecycle schema is `11`. Older databases or checkpoints are rejected at their respective load
boundaries without migration.

`SessionSnapshot.context_window=None` means uninitialized; an explicit `SessionContextWindow()`
means initialized with no memory text. The first input must provide its adopted window through
`initial_context_window`; later inputs must pass `None`. A window stores overview text, its
full/navigation mode, and namespace/path/revision sources outside message history. Full includes
core facts and knowledge scope; navigation contains only knowledge scope. Later runs,
HITL, and recovery continue to read the committed value. Initialization and input share one revision
increment, including initialization with no messages. That revision binds the checkpoint to the window.
Sources are historical metadata; current
runner configuration still determines tool read scope.

### Summary usage and history projection

`record_compaction_usage(RecordCompactionUsage)` adds to `RunUsage.compaction` under the current
activation fence and run CAS, advancing only the run revision and update time. Main token counters,
model-step budget, session, checkpoint, and event sequence stay unchanged; the receipt has
`events=()`. Add the two token categories field by field for combined usage. Child usage stays on
the child run.

`commit_compaction(CommitCompaction)` requires `SAFE/before_model` and one pending model-step
reservation. It atomically replaces `SessionSnapshot.compaction` and the required `context_window`, advances session/run revisions
and checkpoint sequence, and appends one `context.compacted` event. Raw messages, cursor,
reservation, and usage stay unchanged. The store checks the selection snapshot's session revision,
activation fence, cancellation, and advancing coverage. Runtime owns tool-group boundaries and
token limits. The event contains only the covered count and before/after input estimates.
Cancellation, failed candidates, and CAS conflicts preserve the old window. A later SQLite write
failure rolls back the summary and window together.

Recovery after a projection commit but before a main response uses the committed summary, window, and the
same pending reservation. Both mutations require the current revision, so stale command resubmissions conflict;
it does not promise exactly-once external model billing across process restarts.

### Stores and queries

The `iris.store` package exports:

- `InMemoryLifecycleStore` for tests and process-local execution;
- `SQLiteStore` as the schema-v11-only durable `LifecycleStore` implementation.

Both implement the `iris.lifecycle.LifecycleStore` create/begin/reserve/commit/claim/suspend/
resolve/finish/recover/cancel commands and run/session/lane/checkpoint/tool/interaction/event/result
reads. Construct commands and models through `iris.lifecycle`; do not depend on underscored
`iris.store` modules.

`load_session_revision(session_id)` returns `0` for an absent session. SQLite selects only
`sessions.revision`; the in-memory store reads the integer under its lock. Neither decodes or copies
messages, the summary, or the context window; use `load_session()` when those are needed.

`load_session_header(session_id)` reads only revision, message count, and the window. Model context
uses `load_run_context(run_id, *, include_tool_discovery)`, which reads summary coverage, the run's
input start, latest ordinary steer position, and effective originals in one snapshot. SQLite uses
primary-key range reads for the tail and point reads for a few covered anchors, then decodes outside
the lock. Without a summary, the tail is still the entire history. For W tail messages and A actual
prefix anchors, message rows read are bounded by `W+A+2`; locating steer never scans the old prefix.
`include_tool_discovery=False` omits discovery JSON, as do header and control reads. See the
[lifecycle contract](../lifecycle/README.en.md#store-contract) for fields and absolute coordinates.

`read_session_messages(session_id, *, start, limit)` reads a bounded original-message page and
returns `iris.lifecycle.SessionMessagePage(items, next_index, total_count)`. Each item carries its
zero-based source index and message; `next_index=None` marks the end of this read snapshot. SQLite
reads the count and at most `limit` rows in one transaction, then decodes them. The in-memory store
copies the requested slice under its lock without exposing store-owned message objects. Neither
loads the whole conversation before paging or applies summary projection. An absent session or a
start beyond the end returns an empty page; the store raises `IrisRunStateError` for `start < 0` or
`limit <= 0`. Separate pages do not share a fixed snapshot.

`load_tool_call()` returns `None` for an absent composite key even when the run is absent, and
`load_run_control()` follows `load_run()` by returning `None` for an absent run.
`list_tool_calls()` still raises `IrisRunNotFoundError` for an absent run and preserves
`(step_index, ordinal)` ordering. These targeted reads add no extra index or connection pool; the
schema identity is lifecycle v11.
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

Cancellation requests, activation abandon/rebind, and outcome-ready finalization each use aggregate
transactions. After cleanup, `FinishRun` atomically settles WAITING runs and unresolved claims.
Runtime has no old-schema
reader, dual write, or compatibility adapter; incompatible files are rejected directly.

One active activation may hold multiple exact durable claims before any result is committed. Every
claim remains bound to its step, ordinal, call ID, fingerprint, and version. If durable cancellation
commits first, the store rejects a new claim without appending a claim event. If a claim commits
first, that call can only commit a proven result or be closed atomically with every other unresolved
claim as outcome unknown during terminal settlement; it is never replayed.

Preflight failures and `CIRCUIT_OPEN` short-circuit results can commit directly from `PREPARED`
without a claim event. Both stores use the same classification in `_tool_results.py`; actual tool
execution still requires a claim first. Pre-admission subagent `SUBAGENT_CONFIG_ERROR` and
`SUBAGENT_WORKSPACE_DISJOINT`, plus `SUBAGENT_PREPARE_ERROR`, belong to this claimless classification.

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
belongs to runtime and is not persisted; lifecycle schema v11, config, commands, models, and public
exports remain unchanged. Future NETWORK/MCP/write concurrency requires a new durable effect and
recovery protocol and cannot be inferred from current multiple-claim support.

## Session history queries and forks

Both stores provide the same synchronous methods. Import their return types and command from
`iris.lifecycle`:

| Method | Return value |
| --- | --- |
| `list_fork_points(session_id, *, after=None, limit=50)` | `ForkPointPage` |
| `load_session_at_run(source_run_id)` | `RunHistorySnapshot` |
| `fork_session(command)` | `SessionSnapshot` |

Fork points include only terminal top-level runs, accepting every stop reason: `completed`, `failed`,
`cancelled`, `deadline_exceeded`, `interaction_expired`, `budget_exhausted`, and `outcome_unknown`.
A child with an inbound `SubagentRunLink` is ineligible. A top-level parent that called a child
remains eligible; only its session history is copied, without the child transcript or link.

Lists use ascending `(created_at, run_id)` order. Pass the previous page's `next_cursor` as `after`;
`next_cursor=None` means no more results. An absent session or one without eligible runs returns an
empty page. The store checks `limit > 0` and filters children before pagination. SQLite filters in
SQL and reads at most `limit + 1` runs without reading messages. Pagination promises neither
admission order nor a fixed snapshot across pages; refresh from the first page.

Preview returns an independent `RunHistorySnapshot(point, messages)` without a session CAS revision.
Its `point.message_count` comes from the selected run's terminal cutoff, and messages contain only
that prefix, excluding later turns. SQLite reads the source and messages with `ordinal <= count`
in one read transaction. `_sqlite_messages.py` provides the shared decoder for prefix and full
history reads. A run whose deadline expired at creation before committing its input can have an
empty preview; its `ForkPoint.input` still retains a text display of the request, using
`[image: name]` labels for image-only input. Complete `AgentRunRequest.input` and history messages
store ordered data blocks and file references; stores neither read source images nor add attachment tables.

The `ForkSession` command carries `source_run_id`, a new `target_session_id`, and `now`.
The new session starts at `revision=0`, with its direct source in `forked_from_run_id`; later
appends or summary commits advance the revision while preserving the source. Its summary comes
from the source run's frozen `terminal_compaction`, never the source session's later projection.
The target window is always `None`; its first input obtains a fresh overview. History previews
continue to return the raw message prefix.
Message IDs, content blocks, tool references, and metadata remain unchanged, and returned objects
are isolated from stored messages. Fork does not call providers or tools, copy execution-control
facts, or occupy a lane. The source session may be running a later turn while an older cutoff is
copied.

`_session_history.py` shares source checks and result projections. The in-memory store copies the
prefix and inserts the target once under the same `RLock`. SQLite checks the source, creates the
target session with its source field, copies messages through `INSERT ... SELECT`, reads the target,
initializes its discovery projection from that cutoff prefix, and commits within one
`BEGIN IMMEDIATE` transaction. Failure rolls back everything, leaving no
empty target or partial messages. This operation requires neither the source's current session
revision nor a free lane. The current schema v11 policy still provides no migration.

Preview and fork raise `IrisRunNotFoundError` for an absent source. A non-terminal or child source,
or a nonpositive list limit, raises `IrisRunStateError`. An existing target, including an empty
session, raises `IrisRunConflictError` without overwrite or retry. SQLite read/write and parsing
failures use `IrisRunPersistenceError`.

## Maintenance and verification

| Change | Main location | Tests |
| --- | --- | --- |
| Aggregate semantics and CAS | `in_memory.py` | `tests/store/test_lifecycle_store_contract.py` |
| Incremental projection and effective history reads | `_session_projection.py`, both stores | `tests/store/test_session_projection.py`, `tests/store/test_session_context.py` |
| History lists, previews, and forks | `_session_history.py`, `_sqlite_messages.py`, both stores | `tests/store/test_lifecycle_store_contract.py`, `tests/store/test_lifecycle_sqlite_faults.py` |
| Current schema creation and exact validation | `_sqlite_schema.py`, `sqlite.py` | `tests/store/test_lifecycle_sqlite_schema.py` |
| SQLite transactions and fault rollback | `sqlite.py` | `tests/store/test_lifecycle_sqlite_faults.py` |
| Public exports | `__init__.py` | `tests/store/test_lifecycle_store_contract.py` |

```bash
uv run pytest tests/store/test_lifecycle_store_contract.py tests/store/test_lifecycle_sqlite_schema.py tests/store/test_lifecycle_sqlite_faults.py
uv run ruff check src/iris/store tests/store
uv run mypy src/iris/store
```
