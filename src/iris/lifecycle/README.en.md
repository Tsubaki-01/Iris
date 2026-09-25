[中文](README.md)

# `iris.lifecycle`

Public Sub Agent contracts link independent runs through
`SubagentRunLink(parent_run_id, parent_tool_call_id, child_run_id)` while the parent tool stays
PREPARED. The store adds `AdmitChildRun`, `RebindSubagentProxy`, `FinalizeSubagentResult`, and an
exact link point read. Rebind returns complete WAITING `RunCommit` facts. WAITING finalize returns
the ACTIVE run/checkpoint bound to a fresh RESUME activation in the same commit, without an extra
ordinary resume mutation. Child usage remains run-local.
All four types are imported directly from `iris.lifecycle`. Rebind changes only proxy binding and
checkpoint sequence, preserving history/usage/cursor. Finalize commits the parent result, message,
usage, and cursor. The three mutations reuse existing revisions, activation fences, and store
transactions without new concurrency fields.

`iris.lifecycle` is the pure data and synchronous store contract for logical runs. It defines
immutable run/session/activation/checkpoint/tool-call/event/result boundary models, JSON-safe
validation, projections, and CAS commands. Persisted models use Pydantic to validate raw/load data;
same-process commands are frozen slots dataclasses carrying already typed facts. The package owns
neither execution control flow nor a concrete database.

## Dependency boundary

```text
harness -> lifecycle <- store
runtime  -> lifecycle
```

Lifecycle imports none of `iris.harness`, `iris.runtime`, or `iris.store`. `AgentRunner` is the
owner, concrete stores implement the contract, and `AgentRuntime` consumes only option/error facts.

## Aggregate invariants

- A session has at most one non-terminal run lane.
- An active run has exactly one current activation fence; a waiting run has one open interaction.
- Model steps reserve before commit, with at most one outstanding reservation.
- Tool effects claim before execution and commit a result afterward; unresolved claims never replay.
- A terminal run has no current activation, open interaction, or lane.
- `RunRecord.terminal_session_message_count` records the cumulative session message count, including
  tool closers, at its first terminal settlement and remains unchanged afterward. It must be a
  nonnegative integer for terminal runs and `None` for non-terminal runs.
- In a terminal run's durable history, every `tool_use` has exactly one matching result. Tool-call
  phase still distinguishes a committed result, unknown outcome, and never-started execution; a
  synthetic closer must not erase side-effect knowledge.
- Run, checkpoint, session revision, and usage counters cross-validate.
- Mutation events append atomically with aggregate facts and use monotonic sequence numbers.

## Checkpoint v2

`RunCheckpoint.checkpoint_version` is fixed at `2`; loading rejects earlier versions without
migration. New runs start at `before_input` and move to `before_model` after archiving the input
group, so recovery explicitly distinguishes whether the input has already been saved.

`RunCheckpoint.resumability` is `safe`, `outcome_ready`, or `blocked_unknown`. Safe checkpoints may
re-enter the engine. Outcome-ready checkpoints only need terminal settlement. Blocked-unknown facts
must not execute automatically. Checkpoints accept only the current payload shape and never contain
provider clients, tasks, locks, signals, or callbacks.

## Store contract

`LifecycleStore` exposes create/begin/reserve/model-commit/tool-claim/tool-result/suspend/resolve/
cancellation/finish/recover commands plus run/session/lane/interaction/checkpoint/tool/result/event
reads.

`read_session_messages(session_id, *, start, limit)` returns the public `SessionMessagePage`, reading
only a bounded page of original messages. `items` contains `(index, Msg)` tuples with zero-based
indices; `next_index` is the next page's start or `None` at the end, and `total_count` is the original
message count in this read snapshot. The store requires `start >= 0` and `limit > 0`, raising
`IrisRunStateError` otherwise. An absent session or a start at or beyond the end returns an empty
page. This read does not apply summary projection, so compaction cannot renumber the source. It
serves a different purpose from the run-scoped `load_run_message_slice()`.

The read-only `source_id` is a source UUID: reopening a SQLite database preserves it, while an
InMemory identity lasts for one instance. `load_run_message_slice(run_id, after_count=0, *, limit=128)` returns
the public `RunMessageSlice` in one read snapshot, including source/run/session IDs,
`initial_message_count`, `start_message_count`, `end_message_count`, nullable
`terminal_message_count` / `outcome`, and a `messages` tuple. Counts are cumulative session message
counts. The returned interval is `[max(after_count, initial_message_count), end_message_count)`.
Each page contains at most `limit` messages. `end_message_count` is the page end and can be passed
as the next `after_count`; `terminal_message_count` remains the full terminal cutoff on every page.
Active runs never exceed the committed tail, and terminal runs never exceed their own cutoff.
Fewer than `limit` messages means the snapshot tail has been reached; separate pages do not share
a fixed snapshot. Earlier turns, inherited fork history, and later runs are excluded. A nonpositive
`limit`, a negative cursor, or one beyond the run's readable tail raises `IrisRunStateError`;
a missing run raises `IrisRunNotFoundError`. This read supplies
source material and does not generate long-term memory.

`CommitRunInput` / `commit_run_input()` atomically appends BCI and user input and initializes the window under
the existing run/session CAS and activation fence, advancing the checkpoint sequence to
`before_model`. Only the cursor position changes: step index, usage, and model reservations stay
unchanged, and no model event is emitted. Replaying an old command conflicts.
`RunCommit.session_revision` returns the committed revision when raw history, its summary, or the window changes;
it does not contain a full `SessionSnapshot`. Call `load_session()` explicitly when history is
needed. Stores do not promise successful resubmission of historical commands. State-based
idempotence for matching answers, cancellation, and child admission follows each mutation's contract.
Run-state mutations carry the expected revision/fence facts required by their contracts; stale
writers conflict instead of overwriting.
`ResolveInteraction` carries the run ID, current interaction ID, expected run revision, expected
interaction version, typed response, and time; it no longer accepts `expected_fingerprint`.
Pending writes check revision/version. A matching stored answer while WAITING returns current facts
with empty events; a different answer conflicts. Tool argument/workspace fingerprints remain intact.
Stores validate only the phase, counters, identity, and fence affected by the mutation, then apply a
typed delta. They do not dump and fully revalidate an unchanged aggregate for a one-field update.
SQLite rows and checkpoint recovery remain full-validation boundaries, while durable models and
encoders retain JSON-safe guarantees.
`SessionSnapshot` exposes `session_id`, CAS `revision`, complete `messages`, nullable `compaction`,
`context_window`, and direct
source `forked_from_run_id`; later appends preserve that source. Revision advances once per non-empty
message delta regardless of its message count. The terminal cutoff lives in `RunRecord` and cannot
be replaced by the session revision. Persistence ordinals do not enter public models.

### Fixed context window

`SessionSnapshot.context_window: SessionContextWindow | None` is `None` until initialization;
`SessionContextWindow()` represents an initialized window with no memory text. The window stores
the adopted `memory_overview`, `mode` (`full` for core facts and knowledge scope, `navigation` for
knowledge scope only), and a `sources` tuple. Each
`MemoryOverviewSource` contains `namespace`, `path`, and nullable `source_revision`.
These sources describe a historical snapshot and do not grant current tool access.

The first `CommitRunInput.initial_context_window` must explicitly provide a window; later inputs
must pass `None` to retain the adopted text. The input group, window, session revision, and checkpoint
commit together. Initialization advances the revision once even with no messages; changing both
messages and the window also advances it only once. Later runs, HITL, and recovery
reuse that window. Only a successful `CommitCompaction.context_window` replaces it; cancellation,
failure, and CAS conflicts preserve the previous value. Checkpoint v2 binds the window through the
session revision without duplicating its text. `RuntimeExecutionOptions` no longer accepts memory
queries, result snapshots, or character budgets; runtime composition owns reading and selection.

### Summary state and usage

`SessionCompaction(summary, covered_message_count)` stores the complete Markdown body and the raw
prefix `[0,c)` it covers. Raw messages remain available. `RunRecord.initial_session_message_count`
records the run's starting point inside creation. Its first terminal settlement freezes
`terminal_compaction` alongside the message cutoff; later session compaction cannot alter it.
Checkpoints do not duplicate summaries; their session revision binds the current projection.

`RunUsage.compaction: TokenUsage` stores summary input/output/total separately. Existing token fields
still count only main calls; combined usage is derived by adding the two groups. Child usage stays
with the child. Provider totals are preserved without assuming total equals input plus output.

- `RecordCompactionUsage` / `record_compaction_usage()` stores each returned response's usage. It
  changes only run revision and update time, returns no events, and leaves the checkpoint, session,
  and main-step counters unchanged.
- `CommitCompaction` / `commit_compaction()` requires SAFE/before_model with one pending main step.
  It atomically replaces the summary and required `context_window`, advances session/run revisions
  and checkpoint sequence, and
  appends `context.compacted`. Cursor, raw messages, reservation, and usage remain unchanged. The
  event contains only coverage and before/after input estimates.

Both use the active fence and CAS; stale revisions conflict. Neither promises cross-process
exactly-once billing for external model calls.
`load_session_lane()` is only a pure discovery read for the lane owner; it does not recover, repair,
or transfer ownership.
`load_session_revision(session_id)` returns only the current CAS revision, without loading messages,
the summary, or the context window; an absent session returns `0`. Use `load_session()` for history.
`load_tool_call(run_id, tool_call_id)` reads one exact composite identity.
`list_tool_calls(run_id, step_index=...)` limits ordered tool reads to one model step; omitting the
filter returns the whole run.
`load_run_control(run_id)` returns only the session identity and fence/cancellation fields in
`RunControlSnapshot`. Gateways can confirm that a run belongs to the requested session without
loading a complete run. These reads do not replace mutation CAS or change the synchronous store
boundary.

## Session history contract

`history.py` defines four frozen slots dataclasses, all exported from `iris.lifecycle`:

- `ForkPointCursor(created_at, run_id)`: a pagination position for fork points;
- `ForkPoint`: run/session/agent identities, original input, stop reason, creation and finish times,
  and `message_count`;
- `ForkPointPage(items, next_cursor)`: a tuple of fork points and the next-page cursor;
- `RunHistorySnapshot(point, messages)`: history through the selected run, with a tuple of independent
  message objects and no current-session CAS revision.

`LifecycleStore` provides three synchronous methods:

| Method | Return value | Contract |
| --- | --- | --- |
| `list_fork_points(session_id, *, after=None, limit=50)` | `ForkPointPage` | Ascending `(created_at, run_id)` order; `after` accepts `ForkPointCursor`; `limit > 0` |
| `load_session_at_run(source_run_id)` | `RunHistorySnapshot` | Read the complete committed prefix through the terminal message cutoff |
| `fork_session(command)` | `SessionSnapshot` | Accept a `ForkSession` command and atomically create a session |

Sources must be terminal top-level runs; every `RunStopReason` is accepted. A child with an inbound
`SubagentRunLink` is excluded, while a parent with outgoing child links remains eligible. Forking is
allowed while the source session runs a later turn, and new messages do not change the cutoff.
The store owns source eligibility, pagination parameter checks, and target identity checks.

`ForkSession` is a keyword-only frozen slots command exported from `iris.lifecycle`, carrying
`source_run_id`, `target_session_id`, and `now`.
The new session starts at `revision=0`, and `forked_from_run_id` records its direct source.
Inherited messages do not consume revisions; later non-empty appends start at 1 and preserve the
source. Fork inherits the source run's frozen `terminal_compaction`, not the source session's latest
summary. The target starts with `context_window=None` and selects a fresh window on its first input.
Fork does not create a run, activation, checkpoint, tool execution
fact, interaction, event, or lane, and does not restore the source execution position.

## Public API

Public command construction validates the request/checkpoint identity in `CreateRun` and the
disposition/activation combination in `RecoverActiveRun`. Stores trust these established optional
relationships and check current durable facts, mutation CAS, identity, and fences.

All contract models, enums, commands, `LifecycleStore`, `snapshot_run()`, and `project_result()` are
importable from `iris.lifecycle`, including the minimal read projection `RunControlSnapshot`. The
complete-run facade exists only in `iris.harness`.

## Verification

```bash
uv run pytest tests/store tests/harness
uv run ruff check src/iris/lifecycle
uv run mypy src/iris/lifecycle
```
