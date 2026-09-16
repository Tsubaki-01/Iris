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

## Checkpoint v1

`RunCheckpoint.resumability` is `safe`, `outcome_ready`, or `blocked_unknown`. Safe checkpoints may
re-enter the engine. Outcome-ready checkpoints only need terminal settlement. Blocked-unknown facts
must not execute automatically. Checkpoints accept only the current payload shape and never contain
provider clients, tasks, locks, signals, or callbacks.

## Store contract

`LifecycleStore` exposes create/begin/reserve/model-commit/tool-claim/tool-result/suspend/resolve/
cancellation/finish/recover commands plus run/session/lane/interaction/checkpoint/tool/result/event
reads.
`RunCommit.session_revision` returns the committed revision when raw history or its summary changes;
it does not contain a full `SessionSnapshot`. Call `load_session()` explicitly when history is
needed. Exact retries return current facts with empty events, rather than the original snapshot.
Run-state mutations carry the expected revision/fence facts required by their contracts; stale
writers conflict instead of overwriting.
Stores validate only the phase, counters, identity, and fence affected by the mutation, then apply a
typed delta. They do not dump and fully revalidate an unchanged aggregate for a one-field update.
SQLite rows and checkpoint recovery remain full-validation boundaries, while durable models and
encoders retain JSON-safe guarantees.
`SessionSnapshot` exposes `session_id`, CAS `revision`, complete `messages`, nullable `compaction`, and direct
source `forked_from_run_id`; later appends preserve that source. Revision advances once per non-empty
message delta regardless of its message count. The terminal cutoff lives in `RunRecord` and cannot
be replaced by the session revision. Persistence ordinals do not enter public models.

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
  It atomically replaces the summary, advances session/run revisions and checkpoint sequence, and
  appends `context.compacted`. Cursor, raw messages, reservation, and usage remain unchanged. The
  event contains only coverage and before/after input estimates.

Both reuse the active fence, CAS, and existing exact replay. Neither promises cross-process
exactly-once billing for external model calls.
`load_session_lane()` is only a pure discovery read for the lane owner; it does not recover, repair,
or transfer ownership.
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
summary, without creating a run, activation, checkpoint, tool execution
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
