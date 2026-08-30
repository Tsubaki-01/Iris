[中文](README.md)

# `iris.lifecycle`

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
- In a terminal run's durable history, every `tool_use` has exactly one matching result. Tool-call
  phase still distinguishes a committed result, unknown outcome, and never-started execution; a
  synthetic closer must not erase side-effect knowledge.
- Run, checkpoint, session revision, usage counters, and environment fingerprint cross-validate.
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
Every mutation carries expected revision/fence facts; stale writers conflict instead of overwriting.
Stores validate only the phase, counters, identity, and fence affected by the mutation, then apply a
typed delta. They do not dump and fully revalidate an unchanged aggregate for a one-field update.
SQLite rows and checkpoint recovery remain full-validation boundaries, while durable models and
encoders retain JSON-safe guarantees.
`SessionSnapshot` still exposes only `session_id`, CAS `revision`, and complete `messages`. Revision
advances once per non-empty message delta regardless of how many messages that delta contains.
Persistence message counts and ordinals do not enter lifecycle public models or commands.
`load_session_lane()` is only a pure discovery read for the lane owner; it does not recover, repair,
or transfer ownership.
`load_tool_call(run_id, tool_call_id)` reads one exact composite identity, while
`load_run_control(run_id)` returns only the eight fence/cancellation fields in
`RunControlSnapshot`. Neither read replaces mutation CAS or changes the synchronous store boundary.

## Public API

All contract models, enums, commands, `LifecycleStore`, `snapshot_run()`, and `project_result()` are
importable from `iris.lifecycle`, including the minimal read projection `RunControlSnapshot`. The
complete-run facade exists only in `iris.harness`.

## Verification

```bash
uv run pytest tests/store tests/harness
uv run ruff check src/iris/lifecycle
uv run mypy src/iris/lifecycle
```
