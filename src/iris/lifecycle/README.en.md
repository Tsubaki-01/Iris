[中文](README.md)

# `iris.lifecycle`

`iris.lifecycle` is the pure data and synchronous store contract for logical runs. It defines
immutable run/session/activation/checkpoint/tool-call/event/result models, JSON-safe validation,
projections, and CAS commands. It owns neither execution control flow nor a concrete database.

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
- Run, checkpoint, session revision, usage counters, and environment fingerprint cross-validate.
- Mutation events append atomically with aggregate facts and use monotonic sequence numbers.

## Checkpoint v1

`RunCheckpoint.resumability` is `safe`, `outcome_ready`, or `blocked_unknown`. Safe checkpoints may
re-enter the engine. Outcome-ready checkpoints only need terminal settlement. Blocked-unknown facts
must not execute automatically. Checkpoints never contain provider clients, tasks, locks, signals,
or callbacks, and no old payload is migrated.

## Store contract

`LifecycleStore` exposes create/begin/reserve/model-commit/tool-claim/tool-result/suspend/resolve/
cancellation/finish/recover commands plus run/session/interaction/checkpoint/tool/result/event reads.
Every mutation carries expected revision/fence facts; stale writers conflict instead of overwriting.

## Public API

All contract models, enums, commands, `LifecycleStore`, `snapshot_run()`, and `project_result()` are
importable from `iris.lifecycle`. The complete-run facade exists only in `iris.harness`.

## Verification

```bash
uv run pytest tests/harness/test_lifecycle_models.py tests/store/test_lifecycle_store_contract.py
uv run ruff check src/iris/lifecycle
uv run mypy src/iris/lifecycle
```
