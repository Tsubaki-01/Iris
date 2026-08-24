[中文](README.md)

# `iris.harness`

`iris.harness.AgentRunner` is Iris's only complete-run SDK facade. It owns logical-run creation,
resume, durable cancellation, settlement observation, explicit recovery, event delivery, and live
activation resources. `AgentRuntime` is its inner engine. `SessionManager` is an optional,
process-local admission facade for one session; it composes the runner without taking durable
ownership from it.

## Quick start

```python
from iris.harness import AgentRunRequest, AgentRunner

runner = AgentRunner.from_config_path("agent.yaml")
result = await runner.start(
    AgentRunRequest(input="Hello", session_id="default")
)
print(result.run.phase, result.assistant_message)
```

`from_config*()` resolves relative paths from the configuration directory. With an explicit
`store=`, every durable read and write uses that exact object. Otherwise `session.backend: none`
selects `InMemoryLifecycleStore`, while `sqlite` selects lifecycle `SQLiteStore`.

## Public operations

- `start()` atomically creates a run/start activation and advances it to waiting or terminal.
- `resume()` consumes the exact waiting interaction.
- `request_cancel()` guarantees only that the first request is durable. A local active activation
  is signalled after commit; a waiting run can settle cancelled in the same transaction.
- `cancel()` requests cancellation and observes durable settlement. Observation timeout writes no
  new fact.
- `recover()` requires the exact active activation fence. Safe checkpoints create a recover
  activation, outcome-ready checkpoints only finalize, and unresolved claims become
  `outcome_unknown`.
- `get_run()`, `get_result()`, and `list_events(after_sequence=0, limit=None)` are side-effect-free
  durable reads. When provided, `limit` must be a positive integer.

Use `resume()`, not `recover()`, for a valid waiting run. Cancel/recover on terminal runs are
idempotent reads.

## Per-session input management

`SessionManager(runner, session_id)` binds one exact runner and one session. It is intended for a
host that must accept new ordinary input while the current run is executing:

```python
import asyncio

from iris.harness import AgentRunner, SessionManager, SubmissionEvent

runner = AgentRunner.from_config_path("agent.yaml")
manager = SessionManager(runner, "default")

async def consume_events():
    async for event in manager.events():
        if isinstance(event, SubmissionEvent):
            print(event.submission_id, event.state, event.reason)

consumer = asyncio.create_task(consume_events())
initial = await manager.submit("Analyze the current state")
queued = await manager.submit("Focus on concurrency boundaries", mode="steer")

# When the host is done with this manager:
await manager.close()
await consumer
```

When idle, `submit(input, mode=None, options=...)` returns a
`SubmitReceipt(state="delivered")` after the run create is durably committed, without waiting for
the provider or run settlement. While busy, callers must choose explicitly:

- `mode="steer"` targets the exact current run and accepts no new run options. The runtime claims
  one item only at a safe boundary; `SubmissionEvent(state="delivered")` follows the successful
  durable session-history commit.
- `mode="follow_up"` pre-generates a future run ID and may carry options. It creates one run at a
  time, only after the exact current run becomes terminal.

Each mode preserves its own FIFO order, while eligibility is independent: an earlier follow-up
does not block a steer that can still enter the current run. A busy receipt means only `pending`;
the final delivery or failure is reported through `events()`. This single-consumer stream mixes raw
durable `RunEvent` values with transient `SubmissionEvent` values and adds no session-global
sequence. Idle submissions emit no `SubmissionEvent`.

By default, a manager queues at most 64 steers and 64 follow-ups, reserves 256 transient submission
event slots, and tracks 64 durable runs that the consumer has not caught up with. Hosts may set
other finite positive limits through `max_pending_steer`, `max_pending_follow_up`,
`max_buffered_submission_events`, and `max_tracked_durable_runs`. A busy admission reserves both its
pending and terminal event slots. If any required capacity is unavailable, it raises
`IrisRunStateError` before publishing a receipt, queue entry, or event; events are never silently
dropped. An accepted follow-up remains FIFO-blocked while the tracker is full and resumes after the
consumer catches up. A new idle submit is rejected before task creation in the same situation.

HITL responses use only `manager.resume(interaction_id=..., response=...)`; they never enter the
ordinary-input queue. `interrupt()` requests cancellation of the exact current run. An active
cancellation request is not terminal, so follow-ups still wait for actual settlement. `close()`
rejects later operations, fails every pending input with `session_closed`, and ends the event
stream, but neither cancels nor waits for the current run.

The queue, receipt state, submission events, claims, and durable event watermarks exist only in the
current process. Durable event payloads do not accumulate in an unbounded process-local queue: a
callback advances a per-run observed watermark, and the consumer replays bounded batches from the
authoritative store after its delivered watermark, at most 64 events per page. The built-in memory
and SQLite stores apply `limit` before copying or decoding. The runner still slices results from a
legacy custom store that lacks the new keyword, but that compatibility path cannot bound
materialization inside the custom store; custom stores should add `limit`. A new manager does not
scan, recover, or attach an existing active/waiting lane;
a new idle submit is rejected by the store's session-lane CAS in that case. The runner/store remain
authoritative for durable runs, history, checkpoints, interactions, cancellation, results, and
`RunEvent` values.

## Managed composition hooks (package-private)

`AgentRunner._start_managed()` and `_resume_managed()` are package-private hooks for composition
inside `iris.harness`; they are not exported from `iris.harness`. They retain complete-run
semantics: each coroutine still waits for a waiting or terminal `RunResult`, while public
`start()` / `resume()` delegate with empty hooks. There is no managed `recover()` variant.

A managed call may inject an activation-scoped steering port, a synchronous durable event callback,
and an `asyncio.Event` admission signal. The signal is set only after the create/resume durable
mutation succeeds, its events have been relayed, and the exact activation is registered in the
runner's `_active` map. Immediate terminal outcomes and mutation/registration failures do not emit
a false signal.

The store-backed commit port and runner-owned create/resolve/begin/cancel/finish mutations relay
only newly committed durable `RunEvent` values. A callback exception is logged and cannot roll back
the mutation or change the `RunResult`. The public `RunEventObserver` signature is unchanged. Each
observer lane is sequence-ordered, different observers run in parallel, and each event has a
30-second timeout by default, configurable with `observer_event_timeout_s`. A timeout or ordinary
exception is logged and the lane continues without changing the durable result. The synchronous
callback is not a new public observer registry.

## Cancellation and recovery

Cancellation requested is a durable fact, not a settlement claim. A non-cooperative synchronous
tool may delay settlement. If a tool returns after the request, its result is committed before the
run settles cancelled. If an effect cannot be proven after claim, recovery fails closed with
`TOOL_OUTCOME_UNKNOWN`.

The runner's live signal and store-backed commit port use
`iris.exceptions.IrisCancellationRequestedError` to request cooperative runtime settlement; the
type is not part of the `iris.tools` public error surface.

The store-backed commit port rereads minimal run control at every effect/commit safety boundary and
does not cache it across boundaries. It accepts only exact equality or a one-revision,
one-event-sequence cancellation on the same active activation, proven by exactly one
`run.cancellation_requested` event. Phase/fence changes, jumps, repeated cancellation, and event or
payload mismatches fail closed. Mutations still use the original revision and activation CAS as the
final authority.

Runtime's read-only concurrency window has a fixed internal bound of 8 and adds no public config,
schema, or API. Every call in a window has an independent durable claim. Bodies may finish out of
order, but only a continuous known result prefix enters history, cursor, and checkpoint in ordinal
order. Claim telemetry event order is not an ordinal contract. Any uncommitted claim makes
cancellation, deadline, or program interruption settle outcome unknown; the existing terminal
settlement closes every unresolved claim for that activation in one aggregate transaction.

Active recovery validates checkpoint v1, session revision, usage counters, environment
fingerprint, and cursor. Tools are never replayed while unresolved claims exist. Recovery atomically
abandons the old activation, closes every claim as outcome unknown, and creates the terminal result.
Normal parent/control/infrastructure exit waits for runtime children to drain before revoking the
commit port, preventing late child writes. Synchronous blocking callables have no concurrency
speedup guarantee and may still delay settlement.

Initial recovery at `before_model / step 0` reconstructs the uncommitted current-turn input from the
durable `AgentRunRequest.input`. At later checkpoints, that input is already in session history from
the provider commit and is not injected again.

## Public API

`iris.harness` exports `AgentRunner`, `SessionManager`, `SubmitReceipt`, `SubmissionEvent`, and
`SessionEvent`; run request/options/limits/runtime options; phase, stop reason, usage, error,
snapshot, and result; plus run events and observers. Store commands remain in `iris.lifecycle`.

## Verification

```bash
uv run pytest tests/harness
uv run ruff check src/iris/harness tests/harness
uv run mypy src/iris/harness
```
