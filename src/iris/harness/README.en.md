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
try:
    result = await runner.start(
        AgentRunRequest(input="Hello", session_id="default")
    )
    print(result.run.phase, result.assistant_message)
finally:
    await runner.aclose()
```

`from_config*()` resolves relative paths from the configuration directory. With an explicit
`store=`, every durable read and write uses that exact object. Otherwise `session.backend: none`
selects `InMemoryLifecycleStore`, while `sqlite` selects lifecycle `SQLiteStore`.

MCP construction does not connect. Call `aprepare()` to warm up, or let the execution entry prepare
automatically. The complete catalog publishes before computing the final `environment_fingerprint`
and creating a run. Reading it earlier raises `IrisRunStateError`. Required preparation or fingerprint
failure closes resources without creating a run; construct a new runner after correcting configuration.
Runners without MCP retain synchronous fingerprints.

Resume/recover keep pure durable settlement first, preparing only for comparison/execution. They then
reload state, checkpoints, claims, and time. Ordinary fingerprint checks reject catalog, configuration,
or effective read-only-policy drift. Only the digest persists; effective env/header values do not.
Terminal reads, ordinary waiting expiry, unresolved-CLAIMED unknown recovery, queries, history forks,
and cancellation requests do not depend on MCP connections.

Root connections span multiple runs. Stop new calls and await the original start/resume/recover calls
fully before `aclose()`. A cancel result or observation timeout does not prove body cleanup or event
delivery finished. Active closure raises; repeated closure is idempotent, and durable queries remain
available. `SessionManager.close()` does not own runner resources: use `close(cancel_run=True)` before
closing the runner.

Ordinary child YAML can configure MCP independently. Fresh children prepare and compute fingerprints
before admission. Failure returns `SUBAGENT_PREPARE_ERROR` without a child run/link. Every child
WAITING/completion, recovery failure, or early return closes its resources. WAITING retains only durable
links; resume/recover rebuilds and checks the ordinary fingerprint. Parent and child connections remain
independent, with the more restrictive combined permission policy. Live cancellation borrows the runner
and waits for its original task; the creating scope closes resources. Closure failures are logged without
replacing outcomes. Non-live cancellation persists its request before any required preparation.

## Session history branches

`SessionHistory(store)` shares the runner's `LifecycleStore` and provides three synchronous methods.
It generates the new session ID and creation time; the store owns queries, source eligibility, and
atomic copying, with domain errors propagated unchanged to the host. It owns neither store closure
nor the runner's resource lifecycle.

| Method | Return value and behavior |
| --- | --- |
| `list_fork_points(session_id, *, after=None, limit=50)` | `ForkPointPage` in ascending `(created_at, run_id)` order; `after` accepts the previous page's `ForkPointCursor`; `limit > 0` |
| `get_at_run(source_run_id)` | `RunHistorySnapshot` containing the fork point and committed history through it, without a current-session CAS revision |
| `fork(source_run_id)` | `SessionSnapshot` with an automatically generated target ID; each successful call creates a different branch |

Import these history DTOs from `iris.lifecycle`. Lists return an empty page when no eligible run
exists, and `next_cursor=None` when there is no later page. Run the following inside the host's
existing async function, with at least one terminal top-level run already present in `main`:

```python
from iris.harness import AgentRunner, SessionHistory
from iris.lifecycle import AgentRunRequest
from iris.store import SQLiteStore

store = SQLiteStore(".iris/session.db")
runner = AgentRunner.from_config_path("agent.yaml", store=store)
history = SessionHistory(store)

page = history.list_fork_points("main", limit=20)
if page.next_cursor is not None:
    next_page = history.list_fork_points("main", after=page.next_cursor, limit=20)

point = page.items[0]
preview = history.get_at_run(point.run_id)
branch = history.fork(point.run_id)
result = await runner.start(
    AgentRunRequest(input="Try another approach.", session_id=branch.session_id)
)
```

Every terminal stop reason is eligible. Linked children are rejected, while a top-level parent that
called a child remains eligible. Copying stops at the selected run's terminal message cutoff; an
older cutoff can be forked while the source session runs a later turn. The returned session starts
at `revision=0`, records its direct source in `forked_from_run_id`, and preserves that source on
later appends. Target IDs use the `session_` prefix and a UUID; `fork()` accepts no target ID argument.

Fork itself neither calls a provider nor creates a run; it copies the terminal raw-message prefix,
the summary projection frozen at that terminal point, and the direct source. Later compaction in
the source session does not change that branch summary. The next `start()` uses the host-selected
runner's current system, tools, Skills,
memory, and workspace configuration, without restoring an old checkpoint or copying the old
execution environment. Artifact references in messages remain unchanged, and files are not copied.
A host can also construct `SessionManager(runner, branch.session_id)` directly and call
`await manager.submit("Try another approach.")`, observe the existing event stream, and close the
manager when done, without attaching or switching the original manager.

The implementation is in `session_history.py`; related integration tests are in
`tests/harness/test_session_history.py`.

## Public operations

`iris.harness.ChildProviderFactory` defines selected-child provider injection through
`__call__(config: AgentConfig, *, config_path: Path) -> RuntimeProvider`. It receives the loaded
ordinary child configuration, independently of one-off parent provider credential overrides.

`from_config*()` accepts `permission_policy=` and `child_provider_factory=`. With a catalog,
the runner reads one route snapshot and assembles an internal controller. The selected child uses
ordinary AgentConfig, independent session/run IDs, fresh `AgentRunOptions()`, empty request
metadata, no memory service, and the parent's store/clock. CHILD excludes subagent. Linked ACTIVE
runs continue through ordinary recovery; WAITING/TERMINAL paths read the existing result. Dedicated
execution creates a parent proxy when the child waits, keeping the tool PREPARED. The host submits
answers only to parent `resume()`: the response becomes durable before the exact child continues.
Another wait replaces the proxy; terminal finalization advances the parent cursor once, applying
parent identity, artifact handling, and tool error policy.

After a crash following response persistence, `recover(parent_run_id)` continues a RESOLVED proxy
or outer permission from the stored response. Ordinary PENDING waits still require `resume()`;
ACTIVE recovery still requires its activation fence. A fresh process uses the durable selector
against its catalog snapshot and retains ordinary parent/child fingerprint checks. Linked calls
skip outer permission; stored approval without admission still requires execution refresh.
Successful WAITING finalization publishes `tool.completed` with the original tool activation before
running the fresh RESUME activation. SessionManager admits resume before the first child await and
rejects concurrent answers.

Parent cancellation, deadline, and parent-owned proxy expiry settle the exact child before ending
the parent. `request_cancel()` leaves a linked proxy WAITING; `cancel()` or the manager's settlement
task completes it. `settlement_timeout` covers child cleanup and parent observation, preserving
durable cancellation on timeout so later `cancel()`/`recover()` can finish. Repeated interrupts share
the cleanup task; follow-ups start only after true parent terminal settlement.

Child interaction/deadline expiry and outer tool timeout settle the child, then commit
`SUBAGENT_TIMEOUT`. The outer timer starts at durable `child.created_at`, excludes permission waits,
and never resets on resume/rebind/recover. The stored proxy expiry owner determines whether the
parent stops or handles a tool error; parent expiry wins ties. Parent streams contain only parent
start/proxy/final facts, without child streams or usage. Linked recovery does not repeat started;
errors still use `tool.completed` with `is_error`.
The parent retains the final answer and receives only the child's final text. Usage stays on the
child run. Result metadata contains only `agent_selector`, admitted `child_run_id`, and ordinary
artifact metadata. Live events are best-effort, without cross-process exactly-once guarantees.

This minimal Sub Agent example uses three files. Run the parent with the preceding
`AgentRunner.from_config_path("agent.yaml")` example and supply normal provider credentials.

```yaml
# agent.yaml
name: parent
model: openai/gpt-4o-mini
system: Delegate focused tasks to researcher, then use its result to give the final answer.
tools:
  subagent: subagents.yaml
```

```yaml
# subagents.yaml
default: researcher
agents:
  researcher:
    path: agents/researcher/agent.yaml
    description: Analyze a focused task and return concise findings.
```

```yaml
# agents/researcher/agent.yaml
name: researcher
model: openai/gpt-4o-mini
system: Handle only the delegated task, ask the user when needed, and return your findings.
tools:
  builtin: [human.ask]
```

When the host receives parent `pending_interaction`, it uses ordinary typed HITL on parent
`resume()` without managing a child runner. Catalog default/selector/description changes that alter
the ordinary environment fingerprint reject recovery, without bypassing checks or replacing the child.
If the child has closed its HITL interaction but not committed the tool result, ordinary ACTIVE
recovery restores its stored response. Child `IrisRunRecoveryError` propagates unchanged, preserving
recoverable parent/child state.

- `start()` atomically creates a run/start activation and advances it to waiting or terminal.
- `resume()` consumes the exact waiting interaction.
- `request_cancel()` guarantees only that the first request is durable. A local active activation
  is signalled after commit; a waiting run can settle cancelled in the same transaction.
- `cancel()` requests cancellation and observes durable settlement. Observation timeout writes no
  new fact, and settlement does not imply that the original `start()` / `resume()` call has exited.
- `recover()` requires the exact active activation fence. Safe checkpoints create a recover
  activation, outcome-ready checkpoints only finalize, and unresolved claims become
  `outcome_unknown`.
- `get_session()`, `get_run()`, `get_run_control()`, `get_result()`, `list_tool_calls()`, and
  `list_events(after_sequence=0, limit=None)` are side-effect-free durable reads. When provided,
  `limit` must be a positive integer.

Use `resume()`, not `recover()`, for a valid waiting run. Cancel/recover on terminal runs are
idempotent reads.

`get_run_control()` reads only run identity, phase, activation fence, revision, and cancellation
control fields. SessionManager uses it under its lock to decide whether a steer can still enter
the current activation without loading a full run snapshot. Event replay consumes the store's
ordered, unique, bounded pages directly. The store owns sorting and pagination; the manager
advances watermarks and retains empty-page conflicts and submission barriers.

The internal `RunCommit.session_revision` returns only the resulting session revision. The
commit port updates its local revision directly, and mutations do not load complete history for
their receipts. Call `get_session()` explicitly when messages are needed.

## Live publisher composition

A host may inject the same `LivePublisher` (typically a `LiveStreamBroker`) into `AgentRunner`
through `live_publisher=`. The runner synchronously publishes each activation's
`RuntimeStreamEvent` values and every newly committed `RunEvent`; without a publisher, it creates
no runtime sink. The publisher is a best-effort observation plane. An ordinary exception produces
a payload-free warning and cannot roll back a durable mutation, cancel a run, or change its
`RunResult`.

Publisher injection is the sole runtime transport choice: complete without a publisher, streaming
with one. `ModelConfig` has no `stream` field; direct provider calls still use `LLMRequest.stream`.

`AgentRunner.from_config()` and `from_config_path()` pass the publisher only to the runner;
`RuntimeFactory` owns neither the broker nor fan-out. A host can refill durable facts from the
exact runner/store through `get_session()`, `get_run()`, `get_result()`, `list_tool_calls()`, and
`list_events()`. These reads never publish live facts.

## Per-session input management

`SessionManager(runner, session_id, submission_publisher=...)` binds one exact runner and one
session. It is intended for a host that must accept new ordinary input while the current run is
executing:

The host explicitly selects `observation_mode="mixed"` (default) or `"broker_only"`. Mixed mode
provides lossless `events()` and can also publish to a broker. Broker-only mode requires
`submission_publisher`, publishes submission facts without local trackers/transient buffers, and
rejects `events()`. Gateway-only hosts should select broker-only so admission does not depend on an
unused local consumer.

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

Plain-text hosts such as the CLI can use `mode="auto"`. Under the manager lock, durable state selects
a new run when idle or steer when busy. `options` applies only to a new run and is unused in the busy
branch; the UI does not need its own routing-state copy.

Each mode preserves its own FIFO order, while eligibility is independent: an earlier follow-up
does not block a steer that can still enter the current run. A busy receipt means only `pending`;
the selected observation mode reports final delivery or failure. The mixed single-consumer stream mixes raw
durable `RunEvent` values with transient `SubmissionEvent` values and adds no session-global
sequence. Idle submissions emit no `SubmissionEvent`.

In mixed mode, the optional `submission_publisher` is a submission side channel. The manager first writes
the original `SubmissionEvent` to the single-consumer buffer above, then best-effort publishes a
`SessionSubmissionEvent` carrying the session identity. A publisher failure neither repeats the
buffer write nor changes the receipt. Run events, HITL, and results retain their existing
manager/runner contracts.

By default, a manager queues at most 64 steers and 64 follow-ups, reserves 256 transient submission
event slots, and tracks 64 durable runs that the consumer has not caught up with. The latter two
limits apply only in mixed mode. Hosts may set
other finite positive limits through `max_pending_steer`, `max_pending_follow_up`,
`max_buffered_submission_events`, and `max_tracked_durable_runs`. A busy admission reserves both its
pending and terminal event slots. If any required capacity is unavailable, it raises
`IrisRunStateError` before publishing a receipt, queue entry, or event; events are never silently
dropped. An accepted follow-up remains FIFO-blocked while the tracker is full and resumes after the
consumer catches up. A new idle submit is rejected before task creation in the same situation. One
synchronous admission mutation owns both the durable-tracker capacity decision and baseline
registration; there is no check-then-register path.

HITL responses use `manager.resume(interaction_id=..., response=...)` to wait for the complete result,
or `admit_resume(...)` to return `ResumeReceipt(run_id, interaction_id)` after activation admission.
Both share one admission owner; the manager/runner retains background execution ownership and the
response never enters the ordinary-input queue. `interrupt()` requests cancellation of the exact current run. An active
cancellation request is not terminal, so follow-ups still wait for actual settlement. `close()`
rejects later operations, fails every pending input with `session_closed`, and ends the event
stream, but neither cancels nor waits for the current run.

A host about to close its event loop uses `close(cancel_run=True, reason=...)`: close admission and
fail pending input first, prevent another follow-up from starting, then cancel and await the current
run through the runner. It then waits for the original managed `start()` / `resume()` task to end,
including a WAITING parent continuing to await its child. CLI `/exit`, EOF, Ctrl-C, and error exits
all use this path.

The queue, receipt state, submission events, claims, and durable event watermarks exist only in the
current process. Durable event payloads do not accumulate in an unbounded process-local queue: a
callback advances a per-run observed watermark, and the consumer replays bounded batches from the
authoritative store after its delivered watermark, at most 64 events per page. Every
`LifecycleStore` implementation must accept `limit` and apply it before copying or decoding. A new
manager does not scan, recover, or attach an existing active/waiting lane;
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

Within each activation, the runner and commit port share a private `_RunEventCollector`, which
alone owns accumulated events and `(run_id, sequence)` deduplication keys. A new batch checks
only its own events. Even when both paths observe a cancellation event, the synchronous callback
runs only on its first collection. Callback failures do not block the subsequent live publisher.

## Cancellation and recovery

When settling failure, the runner checks the absolute deadline against its injected Clock,
independently of timer scheduling. Provider exceptions, `response.failed`, and cancellation cleanup
failures after the deadline settle as `DEADLINE_EXCEEDED`; errors before it remain `FAILED`.
An uncommitted tool claim still takes precedence as `OUTCOME_UNKNOWN`.

`cancellation_requested` is a durable fact, not settlement. The runner persists the request before
sending the local signal and does not cancel the entire activation task while a claim exists.
`ToolExecutor` translates the signal into cancellation of an ordinary async callable, custom async
`BaseTool`, or THREAD callable body task and waits for body cleanup. Slow middleware, coroutines
that suppress `CancelledError`, and INLINE blocking can still delay settlement; `cancel()` waits
for a durable terminal result without returning cancelled early.

If the body has completed, or returns normally after signal-driven cancellation, its result goes
through postprocessing and the existing ordered durable commit before the run settles cancelled.
Once external task cancellation, timeout, or sibling cancellation interrupts the executor, a cleanup-time return
does not replace the original interruption. Unresolved claims still settle the run as
`TOOL_OUTCOME_UNKNOWN`, including read-only calls. Worker threads may continue, but late returns
cannot change the durable result, history, checkpoint, or events.

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

The recovery fingerprint binds the agent name, effective model route and request options, loaded
structured context, template source versions, current tool definitions, permission policy,
workspace, and checkpoint version. It also includes content versions for every Skill discovered
in the enabled startup directory, including Skills not yet loaded. Session storage paths,
context configuration locations, and duplicate declaration forms are excluded. Moving identical
templates does not change their versions. Use `ToolDefinition.metadata` for an explicit tool
implementation version; the framework neither infers Python source versions nor scans the workspace.

When the factory creates a built-in `ProviderClient`, it saves the effective provider, LiteLLM
provider, endpoint, and headers after global configuration merging in
`RuntimeEnvironment.provider_fingerprint`. API keys are excluded. A host injecting its own
provider should set explicit route or version identifiers in this dictionary before constructing
the runner. Its empty default means the framework does not infer provider internals; request
model names and options still participate in recovery comparisons.

Template sources are frozen when the runner is constructed. The same runtime renders from that
snapshot; a new runtime reads new versions. The snapshot includes nested static dependencies,
optional dependencies, and filename lists. Dynamic filename expressions are unsupported; use
static references in conditional branches instead. A configured empty memory template is frozen
because run options may activate it later, while an empty before-input section is skipped. The
fingerprint does not render context; `StrictUndefined` and character limits remain rendering-time
checks. See [`iris.context`](../context/README.en.md).

Start, resume, subagent parent resume, and recovery pass `run_input` and
`initial_session_message_count` from the durable run. `before_model / step 0` reconstructs input
that has not yet been archived; later steps do not append it again. Recovery after a projection
commit but before the main response uses the new session revision and the same pending model step.

`RunUsage` input/output/total fields count only the main model. Summary calls accumulate under
`usage.compaction`; add the corresponding fields for combined consumption. A summary usage-only
commit advances the run revision without a durable event. The commit port accepts that revision
immediately so later cancellation reads remain valid. Projection commits independently emit
`context.compacted` while retaining raw history.

## Public API

`iris.harness` exports `AgentRunner`, `SessionHistory`, `SessionManager`, `SubmitReceipt`,
`ResumeReceipt`, `SubmissionEvent`,
`SessionSubmissionEvent`, `SessionEvent`, `LiveFact`, and `LivePublisher`; run
request/options/limits/runtime options; phase, stop reason, usage, error, snapshot, and result;
plus run events and observers. Store commands remain in `iris.lifecycle`.

## Verification

```bash
uv run pytest tests/harness
uv run ruff check src/iris/harness tests/harness
uv run mypy src/iris/harness
```
