[中文](README.md)

# `iris.harness`

`iris.harness.AgentRunner` is Iris's only complete-run SDK facade. It owns logical-run creation,
resume, durable cancellation, settlement observation, explicit recovery, event delivery, and live
activation resources. `AgentRuntime` is its inner engine. `SessionManager` is an optional,
process-local admission facade for one session; it composes the runner without taking durable
ownership from it.

## Quick start

Both `from_config*()` entry points accept `observability=` and share that instance with the runtime,
tool executor, and children. Explicit injection overrides each Agent's capture policy; children still
load current YAML for business settings. When enabled without injection, runtime composition creates
a service from global `Config.observability`, and its environment closes it after business resources.
Hosts close injected services last. Disabled observation adds no `init_config()` requirement for a
custom provider. Construction and preparation failures also close a self-created SDK.

Delayed maintenance, automatic Goal continuation, and deadline tasks clear the OTel parent while
preserving business ContextVars. Complete/stream model calls are connected; activation, tool, and
maintenance scopes are not connected yet. See [observability](../observability/README.md).

Both `from_config*()` entry points accept optional `decision_client=` and borrow only its `evaluate`
capability. Runner shutdown never closes an injected evaluator. A client created from configuration
is reused across sessions/Runs and closed by the environment. Children construct independent clients
from their own configuration without inheriting root injection. See [Decision](../decision/README.md).

With `todo.enabled`, `await runner.get_todo(session_id)` reads the current workspace's Markdown
checklist and returns its absolute `path`, immutable `items`, and an optional format `error`.
It also works with a directly constructed runtime, does not create a session or run, and does not
prepare providers, occupy the session lane, or change history revisions. Each query reads current
file contents; a missing file is empty, while disabled access or I/O failure raises `IrisTodoError`.
See [Todo](../todo/README.md) for configuration and the Markdown format.

Ordinary and automatic Goal runs in the same session reuse its file; each run gets its own single
check opportunity. Children use their own configuration and session, including after recovery;
they neither inherit nor merge the parent's list. `SessionHistory(store).fork(source_run_id)`
creates a new session and therefore a new Todo path, read if present and empty otherwise, without
copying a list from history. Direct `RuntimeFactory` construction still requires the existing
`context_access` for context_policy and gains no Todo-specific factory parameter. CLI `/todo`
queries this SDK on demand without consuming a pending human answer.

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

`prompts.root` resolves from the effective root workspace and defaults to `.iris/prompts`. Both
`from_config*()` entry points accept an initialized `prompt_source=`. When omitted, construction
initializes it before Memory and other consumers, adding only missing defaults and preserving
existing project text. Children borrow the same source but take their own construction snapshot;
their YAML and narrower workspace do not relocate it. An injected MemoryService keeps its host
binding; runners never rewrite its source. See [prompts](../prompts/README.md) for adoption timing.

On an open runner, import an image before submitting its data blocks. The main model must support vision through
the selected API protocol:

```python
from pathlib import Path
from iris.message import TextBlock

image = await runner.import_image(Path("invoice.png"), session_id="default", name="invoice")
result = await runner.start(
    AgentRunRequest(input=[TextBlock(text="What is the invoice total?"), image], session_id="default")
)
```

`import_image(Path | bytes, *, session_id, name=None)` resolves relative paths against the runner
workspace, processes and saves the image in a worker thread under
`.iris/image-cache/<encoded session>/`, and returns an `ImageBlock`. Importing does not create a run
or occupy its session lane; later source-file changes do not affect the copy. Image-only `[image]`
input is valid; empty strings/lists and whitespace-only text lists are rejected. Idle, steer, and
follow-up `SessionManager.submit()` calls accept the same `str | list[DataBlock]` input. Import
before submitting so the admission lock only handles saved references. Start, HITL resume, and
recovery reuse complete blocks from the durable request/history without importing again.
Closing the runner preserves image files. SQLite recovery and forks still depend on them, so backups
must include `.iris/image-cache/`; the database alone is insufficient, and paths are not rewritten
automatically when moving to another machine.
`ContextBuildScope.run_input` remains text-only; image-only `ForkPoint.input` displays
`[image: name]` labels. Goal continuation and subagent prompts remain text without implicit parent
image forwarding.

MCP construction does not connect. Call `aprepare()` to warm up, or let the execution entry prepare
automatically. The complete catalog publishes before creating a run. Required preparation failure
closes resources without creating a run; construct a new runner after correcting configuration.

Concurrent entry points on one runner share resource preparation. Cancelling one waiter does not
cancel that preparation; closing the runner settles it before closing owned resources. Each actual
run still registers its own memory foreground activity.

With `goal.enabled`, the runner assembles GoalService, tools, and dynamic context using the same
lifecycle store. A Goal run reuses normal start registration, steering, and command lifecycle.
Its automatic input is marked as context and does not replace the latest ordinary-user anchor.
`start()` still executes one logical run. See [Goal documentation](../goal/README.md) for models
and reporting rules.

When enabled, `SessionManager(runner, session_id).goal` provides async create/get/edit/pause/resume/
complete/clear. Creation explicitly arms continuation; automatic work stays outside the user FIFO
and follows accepted user work. Pause stops later rounds. `interrupt()` also pauses the goal while
cancelling the current run; it returns None when it only stops an idle goal. Disabled managers expose
`goal=None`.

See the [Goal SDK example](../goal/README.md#从配置到执行) for configuration and the full
Runner → Manager → create/get/pause/resume sequence. GoalSession, GoalView, GoalControlResult,
and GoalChanged are also exported from iris.harness. Control dispositions distinguish scheduled,
admitted, running, waiting, needs_recovery, occupied, and stopped; scheduled does not promise
execution has started. Admission consumes a nonrefundable round, resume does not reset it, and
Run deadline_at remains absolute. Completion is reported by the main model or user, not independently certified.

A new manager does not automatically continue a persisted active goal. Explicit `goal.resume()` can
attach the original WAITING run, restore its deadline timer, and accept its pending interaction.
An ACTIVE run without a live invocation requires `expected_activation_id`. Recovery consumes no
new round. Background deadlines and command cleanup errors notify Goal state without requiring a
live publisher. Pending command cleanup retains its lane until explicit settlement retry.
Answer the original typed WAITING interaction through manager.resume(interaction_id=..., response=...);
this preserves the run and round. Default children and history forks do not inherit goals; children
explicitly enabling Goal are rejected.

`GoalChanged` carries the latest state. Mixed streams deliver the related terminal before the goal
result; broker-only mode publishes session-scoped `goal.changed`. Reads neither reconcile nor start
work. Full mixed trackers defer admission until consumption releases capacity. Goal and user follow-ups
share memory handoff; reservations cover only transition to the next foreground run. Waiting for user
input, HITL, or capacity permits normal idle maintenance. Manager close unregisters its attachment,
while Runner keeps ownership of resource shutdown.

Resume/recover keep pure durable settlement first, preparing only for execution. They then reload
state, checkpoints, claims, and time, and continue with the current runner configuration.
Terminal reads, ordinary waiting expiry, unresolved-CLAIMED unknown recovery, queries, history forks,
and cancellation requests do not depend on MCP connections.

Run creation and resume/recover checkpoint validation use `load_session_revision()` instead of
loading full history just for its revision. Validation still independently compares against the
current store revision. Commit-port initialization reuses the loaded checkpoint revision. Input
preparation calls the bound port's `load_session_header()` for metadata; main model steps call
`load_model_context(include_tool_discovery=...)` for a consistent effective-history snapshot.
Both refresh the port's session revision. The latter delegates to the store's
`load_run_context(run_id, ...)`, returning the summary, uncovered suffix, current-run protected
anchors, and requested discovery projection. Public `get_session()` still returns complete raw history.

Root connections span multiple runs. Stop new calls and await the original start/resume/recover calls
fully before `aclose()`. A cancelled terminal includes required command cleanup but does not prove
the original call's event delivery has exited; an observation timeout does not prove settlement.
Active closure raises. Repeated closure is idempotent after success; failed cleanup remains retryable
while new business calls stay closed. Durable queries remain available.
`SessionManager.close()` does not own runner resources: use `close(cancel_run=True)` before
closing the runner.

Ordinary child YAML can configure MCP independently. Fresh children prepare
before admission. Failure returns `SUBAGENT_PREPARE_ERROR` without a child run/link. Every child
WAITING/completion, recovery failure, or early return closes its child-owned resources. Resume/recover
rebuilds with current configuration. The root command service, absolute deadline timers, and pending
cleanup survive temporary child runner closure; settlement reopens the exact child run through its route.
Parent and child connections remain
independent, with the more restrictive combined permission policy. Live cancellation borrows the runner
and waits for its original task; the creating scope closes resources. Closure failures are logged without
replacing outcomes. Non-live cancellation persists its request before any required preparation.

## Run Hooks

Both `AgentRunner.from_config()` and `from_config_path()` accept `hooks=` (a sequence of
`HookRegistration`) and `tool_middlewares=` (constructed `ToolMiddleware` instances), each empty
by default. YAML `hooks` / `middleware.tools` come first and SDK items follow, without name-based
replacement or deduplication. One assembly creates one set of instances; parent SDK additions do
not flow into children. See [Hooks](../hooks/README.md) for configuration and command protocols,
and [Tools](../tools/README.en.md) for the wrapping contract.

```python
from iris.hooks import HookEvent, HookRegistration


async def report_finished(event: HookEvent) -> None:
    print(event.event, event.run_id)


runner = AgentRunner.from_config_path(
    "agent.yaml",
    hooks=[HookRegistration(
        event="run.finished", name="report-finished", handler=report_finished
    )],
)
```

The environment's `HookDispatcher` serves both tool execution and harness. `run.started` runs
after preparation, Run admission, and active-task registration, before the first model call.
Ordinary, Goal, and child starts share that boundary. Preparation failure or rejected Goal admission
does not dispatch it; WAITING, resume, and recover do not repeat it. Started handlers run under
the existing cancellation and absolute deadline. Command unknown outcomes, cleanup failures, and
SDK task cancellation retain their original meanings without fabricated tool claims or checkpoints.

Only a new terminal commit dispatches `run.finished`, including outcome-ready recovery finalization.
Python handlers apply to every terminal reason; command handlers apply only to COMPLETED.
Ordinary failures are logged without changing the durable result. Finished runs after command
settlement removes its pending entry, uses its own timeout after the Run deadline is removed, and
does not wait for slow observers. Reading historical results does not replay it; it is neither a
resource-release mechanism nor a delivery guarantee.

When applicable finished handlers exist, root registers their task before the terminal becomes
visible. Direct SDK admission for that session temporarily raises `IrisRunStateError`.
`SessionManager.submit(mode=None/auto)` waits outside the admission lock and then retries admission
with its original arguments, without reserving a place. Explicit steer rejects terminal Runs;
follow-ups retain FIFO order and completion actively wakes the queue. Goal retains its managed-task
wait and also observes shared completion and admission errors. Without applicable finished handlers,
the existing concurrency between observers and later Runs is preserved.

Cancelling an ordinary submit waiter or detaching a Manager with default close does not cancel
finished. Cancelling the actual start/resume/recover driver, or `close(cancel_run=True)`, interrupts
finished and drains its current command before returning. Postterminal cleanup failure reports
`IrisCommandCleanupError` and blocks new Runs under root, while existing cancel/resume/recover and
resource closure remain available. It does not create retryable pending settlement or finish again;
the host waits for existing drivers, closes root, and rebuilds.

Live and rebuilt children borrow root's completion owner while using their own Agent's handlers.
Child admission also checks root errors after preparation and immediately before its store commit.
Closing a temporary child does not destroy shared tasks. Root `aclose()` waits for actual finished
tasks, including those started by background deadlines after the original child closed, before
closing the shared command service.

## Current-session context reads

`context_policy.enabled` defaults to `true`. `AgentRunner.from_config*()` constructs internal
`ContextAccess` against the same lifecycle store and registers `context_read` and `context_search`
through shared runtime assembly. Hosts need no `file.read`, memory service, or additional storage.
Set `enabled: false` to omit these tools.

Each call uses its tool execution session ID. Root and child read their own committed history;
children do not automatically read the parent. `message:<index>` and
`result:<message_index>:<block_index>` use zero-based original positions that summaries do not
renumber. Forks inherit prefix positions and artifact references without copying files.

Small results come from lifecycle history. Large results use the committed artifact to retrieve
final text or native MCP JSON without rerunning the source tool. Inline text and image references
retain their order in paginated text; image references include names, MIME types, original/model
paths, and dimensions. Search can match image names and references, but reads neither pixels nor
entire artifact files. Existing text_path files are paginated directly, without repeating image
reference prefixes on each page. See [tools](../tools/README.en.md#current-session-context-reads) for page
parameters, errors, and scope. Harness owns the read service; runtime receives a narrow interface
without store ownership.

Images participate in ordinary main-model requests with their messages; viewing one does not
automatically remove it. Compaction preserves complete images in the current run's initial input,
latest steer, and retained tail. Summary calls receive only existing text and image references,
so the summary model needs no vision capability and cannot infer previously unexpressed details
from a reference. After an old prefix image leaves active context, use `context_read` to obtain its
model path, then `read_file` through configured `file.read` to bring the image back. Without the file
tool, references remain readable and the host can resubmit the existing `ImageBlock`; the framework
does not register file tools automatically.

## Dynamic host context

`AgentRunner.from_config()` and `from_config_path()` accept optional `context_source=` implementing
`iris.context.ContextSource.collect(scope)`. Runtime collects once per admitted main model step;
the host chooses content, `required`, and `priority`. See the
[context example](../context/README.en.md#dynamic-host-snapshots). Supplying a source with
`context_policy.enabled=false` raises `IrisConfigError` during assembly.

The source belongs to this runner. Its sessions can collect concurrently, so applications use the
session/run scope to distinguish state. Each snapshot enters only the current request, not lifecycle
history or checkpoints. It supplies current application state while BCI retains the run's initial
background. Recovery at `before_model` recollects; children inherit neither the parent's source nor
its snapshot. Preserve durable evidence through ordinary tool results or host files.

## Deferred tool disclosure and recovery

`context_policy.deferred_tools: true` automatically registers `tool_search`. After a successful
search result commits, its candidates become eligible for complete schemas in the next main model
request. Disclosure facts belong to that session's raw history; sessions sharing a registry do not
share their revealed sets. Forks inherit only discoveries in the copied prefix, and children do not
inherit parent disclosure state. MCP still prepares before execution; deferred schemas do not delay
connections or discovery.

Checkpoint v4 stores the current batch's visible names in `engine_cursor.visible_tool_names`,
committed with its assistant calls. WAITING, partial progress, and recovery retain that set without
rerunning source collection or schema selection. Permissions still refresh before execution.
Completing the batch clears the set for selection at the next main step. See
[runtime](../runtime/README.en.md#deferred-tool-definitions) for budgets, first-candidate protection,
and forced tools.

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
`__call__(config: AgentConfig, *, config_path: Path) -> CompletionProvider`. Import `CompletionProvider`
from `iris.providers`; it requires `complete()` and `estimate_input_tokens()`. The factory receives
the loaded ordinary child configuration, independently of one-off parent provider credential overrides.

`from_config*()` accepts `permission_policy=` and `child_provider_factory=`. With a catalog,
the runner reads one route snapshot and assembles an internal controller. The selected child uses
ordinary AgentConfig, independent session/run IDs, fresh `AgentRunOptions()`, empty request
metadata, and the parent's store/clock. Each child constructs memory from its own configuration and
effective workspace without inheriting the parent's service or dynamic snapshots. CHILD excludes
subagent. Linked ACTIVE
runs continue through ordinary recovery; WAITING/TERMINAL paths read the existing result. Dedicated
execution creates a parent proxy when the child waits, keeping the tool PREPARED. The host submits
answers only to parent `resume()`: the response becomes durable before the exact child continues.
Another wait replaces the proxy; terminal finalization advances the parent cursor once, applying
parent identity, artifact handling, and tool error policy.

After a crash following response persistence, `recover(parent_run_id)` continues a RESOLVED proxy
or outer permission from the stored response. Ordinary PENDING waits still require `resume()`;
ACTIVE recovery still requires its activation fence. A fresh process uses the durable selector
against its current catalog snapshot and continues the existing parent/child runs. Linked calls
skip outer permission; stored approval without admission still requires execution refresh.
Successful WAITING finalization publishes `tool.completed` with the original tool activation before
running the fresh RESUME activation. SessionManager admits resume before the first child await and
rejects concurrent answers.

Parent cancellation, deadline, and parent-owned proxy expiry settle the exact child before ending
the parent. `request_cancel()` leaves a linked proxy WAITING; `cancel()` or the manager's settlement
task completes it. `settlement_timeout` covers child cleanup and parent observation, preserving
durable cancellation on timeout so later `cancel()`/`recover()` can finish. A child stop receipt belongs
only to the current parent call. If a timer already finished the child, delayed proxy handling still
consumes that receipt rather than stopping a later environment. Repeated interrupts share
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
`resume()` without managing a child runner. Recovery uses the stored selector to find the child
configuration in the current catalog. Description changes do not prevent recovery or create a replacement child.
If the child has closed its HITL interaction but not committed the tool result, ordinary ACTIVE
recovery restores its stored response. Child `IrisRunRecoveryError` propagates unchanged, preserving
recoverable parent/child state.

- `start()` atomically creates a run/start activation and advances it to waiting or terminal.
- `resume()` consumes the exact waiting interaction.
- `request_cancel()` guarantees only that the first request is durable. A local active activation
  is signalled after commit; waiting remains waiting until asynchronous cleanup and settlement.
- `cancel()` requests cancellation and observes durable settlement. Observation timeout writes no
  new fact, and settlement does not imply that the original `start()` / `resume()` call has exited.
- `recover()` requires the exact active activation fence. Safe checkpoints create a recover
  activation, outcome-ready checkpoints only finalize, and unresolved claims become
  `outcome_unknown`.
- `get_session()`, `get_run()`, `get_run_control()`, `get_result()`, `list_tool_calls()`, and
  `list_events(after_sequence=0, limit=None)` are side-effect-free durable reads. When provided,
  `limit` must be a positive integer.

`resume()` passes the current interaction ID, run revision, interaction version, and typed response
to the Store without echoing the stored call fingerprint as a resolve argument. Actual tool
execution still retains fingerprint binding.

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
response never enters the ordinary-input queue. `interrupt()` pauses Goal continuation, then requests
cancellation of the exact current run. It returns None when only an idle Goal was paused. An active
cancellation request is not terminal, so follow-ups still wait for actual settlement. For WAITING runs
and ACTIVE runs left without a live continuation after cleanup failure, the manager owns one async
cancel task. A later interrupt retries pending cleanup; a new run never inherits the old cancel owner.
`close()`
rejects later operations, fails every pending input with `session_closed`, and ends the event
stream, but neither cancels nor waits for the current run.

A host about to close its event loop uses `close(cancel_run=True, reason=...)`: close admission and
fail pending input first, prevent another follow-up from starting, then cancel and await the current
run through the runner. It then waits for owned managed tasks to end, including a WAITING parent
awaiting its child and an earlier terminal run still delivering events. Failed cleanup keeps admission
closed while retaining the original run/tasks for a later `close()` retry. Closure becomes idempotent
only after success; cancelling a close waiter does not cancel the owned close task.
The mixed event stream ends even when cleanup fails, so its consumer cannot block host exit.
CLI `/exit`, EOF, Ctrl-C, and error exits
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

The root owns one command service shared across its sessions and children. Normal COMPLETED/WAITING
keeps the environment. Abnormal exits settle linked children, stop and drain command work, then
call `FinishRun` to release the session lane. Docker stops the shared container without cancelling
other runs' model/native/HITL work; Native stops current commands in the target session. A current
causal receipt waits for that stop operation without stopping an environment already restarted.

Process-local `PendingSettlement` retains the original outcome/error, typed target, and receipt.
Concurrent callers join one stop/drain/finish task; cancelling a waiter does not cancel settlement.
`IrisCommandCleanupError` leaves ACTIVE/WAITING and its lane intact, reaches direct callers, and
publishes a small root-owned `CommandCleanupFailed` live fact (or logs without a publisher).
The next cancel/recover/resume retries the original settlement before ordinary dispatch. It neither
changes the original failure cause nor reruns models or commands. Known tool results commit before
cleanup errors propagate and never revert to unknown claims.

Root-owned deadline timers survive WAITING and temporary child runner closure. Typed child routes
rebuild the runner when needed; terminal settlement removes its timer. Root close cancels unfired
timers, waits for fired settlement and pending work, then closes resources. A failed close can be
retried, while new business calls remain closed. Already-expired starts acquire ACTIVE/fence/lane
without committing input, and budget refusal does not terminalize in the store: both await cleanup.

When settling failure, the runner checks the absolute deadline against its injected Clock,
independently of timer scheduling. Provider exceptions, `response.failed`, and provider cancellation cleanup
failures after the deadline settle as `DEADLINE_EXCEEDED`; errors before it remain `FAILED`.
An uncommitted tool claim still takes precedence as `OUTCOME_UNKNOWN`.

`cancellation_requested` is a durable fact, not settlement. The runner persists the request before
sending the local signal and does not cancel the entire activation task while a claim exists.
`ToolExecutor` translates the signal into cancellation of an ordinary async callable, custom async
`BaseTool`, or THREAD callable body task, also interrupts the awaiting Middleware chain, and waits
for started operations to settle. Coroutines that suppress `CancelledError` and INLINE blocking
can still delay settlement; `cancel()` waits
for a durable terminal result without returning cancelled early.

If the body has completed, or returns normally after signal-driven cancellation, its result goes
through postprocessing and the existing ordered durable commit before the run settles cancelled.
Finite local file and artifact IO recovers its known result after cancellation and commits the tool
fact before responding to task cancellation, timeout, or sibling cancellation. Parallel results
still commit only a contiguous ordinal prefix. External task cancellation without an Iris signal
first cleans command resources and linked children, then propagates, leaving recoverable ACTIVE facts.
Repeated task cancellation does not skip that cleanup; failures retain cleanup-only pending work.
When cancellation during Middleware postprocessing also leaves a cleanup error, the Runtime hands
both the original stop reason and the live cleanup error to the runner. Pending settlement preserves
the cancellation or deadline intent; SDK task cancellation remains cleanup-only rather than FAILED.
Unresolved claims still settle the run as
`TOOL_OUTCOME_UNKNOWN`, including read-only calls. Custom THREAD callable workers may continue, but late returns
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

Todo's single check target is stored in the v4 cursor's `todo_reminder_step`, initialized to None
for each new run. Recovery rereads the file at that step and reuses any pending model reservation.
Once its response is committed, later tools or HITL do not repeat the check; outcome_ready only
finalizes. The result contains the actual final assistant response, though streaming may already
have shown the earlier candidate.

Active recovery validates checkpoint v4, session revision, usage counters, and cursor.
Tools are never replayed while unresolved claims exist. Recovery atomically
abandons the old activation and acquires a new RECOVER fence with a BLOCKED_UNKNOWN checkpoint.
Only after cleanup does it close claims as unknown and create the terminal result.
Normal parent/control/infrastructure exit waits for runtime children to drain before revoking the
commit port, preventing late child writes. Synchronous blocking callables have no concurrency
speedup guarantee and may still delay settlement.

Recovery uses the current runner's model, context, tools, and permission configuration. Changes to
system prompts, compaction budgets, or tool catalogs do not trigger a global configuration equality
check. Saved requests, run limits, cursors, call identities, and execution results still come from
the original run. Pending tools must satisfy argument and current permission rules. Run records and
checkpoints do not store an environment fingerprint.

Runner construction freezes system/context templates, Goal/Todo, Memory context guidance, Skill
catalog, and Decision instructions together with their dependencies. Later model steps still pass
current data; an existing runner does not reload template edits. Each complete compaction takes a
new source snapshot, and each automatic Memory cycle takes one shared by all its stages. Rendering
uses the shared [`TemplateRenderer`](../utils/README.md), with autoescape disabled unless an XML
template opts in. `StrictUndefined` applies during rendering, and context character limits apply
after complete text generation. `system` / `context` keep their existing configuration entry points;
see [`iris.context`](../context/README.en.md).

Start, resume, subagent parent resume, and recovery pass `run_input` and
`initial_session_message_count` from the durable run. At `before_input`, a new run archives BCI and
user input. The first session input also atomically commits the selected `SessionContextWindow`
with input and checkpoint before advancing to `before_model`. Window initialization advances the
session revision once even without a message delta; provider failure does not undo committed input.

With an effective runtime memory service, tool loops, later runs, HITL, and recovery replay the committed
overview without re-querying or reloading updated files. Successful compaction replaces summary and window in the same transaction;
failure retains the old state. Fork targets start without an adopted window and choose one at their
first input. Checkpoints bind the session revision without duplicating window text. Lifecycle SQLite
uses schema 11 and checkpoint version 4; old formats are rejected without migration or cleanup.

A new runtime without a memory service omits the saved overview from system messages during ordinary
requests, HITL, and recovery. This does not mutate the saved window or add a session revision change;
static memory and ordinary history remain available. Successful compaction while disabled commits an
empty window in the existing transaction. Failure keeps the previous summary and window, while later
requests continue to omit that overview.

`memory.enabled` defaults to false. Enabling it registers Search/Fetch automatically; writes remain
explicit. Disabled memory does not attach an injected service; enabled memory preserves that service's
dependencies. Rebuild the Agent and start a new session after a change; hot switching is not supported.
Models choose Search/Fetch themselves. Results enter ordinary tool history and may be compacted,
with no cross-turn seen registry or special text pinning. The overview defines topic scope within
2% of the available input budget across namespaces; without one, ordinary chat continues without
long-term memory use. Static `context.yaml` memory stays independent and outside session history.
BCI, original user input, and latest steer keep their existing protection; recovery after compaction
uses the new revision and the same pending model step.

`memory.generation.enabled: true` enables automatic capture and maintenance for root runners;
it defaults to false. The host creates one [`MaintenanceCoordinator`](maintenance.py) and binds
its root runners through `runner.bind_maintenance(coordinator, memory=binding)`. `from_config*`
never creates a private maintenance loop. An enabled but unbound runner fails at its first
prepare/run boundary. The host selects `config.maintenance.idle_seconds` (300 seconds by default).
Runners without Memory can still call `runner.bind_maintenance(coordinator)` to contribute
foreground activity, pausing maintenance from other runners without creating a Memory resource.

First resolve the root workspace and initialize one `PromptSource`, then construct a complete
`MemoryService` with generation and overview providers/models plus a mirror. Pass the same source
to the service and runners, and the same service to runners and `MemoryMaintenanceBinding`.
The binding names the actual SQLite path and write namespace. Injected providers, budgets, IO mode,
and prompt source are preserved. This example assumes Memory automatic generation is enabled:

```python
from pathlib import Path

from iris.agents import AgentConfig
from iris.harness import AgentRunner, MaintenanceCoordinator, MemoryMaintenanceBinding
from iris.lifecycle import AgentRunRequest
from iris.memory import build_memory_service_from_config, resolve_memory_path
from iris.prompts import PromptSource
from iris.providers import CompletionProvider


async def run_sessions(
    config: AgentConfig,
    provider: CompletionProvider,
    config_path: Path,
) -> None:
    workspace = (config_path.parent / config.permissions.workspace).resolve()
    prompt_source = PromptSource.initialize(workspace, config.prompts.root)
    memory = build_memory_service_from_config(
        config.memory, workspace, prompt_source=prompt_source,
        overview_provider=provider, overview_model=config.model.name,
    )
    coordinator = MaintenanceCoordinator(idle_seconds=config.maintenance.idle_seconds)
    binding = MemoryMaintenanceBinding(
        service=memory,
        database_path=resolve_memory_path(config.memory.path, workspace),
        namespace=config.memory.write_namespace,
    )
    runners = [
        AgentRunner.from_config(
            config, config_path=config_path, provider=provider,
            memory_service=memory, prompt_source=prompt_source,
        )
        for _ in range(2)
    ]
    try:
        for runner in runners:
            runner.bind_maintenance(coordinator, memory=binding)
        for index, runner in enumerate(runners):
            await runner.start(AgentRunRequest(input="Process the project task", session_id=f"session-{index}"))
    finally:
        await coordinator.aclose()
        for runner in runners:
            await runner.aclose()
```

[`_capture.py`](_capture.py) records source material without scheduling learning. A root Run
registers its lifecycle source ID, run and session before its first input commit. Compaction
hints, WAITING/terminal exits and close capture only the unrecorded suffix, committing pages
of at most 128 messages. Only the final terminal cutoff seals the source. Completion, failure,
cancellation and recovery without runtime execution retain their actual outcomes. BCI,
system/reasoning and memory readback bodies are not new evidence. Search/Fetch retain item
references; tool calls/results retain their call IDs. Durable watermarks prevent duplication.

Automatic learning consumes only terminal, fully captured Runs. A WAITING session excludes
its older pending sources while other sessions can still be maintained. Tool-origin changes
are linked through captured call IDs and remain pending until matched. Eligibility is reread
before each consumption commit. Missing lifecycle readers leave material pending; SQLite
lifecycle supports restart continuation, whereas lost in-memory state is never reconstructed
as a second eligibility database.

A host runs at most one Memory job and one project evolution job concurrently, each with its own
worker, cancellation state and resource lock. Cancellation requests use each asyncio Task's state
directly rather than a separate mirrored flag. Canonical database path plus namespace determines the
native OS lock. Each bounded cycle rereads material under the lock and holds it through real
IO cleanup, preventing duplicate model calls across independent processes for the same resource.
A busy lock yields the local slot and retries after `max(idle_seconds, 1 second)` without user
input. Multiple resources receive bounded turns. MemoryService owns the dream-first or
flush-then-dream sequence and projection/overview repair; the coordinator does not interpret content.

Hosts construct `build_project_evolution_binding(config, workspace_root=workspace,
prompt_source=prompt_source, provider=provider)` and share it through
`runner.bind_maintenance(coordinator, evolution=project_binding)`, optionally alongside
`memory=binding`. The selected main configuration owns maintenance policy; contributing runners do
not replace it. Evolution uses a workspace lock that never nests with a Memory lock and can run with
Memory disabled. `await coordinator.request_project_experience(project_binding)` requests one pass
without the usual idle delay, while retaining foreground, eligibility and locking rules. No eligible
material returns empty without a model call. After its borrowing runners close,
`await coordinator.unbind_evolution(project_binding)` removes only that resource.
See [evolution](../evolution/README.md) for configuration and artifact adoption.

With `config_targets`, the factory also requires `config_path` pointing to the primary YAML.
Hosts can call `await coordinator.request_revision(project_binding, RevisionRequest(...))` with a
description, finite `RevisionTarget(kind="prompt" or "config", name=...)` values, and an optional
`EvolutionSession(lifecycle_source_id=runner.store.source_id, session_id=...)`. Session-bound requests
respect that session's WAITING state; requests without a Run do not fabricate task evidence.
Explicit A returns an A result; explicit B waits for its own request ID. Failed or cancelled B stays
pending for later external activity. Automatic A saves its issue and releases the project lock before
B reacquires it in another cycle and rereads current targets. Publishing config never rebuilds runners.

Maintenance starts only after foreground admission/activation calls fully exit and the idle
interval passes. New foreground work cancels uncommitted generation without waiting for a
model or resource lock. Goal/follow-up handoffs reserve the same foreground counter. THREAD
services use a dedicated worker; INLINE retains the calling thread. Actual synchronous work
must finish before its slot and lock are released. Model failures wait for external activity
or restart; the cycle's own writes are not new activity. No-change advances consumed inputs.

`runner.aclose()` drains its capture and detaches its borrowing relationship, without closing
shared coordination, services or readers, and without waiting for another runner's foreground
counter. After closing all runners that borrow a resource, the host can call
`await coordinator.unbind_memory(binding)` to cancel/drain and remove just that resource.
Other resources remain available. At host shutdown, close the coordinator before releasing
host-owned providers, services and lifecycle stores. Close never runs extra learning models.
Maintenance usage remains separate from `RunUsage`; a new overview is adopted at a new context
window or successful compaction.

`RunUsage` input/output/total fields count only the main model. Summary calls accumulate under
`usage.compaction`; add the corresponding fields for combined consumption. A summary usage-only
commit advances the run revision without a durable event. The commit port accepts that revision
immediately so later cancellation reads remain valid. Projection commits independently emit
`context.compacted` while retaining raw history.

## Public API

`iris.harness` exports `AgentRunner`, `MaintenanceCoordinator`, `MemoryMaintenanceBinding`,
`ProjectEvolutionBinding`, `build_project_evolution_binding`,
`SessionHistory`, `SessionManager`, `SubmitReceipt`,
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
