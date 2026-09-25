[中文](README.md)

# `iris.runtime`

`iris.runtime` is the low-level inner engine for Agent lifecycle. Starting from a durable
`RuntimeCursor`, it uses a caller-provided `RuntimeCommitPort` to advance provider and tool work
until completion, waiting, budget exhaustion, cancellation, deadline, failure, or unknown outcome.
It does not create logical runs, select stores, or own public cancellation/recovery orchestration.

Use `iris.harness.AgentRunner` for complete runs. Call `AgentRuntime.execute()` directly only when
implementing a custom lifecycle owner.

MCPTool uses the ordinary serial tool path. `IrisMCPOutcomeUnknownError` reaches the existing
`_unknown_tool_outcome` settlement without new stop reasons or persistence protocols. Trusted
read-only SDK failures remain ordinary ToolResults governed by ToolErrorPolicy. MCP adds no
separate cancellation watcher.

Shared assembly reads `AgentConfig.mcp` declarations and binds an `MCPManager` to the original
registry without connecting. `RuntimeEnvironment.aprepare()` / `aclose()` delegate to that manager.
Low-level callers prepare before execute and close after all execution finishes. The environment
does not close injected providers, memory, or stores. Root runners prepare automatically and reuse
one fixed catalog and connections across runs. Harness prepares children before admission and closes
their independent resources at WAITING/completion, rebuilding on recovery.

Assembly resolves the provider first, then lets the memory factory handle `memory.enabled`.
Disabled memory does not attach even an injected service. When enabled, injection takes precedence;
otherwise the factory builds a SQLite service with the resolved provider, model name, overview, and
generation configuration. A resolved service automatically provides Search/Fetch; write tools remain
explicit. Construction makes no model calls. Hosts can call `refresh_overview()` explicitly; root runners
with automatic generation enabled also publish during idle maintenance. Injected services keep their
own generation dependencies. The switch is fixed at construction; rebuild the Agent and start a new
session after changing it.

`RuntimeEnvironment.execution_scope` preserves ROOT/CHILD explicitly; root harness owns automatic
maintenance. Only after `_compact_request()` selects a real compaction range does runtime call the
optional `RuntimeMemoryCapturePort.request_capture(run_id, through_count)` with the committed message
boundary. Ordinary requests and early returns without a compressible prefix send no hint. This
synchronous port waits for neither IO nor memory models and leaves `RuntimeCommitPort` responsible for
durable execution facts. Harness owns background capture/flush/dream and shutdown. Compaction need not
wait for newly generated memories; successful compaction adopts only the overview already published.

## Dependency direction

`iris.providers.CompletionProvider` must implement both `complete()` and synchronous
`estimate_input_tokens(request)`.
The latter estimates the complete request after model options and tool schemas are applied. Custom
providers and test doubles use the same contract. Compaction configuration is carried by
`RuntimeEnvironment.agent_config.compaction`; no separate environment field is needed.

```text
AgentRunner -> AgentRuntime.execute -> RuntimeCommitPort
     |                                  |
     +------------ LifecycleStore <-----+
```

- `RuntimeFactory` assembles context, provider, tools, workspace, and optional memory only.
- `RuntimeEnvironment` contains engine live dependencies, not session/lifecycle stores or an
  interaction service.
- Runtime never imports harness or writes SQLite directly.
- Exact session, checkpoint, tool claim/result, and interaction writes come through the commit port.
- Commit, reservation, and claim DTOs are frozen dataclasses carrying validated in-process facts;
  wrapping a commit does not rescan its cursor. Full durable cursor parsing remains at load/recovery.
- The optional `RuntimeSteeringPort` supplies transient input only at safe boundaries for the
  current activation; it owns neither a queue nor persistence.

## Low-level contract

```python
result = await runtime.execute(
    activation,
    commits=commit_port,
    cancellation=cancellation_signal,
    steering=steering_port,  # optional; omission preserves existing behavior
    stream_sink=stream_sink,  # optional; enables in-process live events
)
```

`RuntimeActivationInput` carries run/activation/session identity, `start | resume | recover` kind,
frozen `RuntimeExecutionOptions`, and a JSON-safe cursor. `RuntimeActivationResult` is an engine
fact only; the owner must reload the final `RunResult` from durable storage.

Every activation carries the original `run_input` and `initial_session_message_count` captured
when the run was created. In `before_input`, the engine prepares the initial window, BCI, and user input,
then saves the window and input atomically through `commit_run_input()` before entering `before_model`. This
does not consume a model reservation or increment the step index. BCI is built only during input
preparation; later steps and recovery after the input commit replay history without appending or
rendering it again. Checkpoint version 2 rejects older checkpoints rather than guessing whether an
old `before_model/step0` cursor had archived its input.

`RuntimeCommitPort.record_compaction_usage(TokenUsage)` independently records each summary
response's usage. `commit_compaction(RuntimeCompactionCommit)` atomically replaces the summary
projection and context window against the session revision used to select its range. It advances session/checkpoint
revisions while preserving raw messages, the cursor, and the pending main-model reservation.
Every `before_model` checks the full input after receiving its main-step reservation; compaction
does not consume an additional model-step budget slot.

### History projection and summary construction

Internal `compaction.py` locates the current run's original input, latest archived steer, and injected
BCI in complete raw history. Projection orders the summary, covered anchors, and uncovered raw suffix.
The assembler places fixed system and static memory before history. BCI is marked with
`context_kind=before_current_input`; Search/Fetch results are ordinary tool history and are not
pinned. A single `<summary>` wrapper is added only in the projection; summaries never append to raw
history. Cuts keep each assistant tool
batch and its results together. Recent retention is a soft target: an oversized group can be summarized
while retaining smaller recent groups, or leaving an empty suffix. Completed steps within the current
run are eligible too.

`_compaction_summary.py` serializes every text block, call argument, result, and required error/artifact
reference in order. Large blocks carry character coverage markers separately from execution status.
Each batch is measured with the current working summary; unprocessed fragments are never dropped.
Complete record prefixes use exponential probes followed by binary refinement, with character splitting
only for the next record when needed. This avoids recounting every growing prefix. Every returned batch
is measured as a complete request; selection does not promise maximum packing or cache working summaries
across batches. Cut-point planning reuses identical empty-suffix estimates without changing message-group
boundaries or the recent-history target.
Summary instructions come from a separate Jinja2 file. The bundled
[`prompts/compaction.j2`](../prompts/compaction.j2) requests seven Markdown headings with body text in
the conversation's primary language. `compaction.prompt` can replace the instructions and output
format; Iris still supplies the previous summary and current history batch, with the user-message
wrapper stored in [`compaction_input.j2`](../prompts/compaction_input.j2). Each compaction obtains
instructions directly from `RuntimeEnvironment.prompt_renderer`, strips leading and trailing whitespace,
and reuses them for all batch estimates and requests.
Jinja reuses compiled templates and detects edits by mtime for the next compaction. There is no heading
parser or format-repair loop. See [agents](../agents/README.en.md) for path configuration.

The environment holds `prompt_renderer` as a shared `iris.utils.TemplateRenderer` instance. Autoescape
is disabled by default, preserving JSON, quotes, and `<>&` in summary inputs. Runtime converts template
loading and rendering failures to `IrisContextError`, retaining the `context` error source.

Summary requests reuse effective main-model options but override streaming, tools, response schema,
output cap S, and `num_retries=0`. Candidates stay in memory until all batches finish and the caller
commits them. Only complete nonempty text is accepted. `IrisContextCompactionError` uses the existing
`context` source with `CONTEXT_COMPACTION_*` codes.

At 80% of usable input budget B, runtime selects a new prefix. With no new prefix, an input no larger
than B continues directly. When summarization starts, each returned response's
`RunUsage.compaction` is recorded before its body is checked. After all batches complete, the full
main request must fit within 80% and be strictly smaller than before to commit the projection.
Summaries never enter the main response's message delta or consume another reservation.

One operation deadline, 300 seconds by default, covers every batch and retry. Each request also
honors the remaining run deadline and any shorter request timeout. Only the failed batch gets one
retry for connection, timeout, or rate-limit errors. Runtime refreshes the remaining run deadline
before the main call; it never extends the original deadline. Cancellation keeps its existing
meaning, and queued steering waits for the existing main-response/tool boundary.

Once compaction starts, failure ends the current run while preserving raw history, the last
committed summary and window, and recorded summary usage. Recovery after the projection but before a main
response uses the new summary, committed window, and the same pending reservation. WAITING first resumes its tool
flow; `outcome_ready` only settles. Actual main-provider overflow has no extra compact-and-retry path.

Cursor positions are `before_input`, `before_model`, `tool_batch`, and `outcome_ready`.
`before_input` prepares and commits the input group before model reservation. A provider response without
tools is committed as `CheckpointResumability.OUTCOME_READY`. Tool effects require a durable claim
before execution and a durable result afterward. If an effect cannot be proven after claim, the
engine returns `TOOL_OUTCOME_UNKNOWN` and never replays it.

When the model returns a tool batch, runtime commits the assistant, all prepared facts, and the
initial tool cursor before advancing in original order. It commits ordinary call results before
atomically suspending the waiting checkpoint and interaction at the human gate identified by the
current `next_tool_index`. Resume does not repeat the committed prefix. Human responses bind to
that durable subject. If dynamic permission changes to ALLOW while waiting, an approval can
execute; if it changes to DENY, approval produces a permission error. User rejection remains
`USER_REJECTED`, and permission is still refreshed before execution.
Tool facts reuse a human request's existing call fingerprint when available; otherwise runtime
computes it from the exact arguments and workspace.

Tool execution uses `ToolBridge.preflight()` to produce a plan, then guarded
`execute_prepared()`. Tool results share the `ToolResult.to_msg()` projection into history.
Ordinary tools and child continuations share the executor's final output handling, retaining the
complete artifact reference while limiting model-visible text.

## Optional live streaming

Summaries always call `complete()` directly and expose neither summary text nor summary model
stream events to the host. Runtime emits `context.compaction.started` only for a new prefix,
`context.compaction.completed` after the projection commits, and then the main `model.step.started`.
An unfinished operation emits `context.compaction.failed`; the terminal run result explains the
cause. These statuses reuse existing identity fields without a separate payload model. Durable
`context.compacted` remains in event history.

`stream_sink=None` preserves the complete-only path exactly: runtime continues to call
`CompletionProvider.complete()` with `stream=False`, overriding `request_options`. With a synchronous
`RuntimeEventSink`, runtime
uses the independent structural `StreamingRuntimeProvider` capability to detect `stream()`.
Missing capability fails with `PROVIDER_STREAM_ERROR/provider`; runtime neither falls back to
`complete()` nor fabricates tokens.

The streaming path copies the trusted request with `stream=True` and direct-pulls the provider
async iterator. Runtime emits `model.step.started`, then synchronously wraps each
`ModelStreamEvent` as `model.event`. Partials, usage, and provider terminals remain live facts.
Only the complete `LLMResponse` carried by `response.completed` enters the existing `to_msg()`,
steering, tool-preflight, and `RuntimeModelStepCommit` path. A failed/cancelled terminal or EOF
before a legal terminal commits no assistant message, history, checkpoint, or tool call. Runtime
explicitly closes the typed provider iterator on terminal, EOF, failure, or cancellation; cleanup
failures produce a warning only. Provider completion does not imply that the durable commit succeeded.

Tool live events preserve the existing effect gate. Runtime emits `tool.preparing` only for a
complete `ToolUseBlock` before preflight; it emits `tool.started` after permission refresh,
activation fencing, and a successful durable claim but before middleware/body; it emits
`tool.completed` with the complete `ToolResult` only after ordered `commit_tool_result()` succeeds.
Parallel tool bodies may finish out of order, while completed events retain model ordinal order.
Runtime does not await the sink, create a queue, or catch custom sink errors; a later
harness-owned sink isolates publisher failures.

## Runtime steering

A custom lifecycle owner may pass an activation-scoped `RuntimeSteeringPort` to one `execute()`
call. `claim(run_id, activation_id)` returns at most one `SteeringInput` per safe boundary. This
frozen model contains only a non-empty `submission_id` and a `Role.USER` message. Runtime creates
no queue and writes claim state to neither the cursor, checkpoint, nor store.

Runtime claims only at two boundaries:

- after producing a no-tool assistant response and before `commit_model_step`; a successful commit
  puts the assistant and steer user message in one delta, advances to the next `before_model`, and
  uses `SAFE` resumability;
- after the final ordered tool result is known and before the final `commit_tool_result`; a
  successful commit puts the tool result and steer user message in one delta and retains the
  existing next-`before_model` cursor.

Runtime never claims between tool results, while a provider or tool effect is in flight, during
HITL waiting, at `outcome_ready`, after cancellation or deadline, or for a STOP terminal error.
There is no `await` between claim return and the synchronous commit plus `acknowledge()` / `fail()`:
only commit success is acknowledged, while a commit exception fails the submission with
`commit_failed` and propagates unchanged. Callback exceptions are logged and cannot replace the
durable result. Passing `None` or receiving `None` from claim preserves existing cursor, message
delta, resumability, and outcome semantics.

## Bounded tool concurrency

Under `RETURN_TO_MODEL`, runtime groups consecutive calls that are both read-only and declared
concurrency-safe into an internal window of at most 8 calls. Eight is a private implementation
bound, not a YAML, `RuntimeExecutionOptions`, or environment setting. This capability does not
change public config, schemas, models, or exports.

Only consecutive candidates share a window. STOP, HITL, preflight results,
WRITE/EXECUTE/NETWORK/MCP/AGENT calls, unsafe calls, and classification failures are serial
barriers; later calls cannot start across them. The batch reuses its first typed tool plan. Before
entering the body, each child refreshes permission and records its own exact durable claim without
repeating schema validation. Bodies may finish out of order, while result
messages, cursors, session history, checkpoints, and committed events advance only as the original
ordinal prefix. The order of multiple `TOOL_CALL_CLAIMED` telemetry events is not contractual.

A control interruption commits only the known `ToolResult` prefix before the first exception or
hole; a later in-memory result never skips that hole. Any uncommitted durable claim makes eventual
cancellation, deadline, or program interruption fail closed as `OUTCOME_UNKNOWN`, including
read-only calls. Runtime cancels and drains the children it created before a parent-task or
infrastructure exit completes.
Cooperative cancellation uses `iris.exceptions.IrisCancellationRequestedError`; runtime converts
it into an activation outcome rather than an ordinary tool error.

Ordinary async callables, custom async `BaseTool` implementations, and THREAD callables share
`ToolExecutor`'s body cancellation bridge: the signal cancels the body task and waits for cleanup.
A completed result or a normal return after signal-driven cancellation still enters existing
postprocessing and commit paths. External cancellation, timeout, or sibling cancellation also drains
the body and commits known results in ordinal order before propagating cancellation or settling the
deadline. `asyncio.timeout().expired()` preserves elapsed tool timeouts even when finite IO returns
normally after cancellation. Missing results retain unknown-outcome semantics.
The body bridge does not interrupt `before_call` / `after_call`; slow
middleware, coroutines that suppress `CancelledError`, and INLINE blocking can still delay exit.

Concurrent file reads share one `ReadFileState` identity. Workers only return immutable
observations, which the event loop merges. The checkpoint snapshot taken after the window settles
contains the combined records, so a later serial write barrier can retain stale-read checks. A raw
checkpoint dictionary is parsed once by `ToolBridge.restore_read_state()`; runtime then carries the
typed state and snapshots it directly.
Built-in write/edit jobs use a snapshot of immutable records, complete their local IO in a worker,
and merge only the returned file observation on the loop. Artifact normalization and writing also
run in workers. These finite local operations recover the actual result before ending the wait.
Synchronous callables remain inline by default; only explicit `CallableExecutionMode.THREAD`
placement moves a blocking body to a worker. Threads cannot be safely forced to stop. Cancellation
or timeout stops only the async waiter; when a claim remains unresolved, runtime settles as
`OUTCOME_UNKNOWN`, and the late result cannot advance history, cursor, checkpoint, or events.
Thread placement does not promise CPU speedup. Future NETWORK/MCP or write concurrency requires a new effect, retry, timeout,
conflict, and crash-reconciliation protocol rather than a relaxed classifier. This work adds no
delta/merge/lock/hash model.

## Memory overview windows and model-directed reads

When an effective memory service exists and a session's `context_window` is `None`, runtime calls
`MemoryService.aload_overviews()` once in
configured `read_namespaces` order. Selection uses the complete pending request, including this input's
BCI/user messages. The overview, input, and checkpoint commit together before the provider call.
An explicit empty window is already initialized. With a service, later runs, tool loops, steer, HITL,
and recovery after the input commit reuse saved text. A new session or successful compaction adopts current
overviews. Fork starts the target with `context_window=None` and adopts on its first input.

When `RuntimeEnvironment.memory_service is None`, the shared request builder passes an empty
`system_addendum`, even if the session retains a previous overview. Ordinary requests and recovery
do not read or rewrite that window or advance the session revision just to suppress it. Static memory,
BCI, and ordinary history, including prior tool results, remain available. Successful compaction
commits the new summary and an empty window in the existing transaction. Failure preserves the previous
summary and window, while subsequent requests without a service continue to omit the overview.

`full` includes core facts and knowledge scope; `navigation` includes only knowledge scope.
All namespaces, status warnings, actual tool guidance, and wrappers share
`floor(compaction.input_budget_tokens * memory.overview.system_budget_ratio)`, with a default ratio
of 2%. Cost is the provider's estimate difference between the same complete request with and without
the overview. Existing system text, static memory, history, and tool schemas are not charged twice.
If full exceeds this allowance or a reducible system/request limit, selection tries all knowledge
scope together. If that still exceeds the allowance, it reports a capacity error without dropping
namespaces or adding a third fallback. Existing compaction handles ordinary history capacity.
Selection also returns the chosen request's complete token count for post-compaction acceptance.
Identical full/navigation candidates are built and measured once. Reuse is confined to the current
adoption operation, with no request cache across persistence boundaries.

[`memory_context.j2`](../prompts/memory_context.j2) owns overview instructions, headings, and wrappers.
It uses the same `RuntimeEnvironment.prompt_renderer`; Python supplies overview and available-tool
data. Window budgeting still measures the complete request containing the rendered text.

`ContextBuilder.build(system_addendum=...)` appends the adopted overview after system-template output,
within the system character limit and outside message history. Static `context.yaml` memory keeps
its original position. Successful compaction commits the new summary, adopted window, checkpoint,
and event together, and the next main request uses that window immediately. Failure or cancellation
preserves the previous window.

Instructions define long-term memory scope through the current overview: unmentioned topics are
treated as absent and are not searched; covered relevant topics may be read as needed. Chat remains
available without an overview or mirror, but the window does not use long-term memory. There is no
database topic filter, so Search/Fetch can read current records within covered topics. Guidance lists
only enabled `memory_search`/`memory_fetch` tools and respects `include_tools`; Fetch alone reads known
IDs. Window guidance stays fixed while execution uses the current registry and permissions.
Search/Fetch results follow ordinary tool-history and compaction rules.

Window guidance asks the model to verify which entity and conditions each fact applies to; merely
mentioning an entity is not enough. It stops when snippets suffice and searches again only for
missing necessary information. The Search tool description owns parameter semantics: ordinary
query terms use OR, with optional `required_terms` phrases. `has_more` does not require exhausting
candidates. Fetch supplies missing body text or source metadata and can recheck a known ID's current
value. This adds neither automatic retrieval nor a fixed number of searches.

A configured SQLite service loads all namespace publications in one worker job. Runtime does not
consume late results after cancellation. Only explicit host calls to `refresh_overview()` generate
an overview; adoption does not generate one.

## Factory

Internal Sub Agent adapters on `ToolBridge` provide raw-only continuation preparation and dedicated
execution. The bridge alone builds `SubagentParentCall` from the existing run/call IDs. They neither
read the store nor add context identity or change ordinary file read state; the lifecycle caller
owns parent-loop integration. Before either batch preflight, runtime checks exact links or stored
outer responses and prepares those calls without permission checks while preserving model order.
Subagent remains a serial boundary under existing concurrency rules. Linked WAITING/terminal
outcomes use commit-port rebind/finalize without a parent effect claim. ACTIVE and WAITING paths
share tool-result cursor projection; WAITING delegates only final result normalization.
Harness owns the special branch's absolute timeout from child admission; runtime does not apply
ordinary tool timeout. After the child await, runtime checks parent cancellation/deadline before
rebind/finalize. Only fresh dispatch emits `tool.started`; linked recovery retains the logical start.

Internal `_assembly.py` assembles context, skills, providers, and tools. ROOT registers `subagent`
only with a preloaded routes/port bundle; CHILD always excludes it. The boundary resolver selects
the narrower parent/child workspace, rejects disjoint roots, and combines the actual parent policy
with the child's default policy. Public `RuntimeFactory.from_config*()` retains its ordinary
parameters and requires `AgentRunner.from_config*()` for `tools.subagent`, since delegation needs
the complete lifecycle owner.

```python
from iris.runtime import RuntimeFactory

runtime = RuntimeFactory.from_config_path("agent.yaml", provider=provider)
```

The factory never reads or creates a lifecycle database. Harness composition interprets the
`session` section of `agent.yaml`; the low-level factory has no persistence side effect from it.

The factory resolves `permissions.workspace` before constructing base context and user-declared
tools. With `skills.enabled: true`, it takes one project-level discovery snapshot against that
workspace. A non-empty result adds the `available_skills` system slot and registers `load_skill`
from the same registry before creating `ToolRegistryView` / `ToolExecutor`. Disabled Skills and an
empty result bypass both additions exactly, preserving the previous context/tool shape. A factory
or runtime instance does not refresh the snapshot automatically.

`RuntimeEnvironment.skill_registry` retains that shared registry.
`load_skill` reads the current text at the registered path each time and returns its first 1000
lines, including frontmatter, without parsing it again. File edits require no new run. Rebuilding
the runtime refreshes catalog names, descriptions, and paths; an old catalog description can
therefore coexist with a new description in the returned file.

A `skills.root` escape, missing `skills.require` entry, or name/alias collision between
`load_skill` and a user tool becomes an assembly-time `IrisConfigError` and fails closed. See
[`iris.skill`](../skill/README.en.md) for the full contract.

## Public API

Package exports cover `AgentRuntime`, factory/environment, `StreamingRuntimeProvider`,
`streaming_provider_for()`, `RuntimeEventSink`, `RuntimeStreamEvent`, assembler/tool
bridge, `RuntimeSteeringPort`, `SteeringInput`, and activation/commit-port contracts. Complete-run
options/status/results, `run_turn()`, `run_loop()`,
`resume()`, and old checkpoint helpers do not exist.
Import the shared non-streaming `CompletionProvider` protocol from `iris.providers`.

## Verification

```bash
uv run pytest tests/runtime
uv run ruff check src/iris/runtime tests/runtime
uv run mypy src/iris/runtime
```
