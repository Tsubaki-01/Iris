[中文](README.md)

# `iris.runtime`

`iris.runtime` is the low-level inner engine for Agent lifecycle. Starting from a durable
`RuntimeCursor`, it uses a caller-provided `RuntimeCommitPort` to advance provider and tool work
until completion, waiting, budget exhaustion, cancellation, deadline, failure, or unknown outcome.
It does not create logical runs, select stores, or own public cancellation/recovery orchestration.

Use `iris.harness.AgentRunner` for complete runs. Call `AgentRuntime.execute()` directly only when
implementing a custom lifecycle owner.

## Dependency direction

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

`start` and initial `recover` activations at `before_model / step 0` carry the current user input;
`resume` and later `recover` activations do not. The engine injects that field into the provider
request exactly once when present, while later recovery relies on committed session history.

Cursor positions are `before_model`, `tool_batch`, and `outcome_ready`. A provider response without
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

Tool execution uses `ToolBridge.preflight()` to produce a plan, then guarded
`execute_prepared()`. Tool results share the `ToolResult.to_msg()` projection into history.

## Optional live streaming

`stream_sink=None` preserves the complete-only path exactly: runtime continues to call
`RuntimeProvider.complete()` with `stream=False`, overriding `request_options`. With a synchronous
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
cancellation, deadline, or program interruption fail closed as `OUTCOME_UNKNOWN`. Runtime cancels
and drains the children it created before a parent-task or infrastructure exit completes.
Cooperative cancellation uses `iris.exceptions.IrisCancellationRequestedError`; runtime converts
it into an activation outcome rather than an ordinary tool error.

Concurrent file reads share one `ReadFileState` identity. Workers only return immutable
observations, which the event loop merges. The checkpoint snapshot taken after the window settles
contains the combined records, so a later serial write barrier can retain stale-read checks. A raw
checkpoint dictionary is parsed once by `ToolBridge.restore_read_state()`; runtime then carries the
typed state and snapshots it directly.
Synchronous callables remain inline by default; only explicit `CallableExecutionMode.THREAD`
placement moves a blocking body to a worker. Threads cannot be safely forced to stop. Cancellation
or timeout stops waiting and discards the late return; when a durable claim exists, runtime settles
as `OUTCOME_UNKNOWN`, and the late result cannot advance history, cursor, or checkpoint state.
Thread placement does not promise CPU speedup. Future NETWORK/MCP or write concurrency requires a new effect, retry, timeout,
conflict, and crash-reconciliation protocol rather than a relaxed classifier. This work adds no
delta/merge/lock/hash model.

## Explicit memory injection

`RuntimeExecutionOptions.memory_query` and `memory_results` are explicit opt-in dynamic memory
inputs. Each logical run injects them only on its first `before_model` step; provider requests
caused by later tool-loop steps or HITL resume do not append the same dynamic memory again. A new
user input creates a new `start` activation and can inject memory once again. Static memory slots
declared in `context.yaml` are not affected by this rule. `memory_results` consumes only the local
snapshot supplied by the caller; only `memory_query` awaits `MemoryService.abuild_context()`. A
configured SQLite service creates, uses, and closes its connection inside one worker job, and the
runtime does not consume a late result after cancellation.

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

When creating `ProviderClient`, the factory projects the effective provider, LiteLLM provider,
endpoint, and headers after global configuration merging into
`RuntimeEnvironment.provider_fingerprint` for harness recovery comparisons. API keys are excluded.
A host injecting its own provider owns its version declaration: set
`runtime.environment.provider_fingerprint = {"version": "my-provider-v2"}` before constructing
the runner. The empty default does not infer custom provider internals, and unused raw route
configuration does not affect the fingerprint.

The factory resolves `permissions.workspace` before constructing base context and user-declared
tools. With `skills.enabled: true`, it takes one project-level discovery snapshot against that
workspace. A non-empty result adds the `available_skills` system slot and registers `load_skill`
from the same registry before creating `ToolRegistryView` / `ToolExecutor`. Disabled Skills and an
empty result bypass both additions exactly, preserving the previous context/tool shape. A factory
or runtime instance does not refresh the snapshot automatically.

`RuntimeEnvironment.skill_registry` retains that shared registry so the harness can include
discovered Skill content versions in the recovery fingerprint without rereading Skill files.
`load_skill` checks the discovery version during its complete read and returns
`SKILL_VERSION_CHANGED` after a file change. Create a new runtime and start a new run to use it.

A `skills.root` escape, missing `skills.require` entry, or name/alias collision between
`load_skill` and a user tool becomes an assembly-time `IrisConfigError` and fails closed. See
[`iris.skill`](../skill/README.en.md) for the full contract.

## Public API

Package exports cover `AgentRuntime`, factory/environment, `StreamingRuntimeProvider`,
`streaming_provider_for()`, `RuntimeEventSink`, `RuntimeStreamEvent`, provider/assembler/tool
bridge, `RuntimeSteeringPort`, `SteeringInput`, and activation/commit-port contracts. Complete-run
options/status/results, `run_turn()`, `run_loop()`,
`resume()`, and old checkpoint helpers do not exist.

## Verification

```bash
uv run pytest tests/runtime
uv run ruff check src/iris/runtime tests/runtime
uv run mypy src/iris/runtime
```
