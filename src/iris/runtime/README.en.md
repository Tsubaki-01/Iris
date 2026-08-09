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

## Low-level contract

```python
result = await runtime.execute(
    activation,
    commits=commit_port,
    cancellation=cancellation_signal,
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

## Bounded tool concurrency

Under `RETURN_TO_MODEL`, runtime groups consecutive calls that are both read-only and declared
concurrency-safe into an internal window of at most 8 calls. Eight is a private implementation
bound, not a YAML, `RuntimeExecutionOptions`, or environment setting. This capability does not
change public config, schemas, models, or exports.

Only consecutive candidates share a window. STOP, HITL, preflight results,
WRITE/EXECUTE/NETWORK/MCP/AGENT calls, unsafe calls, and classification failures are serial
barriers; later calls cannot start across them. Every child still revalidates and records its own
exact durable claim before entering the body. Bodies may finish out of order, while result
messages, cursors, session history, checkpoints, and committed events advance only as the original
ordinal prefix. The order of multiple `TOOL_CALL_CLAIMED` telemetry events is not contractual.

A control interruption commits only the known `ToolResult` prefix before the first exception or
hole; a later in-memory result never skips that hole. Any uncommitted durable claim makes eventual
cancellation, deadline, or program interruption fail closed as `OUTCOME_UNKNOWN`. Runtime cancels
and drains the children it created before a parent-task or infrastructure exit completes.
Cooperative cancellation uses `iris.exceptions.IrisCancellationRequestedError`; runtime converts
it into an activation outcome rather than an ordinary tool error.

Concurrent file reads share one `ReadFileState` identity. The checkpoint snapshot taken after the
window settles contains the combined records, so a later serial write barrier can retain stale-read
checks. Synchronous blocking callables retain correct claim/result/order semantics but receive no
speedup guarantee. Future NETWORK/MCP or write concurrency requires a new effect, retry, timeout,
conflict, and crash-reconciliation protocol rather than a relaxed classifier. This work adds no
delta/merge/lock/hash model.

## Explicit memory injection

`RuntimeExecutionOptions.memory_query` and `memory_results` are explicit opt-in dynamic memory
inputs. Each logical run injects them only on its first `before_model` step; provider requests
caused by later tool-loop steps or HITL resume do not append the same dynamic memory again. A new
user input creates a new `start` activation and can inject memory once again. Static memory slots
declared in `context.yaml` are not affected by this rule.

## Factory

```python
from iris.runtime import RuntimeFactory

runtime = RuntimeFactory.from_config_path("agent.yaml", provider=provider)
```

The factory never reads or creates a lifecycle database. Harness composition interprets the
`session` section of `agent.yaml`; the low-level factory has no persistence side effect from it.

## Public API

Package exports cover `AgentRuntime`, factory/environment, provider/assembler/tool bridge, and
activation/commit-port contracts. Complete-run options/status/results, `run_turn()`, `run_loop()`,
`resume()`, and old checkpoint helpers do not exist.

## Verification

```bash
uv run pytest tests/runtime
uv run ruff check src/iris/runtime tests/runtime
uv run mypy src/iris/runtime
```
