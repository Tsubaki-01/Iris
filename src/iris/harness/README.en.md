[中文](README.md)

# `iris.harness`

`iris.harness.AgentRunner` is Iris's only complete-run SDK facade. It owns logical-run creation,
resume, durable cancellation, settlement observation, explicit recovery, event delivery, and live
activation resources. `AgentRuntime` is its inner engine.

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
- `get_run()`, `get_result()`, and `list_events()` are side-effect-free durable reads.

Use `resume()`, not `recover()`, for a valid waiting run. Cancel/recover on terminal runs are
idempotent reads.

## Cancellation and recovery

Cancellation requested is a durable fact, not a settlement claim. A non-cooperative synchronous
tool may delay settlement. If a tool returns after the request, its result is committed before the
run settles cancelled. If an effect cannot be proven after claim, recovery fails closed with
`TOOL_OUTCOME_UNKNOWN`.

The runner's live signal and store-backed commit port use
`iris.exceptions.IrisCancellationRequestedError` to request cooperative runtime settlement; the
type is not part of the `iris.tools` public error surface.

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

`iris.harness` exports the runner; run request/options/limits/runtime options; phase, stop reason,
usage, error, snapshot, and result; plus run events and observers. Store commands remain in
`iris.lifecycle`.

## Verification

```bash
uv run pytest tests/harness
uv run ruff check src/iris/harness tests/harness
uv run mypy src/iris/harness
```
