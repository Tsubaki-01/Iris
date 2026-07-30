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

Active recovery validates checkpoint v1, session revision, usage counters, environment
fingerprint, and cursor. Only the activation holding the current durable fence may commit.

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
