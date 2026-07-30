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

Cursor positions are `before_model`, `tool_batch`, and `outcome_ready`. A provider response without
tools is committed as `CheckpointResumability.OUTCOME_READY`. Tool effects require a durable claim
before execution and a durable result afterward. If an effect cannot be proven after claim, the
engine returns `TOOL_OUTCOME_UNKNOWN` and never replays it.

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
