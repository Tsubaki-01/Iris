[中文](README.md)

# `iris.streaming`

`iris.streaming` provides a live observation plane embedded in a host application. An in-process
broker owns bounded replay, ordering, and fan-out. A gateway binds an authorized exact session
and routes typed commands to `SessionManager`. SSE and WebSocket adapters handle framing and
the lifetime of tasks within each connection.

This package does not create a server, routes, authentication, authorization, TLS, CORS, a tenant
registry, or artifact download URLs. The host owns those concerns before connections and commands
enter Iris. A cursor identifies an observation position and grants no permissions.

The package ships with Iris, requires Python 3.12 or later, and needs no additional transport dependency.

## Host composition

Inject the same broker into the runner, manager, and gateway:

```python
from iris.harness import AgentRunner, SessionManager
from iris.streaming import LiveStreamBroker, StreamingGateway

broker = LiveStreamBroker(
    replay_capacity_per_scope=256,
    subscription_capacity=64,
    max_replay_scopes=256,
)
runner = AgentRunner.from_config_path(
    "agent.yaml",
    live_publisher=broker,
)
manager = SessionManager(
    runner,
    "default",
    submission_publisher=broker,
    observation_mode="broker_only",
)
gateway = StreamingGateway(
    runner=runner,
    manager=manager,
    broker=broker,
    session_id="default",
    durable_page_size=64,
)
```

Use `observation_mode="broker_only"` when observation runs through the gateway/broker. The manager
requires `submission_publisher`, creates no mixed event buffer, and rejects `events()` calls.
Completed runs do not consume subsequent admission capacity because a mixed stream has no consumer.
Hosts that need a local mixed stream can keep the default `observation_mode="mixed"` and continuously
consume `manager.events()`.

`LiveStreamBroker` must be used on one event loop/thread. `replay_capacity_per_scope` bounds each
run/session ring. `max_replay_scopes` (default 256) bounds the global ring count; publishing or
subscribing updates the scope's LRU position. Evicting a ring does not close active subscriptions:
their published sequence remains until they close and no ring remains. Other old scopes lose both
their ring and counter. The number of counters is therefore bounded by replay scopes plus active
scopes. Recreating a reclaimed scope skips its old sequence values, so old cursors receive an
`unknown_cursor` or `cursor_expired` gap. Do not assume a new scope starts at 1. `close()` clears replay
and counters and queues a terminal item for active subscriptions.

`subscription_capacity` bounds each consumer's future-live backlog. Partials may be coalesced or
dropped. If a critical event cannot be queued, the subscription produces `ReplayGap` and
`SubscriptionTerminal`; the client should reconnect and perform durable sync. Only the active
subscription offer path triggers this slow-consumer transition, producing one gap/terminal pair.

## Gateway and commands

`StreamingGateway` binds the exact runner, manager, broker, and session supplied at construction.
A session scope must match the bound session. For run scopes and durable cursors, the runner must
confirm that the run belongs to that session. The gateway never calls `SessionManager.events()`
and does not mutate the store or perform recovery directly.

```python
from iris.streaming import (
    DurableRunCursor,
    SubmitCommand,
    SubscribeCommand,
)

subscription = gateway.subscribe(
    SubscribeCommand(
        request_id="subscribe-1",
        scope="session",
        scope_id="default",
        durable_cursors=(
            DurableRunCursor(run_id="run-known", after_sequence=0),
        ),
    )
)
receipt = await gateway.handle(
    SubmitCommand(request_id="submit-1", input="继续分析", mode="steer")
)
```

`subscribe()` creates no network task. When the request includes caller-known durable cursors,
`GatewaySubscription` first emits a `DurableSyncItem`, then delegates to the broker's live stream.
`durable_sync()` reads only the explicitly supplied runs and returns bounded event pages in input
order. It neither discovers other runs in the session nor automatically recovers them. `next_cursor`
represents only that run's durable event high-water mark.

`handle()` supports:

- `SubmitCommand` → `SessionManager.submit()`.
- `ResumeCommand` → `SessionManager.admit_resume()`, returning `run_id` and `interaction_id` in
  `ResumeAccepted.receipt` without waiting for the resumed model or tool execution to finish.
- `CancelCommand` → `SessionManager.interrupt()`.
- `SyncCommand` → read-only durable sync.

`request_id` correlates receipts; it provides no idempotency or deduplication. Expected `IrisError`
instances become stable `CommandRejected` receipts. Unexpected errors produce a generic rejection
and a warning that excludes raw frames and payloads. Obtain the final resume result through durable
sync after a live terminal/interaction event. Direct SDK calls to `SessionManager.resume()` still
wait for a complete `RunResult`.

## Disclosure policy

The defaults are `allow_thinking=False` and `allow_tool_arguments=False`. Gateway subscriptions omit
thinking blocks/deltas and tool-argument partials, and remove tool names from live tool facts.
Arguments in durable tool calls and waiting interactions are projected as empty dictionaries. Enable
these options only when the host has authorized the tenant/session and needs those fields.

Durable sync always removes assistant/tool-result metadata, run/tool error details, local artifact
paths, tool result data/stats/metadata, and pending interaction workspace paths. Argument opt-in does
not restore them. The current durable wire shape cannot represent an artifact without its path,
so the gateway returns `artifact=None`; the host owns any authorized download interface.

The earlier projection boundary already removes raw provider chunks/headers/keys/tracebacks,
artifact paths and bytes, and arbitrary metadata. Gateway filtering consumes that trusted allowlist
without reparsing payloads.

## SSE adapter

`SSEAdapter` returns `AsyncIterator[bytes]` and creates no HTTP route:

```python
from iris.streaming import SSEAdapter

adapter = SSEAdapter(heartbeat_interval_s=15)
async for frame in adapter.stream(subscription):
    await host_send_bytes(frame)
```

`LiveEnvelope` frames carry an `id` containing a compact JSON `LiveCursor`. `ReplayGap`,
`SubscriptionTerminal`, and `DurableSyncItem` have no synthetic ID. Heartbeats are always
`: heartbeat\n\n`; they do not enter the broker/ring or consume a live sequence. Heartbeat timeouts
reuse the same pending `anext()` without cancelling or replacing it. Iterator completion or caller
disconnection closes only the subscription.

Hosts can use `encode_live_cursor()` / `decode_live_cursor()` for SSE cursor headers. Decoding is raw
boundary validation; the host should map failures to request errors without sensitive details.

## WebSocket adapter

`WebSocketAdapter.serve(receive, send)` accepts callbacks supplied by the framework. Raw frames must
be UTF-8 JSON typed commands; parsing failures return `INVALID_COMMAND`. The first valid command
must be `subscribe` or `sync`. A sync-first connection must still subscribe before mutation commands.

Only the sender task calls `send()`. Before routing a command, the receiver reserves one receipt slot,
then places its receipt into a capacity-1 queue. The slot is released only after the sender finishes
sending. A slow sender pauses subsequent command admission, so pipelined commands cannot mutate
the manager and then lose their receipt because the queue is full. Resume waits only for admission;
later sync, steer, cancel, or disconnect can be processed before the run completes.
A second subscribe is rejected. `receive() -> None`, receive/send exceptions, or task cancellation
drain/cancel child tasks and close the subscription.

A disconnect is observation loss only: SSE/WS cleanup does not call manager interrupt, runner cancel,
or recover. Only an explicit `CancelCommand` makes the gateway request durable cancellation.

## Restart and recovery

Broker epochs, live cursors, subscriptions, and partials are process-local and not persisted. After
a process restart, an old cursor produces `ReplayGap(reason="epoch_changed")`. Clients must discard
incomplete partials and request sync using their known per-run `DurableRunCursor` values. The runner
and store remain the durable authority; live replay cannot replace durable event/result/tool snapshots.

## Development and verification

`models.py` defines public wire models; `projection.py` projects runner/runtime facts into live
payloads. `broker.py` manages ordering, replay, and subscriptions. `gateway.py` combines commands and
durable sync. `sse.py` / `websocket.py` own transport framing. `__init__.py` exports package-level APIs.

For replay/capacity changes, extend `tests/streaming/test_broker.py`. Update `test_gateway.py` and
`test_models.py` for command contracts, and `test_transports.py` plus `test_system.py` for connection
lifetimes. System tests use real runner/manager/provider adapters with scripted model streams and
no external calls. Broker-only cases use `max_tracked_durable_runs=1` to verify successive admission.

```powershell
$env:UV_CACHE_DIR = "$PWD/tmp/uv-cache"
uv sync --dev
uv run pytest tests/streaming -p no:cacheprovider --basetemp="$PWD/tmp/pytest-streaming"
uv run ruff check src/iris/streaming tests/streaming
uv run mypy src/iris/streaming
```
