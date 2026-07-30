[English](README.en.md)

# `iris.harness`

`iris.harness.AgentRunner` 是 Iris 唯一的 complete-run SDK facade。它拥有 logical run 的创建、
resume、durable cancellation、settlement observation、显式 recovery、事件投递和 activation
live resources；`AgentRuntime` 只作为其内部 engine。

## 快速入门

```python
from iris.harness import AgentRunRequest, AgentRunner

runner = AgentRunner.from_config_path("agent.yaml")
result = await runner.start(
    AgentRunRequest(input="你好", session_id="default")
)
print(result.run.phase, result.assistant_message)
```

`from_config*()` 使用配置文件目录解析相对路径。显式传入 `store=` 时，runner 的所有 durable
reads/writes 使用该 exact object；否则 `session.backend: none` 选择
`InMemoryLifecycleStore`，`sqlite` 选择 lifecycle `SQLiteStore`。

## 公共操作

- `start(request, options=None)`：原子创建 run/start activation，并推进到 waiting 或 terminal；
- `resume(run_id, interaction_id=..., response=...)`：消费 exact waiting interaction；
- `request_cancel(run_id, reason=None)`：只保证首次请求持久化；active 本地 activation 在提交后
  才收到 signal，waiting 可同事务 terminal cancelled；
- `cancel(..., settlement_timeout=None)`：request + 观察 durable terminal result；观察超时不写
  新事实；
- `recover(run_id, expected_activation_id=...)`：对 active run 要求精确 fence。safe checkpoint
  创建 recover activation，outcome-ready 只补 terminal，unresolved claim 结算为
  `outcome_unknown`；
- `get_run()`、`get_result()`、`list_events()`：无副作用 durable reads。

waiting run 应使用 `resume()`，不是 `recover()`。terminal run 的 cancel/recover 是幂等读取。

## Cancellation 与 recovery

`cancellation_requested` 是 durable fact，不等于已取消。同步且不协作的工具可能延迟
settlement；runner 不会提前返回 cancelled。工具 result 若在请求后正常返回，会先 durable
commit result，再结算 cancelled。claim 后 effect/result 无法证明时必须 fail closed 为
`TOOL_OUTCOME_UNKNOWN`。

active recovery 会验证 checkpoint v1、session revision、usage counters、environment fingerprint
与 cursor。旧 activation 被原子 abandon/fence replacement 后才允许新 activation 写入。

## 公开接口

`iris.harness` 导出 `AgentRunner`、run request/options/limits/runtime options、phase/stop reason/
usage/error/snapshot/result，以及 run events/observer。Store commands 仍属于 `iris.lifecycle`。

## 验证

```bash
uv run pytest tests/harness
uv run ruff check src/iris/harness tests/harness
uv run mypy src/iris/harness
```
