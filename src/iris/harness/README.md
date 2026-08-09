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

Runner 的 live signal 与 store-backed commit port 使用
`iris.exceptions.IrisCancellationRequestedError` 通知 runtime 协作式收口；该类型不属于
`iris.tools` 公共错误面。

runtime 的只读并发窗口使用固定内部上限 8；它不增加 public config/schema/API。窗口中每个
调用都有独立 durable claim，body 可以乱序结束，但只有连续的已知 result prefix 会按 ordinal
进入 history/cursor/checkpoint。claim telemetry 的 event 顺序不是 ordinal 契约。任一未提交
claim 都会使 cancellation、deadline 或程序中断结算为 outcome unknown；现有 terminal
settlement 会在同一 aggregate transaction 中关闭该 activation 的全部 unresolved claims。

active recovery 会验证 checkpoint v1、session revision、usage counters、environment fingerprint
与 cursor。只要存在 unresolved claims 就不会重放工具；recovery 会原子 abandon 旧 activation，
把全部 claims 关闭为 outcome unknown，再形成 terminal result。正常 parent/control/
infrastructure 退出会先等待 runtime children drain，随后 revoke commit port；不会允许迟到 child
继续写入。同步阻塞 callable 不保证并发加速，并且仍可能延迟 settlement。

`before_model / step 0` 的初始 recovery 会从 durable `AgentRunRequest.input` 重建尚未提交的
当前轮次输入。后续 checkpoint 的输入已经随 provider commit 进入 session history，因此不会再次
注入。

## 公开接口

`iris.harness` 导出 `AgentRunner`、run request/options/limits/runtime options、phase/stop reason/
usage/error/snapshot/result，以及 run events/observer。Store commands 仍属于 `iris.lifecycle`。

## 验证

```bash
uv run pytest tests/harness
uv run ruff check src/iris/harness tests/harness
uv run mypy src/iris/harness
```
