[English](README.en.md)

# `iris.harness`

`iris.harness.AgentRunner` 是 Iris 唯一的 complete-run SDK facade。它拥有 logical run 的创建、
resume、durable cancellation、settlement observation、显式 recovery、事件投递和 activation
live resources；`AgentRuntime` 只作为其内部 engine。`SessionManager` 是可选的单 session
process-local admission facade，只组合 runner，不接管 durable ownership。

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

## 单 session 输入管理

`SessionManager(runner, session_id)` 绑定一个 exact runner 与一个 session。它适合需要在当前 run
执行期间接收新普通输入的 host：

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
initial = await manager.submit("先分析现状")
queued = await manager.submit("把重点改为并发边界", mode="steer")

# host 结束使用 manager 时：
await manager.close()
await consumer
```

Idle 时，`submit(input, mode=None, options=...)` 在 run create 已 durable commit 后返回
`SubmitReceipt(state="delivered")`，但不等待 provider 或 run settlement。Busy 时必须显式选择：

- `mode="steer"`：绑定 exact current run，不接受新 run options；runtime 只在安全边界 claim
  一条，成功写入 durable session history 后才产生 `SubmissionEvent(state="delivered")`；
- `mode="follow_up"`：预生成 future run id，可携带 options；只在 exact current run terminal 后
  串行创建，一次启动一条。

两种 mode 各自保持 FIFO，但按 eligibility 独立推进，因此较早的 follow-up 不阻塞仍可进入当前
run 的 steer。Busy receipt 只表示 `pending`；最终 delivery/failure 只通过 `events()` 报告。
该单消费者 stream 原样混合 durable `RunEvent` 与 transient `SubmissionEvent`，不创建 session-global
sequence。Idle submit 不产生 `SubmissionEvent`。

HITL response 只走 `manager.resume(interaction_id=..., response=...)`，不进入普通输入队列。
`interrupt()` 只请求取消 exact current run；active cancellation request 不是 terminal，follow-up
仍等待真实 settlement。`close()` 拒绝后续操作、以 `session_closed` 结算全部 pending input 并结束
event stream，但不取消或等待当前 run。

Queue、receipt 状态、submission events、claim 和 event dedup 都只存在于当前进程。新 manager 不扫描、
恢复或 attach 既有 active/waiting lane；此时新的 idle submit 会由 store 的 session-lane CAS 拒绝。
Durable run、history、checkpoint、interaction、cancellation、result 和 `RunEvent` 始终由 runner/store
权威负责。

## Managed 组合钩子（包内）

`AgentRunner._start_managed()` 与 `_resume_managed()` 是供 `iris.harness` 内部组合层使用的
package-private hooks，不是 `iris.harness` 导出。它们不改变 complete-run 语义：coroutine 仍等待
waiting 或 terminal `RunResult`，public `start()` / `resume()` 只是使用空 hook 委托给它们；
`recover()` 没有 managed 变体。

Managed 调用可注入 activation-scoped steering port、同步 durable event callback 和
`asyncio.Event` admission signal。Signal 只会在 create/resume durable mutation 成功、对应 events
已 relay 且 exact activation 注册进 runner `_active` 后置位；立即 terminal 或 mutation/registration
失败不会产生虚假 signal。

Store-backed commit port 与 runner-owned create/resolve/begin/cancel/finish mutation 只在成功后把
新的 durable `RunEvent` 同步 relay，并按 `(run_id, sequence)` 去重。Callback 异常只记录日志，不回滚
mutation 或改变 `RunResult`。公开 `RunEventObserver` 契约不变，仍在 activation settlement 后异步、
best-effort 接收完整事件列表；同步 callback 不是新的 public observer registry。

## Cancellation 与 recovery

`cancellation_requested` 是 durable fact，不等于已取消。同步且不协作的工具可能延迟
settlement；runner 不会提前返回 cancelled。工具 result 若在请求后正常返回，会先 durable
commit result，再结算 cancelled。claim 后 effect/result 无法证明时必须 fail closed 为
`TOOL_OUTCOME_UNKNOWN`。

Runner 的 live signal 与 store-backed commit port 使用
`iris.exceptions.IrisCancellationRequestedError` 通知 runtime 协作式收口；该类型不属于
`iris.tools` 公共错误面。

Store-backed commit port 在每个 effect/commit 安全边界重新读取最小 run control，不跨边界缓存。
它只接受 control 完全相等，或同一 active activation 上 revision/event sequence 各推进一步且由唯一
`run.cancellation_requested` event 证明的取消；phase、fence、跳号、重复取消或 event/payload 不匹配
全部 fail closed。随后 mutation 仍以原有 revision 与 activation CAS 为最终授权。

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

`iris.harness` 导出 `AgentRunner`、`SessionManager`、`SubmitReceipt`、`SubmissionEvent`、
`SessionEvent`，以及 run request/options/limits/runtime options、phase/stop reason/usage/error/
snapshot/result 和 run events/observer。Store commands 仍属于 `iris.lifecycle`。

## 验证

```bash
uv run pytest tests/harness
uv run ruff check src/iris/harness tests/harness
uv run mypy src/iris/harness
```
