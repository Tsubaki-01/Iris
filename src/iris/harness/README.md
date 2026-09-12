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
try:
    result = await runner.start(
        AgentRunRequest(input="你好", session_id="default")
    )
    print(result.run.phase, result.assistant_message)
finally:
    await runner.aclose()
```

`from_config*()` 使用配置文件目录解析相对路径。显式传入 `store=` 时，runner 的所有 durable
reads/writes 使用该 exact object；否则 `session.backend: none` 选择
`InMemoryLifecycleStore`，`sqlite` 选择 lifecycle `SQLiteStore`。

配置 MCP 时构造不连接；`aprepare()` 可显式预热，也会由首次执行入口自动调用。完整目录发布
后才计算唯一 `environment_fingerprint` 并创建 run；此前读取该属性会报 `IrisRunStateError`。
required 准备或指纹计算失败会关闭资源且不创建 run，修正后需新建 runner。无 MCP 保持同步指纹。

resume/recover 先保留纯 durable 结算，确需比较/执行才准备；准备后重新读取状态、checkpoint、
claim 与时间，沿普通指纹规则拒绝目录/配置/有效只读策略漂移。只有最终 digest 进入 store；
有效 env/header 不单独持久化。terminal 读取、普通 waiting 到期、未结算 CLAIMED 的 unknown
恢复、查询、history fork 和取消申请不依赖 MCP 连接。

root 连接跨 run 复用。host 停止新调用后，须等待原 start/resume/recover 完整返回再 `aclose()`；
cancel 的 durable result 或观察超时不代表 body 清理、事件投递已经结束。active 时关闭会报错，
重复关闭幂等；关闭后仍可查询 durable 结果。`SessionManager.close()` 不接管 runner 资源，
使用它时先 `close(cancel_run=True)` 再关闭 runner。child MCP 生命周期在 Phase 06 接入。

## 会话历史分支

`SessionHistory(store)` 复用与 runner 相同的 `LifecycleStore`，提供三个同步方法。它只生成新
session ID 和创建时间，查询、来源资格与原子复制由 store 负责，领域异常原样传给 host。
它不接管 store 的关闭或 runner 的资源生命周期。

| 方法 | 返回值与语义 |
| --- | --- |
| `list_fork_points(session_id, *, after=None, limit=50)` | `ForkPointPage`，按 `(created_at, run_id)` 升序；`after` 接受上一页的 `ForkPointCursor`，`limit > 0` |
| `get_at_run(source_run_id)` | `RunHistorySnapshot`，包含分支点及其末尾的已提交历史，不提供当前 session CAS revision |
| `fork(source_run_id)` | `SessionSnapshot`，自动生成目标 ID 并创建新 session；每次成功调用都产生不同分支 |

上述历史 DTO 从 `iris.lifecycle` 导入。列表没有合格 run 时返回空页，没有后续页时
`next_cursor=None`。以下代码在 host 的现有 async 函数中执行，前提是 `main` 已有至少一个
terminal 顶层 run：

```python
from iris.harness import AgentRunner, SessionHistory
from iris.lifecycle import AgentRunRequest
from iris.store import SQLiteStore

store = SQLiteStore(".iris/session.db")
runner = AgentRunner.from_config_path("agent.yaml", store=store)
history = SessionHistory(store)

page = history.list_fork_points("main", limit=20)
if page.next_cursor is not None:
    next_page = history.list_fork_points("main", after=page.next_cursor, limit=20)

point = page.items[0]
preview = history.get_at_run(point.run_id)
branch = history.fork(point.run_id)
result = await runner.start(
    AgentRunRequest(input="Try another approach.", session_id=branch.session_id)
)
```

来源接受全部 terminal stop reason；linked child 被拒绝，调用过 child 的顶层 parent 仍可
使用。复制只到所选 run 的终态消息截点；来源 session 忙于后续 run 时，也可从旧截点创建分支。
返回的新 session 从 `revision=0` 开始，`forked_from_run_id` 保存直接来源，后续追加保留来源。
目标 ID 使用 `session_` 前缀与 UUID，`fork()` 不接收目标 ID 参数。

Fork 本身不调用 provider 或创建 run，仅复制对话消息和直接来源。下一次 `start()` 使用 host
选定 runner 的当前 system、工具、Skill、memory 与 workspace 配置，不恢复旧 checkpoint 或
复制旧运行环境。消息中的 artifact 引用保持原值，文件不会随 fork 复制。Host 也可直接构造
`SessionManager(runner, branch.session_id)` 并调用 `await manager.submit("Try another approach.")`，
按既有事件流观察运行并在结束使用时关闭 manager；无需 attach 或切换原 manager。

实现位于 `session_history.py`，相关集成用例位于
`tests/harness/test_session_history.py`。

## 公共操作

`iris.harness.ChildProviderFactory` 定义 selected child provider 注入协议：
`__call__(config: AgentConfig, *, config_path: Path) -> RuntimeProvider`。
它接收已加载的普通 child 配置，不依赖 parent provider 的单次凭据覆盖。

`from_config*()` 接受 `permission_policy=` 与 `child_provider_factory=`。配置 catalog 时，
runner 读取一次路由快照并装配内部 controller。Selected child 使用普通 AgentConfig、独立
session/run、fresh `AgentRunOptions()`、空 request metadata、无 memory service，并共享 parent
store/clock。Child 不注册 subagent；linked ACTIVE 通过 ordinary recover 继续原 child，
WAITING/TERMINAL 只读原结果。Child 等待时创建 parent proxy，工具保持 PREPARED。Host 只向
parent `resume()` 提交回答；回答先持久化，再继续 exact child。再次等待只替换 proxy，最终结果
通过单次 finalize 推进 parent cursor，使用 parent identity/artifact 与 tool error policy。

回答已持久化但推进中断时，`recover(parent_run_id)` 从 RESOLVED proxy 或 outer permission
恢复；普通 PENDING waiting 仍需 `resume()`。ACTIVE recovery 仍要求 activation fence。
新进程按 durable selector 取当前 catalog 快照，保留普通 parent/child fingerprint 检查。
Linked continuation 不重复 outer permission；未 admission 的存储批准仍执行 permission refresh。
WAITING finalize 成功后先发布原工具 activation 的 `tool.completed`，再执行 fresh RESUME
activation。SessionManager 在第一次 child await 前完成 resume admission，并拒绝并行回答。

Parent 取消、deadline 或 parent-owned proxy 到期先结算 exact child，再结束 parent。
Linked proxy 的 `request_cancel()` 返回仍为 WAITING 的请求快照；`cancel()` 或 manager
settlement task 完成后才 terminal。`settlement_timeout` 覆盖 child 等待和 parent observation，
超时保留 durable cancellation，后续 `cancel()` / `recover()` 可继续结算。连续 interrupt
共享原 cleanup task，follow-up 等到 parent 真正 terminal 才启动。

Child interaction/deadline 或 outer tool timeout 先结算 child，再提交 `SUBAGENT_TIMEOUT`。
Outer timeout 以 durable `child.created_at` 为起点，权限等待不计入，resume/rebind/recover
不重置。Proxy 保存的 expiry owner 决定停止 parent 还是继续处理工具错误；parent 同刻到期优先。
Parent stream 只含 parent start/proxy/final facts；linked recovery 不重复 started，child 内部
stream 与 usage 不转发，错误仍用带 `is_error` 的 `tool.completed`。
Parent 保留最终回答权，只得到 child 的最终文本；child usage 留在 child run。
结果 metadata 仅含 `agent_selector`、admission 后的 `child_run_id` 和普通 artifact metadata。
Live event 是 best-effort，不承诺跨进程 exactly-once。

最小 Sub Agent 示例包含三个文件，使用前述 `AgentRunner.from_config_path("agent.yaml")`
运行 parent；模型凭据按既有 provider 配置提供。

```yaml
# agent.yaml
name: parent
model: openai/gpt-4o-mini
system: 将专注的子任务交给 researcher，再结合其结果给出最终回答。
tools:
  subagent: subagents.yaml
```

```yaml
# subagents.yaml
default: researcher
agents:
  researcher:
    path: agents/researcher/agent.yaml
    description: 梳理专注的子任务并给出简洁结论。
```

```yaml
# agents/researcher/agent.yaml
name: researcher
model: openai/gpt-4o-mini
system: 只处理交给你的子任务，必要时向用户提问，最后返回结论。
tools:
  builtin: [human.ask]
```

Host 收到 parent `pending_interaction` 后，仍按普通 typed HITL 调用 parent `resume()`。
无需操作 child runner。Catalog default/selector/description 改变若导致普通 environment
fingerprint 不匹配，恢复会拒绝；不会绕过检查或新建替代 child。
Child 已关闭 HITL interaction 但尚未提交工具结果时，普通 ACTIVE recovery 仍从该 interaction
恢复存储回答。Child 的 `IrisRunRecoveryError` 原样传播，保留 parent/child 的可恢复状态。

- `start(request, options=None)`：原子创建 run/start activation，并推进到 waiting 或 terminal；
- `resume(run_id, interaction_id=..., response=...)`：消费 exact waiting interaction；
- `request_cancel(run_id, reason=None)`：只保证首次请求持久化；active 本地 activation 在提交后
  才收到 signal，waiting 可同事务 terminal cancelled；
- `cancel(..., settlement_timeout=None)`：request + 观察 durable terminal result；观察超时不写
  新事实；观察到结算不代表原 `start()` / `resume()` 调用已经退出；
- `recover(run_id, expected_activation_id=...)`：对 active run 要求精确 fence。safe checkpoint
  创建 recover activation，outcome-ready 只补 terminal，unresolved claim 结算为
  `outcome_unknown`；
- `get_session()`、`get_run()`、`get_run_control()`、`get_result()`、`list_tool_calls()` 和
  `list_events(after_sequence=0, limit=None)`：无副作用 durable reads；`limit` 如提供必须是
  正整数。

waiting run 应使用 `resume()`，不是 `recover()`。terminal run 的 cancel/recover 是幂等读取。

`get_run_control()` 只读取 run identity、phase、activation fence、revision 与取消控制字段。
SessionManager 在锁内用它判断 steer 是否仍可进入当前 activation，不装载完整 run snapshot。
事件补读直接消费 store 返回的有序、唯一、有限页；store 是排序和分页的唯一 owner，manager
只推进 watermark 并保留空页冲突与 submission barrier。

内部 `RunCommit.session_revision` 只返回本次会话 revision；commit port 直接更新本地 revision，
mutation 不为回执加载完整 history。需要消息时仍显式调用 `get_session()`。

## Live publisher 组合

Host 可把同一个 `LivePublisher`（通常是 `LiveStreamBroker`）通过 `live_publisher=` 注入
`AgentRunner`。Runner 会把每个 activation 的 `RuntimeStreamEvent` 和每条新 committed
`RunEvent` 同步交给 publisher；未注入时不构造 runtime sink。Publisher 是 best-effort
观察面：普通异常只记录不含 payload 的 warning，不会回滚 durable mutation、取消 run 或改变
`RunResult`。

是否注入 publisher 是 runtime transport 的唯一选择：未注入时 complete，注入时 streaming。
`ModelConfig` 不包含 `stream`；低层直接调用 provider 时仍由 `LLMRequest.stream` 指定接口。

`AgentRunner.from_config()` 与 `from_config_path()` 只把 publisher 交给 runner，
`RuntimeFactory` 不拥有 broker 或 fan-out。Host 可通过 `get_session()`、`get_run()`、
`get_result()`、`list_tool_calls()` 和 `list_events()` 从 exact runner/store 补读 durable facts，
这些读取不会触发 live 发布。

## 单 session 输入管理

`SessionManager(runner, session_id, submission_publisher=...)` 绑定一个 exact runner 与一个
session。它适合需要在当前 run 执行期间接收新普通输入的 host：

Host 显式选择 `observation_mode="mixed"`（默认）或 `"broker_only"`。Mixed 模式提供
lossless `events()`，也可同时发布到 broker。Broker-only 模式要求 `submission_publisher`，
只发布 submission facts，不创建本地 tracker/transient buffer；调用 `events()` 会失败。
仅使用 Gateway 的 host 应选择 broker-only，容量不会取决于一个未使用的本地 consumer。

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

CLI 等普通文本 host 可用 `mode="auto"`：manager 持锁读取当前 durable 状态，idle 时新建 run，
busy 时选择 steer。此时 `options` 只用于新 run，busy 分支不使用它；UI 无需维护路由状态副本。

两种 mode 各自保持 FIFO，但按 eligibility 独立推进，因此较早的 follow-up 不阻塞仍可进入当前
run 的 steer。Busy receipt 只表示 `pending`；最终 delivery/failure 通过所选 observation 模式报告。
该单消费者 stream 原样混合 durable `RunEvent` 与 transient `SubmissionEvent`，不创建 session-global
sequence。Idle submit 不产生 `SubmissionEvent`。

Mixed 模式下，可选 `submission_publisher` 提供 submission side channel。Manager 会先把原
`SubmissionEvent` 成功写入上述单消费者 buffer，再 best-effort 发布带 session identity 的
`SessionSubmissionEvent`；发布失败不重复 buffer 写入，也不改变 receipt。Run events、HITL 和
result 仍以原 manager/runner 契约为准。

Manager 默认最多分别排队 64 条 steer 与 64 条 follow-up，最多保留 256 个 transient submission
event 槽位，并跟踪 64 个尚未被 consumer 追平的 durable run；后两种限制仅用于 mixed 模式。可通过
`max_pending_steer`、`max_pending_follow_up`、`max_buffered_submission_events` 和
`max_tracked_durable_runs` 关键字参数设置其它有限正整数。Busy admission 会同时预留 pending 与
terminal event 槽位；任一容量不足时，在 receipt、队列和 event 发布前抛出 `IrisRunStateError`，不
静默丢弃。已接纳的 follow-up 在 tracker 暂满时保留 FIFO，consumer 追平旧 run 后继续推进；新的
idle submit 则在创建 task 前拒绝。Durable tracker 的容量判断与 baseline 登记由同一个同步
admission mutation 完成，不使用 check-then-register 双阶段路径。

HITL response 通过 `manager.resume(interaction_id=..., response=...)` 等待完整结果，或通过
`admit_resume(...)` 在 activation 已接纳后返回 `ResumeReceipt(run_id, interaction_id)`。
两者共享同一 admission owner；后台执行仍由 manager/runner 持有，不进入普通输入队列。
`interrupt()` 只请求取消 exact current run；active cancellation request 不是 terminal，follow-up
仍等待真实 settlement。`close()` 拒绝后续操作、以 `session_closed` 结算全部 pending input 并结束
event stream，但不取消或等待当前 run。

即将关闭 event loop 的 host 使用 `close(cancel_run=True, reason=...)`：先关闭 admission 并
失败掉 pending input，阻止启动下一条 follow-up，再通过 runner 取消并等待当前 run 结算。
随后等待原 managed `start()` / `resume()` task 结束，包括 WAITING parent 正在继续等待 child 的情况。
CLI 的 `/exit`、EOF、Ctrl-C 和错误退出均使用这条路径。

Queue、receipt 状态、submission events、claim 和 durable event 水位都只存在于当前进程。Durable
event payload 不进入无界进程内队列；callback 只推进每个 run 的 observed watermark，consumer 按
delivered watermark 从权威 store 以最多 64 条的 page 分批补读。所有 `LifecycleStore` 实现必须
支持 `limit`，并在复制或解码前执行上限。新 manager 不扫描、
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
新的 durable `RunEvent` 同步 relay。Callback 异常只记录日志，不回滚 mutation 或改变 `RunResult`。
公开 `RunEventObserver` 签名不变：同一 observer 内按 run sequence 串行保序，不同 observer lane
并行；每个 event 默认最多等待 30 秒，可用 `observer_event_timeout_s` 覆盖。Timeout 或普通异常只
记录 warning 并继续，不改变 durable result；同步 callback 不是新的 public observer registry。

每次 activation 中，runner 与 commit port 共享私有 `_RunEventCollector`，由它唯一持有累计
事件与 `(run_id, sequence)` 去重键。新增批次只检查本批事件；取消事件即使被两条路径观察，
同步 callback 也只对首次收集执行一次。Callback 失败不阻断后续 live publisher。

## Cancellation 与 recovery

结算失败时，runner 使用注入的 Clock 核对 absolute deadline，不依赖 timer 是否已经获得调度。
到期后的 provider 异常、`response.failed` 和取消清理失败结算为 `DEADLINE_EXCEEDED`；
到期前的 provider 错误保持 `FAILED`。未提交的工具 claim 仍优先结算为
`OUTCOME_UNKNOWN`。

`cancellation_requested` 是 durable fact，不等于已取消。Runner 先持久化请求，再发送本地
signal；存在 claim 时不整体取消 activation task。`ToolExecutor` 将 signal 转为普通 async
callable、自定义异步 `BaseTool` 或 THREAD callable 的 body task 取消，并等待 body 清理结束。
慢 middleware、压住 `CancelledError` 的协程及 INLINE 阻塞仍可能延迟 settlement；`cancel()`
只等待 durable terminal result，不提前返回 cancelled。

body 已完成，或响应 signal 取消后仍正常返回时，结果经过后处理并按既有顺序 durable commit 后再结算
cancelled。外层 task cancellation、timeout 或 sibling cancellation 已打断 executor 时，清理期间
的返回值不会替换原中断。未结算 claim 仍使 run 以 `TOOL_OUTCOME_UNKNOWN` 收口，包括只读调用；
worker 线程可以继续运行，晚到返回不能改写 durable result、history、checkpoint 或 events。

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

恢复指纹绑定 agent 名称、有效模型路由与请求参数、已加载的结构化 context、模板来源版本、
当前工具定义、权限 policy、workspace 和 checkpoint 版本。启动时已启用目录发现的全部 Skill
内容版本也会参与，包括尚未加载的 Skill。会话存储路径、context 配置文件位置和重复声明写法
不参与；相同模板移动位置不改变版本。工具实现如需显式版本，应写入
`ToolDefinition.metadata`；框架不推断 Python 源码版本，也不扫描整个 workspace。

Factory 创建内置 `ProviderClient` 时，把合并全局配置后的 provider、LiteLLM provider、endpoint
和 headers 保存在 `RuntimeEnvironment.provider_fingerprint`；指纹不包含 API key。Host 注入
自定义 provider 时，应在创建 runner 前显式设置该字典中的路由或版本标识；默认空字典表示
框架不推断该 provider 的内部行为。模型请求名称与请求参数仍参与恢复比较。

模板来源在 runner 构造期间冻结；同一 runtime 的后续渲染使用这个快照，新 runtime 才读取新
版本。来源范围包括静态嵌套依赖、可选依赖和文件名列表；动态文件名表达式不受支持，可改为
条件分支中的静态引用。配置的空 memory 模板也会冻结，因为 run options 可稍后启用它；空前置
段仍跳过。指纹不渲染 context，`StrictUndefined` 和字符上限保留到实际渲染。详见
[`iris.context`](../context/README.md)。

`before_model / step 0` 的初始 recovery 会从 durable `AgentRunRequest.input` 重建尚未提交的
当前轮次输入。后续 checkpoint 的输入已经随 provider commit 进入 session history，因此不会再次
注入。

## 公开接口

`iris.harness` 导出 `AgentRunner`、`SessionHistory`、`SessionManager`、`SubmitReceipt`、
`ResumeReceipt`、`SubmissionEvent`、
`SessionSubmissionEvent`、`SessionEvent`、`LiveFact`、`LivePublisher`，以及 run
request/options/limits/runtime options、phase/stop reason/usage/error/snapshot/result 和 run
events/observer。Store commands 仍属于 `iris.lifecycle`。

## 验证

```bash
uv run pytest tests/harness
uv run ruff check src/iris/harness tests/harness
uv run mypy src/iris/harness
```
