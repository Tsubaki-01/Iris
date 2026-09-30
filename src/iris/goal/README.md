# Goal：跨 Run 的目标状态

`iris.goal` 保存期望完成的目标、状态、自动执行轮数和 Run 绑定。目标与一次
logical Run 分开：一个 Run 结束不会自行把目标标记为完成。

提供状态服务、原子存储契约、模型工具、动态上下文、终态结算与会话控制 SDK。
GoalDriver 提供续跑策略，实际准入与执行交给 harness，本包不持有异步执行循环。

## 使用状态服务

`GoalService` 使用同一个 lifecycle backend，支持 `InMemoryLifecycleStore` 与
`SQLiteStore`。集成执行系统时，必须传入 Runner 正在使用的 exact store。

```python
from iris.goal import GoalReason, GoalService
from iris.store import InMemoryLifecycleStore

store = InMemoryLifecycleStore()
goals = GoalService(store)
goal = goals.create("session-1", "修复问题并通过相关测试", max_rounds=3)
paused = goals.pause(goal.ref, reason=GoalReason(code="user", text="等待我确认"))
resumed = goals.resume(paused.ref)
assert resumed.rounds_started == 0
assert store.load_session_revision("session-1") == 0
```

创建可以早于首条聊天；它只按需建立空 session，不写历史或占用 lane。
`GoalConfig` 默认 `enabled=False`、`max_rounds=20`，开关由执行装配统一使用；
直接构造领域服务只操作状态。

## 会话控制 SDK

`GoalSession(service, port)` 是异步控制入口。`GoalControlPort` 由宿主绑定一个 session，
在其唯一 admission owner 中处理操作；SDK 不持有 runner、manager、任务或锁。

| 方法 | 作用 |
| --- | --- |
| `create(objective, *, max_rounds=None, run_options=None)` | 保存目标并明确允许推进；不等待模型执行完。 |
| `get()` | 返回只读 GoalView，不补结算或启动执行。 |
| `edit(*, objective=None, max_rounds=None, run_options=None)` | 至少修改一项，编辑后暂停；None 表示未修改。 |
| `pause(*, reason)` / `complete(*, reason)` | 记录非空用户原因并停止后续推进，不取消已准入 Run。 |
| `resume(*, expected_activation_id=None)` | 先处理原执行的附着或精确恢复，再决定后续推进。 |
| `clear()` | 取消当前选择，保留历史和绑定。 |

除 get 外均返回 `GoalControlResult(view, disposition)`。disposition 区分已允许调度
`scheduled`、已创建 Run `admitted`、复用在途调用 `running`、等待人工输入 `waiting`、
缺少接管 fence `needs_recovery`、lane 属于其他执行 `occupied`、仅停止操作或额度用尽
`stopped`。只有宿主实际到达相应阶段才返回该值。

SDK 调用 `GoalService.parse_create/parse_edit` 完成一次原始输入解析，之后把冻结的
`GoalCreateInput/GoalEditInput` 交给 port。宿主取得当前引用后调用
`create_validated/edit_validated`；这两条路径不重复解析或检查模型工具配置。
同步 `GoalService.create/edit` 也复用相同 parser。用户原因以 code=user 的 GoalReason
传递，恢复 fence 在 SDK 入口解析；状态、版本和额度条件由 store 的操作边界检查。

## 状态与版本

- 状态为 `active`、`paused`、`blocked`、`completed`。每个 session 只选择一个当前目标。
- `edit()` 修改正文、总轮数或 Run 选项后转为 paused，保留已经消耗的轮数。
- `pause()` 与 `complete()` 只修改目标状态，不取消或结束已存在的 Run。
- `resume()` 不重置额度。额度耗尽且没有目标在途 Run 时保持或转为 paused；
  已 active 的纯恢复不推进版本。
- `clear()` 取消当前选择，保留记录和所有 Run 绑定。未完成目标不能被 create 隐式覆盖；
  completed 后可以创建新目标。

修改使用 `GoalRef(goal_id, revision)` 做 CAS。实际修改只推进一次 Goal revision，
不会改动 session history revision。重复 pause/complete 且原因相同返回原快照；
已完成目标不能通过编辑或恢复再次运行。

## 存储与职责

`GoalStore` 在本包扩展 `LifecycleStore`。创建、修改和准入接收冻结 typed command；
`GoalService` 是创建、编辑原始字段的解析入口，存储与纯转换直接消费可信输入。

`admit_goal_run(AdmitGoalRun(...))` 原子写入既有 `CreateRun` 的结果、目标绑定和已用轮数。
首轮从 1 开始；失败不留下部分记录，成功后即使模型尚未调用也不退还轮数。
目标必须 current、active、版本匹配、有额度且先前绑定已结算。普通用户 Run
不受目标额度或未结算绑定限制。

持久模型和只读 `GoalView` 位于 `models.py`，控制 command/协议位于 `store.py`，
共享状态规则位于 `transitions.py`。具体数据库实现在 `iris.store`，本包不导入
harness、runtime 或具体 backend。只读 API 不触发状态推进或结算。

## 模型工具与动态上下文

开启 Goal 时装配 `get_goal` 与 `report_goal`，两者默认可见且历史正文保留。
`get_goal()` 返回统一 `GoalView`：当前目标、armed、当前 Run、人工交互、待结算
状态及 driver 错误。未附着控制器时 armed=false；读取不会恢复执行或补结算。
view.run 只表示当前 session lane，run_goal_id 说明它绑定哪个目标；普通 Run 占用
lane 时不伪装成 Goal 执行。无 lane 时 run=None，即使旧终态绑定仍 settlement_pending。

`report_goal(goal_id, revision, decision, reason)` 接受 complete、blocked、continue。
它从真实工具执行上下文取得 Run 绑定，拒绝普通 Run、替换目标或过期版本的报告。
成功时只把 `GoalReport` 写入普通 `ToolResult.data["goal_report"]`，不直接修改目标。

`GoalContextSource(service, host_source=...)` 每个模型 step 调用宿主 source 一次，
保留所有原条目的 required/priority，再追加 required 的 `iris.goal`。该 key 冲突
是配置错误。Goal Run 看到同一目标的最新完整正文、状态、版本与轮数；暂停或完成
要求收尾，已 clear/替换的旧 Run 不会看到新目标正文。普通 Run 只有普通输入说明。

`render_continuation()` 生成简短自动输入；完整目标只从当前动态快照读取。
模板位于 `prompts/`，通过共享 `TemplateRenderer` 渲染；Goal 入口统一包装模板错误。

## 结算与宿主接线

`GoalService.settle_run(run_id, now=...)` 委托同一个 store 原子结算绑定；
`reconcile(session_id)` 显式补结算已 terminal 的未结算绑定，跳过 ACTIVE/WAITING。
正常结束时只采用最新有效申报：同一步工作工具、后续工作、新提交用户输入或
人工回答使旧的完成申报失效，需要模型重新申报。continue 撤回先前申报。
异常结束暂停目标；最后允许的一轮仍先判有效完成，再判轮数耗尽。
用户已 pause/complete/clear 或替换目标时，结算不能覆盖这些决定。

宿主为服务注入只读 `process_state_reader`，以及可选的 `run_options_validator`。
后者只在创建和显式修改 Run 选项时调用；实际执行装配检查 tools 已开启、有效
tool_choice 为 None/auto。领域包不持有宿主任务、不自行启动模型或执行循环。

## 续跑候选与通知

`GoalDriver` 新建时 disarmed，只保存 armed_goal_id 和至多一个
`GoalContinuationIntent(source_run_id, goal_id, run_id)`。`can_continue()` 判断目标当前
是否 active、匹配 armed 身份且有剩余额度；用户优先、lane 与 tracker 容量归 manager。
同一来源终态重复到达时 `offer()` 复用候选；它不创建 Run、不扣轮数，也不操作 memory。

`consume()` 或 `invalidate()` 移除并返回候选身份，`disarm()` 同时停止推进。
宿主据此释放现有 handoff 预留；durable admission 后由启动操作继续持有 run_id，
直到 activation_started 或启动收尾。实际调用的前台计数仍归 Runner，memory 无需感知 Goal。

`GoalChanged(session_id, view)` 是冻结的会话级最新快照通知，不伪造 RunEvent。
通知不承担逐操作重放或调度触发；丢失通知后仍可通过 get 读取真实状态。
