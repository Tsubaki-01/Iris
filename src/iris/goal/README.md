# Goal：跨 Run 的目标状态

`iris.goal` 保存期望完成的目标、状态、自动执行轮数和 Run 绑定。目标与一次
logical Run 分开：一个 Run 结束不会自行把目标标记为完成。

提供状态服务、原子存储契约、模型工具、动态上下文与终态结算。执行装配在开启
Goal 时注册能力；本阶段不包含跨 Run 的自动续跑调度。

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
