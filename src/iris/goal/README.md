# Goal：跨 Run 的目标状态

`iris.goal` 保存期望完成的目标、状态、自动执行轮数和 Run 绑定。目标与一次
logical Run 分开：一个 Run 结束不会自行把目标标记为完成。

当前提供状态服务和原子存储契约；本阶段尚未接入模型工具、上下文与自动续跑。

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
`GoalConfig` 默认 `enabled=False`、`max_rounds=20`，开关由后续执行装配使用；
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
