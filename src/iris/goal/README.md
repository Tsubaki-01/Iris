# Goal：跨 Run 持续推进目标

`iris.goal` 管理目标是否完成、是否允许继续，以及自动执行轮数。一个 Run 内本就可以多次
调用模型和工具；正常结束但目标尚未完成时，SessionManager 才安排下一轮，不要求固定分阶段。

## 从配置到执行

在已配置模型和所需工具的 `agent.yaml` 中启用：

```yaml
context_policy:
  enabled: true
goal:
  enabled: true
  max_rounds: 20
```

默认 `goal.enabled=false`，此时 `manager.goal` 为 None，服务、工具和目标上下文都不挂载。
配置加载不会创建目标。完整自动推进使用 `AgentRunner → SessionManager → manager.goal`；
直接 `runner.start()` 始终只执行调用方的一次普通 Run。

下面示例展示创建、只读查询、暂停、恢复和消费结果；将目标和测试路径换成自己的项目内容。
需要人工回答的应用还应接入后面的 HITL 响应流程。

```python
import asyncio

from iris.harness import AgentRunner, GoalChanged, SessionManager


async def main() -> None:
    runner = AgentRunner.from_config_path("agent.yaml")
    manager = SessionManager(runner, "fix-calculation")
    try:
        goal = manager.goal
        assert goal is not None  # 配置已经显式启用 Goal。
        created = await goal.create(
            "修复计算函数的空输入处理，并让 tests/test_calculation.py 通过",
            max_rounds=3,
        )
        print(created.disposition)
        print(await goal.get())
        await goal.pause(reason="先确认测试范围")
        await goal.resume()

        async for event in manager.events():
            if isinstance(event, GoalChanged) and event.view.goal is not None:
                current = event.view.goal
                print(current.status, current.rounds_started, current.reason)
                if current.status in {"completed", "blocked", "paused"}:
                    break
    finally:
        await manager.close(cancel_run=True)
        await runner.aclose()


asyncio.run(main())
```

`create()` 返回时目标已保存并允许调度，不保证模型已经开始。默认 mixed 模式应持续消费
`manager.events()`，否则 tracker 满时暂停新轮准入；broker-only 模式通过 publisher 观察，
不需要本地 consumer。用户输入及已排队 follow-up 优先于自动续跑。

## 控制与状态

所有 `manager.goal` 方法都是 async；除 get 外返回 `GoalControlResult(view, disposition)`。

| 方法 | 语义 |
| --- | --- |
| `create(objective, *, max_rounds=None, run_options=None)` | 创建当前目标并允许推进；缺省使用配置轮数和 AgentRunOptions。 |
| `get()` | 只读 GoalView，不补结算、arm 或启动。 |
| `edit(*, objective=None, max_rounds=None, run_options=None)` | 至少修改一项；保留已用次数并暂停。None 表示未修改。 |
| `pause(*, reason)` | 暂停后续推进，当前 Run 可以收尾。 |
| `resume(*, expected_activation_id=None)` | 先处理原 Run 的附着或恢复，再决定是否继续。 |
| `complete(*, reason)` | 用户明确完成目标；不取消已准入 Run。 |
| `clear()` | 清除当前选择并停止后续推进，保留目标与绑定历史。 |

持久状态为 active、paused、blocked、completed；armed 是当前 manager 的进程内推进意图。
每个 session 至多一个当前目标，未完成目标不能被 create 隐式覆盖。已完成目标不能编辑或
恢复为新任务，应创建新 Goal。

| disposition | 实际达到的阶段 |
| --- | --- |
| scheduled | 已允许后续调度，可能仍在等待用户工作或 tracker 容量。 |
| admitted | Run 已创建。 |
| running | 复用已有 live invocation。 |
| waiting | 原 Run 等待人工回答。 |
| needs_recovery | 原 ACTIVE Run 无 live invocation，需要明确 activation fence。 |
| occupied | lane 被普通 Run 或其他 Goal 的执行占用。 |
| stopped | 仅暂停、完成、清除，或额度用尽。 |

`GoalView.run` 表示当前 session lane，`run_goal_id` 区分普通 Run 和目标执行。
无 lane 时 run=None；旧终态仍可能 settlement_pending。driver_error 表示未能可靠落盘的
运行控制错误，不能据此推断目标已暂停或完成。

**暂停与立即停止不同。** `await manager.interrupt()` 先停止 Goal 续跑，再按既有契约取消
当前 Run；仅暂停空闲 Goal 时返回 None。`CancelAccepted.run` 同样允许 None。
关闭时先 `manager.close(cancel_run=True)`，再 `runner.aclose()`，让原执行和命令清理收尾。

## 轮数、恢复与人工交互

max_rounds 是自动 Run 总次数，包含 kickoff，不是模型调用次数或 token 上限。轮数在原子
admission 成功时消耗：即使尚未请求 provider 或绝对期限已到，也不退款；准入前失败不计数。
普通用户 Run 不计入 Goal 轮数。resume 不重置次数；增额使用 edit 后再 resume，不能降到
已用次数以下。最后允许的一轮仍可成功完成。

`run_options.limits.max_model_steps` 仍是每 Run 限制，`deadline_at` 始终是绝对时刻，
后续 Run 不会获得重新计时的期限。Goal 执行要求 include_tools=True，按运行选项覆盖模型
配置后，最终 tool_choice 只能为 None/auto。

新 manager 总是 disarmed，不会自动启动持久 active 目标。先调用 `goal.resume()`：

- 原 WAITING Run 仍等待回答时，返回原 interaction。通过
  `manager.resume(interaction_id=..., response=QuestionInteractionResponse(...))` 或
  `PermissionInteractionResponse(...)` 回答它；保持原 run_id 和 round，不另开一轮。
- ACTIVE 无 live invocation 时返回 needs_recovery。读取 view.run.current_activation_id，
  再显式调用 `goal.resume(expected_activation_id=该值)`。不可把存在 run_id 当成仍在执行。
- 命令排空失败保留原 Run 和 pending 收尾；显式恢复重试原收尾。异常终态会暂停，
  不会在同次恢复中自动替换为新尝试。

## 完成意味着什么

模型通过 get_goal 获取最新目标和版本，通过 report_goal 申报 complete、blocked 或 continue。
工具成功只记录本轮申报；正常 Run 结束后，存储才基于已提交事实结算。工作工具与申报同一步、
申报后的新工作、已交付用户输入或人工回答会使原完成判断失效，需要重新申报。continue 撤回
早期申报。用户暂停、完成、清除或替换目标的决定不会被旧 Run 覆盖。
申报不会结束或切换 Run。本轮工作结束后，模型用不含工具调用的普通文本结束回复，随后由
SessionManager 决定是否启动下一轮；提示要求遵守目标规定的阶段和轮次范围，但不强制模型切轮。

**completed 表示主模型或用户声明目标完成，不是独立验收认证。** 模型依据的工具结果和
验收文本决定判断质量；框架保证状态及报告时序一致，不提供额外 judge。

## 存储、范围与错误

Goal 与 Run 使用同一个 InMemoryLifecycleStore 或 SQLiteStore。创建可以早于首次聊天，
不写聊天历史、不占 lane；Goal 控制不会改变 session history revision。
SQLite 当前 lifecycle schema 为 11，不兼容旧库且不迁移；使用新数据库路径，旧文件不重置。
这个 schema 约束在 Goal 关闭时也适用。

默认 child 不继承 Goal，显式为 child 开启会报配置错误；history fork 不复制当前目标和绑定。
目标全文从存储投影为 required 动态上下文，不依赖长期记忆或聊天压缩摘要；旧目标 Run
不会收到替换目标的正文。Goal 共用现有 memory handoff，等待用户/HITL/容量不长期占用前台。

配置不支持、目标身份/版本冲突、额度耗尽、原 Run 待恢复、命令清理失败与存储故障均通过
领域异常或实际 view/result 表达。get 不暗中修复，存储失败不伪报成功。

公开 SDK 类型包括 GoalSession、GoalView、GoalControlResult、GoalChanged；后二者是冻结
快照/回执。GoalChanged 可以合并，重连后用 get 读取当前状态，不能据其逐条重放操作。
底层 GoalService/GoalStore 支持领域操作和自定义 backend；driver、调度 intent 和 admission
helper 留在对应内部模块，不作为顶层稳定 SDK。模型在 [models.py](models.py)，原子存储协议
在 [store.py](store.py)，宿主控制边界在 [session.py](session.py)。
