# 持续推进目标与维护清单

Goal 让一个目标跨多个 Run 推进；Todo 把当前会话的工作清单保存在 Markdown 文件中。二者可以一起用，也可以单独开启。Goal 控制是否启动下一轮，Todo 帮助模型和人查看待办，勾完清单不会自动完成 Goal。

前提：已完成[快速开始](../getting-started/quickstart.md)。本例让 Agent 在 workspace 中整理笔记，允许普通文件工具维护清单。

## 配置和启动

保存 `agent.yaml`：

```yaml
name: notes-assistant
model: deepseek/deepseek-flash
system: |
  帮助用户整理项目笔记，按实际完成结果更新清单。
  Goal 启用时，核对交付物后再申报完成；需要用户补充时说明缺少什么。
permissions:
  workspace: .
  writes: allow
goal:
  enabled: true
  max_rounds: 6
todo:
  enabled: true
tools:
  builtin:
    - file.read
    - file.write
```

```powershell
uv run iris chat agent.yaml
```

Goal 和 Todo 都要求 `context_policy.enabled: true`，这是当前默认值；不要在本配置里关闭它。Goal 自动轮还需要允许工具，并让 `tool_choice` 为 `auto` 或未指定，否则无法按约定读取和申报目标。

先在 workspace 新建 `notes.md`，提供实际输入材料，例如：

```markdown
# 项目笔记

文档正文使用中文，代码标识符保留英文。
第一次运行先用 YAML 和 CLI，再介绍 Python SDK。
待确认：网站最终采用哪种导航样式。
```

在 CLI 输入一个具体目标：

```text
/goal 将当前目录的 notes.md 整理成 summary.md，列出待确认事项并核对遗漏
/goal status
```

创建会保存目标并允许该宿主自动推进，不需要先提交一条普通消息。自动轮仍然是普通的完整 Run：调用模型、执行工具、保存结果，再由宿主判断后继轮是否可准入。终端会显示目标状态、自动推进是否开启、已经使用的轮数以及当前 Run 或人工等待状态。

## 查看和编辑当前清单

输入 `/todo` 可查看当前会话文件路径和条目。运行时也会把路径交给模型；由 `file.write` 等普通工具创建和修改文件，Iris 读取清单本身不会创建文件。

文件使用 UTF-8，内容可以是：

```markdown
# 笔记整理

- [x] 阅读 notes.md
- [-] 归纳主题和关键结论
- [ ] 核对 summary.md 是否遗漏原始事实
```

允许标题、空行和单行条目；`[ ]` 是待处理，`[-]` 是进行中，`[x]` 或 `[X]` 是完成。不要加入普通段落、嵌套列表或条目续行。格式有误时，读取结果会给出行号诊断，不会把半份清单当成有效列表。

清单位于 `.iris/todos/<session ID 的 UTF-8 十六进制>.md`。用 SDK 取得 `snapshot.path` 比手动推导文件名更方便：

```python
snapshot = await runner.get_todo("notes-session")
print(snapshot.path)
print(snapshot.error)
for item in snapshot.items:
    print(item.status.value, item.content)
```

上面是已有 runner 中的片段，完整 Python 宿主见下一节。每个获准的模型步骤都会重新读一次当前文件，因此人工保存的修改会在下一次读取时生效。若模型准备结束而仍有未完成项或格式错误，且尚有步骤与期限额度，runtime 在同一 Run 内安排最多一次结束自查；它不保证清单必须清空，也不会额外开启 Run。

## 从 Python 启动并观察 Goal

在同一工作目录保存 `run_goal.py`。此示例假设任务不需要人工交互；如果出现等待，会退出本次脚本并取消当前运行，实际交互宿主应接入[HITL 响应](hitl-recovery.md)。

```python
import asyncio

from iris import init_config
from iris.harness import AgentRunner, GoalChanged, SessionManager
from iris.lifecycle import AgentRunOptions, RunLimits


async def main() -> None:
    init_config()
    runner = AgentRunner.from_config_path("agent.yaml")
    manager = SessionManager(runner, "notes-session")
    try:
        goal = manager.goal
        assert goal is not None
        receipt = await goal.create(
            "将 notes.md 整理为 summary.md，并核对遗漏",
            max_rounds=6,
            run_options=AgentRunOptions(limits=RunLimits(max_model_steps=12)),
        )
        print(receipt.disposition)
        async for event in manager.events():
            if isinstance(event, GoalChanged):
                view = event.view
                if view.goal is not None:
                    print(view.goal.status.value, view.goal.rounds_started)
                if view.interaction is not None or view.driver_error is not None:
                    print("需要宿主处理", view.interaction, view.driver_error)
                    break
                if view.goal is not None and view.goal.status.value != "active":
                    break
        snapshot = await runner.get_todo("notes-session")
        print(snapshot.path, snapshot.items, snapshot.error)
    finally:
        await manager.close(cancel_run=True)
        await runner.aclose()


asyncio.run(main())
```

```powershell
uv run python run_goal.py
```

`goal.create()` 的返回值是控制回执，不是最终模型答案。通过会话事件读取进展；最终运行结果、消息和工具记录仍在 lifecycle store 中，查询方式见[运行参考](../reference/runtime.md)。需要重启后保留 Goal 和会话时，为 runner 配置或注入 SQLite lifecycle store，见[会话配方](sessions.md)。Todo 文件独立保存在 workspace。

## 暂停、编辑、恢复和中断

| 操作 | 效果 |
| --- | --- |
| `/goal pause` | 停止后续自动轮；当前已准入 Run 继续 |
| `/goal edit 新目标` | 修改目标并暂停后续推进 |
| `/goal edit --max-rounds 10` | 修改总轮数上限；不是再增加十轮 |
| `/goal resume` | 显式允许推进，优先接手原 Run；人工等待仍需回答 |
| `/goal complete` | 用户直接声明完成；不取消当前 Run |
| `/goal clear` | 清除当前目标选择，保留目标和历史记录 |
| Ctrl-C / `manager.interrupt()` | 中断当前工作，并停止目标自动推进 |

`max_rounds` 统计已准入的自动顶层 Run。恢复同一个 Run 不增加一轮；普通用户 Run 不消耗目标轮数。单 Run 的 `max_model_steps`、期限和人工等待超时仍单独生效；模型请求的输入/输出预算也不等同于目标轮数。Goal 当前没有跨轮累计的 token 预算字段。轮数用尽或 Run 异常结束时，目标暂停；它不把额度耗尽算作完成。

进程重建后，持久 `active` 不代表新宿主已经获准自动推进，必须显式 `goal.resume()`。如果原 Run 仍记为 `ACTIVE` 而本进程没有执行任务，回执可能是 `needs_recovery`；确认要接管后，在 SDK 传入查询所得的 `expected_activation_id`。不要用创建新目标代替原 Run 的恢复。

下一步：阅读[Goal 与 Todo 设计](../design/goals.md)理解结束申报、结算和续跑的关系；所有控制返回值和文件格式见[长期能力参考](../reference/memory-goals.md)。

实现入口：[Goal 会话 SDK](../../src/iris/goal/session.py)、[Goal 结算](../../src/iris/goal/settlement.py)、[Todo 解析](../../src/iris/todo/document.py)。
