[English](README.en.md)

# `iris.cli`

`iris chat` 是基于 `AgentRunner` 和单个 `SessionManager` 的终端宿主。主线程读取输入，
后台 event loop 执行 Agent、消费事件和接收人工回答；CLI 不拥有运行、目标或历史的权威状态。

## 启动与普通输入

```bash
iris chat agent.yaml --session-id work --max-steps 8
```

`agent.yaml` 的模型和工具使用 [agents 配置](../agents/README.md)。可通过 `--env-file`
加载指定 dotenv 文件。`--max-steps` 限制每个 Run 的模型步数；`--no-tools` 关闭模型工具，
此选项不能用于创建自动 Goal。

启动时先按有效 root workspace 解析 `prompts.root`（默认 `.iris/prompts`），只补齐缺少的
命名模板，然后把同一个 `PromptSource` 传给 Memory 服务和 runner。已有项目正文保留。
手工修改策略后，Goal/Todo、system/context 等指引由新建 runner 采用；压缩和自动 Memory
分别在下一次完整压缩、下一轮维护开始时采用。原 `system` / `context` 配置入口不变，详见
[项目提示来源](../prompts/README.md)。

启用 `evolution.enabled`（同时启用 `skills.enabled`）时，CLI 用主配置装配项目经验服务，
与 Memory 共用宿主维护协调器；两类任务各自持锁、取消和排空。关闭 Memory 仍可维护项目
经验；退出只收尾，不补跑总结。新生成 Skill 由下次启动的 runner 发现，没有新增维护命令。

`prompt_targets/config_targets` 允许按真实任务中的具体问题修订有限目标。CLI 始终把本次
主 YAML 路径交给候选解析器；Config 保存后需重新启动 `iris chat` 才采用，新 session 不会
热切换已有 runner 的配置。

- 普通输入：空闲时创建 Run，执行中按现有安全边界 steer 当前 Run。
- `/follow-up <消息>`：排入下一轮，等待当前 Run 收尾。
- `/todo`：只读查看当前会话待办及实际文件路径。
- 人工交互：permission 输入 `y/yes/n/no`，空输入为拒绝；question 输入答案或选项编号。
- `/help` 显示命令；`/exit`、`/quit`、EOF 退出；Ctrl-C 取消当前执行并退出，退出码为 130。

## 自动维护

启用 `memory.generation.enabled` 后，CLI 将已构造的 Memory 服务绑定到一个共享
`MaintenanceCoordinator`，在接收输入前绑定 runner。`maintenance.idle_seconds` 控制安静时间，
默认 300 秒；Memory 的生成预算继续位于 `memory.generation`。

自动维护仅消费已结束且完整捕获的 Run。等待用户回答时，该会话的材料保留 pending；
新前台输入取消未提交的生成。退出时先停止 SessionManager，再排空维护的实际 IO，
最后关闭 runner 自有资源和后台 event loop，不在退出阶段补跑模型。

启用 Agent 的 `observability.enabled` 后，CLI 从全局 `observability` 导出配置创建一个
共享观测服务，传给 runner、Memory、Evolution 与维护协调器。安装要求和完整 OTLP 地址配置
见 [observability](../observability/README.md)。正文默认不采集；关闭观测时不创建 SDK/exporter。
退出时先完成上述业务收尾，再关闭观测服务；同步装配或准备失败也由 CLI 释放自己的服务。
直接调用 `run_chat_loop(runner=...)` 不接管 runner 借用的观测资源；只有显式传入
`observability=...` 才将该共享服务的关闭责任交给 chat 宿主。

## 查看 Todo 工作清单

在现有 Agent 配置中设置 `todo.enabled: true`，并保持 `context_policy.enabled: true`，
重建 Agent 后即可使用 `/todo`。该命令显示已完成数/总数、文件绝对路径，以及
`[ ]`（待处理）、`[-]`（进行中）、`[x]`（已完成）的全部条目。

清单缺失或为空时显示“暂无待办”；格式错误时显示路径和诊断，不当作完成。
人工修改文件后再次执行 `/todo` 即读取最新内容。命令不创建文件，不自动清除完成项；
模型维护文件所需的普通文件工具仍须显式配置，详见 [Todo](../todo/README.md)。

普通或 Goal Run 执行中、等待 question/permission 回答时都可查看；`/todo` 不成为
聊天输入、steer 或人工答案，不改变待回答交互，下一条实际回答仍恢复原交互。
命令不接受参数，`/todo extra` 显示“用法：/todo”。未启用或文件读取失败只显示错误，
聊天继续；不会自动启用能力，也不轮询文件变化。

## 自动 Goal

在现有 Agent 配置中加入：

```yaml
context_policy:
  enabled: true
goal:
  enabled: true
  max_rounds: 20
session:
  backend: sqlite
```

然后输入明确目标与验收要求，例如：

```text
/goal 修复计算函数的空输入处理，并让指定测试通过
```

Goal 可以跨多个 Run；单个 Run 内本来就能多次调用模型和工具，不按固定步骤拆轮。
CLI 只等待目标控制操作的回执，主输入仍可继续。用户输入和已排队的 follow-up 优先于自动续跑。

| 命令 | 行为 |
| --- | --- |
| `/goal` | 显示用法。 |
| `/goal <目标>` | 创建并允许自动推进，采用当前 CLI 的模型步数与工具选项。 |
| `/goal status` | 只读目标、状态、是否允许自动推进、轮数、原因、占用 Run、交互和错误。 |
| `/goal edit <目标>` | 替换正文并暂停，保留 Goal ID 与已用轮数。 |
| `/goal edit --max-rounds 30` | 调整总轮数并暂停，保留正文与已用次数。 |
| `/goal pause` | 暂停后续轮次；当前 Run 可以继续收尾。 |
| `/goal resume` | 显式恢复；已有执行或人工交互继续使用原 Run。 |
| `/goal complete` | 用户声明目标完成；当前 Run 可以继续收尾。 |
| `/goal clear` | 清除当前选择，保留历史目标和 Run 绑定。 |

保留词开头的目标使用 `/goal -- status 分析要求`；选项前缀开头的编辑正文使用
`/goal edit -- --max-rounds 是正文`。正文内部的空格和引号保留；非法参数显示用法，
不会作为普通聊天或 steer 发送。Goal 未启用时提示配置，CLI 不会修改 YAML。

轮数在 Run 准入时消耗且不退款；resume 不清零。最终允许轮仍可完成目标。
达到额度后可以先 `edit --max-rounds` 提高总额，再 `resume`。完成状态来自用户声明或模型的
已提交报告，不代表独立验收。立即停止当前执行使用 Ctrl-C。

新 manager 默认不自动恢复旧目标。WAITING 继续回答原问题；遗留 ACTIVE 缺少本进程调用时，
`resume` 显示 run/activation ID，并提示显式 SDK 调用
`await manager.goal.resume(expected_activation_id="...")`。CLI 不提供通用 lifecycle 恢复命令。
SQLite schema 按新契约使用，不提供旧 schema migration；child/fork 不继承 Goal。
更多控制回执、绝对 deadline 与恢复规则见 [goal](../goal/README.md)。

## 实现与验证

`chat.py` 的 `_ChatSessionHost` 将所有 Goal 控制交给 manager 所属 event loop。
`GoalChanged` 与 Run 事件复用 mixed stream，按终态顺序显示最新目标快照；配置 live 文本输出时
使用同一 GoalView 格式，不重复打印另一条 Goal 路径。通知可以合并，`status` 用于读取当前事实。
命令排空失败和目标存储错误保留实际错误内容，不伪造完成状态。

`tests/cli/test_chat_goal.py` 使用真实 host、manager、runtime 与存储以及可控 provider，覆盖
两轮完成、命令派发、正文保留、关闭开关和显式恢复提示；它不是一次真实模型质量评估。

`tests/cli/test_chat_todo.py` 通过真实 `run_chat_loop` 验证清单三态、人工编辑刷新、
查询错误与参数提示，以及普通/Goal 执行和 question/permission WAITING 时的只读行为。
