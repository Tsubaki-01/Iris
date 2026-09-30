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

- 普通输入：空闲时创建 Run，执行中按现有安全边界 steer 当前 Run。
- `/follow-up <消息>`：排入下一轮，等待当前 Run 收尾。
- 人工交互：permission 输入 `y/yes/n/no`，空输入为拒绝；question 输入答案或选项编号。
- `/help` 显示命令；`/exit`、`/quit`、EOF 退出；Ctrl-C 取消当前执行并退出，退出码为 130。

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
