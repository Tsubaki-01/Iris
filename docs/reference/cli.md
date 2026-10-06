# CLI 参考

`iris` 当前提供一个子命令：`chat`。它是使用 AgentRunner 和 SessionManager 的终端宿主。恢复、事件查询等仓库示例脚本不是额外的已安装 CLI 子命令。

## 启动参数

```text
iris chat AGENT_CONFIG [--session-id ID] [--max-steps N]
                       [--env-file PATH] [--no-tools]
```

从源码环境执行时加 `uv run`：

```powershell
uv run iris chat examples/chat/agent.yaml --session-id example --max-steps 8
```

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `AGENT_CONFIG` | 必填 | Agent YAML 文件路径，相对当前命令目录 |
| `--session-id` | `cli` | 使用的会话 ID；是否跨进程保留取决于 store |
| `--max-steps` | `8` | 每轮 logical Run 的最大模型步骤数 |
| `--env-file` | 不加载 | 指定 dotenv 文件 |
| `--no-tools` | 不启用此选项 | 设置后不向模型提供工具 |

`iris --help` 与 `iris chat --help` 可查看当前 parser 的帮助。命令没有 version、init、status、events 或 recover 子命令；模板生成使用[Python scaffold](configuration.md#模板-scaffold)。

## 普通输入和会话命令

| 输入 | 行为 |
| --- | --- |
| 普通文本，当前空闲 | 在当前 Session 启动新 Run |
| 普通文本，Run 执行中 | 作为 steer 排队，在可接收的边界进入当前 Run |
| `/follow-up 文本` | 为下一轮排队，沿用当前会话 |
| `/todo` | 按需读取当前会话清单及路径；启用方式见[Goal/Todo](memory-goals.md) |
| `/help` | 显示交互帮助 |
| `/exit` 或 `/quit` | 结束 chat |
| Ctrl-C | 请求中断当前 Run，并退出 chat |

`/follow-up` 不带正文时显示用法；`/todo` 不接受附加参数。未知的 `/` 命令会提示查看帮助，不作为普通输入发送。

人工 permission/question 提示出现后，下一行输入被解析为该交互的 typed response，不作为 steer 或新任务。具体可选值以当时终端提示为准；宿主实现对应[HITL 契约](runtime.md)。

## Goal 命令

需要 `goal.enabled: true`：

| 输入 | 行为 |
| --- | --- |
| `/goal 目标正文` | 创建自动推进目标 |
| `/goal status` | 查看持久目标、当前 Run 和进程内自动推进状态 |
| `/goal edit 新目标正文` | 修改目标 |
| `/goal edit --max-rounds 正整数` | 调整自动轮数额度 |
| `/goal pause` | 暂停目标推进 |
| `/goal resume` | 恢复目标推进 |
| `/goal complete` | 由用户声明目标完成 |
| `/goal clear` | 清理当前目标，条件见[Goal 参考](memory-goals.md) |
| `/goal` | 显示用法 |

目标正文若以 `status`、`pause` 等保留词开头，用 `/goal -- 目标正文`。编辑正文若形似选项，用 `/goal edit -- 正文`。这些命令操纵 Goal，不等于手工修改 Todo Markdown。

## 输出和持久化

终端在模型增量到达时显示文字；成功后不重复打印已经显示的同一回答，没有文字增量时补显完整结果。压缩会显示开始、完成或未完成的状态，不输出摘要正文。

CLI 的人类可读输出不是稳定 JSON API。需要可靠读取状态、事件和结果时，使用[运行 SDK](runtime.md)，或查看[仓库 lifecycle 脚本](../../examples/lifecycle)。普通终端输入、提示渲染和退出都由 CLI 管理，不需要应用自己实现。

当前界面不包含图片发送、图片渲染、录音或语音播放。对应能力由[媒体 SDK](media.md)提供给其他宿主。

源码：[命令参数](../../src/iris/cli/main.py)、[交互循环](../../src/iris/cli/chat.py)。
