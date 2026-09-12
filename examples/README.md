# Iris 示例

`examples/` 是 Iris 仓库的统一教学入口。以下命令都从仓库根目录运行。

## 运行前提

会调用 DeepSeek provider 的示例需要配置 API key。PowerShell 中可以直接设置：

```powershell
$env:IRIS_PROVIDER_API_KEYS__DEEPSEEK = "sk-..."
```

也可以把同名变量写入 `.env.local`，并在命令末尾追加 `--env-file .env.local`。

## 第一次运行

使用 `examples/chat/agent.yaml` 启动交互式会话：

```powershell
uv run iris chat examples/chat/agent.yaml --session-id example
```

Chat 的本地输出器在 runtime 所属 event loop 中直接消费 typed 文本 delta 并同步写入终端。
`SessionManager` 的单一事件消费者负责 durable 终态与 HITL 提示；已显示的成功响应不会重复
打印，没有文本增量时仍补显完整结果，失败或取消时会收尾已输出的文本行。

当前 run 执行期间可以继续输入普通文本，它会作为 `steer` 在下一个安全边界进入当前 run；使用
`/follow-up <消息>` 可以排入下一轮。Ctrl-C 会先请求中断当前 run，再保持原有行为退出 chat。
permission / question 提示出现后，下一行输入会作为 typed HITL response，而不是普通消息。

## MCP

本地 MCP 的 JSON/Codex TOML 双格式示例见 [mcp/README.md](mcp/README.md)。服务提供无需凭据的
只读 echo；可通过离线 provider 测试实际调用，再使用 `uv run iris chat examples/mcp/agent.yaml`
体验模型调用。

## Skill

Chat 示例在 `examples/chat/workspace/.agents/skills/` 内提供 `review-python`，并通过
`agent.yaml` 的 `skills.require` 保证启动时能够发现它。可以直接输入：

```text
请使用 review-python 检查 hello.py，只报告问题，不修改文件。
```

模型会先从 system context 的 Skill catalog 发现该名称，再调用自动注册的 `load_skill` 读取
`SKILL.md`；无需把 `load_skill` 写进 `tools.builtin`。

## Sub Agent

`examples/subagent/` 演示父 Agent 通过 `subagent` 工具委派任务，再根据子 Agent 的最终文本
汇总回答。它包含以下配置：

| 文件 | 用途 |
| --- | --- |
| [subagent/agent.yaml](subagent/agent.yaml) | 父 Agent，通过 `tools.subagent` 启用 catalog |
| [subagent/subagents.yaml](subagent/subagents.yaml) | 声明默认子 Agent、selector、配置路径和职责 |
| [researcher/agent.yaml](subagent/agents/researcher/agent.yaml) | 分析 prompt 中提供的文本 |
| [interviewer/agent.yaml](subagent/agents/interviewer/agent.yaml) | 调用 `ask_question` 澄清需求 |

父子 Agent 都使用 DeepSeek，按前面的运行前提配置 API key，然后启动父 Agent：

```powershell
uv run iris chat examples/subagent/agent.yaml --session-id subagent-example
```

在 chat 中输入以下内容，体验默认 `researcher` 委派：

```text
请使用默认子 Agent 分析这段需求，归纳两个要点：Iris 面向 Python 开发者，使用 YAML 配置本地 Agent，由宿主应用负责界面和输入输出。
```

父 Agent 会调用 `subagent({"prompt": "..."})`；省略 `agent` 时使用 catalog 的 `default`。
要体验显式选择和子 Agent 提问，可以输入：

```text
请委派给 interviewer，先询问我这个项目的目标用户是谁，再整理一段需求摘要。
```

此时父 Agent 调用 `subagent({"agent": "interviewer", "prompt": "..."})`。终端显示问题后
输入答案，Iris 会继续原来的子任务，最后由父 Agent 回答。

也可以复用 lifecycle 脚本，观察子 Agent 的问题如何让父 run 进入 `waiting`：

```powershell
uv run python -m examples.lifecycle.start --config examples/subagent/agent.yaml --session-id subagent-hitl --input "请委派给 interviewer，先询问目标用户是谁，再整理需求摘要。"
```

取输出中父 run 的 `run.run_id` 和 `pending_interaction.interaction_id`，在新进程中提交答案：

```powershell
uv run python -m examples.lifecycle.resume --config examples/subagent/agent.yaml --run-id RUN_ID --interaction-id INTERACTION_ID --answer "希望用 YAML 快速搭建本地 Agent 的 Python 开发者"
```

父子 run 共用 `examples/subagent/.iris/subagent.db`，各自有独立的 session/run identity。
回答始终提交给父 run，无需操作 child runner。读取状态和事件也可复用下方 lifecycle 命令，
每次都传入同一个 `--config examples/subagent/agent.yaml`。

Catalog 路径相对父 YAML，child 配置路径相对 catalog；只有被选中的 child YAML 才会加载。
Child 是普通 Agent 配置，当前只支持一层委派；它仅接收委派 prompt，不自动继承父会话历史。
需要传递的材料应直接写入 prompt。Python SDK 的装配与恢复接口见
[harness README](../src/iris/harness/README.md)。

## Provider

基础调用展示 provider-neutral 请求和流式事件；文本 delta 会在到达时立即写入终端，成功终态
仍携带标准化完整响应：

```powershell
uv run python -m examples.provider.basic --model deepseek/deepseek-chat
```

进程内 trace 包装器会逐条转发同一事件流，并在结束后额外打印标准化请求、最终响应或安全错误：

```powershell
uv run python -m examples.provider.trace --model deepseek/deepseek-chat
```

## Lifecycle

启动一个会停在问题交互上的 logical run：

```powershell
uv run python -m examples.lifecycle.start --input "调用 ask_question 一次，询问我选择 test 还是 production。" --session-id example
```

使用输出中的 `run_id` 读取状态和事件。首次读取事件时从 `0` 开始：

```powershell
uv run python -m examples.lifecycle.status --run-id RUN_ID
uv run python -m examples.lifecycle.events --run-id RUN_ID --after-sequence 0
```

使用 waiting 结果中的 `interaction_id` 回答问题；权限交互则把 `--answer test` 换成
`--decision approve` 或 `--decision reject`：

```powershell
uv run python -m examples.lifecycle.resume --run-id RUN_ID --interaction-id INTERACTION_ID --answer test
```

取消另一个仍为 active 的 run，并等待 durable settlement：

```powershell
uv run python -m examples.lifecycle.cancel --run-id RUN_ID --reason "停止示例" --settlement-timeout 30
```

恢复 active run 时必须传入公开状态中的精确 activation fence：

```powershell
uv run python -m examples.lifecycle.recover --run-id RUN_ID --activation-id ACTIVATION_ID
```

`run_id`、`interaction_id`、`activation_id` 和 `next_after_sequence` 必须取自这些脚本打印的
公开模型输出。后续读取事件时把 `next_after_sequence` 作为新的 `--after-sequence`；不要猜测
identity，也不要按 `session_id` 发现或恢复 run。

`start`、`resume` 和 `recover` 会执行 provider，因此需要 API key；`status`、`events` 和
`cancel` 会注入禁止执行的 provider，只访问 durable lifecycle 状态，不调用网络。

这些模块是仓库内的教学脚本，不是随 Iris 安装的 console command；其打印内容也不是稳定的
JSON API。
