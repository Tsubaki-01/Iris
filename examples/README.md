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

当前 run 执行期间可以继续输入普通文本，它会作为 `steer` 在下一个安全边界进入当前 run；使用
`/follow-up <消息>` 可以排入下一轮。Ctrl-C 会先请求中断当前 run，再保持原有行为退出 chat。
permission / question 提示出现后，下一行输入会作为 typed HITL response，而不是普通消息。

## Provider

基础调用展示 provider-neutral 请求和响应：

```powershell
uv run python -m examples.provider.basic --model deepseek/deepseek-chat
```

进程内 trace 包装器会额外打印标准化请求和响应：

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
