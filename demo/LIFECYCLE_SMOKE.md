[English](LIFECYCLE_SMOKE.en.md)

# `iris run` Lifecycle 人工 Smoke

本指南从仓库根目录用真实 DeepSeek `deepseek-chat` 和专用 SQLite 数据库验收
`start/status/events/resume/cancel/recover`。其中 events 是 provider-free 只读操作；其余场景会
访问网络并消耗真实 provider 配额，不要把它放进
pytest 或 CI。三个场景使用不同 run；cancel 和 recover 是替代分支，不能在同一个已经 terminal
的 run 上依次执行。

## 准备与重置

在每个新 PowerShell 窗口中设置 key 和本地 uv cache：

```powershell
$env:IRIS_PROVIDER_API_KEYS__DEEPSEEK = "sk-xxx"
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
```

也可在后续命令加 `--env-file .env.local`，由文件提供 key。以下重置命令具有破坏性；它只删除
专用 smoke 数据库及其 SQLite sidecar，不影响 `demo/.iris/demo-session.db`。先关闭仍在运行的
smoke 进程，再执行：

```powershell
Remove-Item -LiteralPath @(
  "demo\.iris\lifecycle-smoke.db",
  "demo\.iris\lifecycle-smoke.db-wal",
  "demo\.iris\lifecycle-smoke.db-shm"
) -Force -ErrorAction SilentlyContinue
```

## 场景 A：HITL waiting、重启与 resume

启动固定 identity 的 run：

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run start demo/lifecycle-agent.yaml --input "Call ask_question exactly once with question 'Choose smoke environment' and options ['test', 'production']. Do not answer it yourself." --session-id lifecycle-hitl --run-id lifecycle-hitl-001 --json
```

验收 start 输出和退出码：

- `$LASTEXITCODE` 为 `0`；
- `ok=true`、`run.phase="waiting"`；
- `run.run_id="lifecycle-hitl-001"`；
- 保存 `pending_interaction.interaction_id`，以下以 `INTERACTION_ID` 表示。

原命令已经退出。在新 PowerShell 进程中显式提交 question answer：

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run resume demo/lifecycle-agent.yaml --run-id lifecycle-hitl-001 --interaction-id INTERACTION_ID --answer "test" --json
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run status demo/lifecycle-agent.yaml --run-id lifecycle-hitl-001 --json
```

resume 和 status 都应退出 `0`，返回同一个 `run_id`，最终
`run.phase="terminal"`、`run.stop_reason="completed"` 且包含 assistant message。若模型没有调用
`ask_question`，删除专用 DB 后重试，并保留更明确的 “exactly once / do not answer” 输入；不要手工
伪造 interaction ID。

## 场景 B：双终端跨进程 cancel

终端 A 启动独立 run；看到 marker 后保持窗口运行：

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run start demo/lifecycle-agent.yaml --input "Call wait_for_seconds exactly once with seconds=30, then report its result." --session-id lifecycle-cancel --run-id lifecycle-cancel-001 --json
```

预期 marker：

```text
IRIS_LIFECYCLE_SMOKE_TOOL_STARTED seconds=30
```

终端 B 在 marker 出现后执行：

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run cancel demo/lifecycle-agent.yaml --run-id lifecycle-cancel-001 --reason "phase 6 cross-process smoke" --settlement-timeout 45 --json
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run status demo/lifecycle-agent.yaml --run-id lifecycle-cancel-001 --json
```

同步工具不会协作式中断；cancel 必须等待工具返回后的安全检查点，不能提前声称 settled。最终两条
输出都应退出 `0`，且 `run.phase="terminal"`、`run.stop_reason="cancelled"`。

## 场景 C：claimed effect、进程中断与 fenced recover

终端 A 启动第三个独立 run：

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run start demo/lifecycle-agent.yaml --input "Call wait_for_seconds exactly once with seconds=60, then report its result." --session-id lifecycle-recover --run-id lifecycle-recover-001 --json
```

看到下面的 marker 后立即在终端 A 按 Ctrl+C；不要运行 cancel：

```text
IRIS_LIFECYCLE_SMOKE_TOOL_STARTED seconds=60
```

新进程先读取 durable active snapshot：

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run status demo/lifecycle-agent.yaml --run-id lifecycle-recover-001 --json
```

status 应退出 `0`，返回 `run.phase="active"`。保存精确的
`run.current_activation_id`，以下以 `ACTIVATION_ID` 表示；不要猜测或改用较新的 ID。然后执行：

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run recover demo/lifecycle-agent.yaml --run-id lifecycle-recover-001 --activation-id ACTIVATION_ID --json
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run status demo/lifecycle-agent.yaml --run-id lifecycle-recover-001 --json
```

recover 和最终 status 都应退出 `1`，返回 `ok=false`、`run.phase="terminal"`、
`run.stop_reason="outcome_unknown"` 以及 durable tool error。recover 期间不得再次出现 wait-tool
marker；这证明 claimed effect 没有被重放。

## 通用检查：durable event 时间线

任一场景创建 run 后，都可以在新 PowerShell 进程中只读检查其完整 durable event 时间线：

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"; uv run iris run events demo/lifecycle-agent.yaml --run-id lifecycle-hitl-001 --after-sequence 0 --json
```

命令应退出 `0`，`events` 按 sequence 升序保持存储顺序，每一项的 sequence 都严格大于输入的
`after_sequence`。把响应中的 `next_after_sequence` 作为下一次命令的 cursor；如果没有新事件，
返回 `events=[]`，且 `next_after_sequence` 保持输入值。events 每次只读取一次，不 watch、不调用
provider，也不改变 run，因此不要求 DeepSeek key。

## 通用输出与排查

- `--json` 正常 durable 输出只在 stdout 写一个紧凑 JSON 对象；无 durable run 的操作异常写 stderr。
- wait-tool marker 写 stderr，因此不会污染供脚本解析的 JSON stdout。
- exit `0` 表示 waiting、active 或非失败 terminal outcome；exit `1` 表示操作失败、failed 或
  outcome_unknown；argparse 错误为 `2`；Ctrl+C 为 `130`。
- status/events/cancel 不要求 DeepSeek key；start/resume/recover 需要真实 key。
- `RUN_NOT_FOUND` 通常表示 run ID、配置路径或专用 DB 不一致。
- `RUN_CONFLICT` 表示 interaction/activation identity 已过期或不匹配；不要自动重试另一个 ID。
- 如果 provider 没按要求选择工具，重置专用 DB、换一个新的固定 run ID，并把输入写得更明确；
  不要把非工具回复记为通过。
- 为保证 human next command 可直接复制到 PowerShell，本指南的自定义 run/session identity 只使用
  ASCII 字母、数字和连字符；不要在 identity 中使用引号或 PowerShell 元字符。

## 验收记录

| 项目 | 记录 |
| --- | --- |
| 日期 | |
| provider/model | DeepSeek / `deepseek-chat` |
| 场景 A run ID / phase / stop reason / PASS-FAIL | |
| 场景 B run ID / phase / stop reason / PASS-FAIL | |
| 场景 C run ID / phase / stop reason / PASS-FAIL | |
| interaction ID | |
| activation ID | |
