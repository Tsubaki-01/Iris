[English](README.en.md)

# Iris Demo

这个目录提供一个可直接用于本地试用的 Iris CLI Agent 配置。它展示了 Iris 的
config-first 使用方式：通过 `agent.yaml` 声明模型、上下文、工具、workspace 和
session，然后交给 `iris chat` 启动多轮交互。

## 目录内容

- `agent.yaml`: demo agent 主配置，使用 DeepSeek `deepseek-chat`，加载
  `context.yaml`，并把 `workspace/` 作为工具工作区。
- `context.yaml`: 结构化上下文配置，定义 system、memory 和当前轮次前置提示。
- `workspace/`: demo agent 可访问的本地工作区，内含一个简单的 `hello.py` 示例文件。
- `.iris/demo-session.db`: SQLite session 文件，用于保存 demo 会话状态。
- `trace.jsonl`: 可选的 provider request / response trace 输出文件。
- `lifecycle-agent.yaml`: `iris run` 专用的 DeepSeek + SQLite 配置。
- [`LIFECYCLE_SMOKE.md`](LIFECYCLE_SMOKE.md): 显式 lifecycle identity 的人工验收指南。

## 运行前提

在仓库根目录运行命令，并确保当前环境可以读取 DeepSeek API key。Bash / Git Bash /
WSL 使用：

```bash
export IRIS_PROVIDER_API_KEYS__DEEPSEEK=sk-xxx
```

PowerShell 使用：

```powershell
$env:IRIS_PROVIDER_API_KEYS__DEEPSEEK = "sk-xxx"
```

也可以把 key 写入 `.env` 或 `.env.local`，再通过 `--env-file` 显式传给 CLI。
文件内容示例：

```dotenv
IRIS_PROVIDER_API_KEYS__DEEPSEEK=sk-xxx
```

## 启动

在仓库根目录执行。Bash / Git Bash / WSL 使用：

```bash
uv run iris chat demo/agent.yaml \
  --session-id demo \
  --trace compact \
  --trace-file demo/trace.jsonl
```

PowerShell 使用反引号续行：

```powershell
uv run iris chat demo/agent.yaml `
  --session-id demo `
  --trace compact `
  --trace-file demo/trace.jsonl
```

如果使用单独的 env 文件，Bash / Git Bash / WSL 使用：

```bash
uv run iris chat demo/agent.yaml \
  --env-file .env.local \
  --session-id demo \
  --trace compact \
  --trace-file demo/trace.jsonl
```

PowerShell 使用：

```powershell
uv run iris chat demo/agent.yaml `
  --env-file .env.local `
  --session-id demo `
  --trace compact `
  --trace-file demo/trace.jsonl
```

进入交互后可以尝试：

```text
读取 workspace 里的文件列表
看一下 hello.py 做了什么
在 workspace 新建 hitl-demo.txt，内容为 approved once
调用 ask_question 询问我部署到测试还是生产环境
```

常用 slash command：

- `/help`: 查看 CLI 支持的命令。
- `/trace off|compact|full`: 切换当前进程内的 trace 展示模式。
- `/exit` 或 `/quit`: 退出交互。

## 工具与写入权限

`agent.yaml` 当前暴露了 `file.read`、`file.list`、`file.grep`、`file.write` 和
`file.edit`，以及 `human.ask` 对应的 `ask_question`；这些工具的工作区限制在
`demo/workspace/` 下。`permissions.writes` 设置为 `confirm`，因此每个写调用都会先显示
一次 `PERMISSION [y/N]`：

- 输入 `y` 或 `yes` 只批准面板中展示的精确调用一次。
- 输入空值、`n` 或 `no` 会 reject，并向模型回灌 `USER_REJECTED`；目标文件不会改变。
- 其它输入不会调用 `resume()`，CLI 会继续要求输入有效选项。

`allow` 会直接执行写调用；`deny` 会直接拒绝，不能被 CLI 的 `y` 覆盖。

`ask_question` 会显示 `QUESTION` 面板。有 options 时可输入从 `1` 开始的编号，也可以输入
自由文本；空回答会重新询问。同一个 assistant response 若依次触发 permission/question，
CLI 会按 runtime 返回的 gate 顺序处理，最后只渲染一次终态答复。

### 显式跨进程 lifecycle

`iris chat` 只承载当前进程内的交互循环，不会在重启后按 session 自动猜测或恢复 run。
Ctrl+C/EOF 也不等于 reject、cancel 或 recover。需要跨进程操作 durable run 时，使用
`iris run start/status/events/resume/cancel/recover` 并显式传递 run、interaction 和 activation
identity。`events` 使用显式 run ID 和 exclusive `--after-sequence` cursor 做一次 provider-free
durable read；它不 watch、不自动发现 run，也不修改 lifecycle 状态。

完整的真实 DeepSeek + SQLite 三场景验收见
[`LIFECYCLE_SMOKE.md`](LIFECYCLE_SMOKE.md)。它使用独立的 `lifecycle-agent.yaml` 和
`.iris/lifecycle-smoke.db`，不会改变本页普通 chat demo 的工具面或数据库。

`context.yaml` 要求只有在用户明确提出时才执行写入，并继续禁止删除、提交或权限提升。
验证 approve/reject 时请给出明确的相对路径和内容，并检查 `demo/workspace/` 内的文件变化。

## 产物

- session 数据写入 `demo/.iris/demo-session.db`。
- 启动命令包含 `--trace-file demo/trace.jsonl` 时，每次 provider 调用会追加一行 JSONL。
- demo 过程中产生或修改的文件应限制在 `demo/workspace/`。

## 排查

- 如果启动时报缺少 API key，确认 `IRIS_PROVIDER_API_KEYS__DEEPSEEK` 是否已设置，或
  `--env-file` 是否指向正确文件。
- 如果模型没有使用工具，可以尝试用更明确的问题要求它读取或搜索 `workspace/`。
- 如果不想暴露工具，可加 `--no-tools` 启动，只验证纯对话链路。

## 维护与验证

这个目录是 CLI/runtime 的集成示例，不是独立 Python 包。修改时应同步核对
`agent.yaml`、`context.yaml`、lifecycle smoke 资产和 `src/iris/cli/main.py` 的参数。

```bash
uv run pytest tests/cli tests/context tests/agents
```
