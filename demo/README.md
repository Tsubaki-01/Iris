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
```

常用 slash command：

- `/help`: 查看 CLI 支持的命令。
- `/trace off|compact|full`: 切换当前进程内的 trace 展示模式。
- `/exit` 或 `/quit`: 退出交互。

## 工具与写入权限

`agent.yaml` 当前暴露了 `file.read`、`file.list`、`file.grep`、`file.write` 和
`file.edit`，并将 `permissions.writes` 设置为 `allow`。这些工具的工作区限制在
`demo/workspace/` 下。

`context.yaml` 同时要求 demo agent 不主动执行写入、删除、提交或权限提升操作。因此这个
demo 更适合验证读取、搜索、trace 和 session 行为；如需验证写入工具，建议先明确给出写入
目标，并检查 `demo/workspace/` 内的文件变化。

## 产物

- session 数据写入 `demo/.iris/demo-session.db`。
- 启动命令包含 `--trace-file demo/trace.jsonl` 时，每次 provider 调用会追加一行 JSONL。
- demo 过程中产生或修改的文件应限制在 `demo/workspace/`。

## 排查

- 如果启动时报缺少 API key，确认 `IRIS_PROVIDER_API_KEYS__DEEPSEEK` 是否已设置，或
  `--env-file` 是否指向正确文件。
- 如果模型没有使用工具，可以尝试用更明确的问题要求它读取或搜索 `workspace/`。
- 如果不想暴露工具，可加 `--no-tools` 启动，只验证纯对话链路。
