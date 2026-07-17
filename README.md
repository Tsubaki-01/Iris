<img src="./imgs/logo/logo.png" alt="project logo">

# Iris

Iris 是面向 Python 开发者的本地优先、config-first Agent Kit。它以 YAML 组装 provider、
context、tools、permissions 和 session，并提供 Python runtime 与 `iris chat` 终端入口。

## 快速开始

```powershell
uv sync
$env:IRIS_PROVIDER_API_KEYS__DEEPSEEK = "sk-xxx"
uv run iris chat demo/agent.yaml --session-id demo --trace compact
```

也可以把 key 放入 `.env.local`，然后增加 `--env-file .env.local`。完整 demo、trace 和
PowerShell/Bash 示例见 [`demo/README.md`](demo/README.md)。

## CLI HITL

`iris chat` 是 runtime 的 terminal host adapter：

- `permissions.writes: confirm` 会在精确写调用前显示 `[y/N]`；空输入拒绝，`y` 只批准该次调用。
- `human.ask` 支持编号选项和自由文本；同一 run 的多个 gate 会按顺序处理。
- Ctrl+C/EOF 不代表 reject 或 cancel，interaction 仍保持 pending。
- `session.backend: sqlite` 支持相同 session 的进程重启恢复；内存 backend 不承诺跨进程。
- 已领取但执行结果未知的工具调用会 fail closed，不会自动重放。

`writes: allow` 会直接执行，`writes: deny` 不能被 CLI 覆盖。HITL 的权威状态机与 checkpoint
位于 `iris.hitl`/`iris.runtime`；CLI 只负责呈现和提交 typed response。
