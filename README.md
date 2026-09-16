<img src="./imgs/logo/logo.png" alt="project logo">

# Iris

Iris 是面向 Python 开发者的本地优先、配置优先 Agent Kit，并以 Python SDK 提供底层能力。

## 安装

```powershell
uv sync
```

## 快速开始

配置 provider API key 后，从仓库根目录启动示例 Agent：

```powershell
uv run iris chat examples/chat/agent.yaml --session-id example
```

Agent 默认自动压缩长上下文：可用输入预算为 96,000 tokens（已扣除输出预留），完整输入
达到 80% 时摘要旧历史，并保留当前任务锚点与近期原文；当前 run 已提交的工具步骤也可压缩。
原文完整保留，摘要随 session 持久化，恢复沿用已提交的摘要。主调用和摘要 token 用量分开累计。
配置与预算含义见 [`CompactionConfig`](src/iris/agents/README.md#compactionconfig)。

`iris chat` 会显示“正在压缩上下文”“上下文压缩完成”或“上下文压缩未完成”，不输出摘要正文。
一旦开始压缩且失败，当前 run 结束；原始会话和上次成功提交的摘要可继续使用。

Provider 与 lifecycle 示例见 [`examples/README.md`](examples/README.md)。
