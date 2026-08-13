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

Provider 与 lifecycle 示例见 [`examples/README.md`](examples/README.md)。
