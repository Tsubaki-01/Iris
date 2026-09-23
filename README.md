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

## 长期记忆

使用以下配置启用长期记忆，默认关闭：

```yaml
memory:
  enabled: true
```

开启后自动接入概览和 Search/Fetch 两个读取工具，以 SQLite 保存权威条目，Markdown 提供
人工可读投影。概览包含“核心事实＋可查询知识”，可由宿主显式生成或后台维护发布；
新会话首次输入和成功压缩时采用，全部读取空间
共用可用输入预算的 2%，会话窗口内保持稳定。没有概览时仍可正常聊天，但不使用长期记忆。

模型根据概览决定是否调用 `memory_search` / `memory_fetch`：Search 返回匹配
片段，信息充分即可回答；需要完整当前记录时再 Fetch。普通对话轮次不自动检索，工具结果
进入正常会话历史，静态 `context.yaml` memory 槽位保持独立。

写工具仍需显式声明。开关在构建 Agent 时确定，改配置后重建 Agent 并使用新会话，暂不支持热切换。

需要自动生成时，额外开启 `memory.generation.enabled`：

```yaml
memory:
  enabled: true
  generation:
    enabled: true
    idle_seconds: 30
```

root run 的已提交经历先保存为 Episode，flush 提炼为带证据的 Observation，dreaming 再统一
为正式记忆并发布概览。生成在前台空闲时运行，新输入优先；中间观察不参与检索，关闭后保留
待处理材料。自动生成默认关闭，维护调用的模型用量与主任务分开记录。

本次模型重构使用 memory schema v5 与 lifecycle schema v9；旧数据库会明确拒绝，
不自动迁移或删除已有数据。

配置、显式概览生成和 SDK 用法见 [`iris.memory`](src/iris/memory/README.md)。
