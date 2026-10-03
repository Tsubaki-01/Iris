<img src="./imgs/logo/logo.png" alt="project logo">

# Iris

Iris 是面向 Python 开发者的本地优先、配置优先 Agent Kit，并以 Python SDK 提供底层能力。

## 安装

```powershell
uv sync
```

## 快速开始

配置 provider API key 后，从仓库根目录启动示例 Agent：

默认模型调用使用原生 Responses，示例使用 `deepseek/deepseek-flash`。
在 Agent YAML 中设置 `model.api_style: chat_completions` 可显式选择 Chat Completions；默认值为
`responses`。逻辑 DeepSeek 凭据保持不变，协议由 provider 构造时确定，请求失败不会切换协议。

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

## 图片输入

Python SDK 支持静态 PNG、JPEG 和 WebP：先用 `runner.import_image(..., session_id=...)`
保存图片，再以 `AgentRunRequest(input=[TextBlock(...), image], session_id=...)` 提交；纯图片
使用 `input=[image]`。主模型须支持所选协议的视觉输入与工具调用，默认 Responses，也可显式
选择 Chat Completions。可运行示例见 [`examples/image`](examples/image/README.md)。

图片的 `original` 与 `model` 引用随会话持久化，普通模型调用读取 `model` 版。
run 或 runner 结束保留缓存，fork 继续依赖源 session 图片目录；备份需同时带上数据库与
`image-cache`。文字摘要只保存已表达的视觉结论和图片引用，需要细节时通过 `file.read`
对应的 `read_file` 回读。图片预算是近似值，实际 usage 以服务端返回为准。
`iris chat` 当前没有发图或图片渲染界面。

## 本地命令与可选 Docker 沙箱

普通文件、Python 扩展和 MCP 工具在宿主执行。显式声明的命令工具和配置式命令 Hook 使用所选
命令环境；`exec.command` 对应的模型调用名为 `exec_command`。Native 是默认模式，无需 Docker：

```yaml
name: local-command-agent
model: deepseek/deepseek-flash
system: 根据任务使用文件和命令工具，先检查结果再继续。
tools:
  builtin: [file.read, file.write, exec.command]
permissions:
  workspace: .
  writes: confirm
  execute: confirm
command:
  mode: native
  timeout_seconds: 120
```

`execute` 默认需要确认，独立于原生文件写权限。Native 用宿主用户权限运行：Windows 为
`cmd.exe`，Linux 为 `/bin/sh`，它不是 OS 隔离沙箱。`cwd` 只决定起始目录；原生只读
workspace 不允许注册命令工具。移动端可保留普通工具而不提供代码执行。

需要本地隔离时，先安装并启动 Linux Docker Engine 或 Docker Desktop 的 Linux engine，
显式安装可选依赖，并在仓库根目录用 [Dockerfile](Dockerfile) 构建命令镜像：

```powershell
uv sync --extra sandbox
docker build --load -t iris-command:local .
```

镜像以 `python:3.12-slim` 为最小基底，可按项目需要预装依赖；Iris 应用仍在宿主运行。
下面是完整的 Docker Agent 配置；运行时不会自行构建或拉取镜像，也不会在 Docker 不可用时回退宿主：

```yaml
name: docker-command-agent
model: deepseek/deepseek-flash
system: 命令运行在本地Linux容器，按工具说明使用共享项目文件。
tools:
  builtin: [file.read, file.write, exec.command]
permissions:
  workspace: .
  writes: allow
  execute: confirm
command:
  mode: docker
  timeout_seconds: 120
  docker:
    image: iris-command:local
    network: none
    cpus: 2
    memory_mb: 1024
    pids_limit: 128
```

一个 root runner 拥有一个容器，其 session 和 child 共用 `/workspace`、依赖、后台服务与
额度；不同 root 实例各自独立。只挂载 root 项目目录，child workspace 仅指定默认 cwd，
并不隔离 root 里的其他文件。Docker child 的 `writes: deny` 仍可开放命令：它只限制原生
文件工具，命令能否写入取决于 root 的实际挂载；不能据此称 child 整体只读。

每次命令启动新 shell，`cd`/`export` 不跨调用保留。单命令超时只停止该命令；run 取消、
总期限、预算或模型失败会停止共享容器，其他 session 的模型推理和 HITL 继续。停止保留文件
和已安装依赖，下次命令重启容器，后台服务须重新启动；`runner.aclose()` 删除其拥有的容器，
保留宿主项目文件。未知结果不自动重放，停止也不回滚已写文件。Native 不追踪历次后台后代。

清理无法确认时 run 保留原状态与占用，随后调用可重试清理；错误会通知宿主。
模式、镜像、挂载及额度在 root 构建时固定，调整后关闭并重建 runner，不承诺运行中切换的行为。
Iris 实现的是工具权限、命令生命周期和 run 结算协调；OS 隔离由 Docker 提供。

不需要模型 API key 的可重复示例见 [`examples/command`](examples/command/README.md)，
命令执行契约见 [`iris.command`](src/iris/command/README.md)，镜像与容器配置见
[`iris.sandbox`](src/iris/sandbox/README.md)。

## Hooks 与工具 Middleware

Hooks 在明确的事件时点执行附加逻辑，可使用 Python 处理器或一次性 JSON 命令脚本：

| 事件 | 时点与能力 |
| --- | --- |
| `run.started` | 新 Run 的初始受控任务内、首个模型步骤前；附加动作 |
| `run.finished` | 新终态提交后；Python 覆盖各停止原因，命令脚本仅在 `COMPLETED` 后运行 |
| `tool.before` | 权限和 claim 后；允许拒绝本次工具调用 |
| `tool.after` | 实际工具 body 结果已知后；允许追加模型反馈 |

工具 Middleware 使用 `wrap_tool_call(call, call_next)` 包裹单次工具调用，可短路或返回新结果；
`call_next()` 最多执行一次。它不提供 Run/model 包装、重试或参数改写。Hook 不修改工具参数、
不替换结果，也不决定 Run 续跑。父侧 subagent 委派不进入工具扩展链，child 独立配置自己的扩展。

YAML 通过 `hooks` 和 `middleware.tools` 声明；Runner 与 RuntimeFactory 的配置构造入口
支持 `hooks`、`tool_middlewares`，SDK 项追加在 YAML 项之后。直接运行低层 Runtime 只派发
工具事件；Run 事件由 harness 拥有。结束处理可能延长 SDK 返回时间，但不改写已提交结果；
同 session 的后续输入由 SessionManager 等待完成。Hook 尽力执行，不为旧结果补发或重放。

配置、脚本协议和限制见 [Hooks](src/iris/hooks/README.md)，包装接口见
[工具系统](src/iris/tools/README.md#middleware)。

## Context Engineering

Compaction、Offload 与回读、Pruning 与 Trim、动态上下文、统一上下文预算和工具按需披露，
见 [`Context Engineering 实现机制`](docs/context-engineering.md)。

## 评测接入

仓库提供 [Inspect AI 接入接口](evals/README.md)，通过独立的 `eval` 依赖组使用。
接口调用现有 `AgentRunner`，支持任务回调、运行结果投影和资源收尾；当前未接入实际
题集、benchmark 评分或真实模型跑分。

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

## 可选 Decision

[`iris.decision`](src/iris/decision/README.md) 提供独立 Choice/Boolean/Score SDK，首个后端为 Jev。
在 Agent YAML 通过 `decision.path` 引用独立配置，分别启用 `tools.discovery` 和 `memory.recall`。
前者一次批量 Choice 发现 deferred 工具；后者从满足显式必要词组的全部允许记忆中一次 Score
直接召回。两者默认关闭，eager 工具、Skill 和子 Agent 仍直接调用。配置方式与 SDK 用法见
[`iris.decision` 使用说明](src/iris/decision/README.md)。
