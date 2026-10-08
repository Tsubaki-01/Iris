# 运行第一个 Iris Agent

这一页从一次简单对话开始，再让 Agent 读取本地项目文件。你需要会运行 Python 命令，不需要先理解运行时、检查点或工具执行器。

## 准备环境

需要 Python 3.12 或更新版本、uv，以及一个可用的模型服务凭据。下面使用仓库已有示例采用的 `deepseek/deepseek-flash`。模型名称、可用协议与能力应以你的服务账号为准；换服务时参阅[模型配置](../reference/configuration.md#模型与协议)。

本文按源码安装，不假定公共包仓库中的同名包就是本项目。在终端执行：

<!-- iris-web:install:start -->
```powershell
git clone https://github.com/Tsubaki-01/Iris.git
cd Iris
uv sync
```
<!-- iris-web:install:end -->

之后所有命令都在这个仓库根目录执行。`uv sync` 创建项目环境并安装锁定的依赖；使用 `uv run` 执行该环境里的 Iris。

配置凭据。PowerShell：

```powershell
$env:IRIS_PROVIDER_API_KEYS__DEEPSEEK = "替换为你的 API key"
```

Bash / Zsh：

```bash
export IRIS_PROVIDER_API_KEYS__DEEPSEEK="替换为你的 API key"
```

也可以把同名变量写入 `.env.local`，随后给命令加上 `--env-file .env.local`。Iris 默认不会自动加载 `.env`。凭据属于进程配置，无需放入 Agent YAML。

## 创建最小配置

在仓库根目录新建 `agent.yaml`：

<!-- iris-web:minimal-agent:start -->
```yaml
name: first-agent
model: deepseek/deepseek-flash
system: 你是一个中文助手。清楚说明依据，不编造项目文件内容。
```
<!-- iris-web:minimal-agent:end -->

三个字段分别声明 Agent 名称、模型路由和系统要求。`model` 的短写等价于分别指定 `provider` 与 `name`；默认协议是 `responses`。需要 Chat Completions 的服务应显式设置 `model.api_style: chat_completions`，见[配置自己的 Agent](../cookbook/configure-agent.md)。

启动：

<!-- iris-web:run:start -->
```powershell
uv run iris chat agent.yaml
```
<!-- iris-web:run:end -->

输入：

```text
请用三句话解释 Python 的上下文管理器。
```

成功标志是终端显示模型回答，随后还能继续输入。回答正文不是固定测试输出，不要求与某段示例逐字一致。输入 `/exit` 退出，`/help` 查看交互命令。

此时你已经运行了一个由 YAML 装配的 Agent。CLI 负责读取输入、提交运行、展示输出和关闭资源；模型调用与后续工具循环由 Iris 完成。

## 让它读取项目

将 `agent.yaml` 改为以下完整内容：

```yaml
name: project-reader
model: deepseek/deepseek-flash
system: |
  你是项目阅读助手。需要了解文件内容时先使用工具读取。
  回答时给出你依据的文件路径，区分已读到的事实和推测。
tools:
  builtin:
    - file.list
    - file.read
    - file.grep
permissions:
  workspace: .
  writes: deny
```

退出旧进程，再运行同一条启动命令。输入：

```text
读取 pyproject.toml，告诉我这个项目要求的 Python 版本，以及它注册了什么命令行入口。
```

检查答案是否指出 Python `>=3.12` 和 `iris = "iris.cli:main"`。这次模型可以主动选择文件工具，Iris 执行工具并把结果放回对话，再由模型回答。工具是否被调用以及调用次数由模型决定，最终答案仍需对照文件。

`workspace: .` 相对 `agent.yaml` 所在目录解析。本例只注册三个读取工具，没有注册命令执行和文件写入工具；它适合先体验基于文件事实的问答。

## 保存多轮会话

默认使用进程内存储，退出进程后不保留这些会话记录。要在重启后继续读取历史，在上面的配置中增加：

```yaml
session:
  backend: sqlite
  path: .iris/session.db
```

然后指定会话 ID：

```powershell
uv run iris chat agent.yaml --session-id project-reading
```

退出后使用同一配置和 ID，可以继续这个会话。SQLite 路径相对 Agent YAML；不要只复用 ID 却换到另一份数据库。中断中的运行如何恢复、历史如何分支，见[管理会话](../cookbook/sessions.md)和[人工交互与恢复](../cookbook/hitl-recovery.md)。

Runner 构造时还会在 workspace 初始化 `.iris/prompts` 中的项目模板；只有配置 SQLite 时才将 lifecycle 状态写入数据库。若使用图片或其他会话材料，备份时还应保留对应文件，不能把数据库视为所有资源的容器。

## 遇到问题时

| 表现 | 先检查什么 |
| --- | --- |
| 提示缺少 provider API key | 环境变量是否设置在运行命令的同一终端；使用 dotenv 时是否传入 `--env-file` |
| 服务拒绝模型或协议 | 核对服务支持的模型与 `model.api_style`；Iris 不会在失败后自动切换协议 |
| 无法读取预期文件 | 检查 workspace 的解析基准以及实际文件路径 |
| 配置报错 | `system` 与 `context` 必须恰好选一个；字段名以[配置参考](../reference/configuration.md)为准 |

接下来阅读[必要概念](concepts.md)，或者直接把这个 Agent [接入 Python 应用](python-sdk.md)。想了解一次请求内部如何流转，可转到[架构总览](../design/architecture.md)。
