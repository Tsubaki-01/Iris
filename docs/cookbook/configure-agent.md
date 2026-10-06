# 配置自己的 Agent

Agent YAML 用于声明“这个 Agent 具备哪些能力”，宿主负责“这次让它做什么”。本页在[快速开始](../getting-started/quickstart.md)之后使用；完整字段查[配置参考](../reference/configuration.md)。

## 明确模型与协议

使用展开形式可以配置请求参数：

```yaml
name: project-assistant
model:
  provider: deepseek
  name: deepseek-flash
  api_style: responses
  temperature: 0.2
  max_tokens: 2048
  timeout: 60
system: 根据用户任务使用已配置能力，清楚区分观察和推测。
permissions:
  workspace: .
session:
  backend: sqlite
  path: .iris/session.db
```

这是完整 Agent 配置，可保存为 `agent.yaml` 后运行 `uv run iris chat agent.yaml`。`max_tokens` 限制一次模型回答，不等于整个 Run 的用量上限；`timeout` 是模型请求级设置，不是命令工具或完整任务的总时限。

协议在创建 provider 时确定。若你的服务使用 Chat Completions，把 `api_style` 明确改成 `chat_completions` 后重建 Runner。框架不会在某次请求失败后悄悄更换协议。

内置逻辑 provider 为 `openai`、`anthropic` 和 `deepseek`。这不代表每一家、每个模型都支持两种协议的所有能力；应选择服务实际支持的组合。自定义网关在进程配置的 `providers` 中注册，见[配置参考](../reference/configuration.md#自定义-provider)。

## 把系统要求拆成结构化 context

少量要求放在 `system` 即可。需要分别维护身份、规则、固定资料和当前任务背景时，改用 `context.path`：

```yaml
name: project-assistant
model: deepseek/deepseek-flash
context:
  path: context.yaml
```

同目录创建 `context.yaml`：

```yaml
system:
  slots:
    - name: identity
      order: 10
      content: 你是项目阅读助手。
    - name: rules
      order: 20
      content:
        - 先读取实际文件，再描述实现。
        - 将事实、推断和建议分别表达。
before_current_input:
  slots:
    - name: task_background
      content: 当前使用者会 Python，但可能不熟悉此项目。
```

`system` 和 `context` 必须二选一，不能同时声明。这里的 slot 是结构化内容，不是自动执行的插件。它们怎样排序、如何进入消息以及与每步动态快照的区别，见[配置上下文](context.md)。

## 按需求启用能力

功能开关不会自动代替所有接线。例如配置命令环境不会自动注册命令工具；SDK 开启自动记忆生成后，还需要绑定宿主维护协调器。

| 需要的能力 | 配置或入口 | 后续步骤 |
| --- | --- | --- |
| Python / 内置工具 | `tools.builtin`、`tools.python` | [工具配方](tools.md) |
| 外部工具服务 | `mcp.path` | [MCP 与检索](mcp.md) |
| 本地命令或 Docker | `command`，显式注册 `exec.command` | [命令执行](commands.md) |
| 项目 Skill / 子 Agent | `skills` / `tools.subagent` | [Skill 与委派](skills-subagents.md) |
| 长期记忆和自动生成 | `memory` | [记忆维护](memory.md) |
| 项目经验修订 | `evolution`，同时启用 Skill | [项目经验](evolution.md) |
| 持续目标与会话清单 | `goal` / `todo` | [Goal 与 Todo](goals-todos.md) |
| 观测 / 语音输入 | `observability` / `speech` | [观测](observability.md) / [媒体](media.md) |

配置在构造 Agent 时采用。改变工具、路径、协议或可选服务后，关闭旧实例并重建；已有 Session 的持久状态不等于一份可任意热替换的配置快照。

## 使用内置模板起步

模板系统提供 Python scaffold 入口，目前内置模板为 `file-agent`。从仓库根目录执行：

```powershell
uv run python -c "from iris.templates import scaffold_template; print(scaffold_template('file-agent', 'my-agent'))"
```

生成内容与目录行为以[模板参考](../reference/configuration.md#模板-scaffold)为准。CLI 当前只有 `chat`，没有 `iris init` 或 scaffold 子命令。

实现入口：[Agent 配置](../../src/iris/agents/config/base.py)、[模型工厂](../../src/iris/providers/factory.py)、[Context 配置解析](../../src/iris/context/config.py)。
