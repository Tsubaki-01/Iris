# 配置参考

Iris 有两层配置：进程配置提供服务凭据、provider 注册和统一导出设置；Agent YAML 声明一个 Agent 的模型、上下文、工具和可选能力。首次使用见[快速开始](../getting-started/quickstart.md)，常见组合见[配置配方](../cookbook/configure-agent.md)。

## 加载入口与路径

`iris.agents.load_agent_config(path: str | Path) -> AgentConfig` 读取 YAML 并校验；`parse_agent_config(raw_config: dict[str, Any], *, config_path: Path) -> AgentConfig` 解析已有对象。解析配置不等于创建 Runner，也不会执行工具。

`AgentRunner.from_config_path(path)` 同时完成配置加载和运行组件装配。直接调用 `from_config(config, config_path=...)` 时，`config_path` 为相对路径提供基准；不提供时，装配阶段以当前工作目录为基准。

| 路径 | 相对基准 |
| --- | --- |
| `permissions.workspace`、`session.path` | Agent YAML 所在目录 |
| `context.path`、`mcp.path`、`decision.path`、`tools.subagent` | 声明它的 Agent YAML 所在目录 |
| Context section 的 `template` | `context.yaml` 所在目录 |
| 子 Agent catalog 内的配置路径 | catalog 所在目录 |
| `skills.root`、`prompts.root`、记忆数据路径 | workspace；各域额外规则见对应参考 |
| 交给文件工具的相对路径 | 当前 Agent 的有效 workspace |

路径可以使用绝对路径。workspace 与数据库路径是两回事，例如 `permissions.workspace: workspace` 不会使 `session.path: .iris/session.db` 自动变成 `workspace/.iris/session.db`。

## Agent 顶层字段

配置对象不接受未知字段，`system` 与 `context` 必须恰好提供一个。下表列全顶层声明；大型领域的精确子字段只在其对应参考维护。

| 字段 | 默认值 / 必填 | 含义与详细规则 |
| --- | --- | --- |
| `name` | 必填非空字符串 | Agent 名称 |
| `model` | 必填 | `provider/model` 字符串或下节的模型对象 |
| `system` | `null` | 简单系统要求，与 `context` 互斥 |
| `context` | `null` | `{path: context.yaml}`；[Context 格式](context.md) |
| `tools` | 内置列表和 Python 引用为空，subagent 为 `null` | [工具参考](tools.md) |
| `permissions` | workspace `.`，writes/execute 均 `confirm` | 本页下节 |
| `session` | backend `none`，path `null` | 本页下节 |
| `compaction` | 自动上下文压缩的默认预算 | [Context 参考](context.md#压缩预算) |
| `context_policy` | enabled `true`，deferred_tools `false` | [Context 参考](context.md#上下文回读与选材) |
| `skills` | `null` | 未启用；声明对象后 enabled 默认仍为 `false`；[工具参考](tools.md) |
| `mcp` | `null` | 外部配置文件及本地覆盖项；[工具参考](tools.md) |
| `decision` | `null` | `{path: decision.yaml}`；[工具参考](tools.md)及[记忆参考](memory-goals.md) |
| `hooks` | `[]` | 四事件处理器，按声明顺序执行；[扩展配方](../cookbook/extensions.md) |
| `middleware` | tools `[]` | 工具包装链；[工具参考](tools.md) |
| `command` | Native 环境 | 不会自动注册命令工具；[工具参考](tools.md) |
| `memory` | enabled `false` | [记忆配置与 SDK](memory-goals.md) |
| `prompts` | root `.iris/prompts` | [命名模板](context.md#项目命名模板) |
| `maintenance` | idle_seconds `300` | 宿主共享维护的空闲等待，秒，允许 `0`；领域生成预算另设 |
| `evolution` | enabled `false` | 要求同时启用 Skill；[长期能力参考](memory-goals.md) |
| `goal` | enabled `false` | 要求 context policy 开启；[长期能力参考](memory-goals.md) |
| `todo` | enabled `false` | 要求 context policy 开启；[长期能力参考](memory-goals.md) |
| `observability` | enabled `false` | [采集与导出参考](streaming-observability.md) |
| `speech` | enabled `false` | 由宿主装配 ASR，不自动提供录音 UI；[媒体参考](media.md) |

配置一般在 Runner 构建时确定。命名模板有自己的操作快照语义；这不等于 Agent 所有配置支持热更新。

## 模型与协议

| 字段 | 默认 / 类型 | 含义 |
| --- | --- | --- |
| `provider` | 必填 `str` | 凭据和注册配置使用的逻辑服务名 |
| `name` | 必填 `str` | 服务内的模型名称 |
| `api_style` | `responses` | `responses` 或 `chat_completions`，在构造时选定 |
| `base_url` | `null` | 覆盖 provider 的 endpoint |
| `temperature`、`top_p` | `null` | 交由服务解释的采样参数 |
| `max_tokens` | `null` | 单次模型输出的 token 上限 |
| `timeout` | `null` | 单次模型请求时限，秒 |
| `tool_choice` | `null` | `auto`、`none`、`required`，或 `{name: 工具调用名}` |
| `response_format` | `null` | `text`、`json_object`，或 `{name, schema, strict?}` |
| `provider_options` | `{}` | provider 专属选项；当前客户端透传 `num_retries`，不得借此设置 `api_style` |
| `metadata` | `{}` | Iris 请求元数据，不应假定每个键都发送给服务商 |

短写 `model: deepseek/deepseek-flash` 使用同样的默认值，不会自动探测协议。完整请求和响应结构见[模型与媒体参考](media.md)。模型是否支持工具、视觉和某种输出格式由具体服务决定。

## Workspace、权限与会话

| 字段 | 默认 | 规则 |
| --- | --- | --- |
| `permissions.workspace` | `.` | 配置及文件操作的工作区位置 |
| `permissions.writes` | `confirm` | `confirm / allow / deny`，普通写入策略 |
| `permissions.execute` | `confirm` | 同样三种值，命令执行的独立策略 |
| `session.backend` | `none` | `none` 选择进程内 LifecycleStore；`sqlite` 选择本地 SQLiteStore |
| `session.path` | `null` | SQLite 省略时规范化为 `.iris/session.db`；显式注入 `store` 优先于配置装配 |

写入确认与命令确认不是同一项。Native 下只读 workspace 不能注册命令工具；Docker 的挂载和 root 共享语义见[命令指南](../cookbook/commands.md)。等待确认后的继续由[HITL SDK](runtime.md)处理。

## 进程配置

公开入口位于 `iris`：

```text
init_config(*, env_file: str | None = None, **kwargs) -> Config
get_config() -> Config
reset() -> None
```

`init_config` 在进程中只能初始化一次，重复调用抛出 `IrisConfigError`；`get_config` 在未初始化时也会报错。`reset` 用于测试或重新引导，不是正在运行的 Agent 的配置热更新接口。

配置优先级为：显式参数 > 环境变量 > 显式选择的 dotenv 文件 > 默认值。

| 字段 | 默认 | 常见环境变量 |
| --- | --- | --- |
| `api_key` | `null` | `IRIS_API_KEY`，全局兜底凭据 |
| `provider_api_keys` | `{}` | `IRIS_PROVIDER_API_KEYS__DEEPSEEK` 等，按逻辑 provider 查找；Tavily 也使用此字典 |
| `providers` | `{}` | `IRIS_PROVIDERS__名称__BASE_URL` 等，自定义非 secret 配置 |
| `observability` | 默认导出配置 | `IRIS_OBSERVABILITY__TRACES_ENDPOINT` 等；[详细字段](streaming-observability.md) |

环境变量前缀为 `IRIS_`，嵌套分隔符为 `__`，不区分大小写。API key 查找顺序是 provider factory 的显式 `api_key`、逻辑 provider 专属 key、全局 `api_key`。服务配置自身不应写入 secret。

### 自定义 provider

`ProviderConfig` 有 `litellm_provider="openai"`、`base_url=None`、`headers={}` 三个字段。自定义逻辑 provider 需要非空 `base_url` 才进入注册表；内置 provider 可以只覆盖其明确声明的字段。

以下是程序启动时的一种配置方式；`gateway` 是自定义逻辑名称，endpoint 与模型名需换成实际服务配置：

```python
from iris import init_config

init_config(
    providers={
        "gateway": {
            "litellm_provider": "openai",
            "base_url": "https://your-gateway.example/v1",
        }
    }
)
```

凭据另以 `IRIS_PROVIDER_API_KEYS__GATEWAY` 提供。Agent YAML 的 `model.provider` 设置为 `gateway`；`model.api_style` 仍单独选择，`litellm_provider` 与协议不是同一个概念。

## 模板 scaffold

`iris.templates.scaffold_template(template_name, target_dir, *, overwrite=False) -> list[Path]` 复制内置模板并返回实际写入文件。当前模板名为 `file-agent`。默认遇到待写文件冲突时拒绝，`overwrite=True` 才允许覆盖。

它是 Python SDK，不是 CLI 子命令。可以先生成到新目录，再编辑 YAML、设置凭据和启动 chat。

## 可选安装项

从仓库运行时使用：

| 命令 | 用途 |
| --- | --- |
| `uv sync` | 核心运行与默认开发依赖 |
| `uv sync --extra sandbox` | Docker 命令环境的可选依赖 |
| `uv sync --extra observability` | OTel SDK 与 OTLP HTTP 导出 |
| `uv sync --group eval` | 仓库内 Inspect AI 评测接入辅助代码 |

多个 extra 可以同时传入。普通文件工具和 Native 命令不需要 Docker；安装 Docker extra 也不会替你启动引擎或构建镜像。

依据：[Agent 模型](../../src/iris/agents/config/base.py)、[进程配置](../../src/iris/config.py)、[provider factory](../../src/iris/providers/factory.py)、[scaffold](../../src/iris/templates/scaffold.py)、[依赖声明](../../pyproject.toml)。
