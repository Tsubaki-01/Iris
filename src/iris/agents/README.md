[English](README.en.md)

# iris.agents

`iris.agents` 提供 config-first agent 的配置加载和工具注册入口。它负责把
`agent.yaml` 解析成强类型配置，并根据配置创建 `ToolRegistry`。本模块不直接调用模型；
模型调用、context 组装、工具桥接和 session 写入由 `iris.runtime` 消费这些配置后完成。

## 架构

`AgentConfig.mcp` 使用 `AgentMCPConfig` 文件引用与 `MCPServerOverride` 本地策略模型，
均从 `iris.agents` 导出。`mcp.path` 相对 agent YAML 解析；加载 YAML 不连接外部服务。
shared assembly 读取 MCP 声明，runner 首次执行前准备并发布工具。外部格式见
[MCP 包说明](../mcp/README.md)。例如：

```yaml
mcp:
  path: mcp.json
  overrides:
    local-server:
      required: true
      trust_annotations: false
```

`tools.subagent: subagents.yaml` 可声明内部 Sub Agent catalog，默认不启用。
路径相对 parent YAML；catalog 的 `default` 必须命中 `agents` 的 exact kebab-case key，
每个 entry 包含相对 catalog 的 `path` 和非空 `description`。内部 loader 只读取 catalog、
冻结路由，不加载 child YAML。`build_tool_registry()` 仍只处理 builtin/Python 工具。
Runner 启动只读一次 catalog，选择执行时才懒加载对应 child YAML；CHILD 不注册 subagent，
也不读取 nested catalog。完整三个 YAML 示例见 [harness README](../harness/README.md)。

```mermaid
flowchart LR
    YAML["agent.yaml"] --> Loader["load_agent_config"]
    Loader --> Config["AgentConfig"]
    Config --> Route["ModelRoute"]
    Config --> Tools["build_tool_registry"]
    Tools --> Registry["ToolRegistry"]
```

## 快速入门

```python
from iris.agents import build_tool_registry, load_agent_config

config = load_agent_config("agent.yaml")
route = config.to_model_route()
registry = build_tool_registry(config.tools)
```

`load_agent_config()` 只读取 YAML 并校验结构。运行时入口可以复用返回的
`AgentConfig`、`ModelRoute` 和 `ToolRegistry`，但 loop 本身仍不在本包内。

## 配置示例

```yaml
name: notes-agent
model:
  provider: openai
  name: gpt-4o-mini
  temperature: 0.2
  max_tokens: 512
system: |
  你是一个本地笔记助手。
skills:
  enabled: true
  root: .agents/skills
  require:
    - review-python
tools:
  builtin:
    - file.read
    - file.list
    - file.grep
    - human.ask
  python:
    functions:
      - my_project.tools:search_notes
    registrars:
      - my_project.tools:register_tools
permissions:
  workspace: .
  writes: confirm
session:
  backend: sqlite
```

结构化 context 模式使用独立 `context.yaml`：

```yaml
name: notes-agent
model: openai/gpt-4o-mini
context:
  path: context.yaml
tools:
  builtin:
    - file.read
```

`system` 和 `context` 必须二选一，不能同时配置。`load_agent_config()`
只校验 `context.path` 字段并把相对路径按 `agent.yaml` 所在目录解析为绝对路径；
它不会读取或校验该 context 文件。context 文件存在性、内容结构和模板路径由
`iris.runtime.RuntimeFactory` 调用 `iris.context.load_context_build_input()` 时校验。

`model` 推荐使用结构化对象。为了兼容简单配置，也可以写成 route string：

```yaml
model: openai/gpt-4o-mini
```

## 核心定义

### `AgentConfig`

顶层 agent 配置模型，字段包括：

- `name`: agent 名称，不能为空。
- `model`: `ModelConfig`，也兼容 `provider/model` route string。
- `system`: 简单模式的 system prompt，和 `context` 互斥。
- `context`: `AgentContextConfig`，声明独立 context 配置路径，和 `system` 互斥。
- `skills`: 可选的 `AgentSkillsConfig`；默认 `None`，不启用 Skill。
- `mcp`: 可选的 `AgentMCPConfig`；引用 JSON/JSONC/TOML 文件，默认 `None`。
- `compaction`: 默认构造的 `CompactionConfig`，声明自动压缩的预算、超时与摘要指令文件。
- `context_policy`: 默认构造的 `ContextPolicyConfig`，控制当前会话回读、宿主动态快照选材及历史工具正文减载。
- `memory`: 复用 `iris.memory.MemoryConfig`，默认 `enabled: false`，不接入长期记忆服务。
- `tools`: `ToolsConfig`，声明 builtin/Python 工具，默认声明为空；框架自动注册的工具由对应功能开关控制。
- `permissions`: `PermissionsConfig`，默认 `workspace: .`、`writes: confirm`。
- `session`: `SessionConfig`，默认 `backend: none`。

### `AgentContextConfig`

结构化 context 配置声明：

- `path`: 独立 `context.yaml` 文件路径。相对路径按 `agent.yaml` 所在目录解析。

`AgentContextConfig` 只保存路径声明，不读取 context 文件。

### `ContextPolicyConfig`

`context_policy` 默认配置：

```yaml
context_policy:
  enabled: true
  preserve_recent_tool_groups: 2
  old_result_preview_chars: 512
```

完整 `AgentRunner` 根据这个开关注册 `context_read` 与 `context_search`，读取当前会话的已提交
消息和当时保存的工具结果；不依赖长期 memory，也不需要显式配置 `file.read`。设为 `false`
时不注册这两个工具，也不进行新增的历史正文裁剪或动态注入；现有工具 artifact 和 LLM
compaction 继续工作。同时传入 `context_source` 会在装配时报 `IrisConfigError`，不会忽略宿主输入。

`preserve_recent_tool_groups` 是确定性裁剪时完整保留的最近已闭合工具批次数，默认 2，允许
设为 0；不改变原有 LLM 摘要的近期原文软目标。`old_result_preview_chars` 默认 512，按
384 字符开头与 128 字符结尾保留旧观察正文，说明与引用另外计量；设为 0 时只留说明和引用。
两项均须为非负整数。只有工具作者声明为 `observation` 的成功结果可以参与裁剪。

完整请求达到既有 80% 压力线时，runtime 先尝试精确重复正文折叠，再按优先级移除宿主明确
标为可选的动态贡献，最后按历史顺序短化旧结果。
正文替换仅采用能减少完整请求 token 估算的候选；不足时继续原有 LLM compaction。每次实际工具调用
照常执行，已提交原文保持不变。具体回读可用条件与保护规则见
[runtime 说明](../runtime/README.md#历史投影与摘要构造)。

配置加载不读取历史。Runner 提供绑定 lifecycle store 的读取服务；直接使用
`RuntimeFactory.from_config*()` 且启用该策略时，必须传入 `context_access`，否则装配报告
`IrisConfigError`。参数、分页和引用范围见 [tools 说明](../tools/README.md#当前会话上下文回读)。

动态状态由 Python SDK 的 `context_source=` 注入，`AgentRunner` 与 `RuntimeFactory` 的两个
`from_config*()` 入口均支持；不在 YAML 中配置 callback。source 每主模型步骤返回完整当前
快照，与输入阶段一次归档的 BCI 不同。required 条目保持，只有显式 optional 条目参与选材；
协议与示例见 [context 说明](../context/README.md#宿主动态快照)。

### `CompactionConfig`

`CompactionConfig` 从 `iris.agents` 和 `iris.agents.config` 导出，所有 agent 默认使用：

```yaml
compaction:
  input_budget_tokens: 96000
  keep_recent_ratio: 0.15
  summary_ratio: 0.05
  timeout_seconds: 300
```

`input_budget_tokens` 是已扣除输出预留的可用输入预算，不再自动减去 `model.max_tokens`。
触发和压缩后验收额度固定为预算的 80%（向下取整）；近期原文目标与摘要输出上限分别为
预算乘以对应比例（向上取整）。近期原文只是软目标，不要求两个比例之和小于 80%。
输入预算与超时必须为正数，两个比例必须位于 0 与 1 之间。

可用独立 Jinja2 文件自定义摘要指令和输出格式：

```yaml
compaction:
  prompt: ./prompts/summary.j2
```

`prompt` 相对 `agent.yaml` 所在目录解析；省略或设为 `null` 时使用包内的
[默认七栏 prompt](../prompts/compaction.j2)。文件内容作为摘要请求的 system 消息，
旧摘要与本批历史由框架作为 user 消息提供，无需在模板里插入数据变量。自定义文件可以
改变栏目和措辞；摘要正文仍由框架统一包裹 `<summary>` 后注入主请求。

模板通过共享的 [`iris.utils.TemplateRenderer`](../utils/README.md) 使用 Jinja2 原生按需加载
与编译缓存，支持动态 include/import/extends。默认不进行 XML 转义；XML 模板可显式使用
`{% autoescape true %}` 或 `|e`。Runtime 在消费摘要指令时去除首尾空白。
同一 runtime 在下次压缩操作检测文件修改；一次操作的全部分块与重试共用同一份已渲染指令。
配置加载只解析路径，首次压缩时读取文件。直接使用 Python SDK
时，相对路径以 `config_path` 所在目录为基准，未提供时以当前工作目录为基准。

配置仍由 `load_agent_config()` 或 `AgentConfig` 校验，随后通过
`RuntimeEnvironment.agent_config` 传递；加载时不调用模型或 tokenizer。每次主模型调用前，
runtime 先按 `context_policy` 尝试确定性正文减载；完整输入仍达到 80% 时才自动摘要旧历史，
包括当前 run 已提交的步骤，并保留当前任务
锚点与近期原文。摘要使用当前模型；`summary_ratio` 是生成输出上限，可能包含模型 reasoning。
压缩后的完整请求必须不超过 80% 且严格缩小。已开始的压缩失败会结束当前 run，原文与上次
已提交摘要保留。主调用 usage 与 `RunUsage.compaction` 分开累计；provider 估算边界见
[providers 说明](../providers/README.md)，执行与恢复见 [runtime](../runtime/README.md)。

### `MemoryConfig`

通过 `memory` 启用项目记忆；配置模型从 `iris.memory` 导入：

```yaml
memory:
  enabled: true
  read_namespaces: [project]
  write_namespace: project
tools:
  builtin:
    - memory.remember
    - memory.update
    - memory.forget
```

`memory.enabled` 同时接入服务、概览和自动注册的 Search/Fetch；写工具仍按需声明。旧
`memory.backend` 和手工 `memory.search/fetch` 声明不再接受。模型根据当前概览自主选择
Search/Fetch，查询使用全部词项；旧 recall_mode/max_query_terms/mirror 配置不再接受。
概览通过宿主显式生成，内容为核心事实和知识范围，新会话或成功压缩时采用，
全部 namespace 合计使用可用输入预算的 2%。概览未提及的主题默认没有，无概览时正常聊天但
暂不使用长期记忆。完整规则见 [memory 说明](../memory/README.md)。

YAML 加载不打开数据库。Runtime 确定 effective workspace 和 provider 后构建一个 service，
供概览及记忆工具使用。默认数据库为该 workspace 的 `.iris/memory/memory.db`；同项目默认
`project` namespace 可共享，不同 workspace 使用各自数据库。开启时显式传入的 `memory_service`
优先于配置构造；关闭时不挂载注入对象。CLI 使用同一装配链；child 根据自己的配置和收窄后的 effective workspace
构建服务和自己的概览窗口，不继承父 Agent 的 service。数据库初始化错误直接沿装配入口报告。

开关在构建 Agent 时确定，改配置后重建 Agent 并使用新会话，暂不支持热切换。静态
`context.yaml` memory 与已有历史不受关闭影响；`include_tools=False` 仍控制当次工具 schema。

### `ModelConfig`

模型路由配置：

- `provider`: provider 名称，例如 `openai`。
- `name`: 模型名称，例如 `gpt-4o-mini`。
- `base_url`: 可选自定义 endpoint。
- `temperature`、`top_p`、`max_tokens`、`tool_choice`、`response_format`、
  `timeout`、`provider_options`、`metadata`: 可选请求级参数，会由 runtime 透传给
  `LLMRequest`。
- Streaming 由 host 给 runner 注入 `live_publisher` 开启，模型配置不声明 `stream`。
- 调用契约固定为 LiteLLM Chat Completion，不提供 `api_style` 配置字段。

调用 `to_model_route()` 可转换为 providers 层使用的 `ModelRoute`。
调用 `to_llm_request_options()` 可得到 `LLMRequest` 支持的请求级参数；`provider`、
`name` 和 `base_url` 不会进入该结果。

### `ToolsConfig`

`tools.builtin` 使用 Iris 内置工具名：

- `file.read`
- `file.list`
- `file.grep`
- `file.write`
- `file.edit`
- `human.ask`
- `web.search`
- `web.fetch`
- `memory.remember`、`memory.update`、`memory.forget`

`human.ask` 向模型暴露的工具名是 `ask_question`。它只声明人工问题；实际呈现问题、
收集回答与调用 `AgentRuntime.resume()` 仍由 runtime 和宿主 adapter 完成。

`web.search` 和 `web.fetch` 分别暴露 `web_search`、`web_fetch`，使用 Tavily Search/Extract。
两者固定 basic，可独立启用，不增加专用 Web 配置块：

```yaml
tools:
  builtin:
    - web.search
    - web.fetch
    - file.read
```

注册 Web 工具前，host 需要通过现有 `init_config()` 初始化全局配置。凭据来自
`Config.provider_api_keys["tavily"]`，对应环境变量 `IRIS_PROVIDER_API_KEYS__TAVILY`；
也可在初始化时传入 `provider_api_keys`。需要 dotenv 时显式传 `env_file`。缺少 Tavily key
会抛出 `IrisConfigError`，不会改用聊天模型的通用 key。

两个具体 Web builtin 在默认策略下自动执行，自定义策略仍然生效。默认 context policy 提供
`context_read`，可按历史引用继续读取长网页结果；`file.read` 是需要显式启用的文件工具。
检索筛选、批量 URL、正文/摘录和错误
行为见 [Web 工具说明](../tools/README.md#web-搜索与网页读取)。

`tools.python` 必须使用结构化对象，不支持混合列表：

```yaml
tools:
  python:
    functions:
      - my_project.tools:search_notes
    registrars:
      - my_project.tools:register_tools
```

`functions` 是默认推荐路径。Iris 会导入 `module:function` 指向的 Python 函数，并用
`ToolRegistry.register_function()` 注册为工具。

`registrars` 是高级路径。Iris 会导入 registrar，并调用
`registrar(registry)`，由它批量注册多个工具。

YAML 中不支持 inline Python 脚本。Python 扩展必须通过可导入的模块引用提供。

### `AgentSkillsConfig`

项目级 Skill 发现配置：

- `enabled`: 严格布尔值，默认 `false`。关闭时 runtime 完全绕过 Skill discovery。
- `root`: 默认 `.agents/skills`，相对 `permissions.workspace` 解析，且不得越出 workspace。
- `require`: 默认空元组；每个名称必须是小写 kebab-case，启用后缺少任一必需 Skill 都会让
  `RuntimeFactory` 抛出 `IrisConfigError`。

Skill 目录约定和 `SKILL.md` 格式见 [`iris.skill`](../skill/README.md)。启用且发现结果非空时，
`RuntimeFactory` 会自动注册 `load_skill`，并让它与 context catalog 共用同一个 registry。
`load_skill` 不是 `tools.builtin` 配置项；不要在 `tools.builtin` 中声明它。

### `PermissionsConfig`

当前只承载配置形状：

- `workspace`: 文件工具工作区路径。
- `writes`: 写入策略，取值为 `confirm`、`allow` 或 `deny`。

具体执行权限仍由工具层的 permission policy 和 executor 决定。

### `SessionConfig`

`backend` 目前支持：

- `none`: 不启用持久化。
- `sqlite`: 使用 SQLite，本地默认路径为 `.iris/session.db`。

## API

### `load_agent_config(path)`

读取 UTF-8 YAML 文件并返回 `AgentConfig`。配置缺失、YAML 格式错误、字段类型错误、
未知字段、不可读路径都会包装为 `IrisConfigError`。

### `build_tool_registry(config, *, memory_service=None, memory_config=None)`

根据 `ToolsConfig` 构建 `ToolRegistry`：

1. 有有效 memory service 时先注册 Search/Fetch，再按 `tools.builtin` 注册其它内置工具。
2. 注册 `tools.python.functions` 中的直接函数引用。
3. 调用 `tools.python.registrars` 中的批量注册入口。

未知内置工具、错误引用格式、模块不存在、函数不存在、引用对象不可调用、registrar
签名不兼容都会抛出 `IrisConfigError`。
显式 memory 写工具要求提供 service；手工 Search/Fetch 声明在本装配边界报配置错误。
`memory_config` 绑定工具的 read/write namespaces，不传时使用 `MemoryConfig()` 默认范围；
此处只消费来源工厂已解析的 service，不重新判断 enabled。实际工具名称或别名冲突继续
由 `ToolRegistry` 报错。需要按 YAML 自动创建服务时使用完整 runner
或 RuntimeFactory；此函数不解析 workspace 或打开数据库。

## 边界

本模块只固定配置契约和工具注册方式。它不做以下事情：

- 不实现 agent loop。
- 不自动调用模型。
- 不提供长期记忆系统。
- 不引入 Redis、向量数据库或 ORM。

需要从 `agent.yaml` 创建完整 logical-run agent 时，使用 `iris.harness.AgentRunner`：

```python
from iris.harness import AgentRunner

runner = AgentRunner.from_config_path("agent.yaml")
```

## 维护与验证

| 修改内容 | 主要位置 | 对应测试 |
| --- | --- | --- |
| `agent.yaml` 加载与相对 context 路径 | `config/base.py`, `../runtime/factory.py` | `tests/runtime/test_factory.py` |
| 压缩预算与摘要指令配置 | `config/compaction.py`, `config/base.py` | `tests/agents/test_compaction_config.py`, `tests/runtime/test_compaction_prompt.py` |
| Skill 配置与 factory 集成 | `config/base.py`, `../runtime/factory.py` | `tests/agents/test_skill_config.py`, `tests/runtime/test_factory_skills.py` |
| Memory 配置与 ROOT/CHILD 装配 | `config/base.py`, `config/tools.py`, `../runtime/_assembly.py` | `tests/agents/test_memory_config.py`, `tests/runtime/test_memory_assembly.py` |
| 内置工具与 Python 引用加载 | `config/tools.py` | `tests/agents/test_tools_config.py` |

```bash
uv run pytest tests/agents tests/runtime/test_factory.py
uv run ruff check src/iris/agents tests/agents
```
