# 工具与扩展参考

本页维护工具开发、发现、MCP、命令、Skill、子 Agent 和执行扩展的精确规则。首次接入请先读[工具 Cookbook](../cookbook/tools.md)；Agent 顶层配置与模型配置见[配置参考](configuration.md)，完整运行、HITL 和资源关闭见[运行 SDK](runtime.md)。

## 工具定义与函数注册

主要入口从 `iris.tools` 导入：`tool`、`CallableTool`、`BaseTool`、`ToolDefinition`、`ToolRegistry`、`ToolExecutor`、`ToolExecutionContext`、`ToolResult`、`ToolArtifact`、`ToolErrorInfo`。

`@tool(...)` 给原函数附加元数据，返回的仍是原函数。提供 `registry` 时立即注册；否则由 YAML 或 `registry.register_function` 后续注册。

`CallableTool(func, *, ...)` 与 `ToolRegistry.register_function(func, *, ...)` 支持下表参数；后者返回已经注册的 `BaseTool`。装饰器支持相同元数据参数，但用 `registry=None` 代替 `input_model`，即装饰器本身不接受 `input_model`。

| 参数 | 类型与默认值 | 规则 |
| --- | --- | --- |
| `name` | `str \| None = None` | 未指定时读取装饰器名称，再用函数名 |
| `description` | `str \| None = None` | 未指定时读取装饰器描述、函数 docstring，最后用工具名 |
| `input_model` | `type[BaseModel] \| None = None` | 默认按函数注解生成；显式模型作为输入解析与 schema 来源 |
| `preset_kwargs` | `dict[str, Any] \| None = None` | 宿主绑定参数，从可见 schema 移除；模型不得覆盖 |
| `capabilities` | `set[ToolCapability] \| None = None` | 默认无能力标签；按实际副作用声明 |
| `group` | `str = "core"` | 目录过滤与发现分类 |
| `deferred` | `bool = False` | 标记延迟披露；不等于延迟导入函数或延迟连接资源 |
| `examples` / `tags` | `list[dict] \| None` / `list[str] \| None` | 默认空；示例与检索元数据 |
| `version` | `str \| None = None` | 版本元数据 |
| `deprecated` / `deprecation_message` | `False` / `None` | 将弃用提示附入描述，不提供旧接口兼容 |
| `execution_mode` | `CallableExecutionMode \| None = None` | 默认 `INLINE`；`THREAD` 使用线程执行同步函数；async 函数禁止 THREAD |
| `concurrency_safe` | `bool \| None = None` | 未声明时默认允许并发；是否真正并发还取决于只读判定与执行窗口 |

参数须有可解析的类型注解，支持普通和 keyword-only 参数；不支持 positional-only 参数。`*args`、`**kwargs` 不生成模型参数字段。无默认值的参数必填。docstring 的 `Args:` 提供字段说明。通过 `schema_from_pydantic_model(model)` 导出 schema，或 `schema_from_callable(func, *, preset_kwargs: set[str])` 按函数生成。

函数返回 `str` 时作为正文，`None` 对应空正文，其他普通值先尝试 JSON 文本；直接返回 `ToolResult` 可控制更丰富的结果。执行器最终绑定实际调用 identity。

### ToolDefinition

必填字段是 `name: str`、`description: str`、`input_schema: dict`。名称最长 64 字符，以字母或下划线开始，仅含字母、数字、下划线；描述非空；schema 根节点必须是 object。

| 字段 | 默认值 | 用途 |
| --- | --- | --- |
| `capabilities` | 空 set | `read`、`write`、`execute`、`network`、`mcp`、`agent` |
| `group` / `aliases` | `"core"` / `()` | 分类与执行名称别名 |
| `deferred` | `False` | 延迟 schema 披露 |
| `max_result_chars` | `50000` | 最终模型正文超过此长度时保存正文 artifact |
| `preview_chars` | `8000` | 保存正文后保留的预览字符数 |
| `preview_mode` | `"head"` | `head` 或 `head_tail` |
| `context_retention` | `"keep"` | `keep` 保留，`observation` 允许旧结果在上下文视图中减载 |
| `metadata` | `{}` | 工具元数据 |

`BaseTool` 必须实现 `async arun(params: BaseModel | dict, context: ToolExecutionContext) -> ToolResult`。默认 `validate_input(params)` 直接返回字典；类工具须自行实现原始输入解析。默认 `is_read_only` 只在不含 write/execute/network/mcp/agent 标签时返回真，`is_destructive` 检查 write/execute，`is_concurrency_safe` 返回真。`timeout_owner` 默认为 `ToolTimeoutOwner.RUNTIME`；内置命令工具使用 `TOOL`。

### Registry 与视图

| 方法或属性 | 语义 |
| --- | --- |
| `register(tool)` / `register_many(tools)` | 注册对象；批量注册先检查全部名称和 alias 冲突，再整体发布 |
| `get(name)` | 按名称或 alias 获取；不存在抛 `IrisToolNotFoundError` |
| `view(*, include_groups=None, allow=None, deny=None)` | 创建只读过滤视图，集合使用 canonical 工具名 |
| `view.available_tools` | 静态范围内的完整目录，含尚未披露的 deferred |
| `view.active_tools` | available 中的 eager 与显式 allow 工具 |
| `active_specs()` | 返回当前活动 `ToolSpec`，不选择供应商协议 |
| `view.specs_for(names: tuple[str, ...])` | 按注册顺序投影选定工具 schema |
| `search_deferred(query, *, include_groups=None, limit=10, allowed_names=None)` | SDK 的本地候选检索，返回 `list[ToolDefinition]`；与模型工具 `tool_search` 的参数不同 |

过滤顺序是 deny 优先，group 再限制，allow 可以放行组外工具或使 deferred 活跃。`view.get()` 委托原 registry 查找，不应当用它代替执行权限或当前请求的工具可调用检查。

YAML 的 `tools.python.functions` 是 `module:function` 引用列表；`registrars` 是接收 `ToolRegistry` 的同步注册函数列表。`build_tool_registry(config: ToolsConfig, *, memory_service=None, memory_config=None, memory_decision_client=None, command_binding=None, prompt_snapshot=None)` 是配置到 registry 的公开装配入口。命令工具需要现成 command binding；普通集成优先使用 `AgentRunner.from_config_path` 自动装配。

## 调用、结果与权限

`ToolExecutor(registry, *, permission_policy=None, middleware=None, circuit_breaker=None, hook_dispatcher=None, command_binding=None, observability=None)` 提供独立执行能力。底层 HookDispatcher 位于 `iris.hooks.dispatcher`，普通 Agent 集成用 runner 的 `hooks` 参数，不必自行装配派发器。

```text
await executor.execute_one(tool_use, context, *, approved_tool_call_id=None)
await executor.execute_many(tool_uses, context)
executor.prepare_many(tool_uses, context)
```

前两者返回单个 `ToolResult` 或按原调用顺序对应的列表；`prepare_many` 返回 `ToolBatchPlan`。相邻只读且并发安全的调用可并行。独立 executor 不负责创建持久化 run、呈现人工问题或恢复等待；`ask_question` 与 `subagent` 应由 runner/runtime 驱动。

`ToolExecutionContext` 必填 `workspace_root: Path`，其余包括 `call_id/tool_name/session_id/agent_id=""`、`permission_mode="default"`、`metadata={}`、`read_state=None`、`cancellation=None`、`tool_timeout_seconds=None`。它还携带内部命令停止与控制槽，工具开发通常只消费已有 context，不自行伪造这些运行控制事实。`CancellationSignal` 提供 `requested` 属性与 `raise_if_requested()`。

| ToolResult 字段 | 默认与含义 |
| --- | --- |
| `tool_use_id` / `tool_name` | 必填字符串，executor 会归一为本次 identity |
| `content` | `[]`，有序文本/图片 DataBlock |
| `is_error` / `error` | `False` / `None`；错误为 `ToolErrorInfo(code, message, retryable=False, details={})` |
| `data` | `{}`，宿主使用的结构化结果，不自动等于模型正文 |
| `artifact` | `None` 或 `ToolArtifact` |
| `stats` / `metadata` | `{}` / `{}`，统计与额外信息 |
| `hook_feedback` | `()`，当前调用 after Hook 追加的文本 |

`model_blocks` 返回模型内容投影：错误文本使用 `Error[code]: message`，保留图片，最后附 Hook feedback；`model_content` 汇总该投影的文本。`to_msg()` 转成带 `ToolResultBlock` 的消息。

`DefaultPermissionPolicy(*, workspace_policy=None, write_mode="confirm", execute_mode="confirm")` 实现 `PermissionPolicy.check(tool, params, context) -> PermissionDecision`。effect 为 `allow`、`deny` 或 `require_human`；后两种需要 reason。执行权限先判定；普通只读自动允许，纯 read/write 能力按 write_mode；内置 Web、工具发现、Memory Search、subagent 有专门的自动允许规则；MCP 只有被有效信任为只读时自动允许，其他一般请求确认。能力声明不会把任意 Python 函数放入 OS 沙箱。

`WorkspacePolicy.resolve_path(path, *, workspace_root)` 解析并限制文件路径范围。真正的人工批准走[运行 SDK](runtime.md)，不要把 standalone executor 的参数当成通用 UI 授权流程。

`CircuitBreaker(failure_threshold=3, cooldown_seconds=30.0)` 可选注入：连续真实 body 失败达到阈值后暂时返回 `CIRCUIT_OPEN`，冷却后允许试探；成功清除记录。普通 `ToolExecutor` 不默认创建熔断器。

## Artifact

`ToolArtifact(path=..., mime_type="text/plain", size_bytes=0, preview="", text_path=None)` 表示本地文件引用，所有字段以关键字传入。`path` 是原生或发布产物；`text_path` 在最终模型正文被保存时指向完整已保存正文，两者可以不同。

`ToolArtifactStore(root: Path, preview_chars=8000, *, preview_mode="head")` 的公共操作：

| 方法 | 返回与副作用 |
| --- | --- |
| `persist_file(tool_use_id, source: Path, *, preview)` | 复制已解析源文件，返回独立 `ToolArtifact` |
| `persist_json(tool_use_id, payload, *, preview)` | 保存完整 JSON 原生结果，返回 artifact |
| `persist_if_large(result, *, max_chars)` | 短结果不变；长结果保存正文并返回带预览和引用的新结果 |

普通工具结果由 executor 统一处理，存入 workspace `.iris/tool-results/{session}/` 范围。显式发布一个文件用 `file.publish`。模型可见图片应使用[消息与媒体接口](media.md)，只设置 artifact 不会自动把像素放进模型请求。

## 内置工具目录

左列用于 `tools.builtin`，右列用于模型调用、Hook 过滤和工具记录。

| YAML 键 | 模型工具名 | 参数与行为 |
| --- | --- | --- |
| `file.read` | `read_file` | `file_path`；文本支持 `offset=None`、`column=0`、`limit=None`、`with_line_numbers=False` |
| `file.list` | `list_files` | `path="."`、`pattern=None`、`max_results=200`（0–1000）；递归列出文件，pattern 为 glob |
| `file.grep` | `grep_search` | `pattern` 为 Python 正则；`path="."`、`max_results=200`（0–1000）；结果为路径、行号与文本 |
| `file.write` | `write_file` | `file_path`、`content`；创建或写入全文 |
| `file.edit` | `edit_file` | `file_path`、非空 `old_string`、`new_string`；唯一片段替换 |
| `file.publish` | `publish_artifact` | `file_path`；复制已有文件作为发布副本 |
| `exec.command` | `exec_command` | 非空 `command`、`cwd="."`、`timeout_seconds=None` |
| `exec.python` | `run_python` | 非空完整 `code`、`cwd="."`、`timeout_seconds=None` |
| `human.ask` | `ask_question` | 非空 `question`、`options=[]`；选项不可为空或重复，通过 HITL 返回回答 |
| `web.search` | `web_search` | 见下表，需要 Tavily 凭据 |
| `web.fetch` | `web_fetch` | 见下表，需要 Tavily 凭据 |

文本 `offset` 和 `column` 从 0 开始；`limit` 0–1000，省略使用 1000。正文受单次字符预算限制，按结果 `next_offset` 与 `next_column` 继续，避免超长单行无法读完。PNG/JPEG/WebP 内容走图片导入，忽略分页与行号参数。读取更新文件观测；编辑或覆盖已有文件需要满足读后未变化的约束。`file.publish` 不更新编辑前读取状态。

Web 参数：

| 工具 | 字段 | 默认与约束 |
| --- | --- | --- |
| `web_search` | `query` | 必填、去掉首尾空白后非空 |
| | `max_results` | 10，严格整数 1–20 |
| | `time_range` | None，或 day/week/month/year |
| | `include_domains` | []，最多 300 个 |
| | `exclude_domains` | []，最多 150 个 |
| `web_fetch` | `urls` | 必填，1–20 个 HTTP(S) URL |
| | `query` | None 表示完整提取正文，非空字符串表示相关摘录 |

两个 Web 工具都固定使用 Tavily basic；搜索不让服务端生成最终答案。`web_fetch` 部分失败时保留成功与失败信息，全失败返回工具错误。凭据由 `IRIS_PROVIDER_API_KEYS__TAVILY` 提供，操作例子见[外部检索指南](../cookbook/mcp.md)。

以下工具由能力配置自动装配，不是手写 builtin 键：`load_skill`、`subagent`、`tool_search`、`context_read`、`context_search`、`get_goal`、`report_goal`、`memory_search`、`memory_fetch`。Memory 写工具按其[状态与长期能力参考](memory-goals.md)显式配置。Todo 由普通文件工具维护，没有专用写 Todo 工具。Evolution 的经验与修订入口见[Evolution 指南](../cookbook/evolution.md)。

`context_read` 的 message/result 引用、text/raw 表示和字符分页，以及 `context_search` 的扫描游标，集中在[回读工具参考](context.md#回读与搜索工具)。

## 工具发现

`context_policy.enabled` 默认 true，`context_policy.deferred_tools` 默认 false。启用 deferred_tools 要求 enabled=true；配置该开关会装配 `tool_search`，并把接入的 MCP 工具作为 deferred 发布。Python 工具的 deferred 由自身定义声明；该开关不把所有普通 eager 工具都改为 deferred。

模型输入是 `ToolSearchInput(queries=[...], include_groups=None)`：

- `queries` 至少一项，每项为去掉首尾空白后的非空意图；每项至多选择一个 deferred 工具。
- `include_groups=None` 不再缩小组范围；空列表表示不允许任何组。
- 只搜索静态视图允许的 deferred 候选；不包含 eager 工具、Skill 或子 Agent。
- 结果 `data.selections` 按输入顺序保存 `{query, tool}`，无匹配时 `tool=null`；`data.tools` 按首次命中去重，提供名称、最多 240 字符描述和 group。
- executor 认证真实搜索结果的 `context_revealed_tools`，交给 runtime 更新披露状态。后续请求只有实际加载完整 schema 的工具才可调用；命中不绕过预算和权限。

默认本地检索使用名称、标签、组和描述的相关性排序。Decision 后端使用同一输入和输出契约，见下文。机制解释见[工具生命周期](../design/tool-execution.md)。

## MCP

`AgentMCPConfig` 通过 `mcp.path` 引用单个 JSON/JSONC/TOML 文件；路径相对 Agent YAML。`mcp.overrides` 按原始 server 名覆盖 `required`、`trust_annotations`、`startup_timeout_sec`、`tool_timeout_sec`；未给值保留服务声明和默认值。

配置根必须恰含 `mcpServers`、`servers`、`mcp_servers` 中一个 server mapping。每个 server 支持：

| 字段 | 默认或规则 |
| --- | --- |
| `type` | 有 command 时默认 stdio，否则 streamable-http；支持 stdio、streamable-http、sse；http/streamable_http 归一为 streamable-http |
| `command` / `args` | STDIO 必填 command，args 默认 [] |
| `cwd` | STDIO 省略时使用 workspace；显式相对路径基于 MCP 文件目录 |
| `env` / `env_vars` / `envFile` | 显式环境 mapping / 继承的宿主变量名 / dotenv 路径；默认空或省略 |
| `url` | HTTP/SSE 必填；支持来源别名 serverUrl |
| `headers` / `http_headers` | HTTP 静态 header mapping |
| `env_http_headers` | HTTP header 名到宿主环境变量名的 mapping |
| `bearer_token_env_var` | 从指定环境变量构造 Authorization Bearer header |
| `enabled` / `disabled` | 默认启用；两者同时给出时不得矛盾 |
| `required` | true；失败是否阻止准备 |
| `startup_timeout_sec` | 30，正有限数；来源 startup_timeout_ms 转换为秒 |
| `tool_timeout_sec` | 30，正有限数 |
| `enabled_tools` | 省略表示全部；指定时按原始 wire name 选择 |
| `disabled_tools` | []；按原始 wire name 排除，优先于 enabled_tools |

`trust_annotations` 是 Iris override 策略，默认 false，通过 Agent YAML 的 overrides 配置。当它为 true 且工具 `read_only_hint` 为 true，才使用只读执行策略。STDIO 字段和 HTTP 字段不能混用。

环境表达式支持 `${VAR}`、`${env:VAR}`、`${VAR:-default}`；只替换一次。STDIO 显式环境按 env_vars → envFile → env 顺序覆盖，表达式读取准备时的宿主环境；不解释 `${workspaceFolder}` 等其他客户端专属占位符。HTTP header 名归一成小写，同一有效键不同值报配置错误。

常规工具名为 `mcp__server__tool`；需替换字符或截断时带稳定身份后缀，应读取目录实际名称。输入 schema 使用 JSON Schema 2020-12，引用必须本地可解析。准备获取完整目录后固定发布；deferred 仅推迟模型 schema 披露，不推迟连接与发现。

低层公开入口：`iris.mcp.load_mcp_config(path: Path, *, overrides)` 返回 `MCPConfig`；`MCPManager(config, *, registry, workspace_root, defer_tools=False)` 的 `await prepare()` 返回 `MCPCatalogSnapshot`，`.snapshot` 准备前为 None，`await aclose()` 关闭自有连接。快照包含 server、实际协议、工具 descriptor 和 diagnostics；普通 Agent 由 runner 管理这些资源。

原生 MCP 结果保存为 artifact，文字和支持的图片投影到模型内容。使用方式见[MCP Cookbook](../cookbook/mcp.md)。

## 命令环境

`CommandConfig` 从 `iris.command` 导入，配置 root 命令资源：`mode="native"`、`timeout_seconds=120`、`docker=None`。native 不能声明 docker；docker 模式省略 docker 对象时采用下列默认值。

| Docker 字段 | 默认值 |
| --- | --- |
| `image` | `iris-command:local` |
| `endpoint` | None；显式值只支持本地 unix socket 或 Windows npipe |
| `network` | none，可选 bridge |
| `cpus` | 2.0，正有限数 |
| `memory_mb` | 1024，正整数 |
| `pids_limit` | 128，正整数 |
| `environment` | {} |

需要 `sandbox` extra、本机 Linux engine 和已存在的镜像；prepare 不拉取或构建，也不降级 Native。Native 的 shell 为 Windows cmd.exe 或其他宿主 /bin/sh；Docker 为 Linux /bin/sh。`run_python` 在 Native 使用当前解释器，在 Docker 使用镜像 Python；每次独立进程，无 notebook 变量续存、无交互 stdin/TTY、不自动安装依赖。

工具参数 `cwd` 必须是当前 Agent workspace 内的已有目录。最终期限为 root command timeout、显式调用 timeout 和 runtime tool timeout 的最小值。普通命令结果提供 mode/status/exit_code/cwd/duration_seconds/output_truncated/output_stats；status 为 exited、timed_out、cancelled、environment_interrupted。不能确认真实退出时不伪造 exit code。

错误码包括 `COMMAND_FAILED`（非零退出）、`COMMAND_TIMEOUT`、`COMMAND_CANCELLED`、`COMMAND_ENVIRONMENT_INTERRUPTED`、`COMMAND_UNAVAILABLE`（未启动）。采集端丢弃的输出字节不保存；artifact 保存的是已有工具正文。

child 借用 root command binding，不能显式声明 command；权限与 workspace 仍按父子规则收窄。root `aclose()` 清理自有资源，borrower 不自行关闭服务。操作示例与资源保留规则见[命令指南](../cookbook/commands.md)。

面向后端实现者，`CommandService` 以类型化对象提供：

| 方法 | 契约 |
| --- | --- |
| `await prepare()` | 准备自有资源 |
| `await execute(scope: CommandScope, request: CommandRequest)` | 返回 `CommandOutcome` |
| `stop(scope) -> StopOperation` | 同步登记并调度或加入停止 |
| `await operation.wait_stopped()` | 物理停止确认，返回 live service 的 `CommandStopReceipt` |
| `await operation.wait_drained()` / `await service.wait_drained(receipt)` | 等待旧调用收尾，不再次停止新环境 |
| `await aclose()` | 关闭服务资源 |

`CommandScope(run_id, session_id)` 定义归属；`CommandRequest(call_id, payload, cwd, timeout_seconds, stdin=None)` 的 payload 为 `ShellCommand(command)` 或 `PythonCode(code)`。低层 stdin 字节供脚本 Hook 等内部适配使用，不是模型命令工具的交互输入 API。

## Skill 与子 Agent

Agent `skills` 默认不启用；对象字段为 `enabled=False`、`root=".agents/skills"`、`require=()`。root 相对 workspace，扫描直接子目录，不递归搜索任意层次。目录名是小写 kebab-case，包含 `SKILL.md`，frontmatter description 非空；声明 name 与目录名不一致会产生诊断，catalog 使用目录身份。目录描述默认最多 1024 字符，完整正文按名称加载。require 中缺失名称使装配失败。

`iris.skill` 导出 `resolve_skills_root`、`discover_skills`、`SkillDiscoveryOptions`、`SkillDiscoveryResult`、`SkillMetadata`、`SkillRegistry`、`SkillCatalog`、`LoadSkillTool`。独立扫描使用 `SkillDiscoveryOptions(workspace_root, roots, max_description_chars=1024)` 的关键字字段；roots 是 `(SkillScope.PROJECT, Path)` 元组序列。`discover_skills(options)` 返回 skills 与 diagnostics；`SkillRegistry(result)` 提供 `get(name)`、`has(name)`、`names()`、`missing(names)`、`diagnostics`、`len(registry)`。`LoadSkillTool(registry, *, file_service=None, max_result_chars=50000)` 接受 `{"name": "kebab-name"}`，只读取已发现名称的当前正文。

`tools.subagent` 指向目录文件。目录必填 `default` 和 `agents`，每项必填 `path`、非空 `description`；default 必须精确命中 selector，selector 使用小写 kebab-case。目录相对父 YAML，child path 相对目录文件；只加载选中 child。

模型工具 `subagent` 输入为非空 `prompt`、`agent=None`；省略 selector 用 default。child 独立 session，不继承父历史，共享 lifecycle store 和 root command 服务。父模型只接收 child 最终文本，工具 metadata 提供 agent_selector 与 child_run_id，usage 留在 child。

child workspace 与父不相交时报配置错误，存在包含关系时使用较小范围；权限按父子更严格者执行。child 不开放递归 subagent。父子 WAITING 用代理 interaction 连接，宿主恢复父 interaction 即可；详见[委派指南](../cookbook/skills-subagents.md)。测试/自定义宿主可向 runner 注入 `ChildProviderFactory`，签名为 `factory(config: AgentConfig, *, config_path: Path) -> CompletionProvider`，让每次选中 child 构造自己的 provider。

## Hook 与 Middleware

公共事件类型从 `iris.hooks` 导入。handler 签名是 `async handler(event: HookEvent) -> HookResult`。事件含 agent_id、session_id、workspace、run_id、activation_id、occurred_at，以及下表字段：

| 事件 | 额外字段 | 允许返回 |
| --- | --- | --- |
| `run.started` | run、input | None |
| `run.finished` | result | None |
| `tool.before` | call_id、tool_name、arguments | None，或 `ToolBeforeResult(deny_reason)` |
| `tool.after` | 上述工具字段、result、body_status（success/error） | None，或 `ToolAfterResult(feedback)` |

`HookRegistration(event, name, handler, tool_names=None, timeout_seconds=10)` 使用关键字构造；name 非空，期限为正有限数，非空 tool_names 仅用于工具事件。每个 handler 收到独立事件快照。before 普通失败拒绝本次调用；其他事件普通失败记录并继续；取消、结果未知、资源清理失败保持控制语义。

YAML `hooks` 为有序列表，单项字段为 `name`、`event`、`handler`，可选 `tools`、`timeout_seconds=10`。Python handler 是 `{type: python, factory: module:factory, options: {}}`，同步 `factory(**options)` 返回异步 handler。命令 handler 是 `{type: command, command: ...}`，不接受 options。

命令 Hook 用 stdin 接收 `event_to_dict(event)` 的 JSON；完整 stdout 必须是单个 JSON object：`{}` 表示无结果，tool.before 可输出 `{"deny_reason":"..."}`，tool.after 可输出 `{"feedback":"..."}`。stderr 用于诊断。非零退出、截断输出、非法协议视为 Hook 错误。命令使用独立 Hook 期限；非 completed 的 run.finished 跳过命令 Hook。

`ToolMiddleware` 从 `iris.tools` 导入，实现 `async wrap_tool_call(call: ToolCall, call_next: ToolNext) -> ToolResult`。`ToolCall` 含 tool_use_id、tool_name、arguments、agent_id、session_id、run_id、activation_id、workspace_root。arguments 为独立快照；`ToolNext` 是无参数异步调用，当前包装器内至多执行一次。首项最外层，可以不调用下游直接返回；修改下游结果须返回新对象。

YAML `middleware.tools` 为 `{factory: module:factory, options: {}}` 列表，工厂返回 `ToolMiddleware`。runner 的 SDK 参数为 `hooks=()` 与 `tool_middlewares=()`；顺序为 YAML 项在前、SDK 项在后。child 不继承父扩展。

普通路径为输入/权限 → before → Middleware/body → 有资格的 after → artifact。permission 等待、ask_question 与外层 subagent 不进入这条普通链；没有运行 body 的短路也没有 after。逐步用法见[扩展 Cookbook](../cookbook/extensions.md)，observer 的区别见[扩展设计](../design/extensions.md)。

## Decision

Agent 的 `decision.path` 相对 Agent YAML 指向独立 YAML。配置字段：`provider="typesafe"`、`model="jev-1.13.0"`、`timeout_seconds=5.0`、`tools.discovery=False`、`memory.recall=False`。只有启用接点才需要 evaluator；工具发现要求 context_policy.deferred_tools=true，记忆召回规则见[长期记忆](../cookbook/memory.md)。

默认工厂从 `IRIS_PROVIDER_API_KEYS__TYPESAFE` 构造 JevClient。也可给 runner 注入 `decision_client`，只要求实现 `DecisionEvaluator.evaluate(request)`；借用对象的关闭权留给宿主。由 Iris 创建的 client 随环境关闭。Decision 错误不静默回退本地搜索。

公开类型均从 `iris.decision` 导入：

| 类型 | 契约 |
| --- | --- |
| `DecisionRequest` | `state: JsonValue`，非空 `questions: dict[str, DecisionQuestion]` |
| `ChoiceQuestion` | 非空 instructions、非空 options mapping；选项值为说明字符串或 None |
| `BooleanQuestion` | 非空 instructions；判断命题成立概率 |
| `ScoreQuestion` | 非空 instructions，至少两个从低到高的 levels |
| `DecisionResponse` | provider、model、按原问题 ID 对应的 answers、usage |
| `ChoiceAnswer` | choice、probabilities、confidence |
| `BooleanAnswer` | probability |
| `ScoreAnswer` | score、probabilities、levels、confidence |
| `DecisionUsage` | input_tokens、output_tokens；独立于主模型 |

`JevClient(api_key, model="jev-1.13.0", timeout_seconds=5.0)` 惰性建立 HTTP client，`await evaluate(request)` 在一次总期限内发送单次请求，`await aclose()` 关闭资源，也支持异步上下文管理。Jev 每个 Choice 最多 255 个选项，Score 最多 10 个等级；工具发现增加了一个 none 选项，因此单次候选目录最多容纳 254 个工具，超过限制会报错，不自动裁剪或分页。工具发现结果在 metadata.decision 记录 feature/provider/model/question_count 和 usage。

源码契约入口：[工具基础类型](../../src/iris/tools/base.py)、[内置工具装配表](../../src/iris/agents/config/tools.py)、[MCP 模型](../../src/iris/mcp/models.py)、[命令模型](../../src/iris/command/models.py)、[Hooks](../../src/iris/hooks/models.py)、[Decision](../../src/iris/decision/models.py)。
