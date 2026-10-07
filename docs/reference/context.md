# Context、压缩与项目模板参考

## 观察真实准备过程

Runtime 在每个获准的 `before_model` 步骤产生 `ContextPreparation`，由 live publisher
交给宿主。它包含 preparation_id、configuration_snapshot_id、run/session/activation、
step_index、输入预算、压力线、真实 stages/decisions、最终工具与贡献 key、保护来源和
final_input_tokens。phase 为 ready、failed 或 cancelled；失败保留对应错误。该事实不进入
checkpoint，也不参与恢复裁决。

阶段记录来自已有的完整请求计量：装配、重复观察折叠、可选贡献与工具撤下、旧正文短化、
压缩和最终选择。未进入的阶段标 skipped；压缩规划与摘要候选标 candidate，拒绝候选标
rejected，只有真实提交后的压缩标 applied。required、近期结果、重复正文回读位置与
可选优先级理由均由实际分支产生，不重新运行选材或为展示额外调用 estimator。

before/after_input_tokens 是整个请求的估算，不是区块独立成本；不能将其拼成相加等于
输入总量的饼图。provider 报告的 usage、摘要额外 usage 与本地估算分别展示。
完整快照通过原始 publisher 交给宿主证据层；Broker 的 `context.preparation` 只发送
preparation_id、phase、步骤、阶段数与最终计量。宿主可按 ID 保存和读取详情，观察失败不
改变执行。真正 provider 输入仍由原 OTel capture_content 接点采集，final_request_ref
尚未建立外部证据引用时为 None。

本页集中维护上下文相关格式和默认值。先理解行为可读[上下文工程](../design/context-engineering.md)，需要接入步骤可读[上下文配方](../cookbook/context.md)。

## context.yaml 格式

`load_context_build_input(path) -> ContextBuildInput` 读取 YAML。顶层只有 `system`、`memory`、`before_current_input`，其中 system 必填且至少有一个启用的 slot，另外两项可省略。

每个 section：

| 字段 | 默认 | 语义 |
| --- | --- | --- |
| `slots` | `[]` | 结构化内容列表 |
| `template` | `null` | 自定义 Jinja 文件；YAML 中相对 context 文件解析，直接构造 typed section 时使用绝对路径 |
| `max_chars` | `null` | 最终 section 字符上限，提供时须为正整数；超出报错，不自动截断 |

每个 slot：

| 字段 | 默认 | 语义 |
| --- | --- | --- |
| `name` | 必填 | 有效 XML 标签名，用于默认结构化渲染 |
| `content` | 必填 | 文本、列表或其他可渲染内容 |
| `order` | `100` | 整数，先按此值再按名称排序 |
| `attributes` | `{}` | 字符串属性，属性名满足 XML 名称规则 |
| `enabled` | `true` | 关闭后不进入 section 渲染 |

自定义模板接收 `slots`：已启用且排序后的 slot 字典列表。`TemplateRenderer` 默认按纯文本处理，若模板自行输出 XML，应在适当位置显式转义。未指定模板时，由默认 XML renderer 处理结构化内容。

公开构造类型从 `iris.context` 导入：`ContextSlot`、`ContextSection`、`ContextBuildInput`、`ContextBuildOutput`。`ContextBuilder.build(input_data, *, system_addendum="")` 返回三个位置的 `Msg`；system 使用 system role，另外两者使用 context sender 的 user 消息。空的可选 section 不输出消息。

## 每步动态快照

同样从 `iris.context` 导入：

```text
async ContextSource.collect(scope: ContextBuildScope) -> ContextSnapshot
ContextBuildScope(session_id, run_id, step_index, workspace_root, run_input)
ContextContribution(key, text, required=True, priority=100)
ContextSnapshot(contributions=())
```

这些是进程内 typed 对象。每份快照的 key 由 source 保证唯一；返回值代表本步全部状态，缺席的旧条目不会自动继承。`scope.run_input` 是运行输入的文本视图，不是让 source 自行读取整段历史。

通过 `AgentRunner.from_config*(..., context_source=source)` 绑定。不同 session 可以并发调用同一个 source，宿主需要提供与 scope 相符的状态。动态快照不追加到 durable 对话；只有被选入本次请求的内容占输入预算。

可选条目按较低 priority 优先移除；相同 priority 按靠后的条目先移除。required 条目不参与这一选择。同一步的选材固定，摘要之后不会重新填回已撤下的条目。

## 上下文回读与选材

`context_policy` 的所有配置字段：

| 字段 | 默认 | 约束 / 行为 |
| --- | --- | --- |
| `enabled` | `true` | 注册并使用会话回读能力；完整应用由 Runner 提供访问端口 |
| `preserve_recent_tool_groups` | `2` | 非负整数，保留近期完整工具组不做旧正文裁剪 |
| `old_result_preview_chars` | `512` | 非负整数，旧结果预览的字符额度，`0` 只保留说明与引用 |
| `deferred_tools` | `false` | 启用工具按需披露；要求 enabled 为 true |

Goal 和 Todo 的每步投影也要求 `context_policy.enabled=true`。不要把关闭回读当作关闭所有上下文组织或关闭自动摘要。

正文减载要求本步具有可用的 `context_read`。错误结果、未闭合工具批次和不允许减载的结果不参与旧观察正文的 Trim。通用工具规则见[工具参考](tools.md)，历史读取参数如下。

### 回读与搜索工具

开启 context policy 时，Runner 自动提供 `context_read` 与 `context_search`，不需要把它们写进 `tools.builtin`。它们只访问当前 session 已提交材料，不重跑来源工具。

`context_read`：

| 参数 | 默认 | 语义 |
| --- | --- | --- |
| `ref` | 必填 | `message:N` 读取第 N 条消息，或 `result:N:B` 读取该消息的第 B 个内容块中的工具结果；下标从 0 开始 |
| `offset` | `0` | 非负 Python Unicode 字符位置，不是字节或 token |
| `limit` | `4000` | 每页字符数，范围 1–8000 |
| `representation` | `text` | `text` 读当时保存的最终模型文本；`raw` 只适用于带 artifact 的 result 引用，读取原生材料 |

引用来自模型视图的回读提示或搜索结果。返回 `ContextReadPage(ref, representation, offset, next_offset, has_more, content)`；继续读取时使用返回的 `next_offset`。普通内联结果和 message 引用不支持 raw。图片的 text 表示名称和 original/model 引用，查看像素需用已注册的 `read_file` 读取 model 路径。

例如，已有提示明确提供 `result:3:0` 时，可以调用以下参数；引用必须来自本会话实际材料，不能猜测下标：

```json
{"ref":"result:3:0","offset":0,"limit":4000,"representation":"text"}
```

`context_search` 接受 `query`（非空白字符串）、`after=0`（非负消息扫描起点）和 `limit=10`（返回上限 1–20）。它按 Unicode casefold 子串搜索已提交正文、工具预览与图片名称/引用，每次最多扫描 200 条消息，每条最多返回一个命中；不打开全部外置全文，也不做 OCR。

返回 `ContextSearchPage(matches, next_after, has_more)`，命中含 ref、role、tool_name、snippet。即使 matches 为空，只要 has_more 为 true，仍可用 next_after 继续扫描。找到线索后，再用 `context_read` 展开原文。

SDK 类型和 `ContextAccessPort.read(session_id, params, workspace_root)` / `search(session_id, params)` 契约位于 [context_access.py](../../src/iris/tools/context_access.py)；完整应用由 Runner 提供端口，工具本身不持有 lifecycle store。

## 压缩预算

`compaction` 没有 enabled 字段，包含：

| 字段 | 默认 | 约束与含义 |
| --- | --- | --- |
| `input_budget_tokens` | `96000` | 正整数；已扣除输出预留的完整输入额度 |
| `keep_recent_ratio` | `0.15` | `0 < 值 < 1`；近期原文的软保留目标 |
| `summary_ratio` | `0.05` | `0 < 值 < 1`；摘要请求的最大输出额度比例 |
| `timeout_seconds` | `300` | 正有限秒数；一次完整压缩操作的时限 |

派生值：压力线为 `floor(input_budget_tokens * 0.8)`；近期原文目标与摘要生成上限分别使用上述比例并向上取整。这些派生值不是额外可配置 YAML 字段。

保留近期原文是软目标，切点必须尊重完整消息组和任务锚点。成功摘要后的完整请求必须不超过压力线且比之前更小。无新切点但仍在硬输入上限内时可以继续；有摘要尝试却失败时结束当前 Run，保留旧有效状态。

压缩 token 记入 `RunUsage.compaction`，主模型计数另存。具体预算投影见[运行参考](runtime.md)。

## 项目命名模板

`prompts.root` 默认 `.iris/prompts`，相对 root workspace。`PromptSource.initialize(workspace_root, root)` 只补齐缺失模板，不覆盖已有内容。

当前固定 ID（对应同名 `.j2` 文件）：

| 用途 | ID |
| --- | --- |
| 压缩 | `compaction`、`compaction_input` |
| 长期记忆 | `memory_context`、`memory_flush`、`memory_dream`、`memory_overview`、`memory_recall_instruction` |
| Goal / Todo | `goal_context`、`goal_continuation`、`todo_context`、`todo_reminder` |
| Skill / 工具发现 | `skill_catalog_usage`、`tool_discovery_instruction` |
| 项目经验 | `project_skill_update`、`evolution_review` |

`iris.prompts` 公开 `PROMPT_IDS`、`PromptConfig`、`PromptSource`、`PromptSnapshot`。`source.snapshot()` 冻结目录的模板源；`snapshot.render(prompt_id, variables)` 在该冻结来源上渲染；`snapshot.with_template(prompt_id, source)` 返回替换一个模板源的新快照，不写磁盘。

正文模板描述策略，消费领域的代码仍拥有输入变量及输出 schema。修改模板不能任意更换模型返回结构。各完整操作采用自己的快照，修改后的内容由下一次相应操作采用，不影响已开始操作中途的模板来源。

要了解自动修订哪些模板、何时使用新配置，见[项目经验](../cookbook/evolution.md)。

依据：[Context 模型与构建器](../../src/iris/context/models.py)、[动态 source](../../src/iris/context/source.py)、[压缩配置](../../src/iris/agents/config/compaction.py)、[选材](../../src/iris/runtime/_context_projection.py)、[模板来源](../../src/iris/prompts/source.py)。
