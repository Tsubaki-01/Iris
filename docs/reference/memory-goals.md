# 记忆、经验、Goal 与 Todo 参考

本页是长期能力的配置与 SDK 查询入口。第一次使用请先读[记忆](../cookbook/memory.md)、[经验与修订](../cookbook/evolution.md)或[目标与清单](../cookbook/goals-todos.md)配方。一般运行、会话和存储配置见[配置参考](configuration.md)与[运行参考](runtime.md)。

## 配置字段

以下路径都位于 `agent.yaml`。配置在构造 runner 时采用；修改配置文件后重建 runner，不支持用旧 runner 的新会话热更新配置。路径所指的 workspace 由 `permissions.workspace` 确定。

### Memory

| 字段 | 类型与默认值 | 规则 |
| --- | --- | --- |
| `memory.enabled` | `bool = false` | 统一控制服务挂载、概览和自动注册的 Search/Fetch；关闭时不使用注入服务 |
| `memory.root` | `str = ".iris/memory"` | Markdown 投影目录；相对 workspace，必须位于 workspace 内 |
| `memory.path` | `str = ".iris/memory/memory.db"` | SQLite 数据库；与 root 相同的路径解析规则 |
| `memory.read_namespaces` | `list[str] = ["project"]` | 模型可读范围；空列表表示不读取任何 namespace；每项必须含非空白字符 |
| `memory.write_namespace` | `str = "project"` | 写工具及自动捕获的单一目标 namespace，必须含非空白字符 |
| `memory.overview.input_budget_tokens` | `int = 96000` | 显式概览生成的输入预算，必须大于 0 |
| `memory.overview.max_tokens` | `int = 4096` | 概览生成输出上限，必须大于 0 |
| `memory.overview.system_budget_ratio` | `float = 0.02` | 所有 namespace 概览共用 `compaction.input_budget_tokens × ratio` 额度；取值 `(0, 1]` |
| `memory.generation.enabled` | `bool = false` | 开启 Run 材料捕获与宿主自动维护；SDK 需绑定协调器 |
| `memory.generation.flush_input_budget_tokens` | `int = 32000` | flush 输入预算，必须大于 0 |
| `memory.generation.flush_output_budget_tokens` | `int = 4000` | flush 输出预算，必须大于 0 |
| `memory.generation.dream_input_budget_tokens` | `int = 32000` | dream 输入预算，必须大于 0 |
| `memory.generation.dream_output_budget_tokens` | `int = 4000` | dream 输出预算，必须大于 0 |

`memory.generation.enabled` 不取代 `memory.enabled`。所有生成预算都是对应独立请求的额度，概览占用比例则用于主模型请求的窗口采用。新会话首次输入、成功压缩采用概览；普通后续 Run 和恢复重用持久窗口。完整概览超额时尝试完整知识范围，知识范围仍超额抛出 `IrisContextError`。

### Evolution 与共享维护

| 字段 | 类型与默认值 | 规则 |
| --- | --- | --- |
| `maintenance.idle_seconds` | `float = 300` | 宿主共享空闲时间，有限且非负；只有启用维护能力时才有维护工作 |
| `evolution.enabled` | `bool = false` | 开启项目经验维护；要求 `skills.enabled: true` |
| `evolution.policy_skill` | `str \| null = null` | 非空时相对 workspace 读取策略 Skill；空值使用包内策略 |
| `evolution.skill_max_chars` | `int = 8000` | 经验 Skill 正文字符上限，必须大于 0 |
| `evolution.input_budget_tokens` | `int = 32000` | 经验整理与修订请求的独立输入上限，必须大于 0 |
| `evolution.output_budget_tokens` | `int = 8000` | 独立输出上限，必须大于 0 |
| `evolution.prompt_targets` | 字符串序列，默认空 | 显式开放的命名 prompt，见下表 |
| `evolution.config_targets` | 字符串序列，默认空 | 显式开放的主 YAML 叶字段，见下表 |

经验发布到 `<skills.root>/project-experience/SKILL.md`，默认 `.agents/skills/project-experience/SKILL.md`。完整文件还需满足现有 Skill loader 的 50000 字符、1000 行限制。`skills` 自身配置见[工具与扩展参考](tools.md)。

允许开放的修订目标：

| 类型 | 完整取值 |
| --- | --- |
| prompt | `memory_flush`、`memory_dream`、`memory_overview`、`project_skill_update`、`compaction`、`compaction_input` |
| config | `context_policy.preserve_recent_tool_groups`、`context_policy.old_result_preview_chars`、`compaction.input_budget_tokens`、`compaction.keep_recent_ratio`、`compaction.summary_ratio`、`todo.enabled`、`system` |

config 候选按完整 AgentConfig 重新解析；`system` 只允许修改已有简单模式的文本，不能从外部 context 模式切换。一次候选只发布一个 prompt 或一份主 YAML 的有限字段。开放 config 目标时，装配函数必须收到明确 `config_path`。YAML 发布会重新序列化，可能改变排版并移除注释。

### Goal 与 Todo

| 字段 | 类型与默认值 | 规则 |
| --- | --- | --- |
| `goal.enabled` | `bool = false` | 启用 `SessionManager.goal`、目标上下文与目标工具；要求 context policy 开启 |
| `goal.max_rounds` | `int = 20` | 新目标的默认自动顶层 Run 总数上限，必须大于 0 |
| `todo.enabled` | `bool = false` | 每个获准模型步骤读取当前会话清单；要求 context policy 开启 |

Goal 自动执行的 `AgentRunOptions.runtime.include_tools` 必须是 `true`；最终 `tool_choice` 必须为 `auto` 或未指定。子 Agent 的 Todo 由自己的配置控制，使用自己的 session 文件。Goal 不向子 Agent 传播一套独立的自动目标循环。

## Memory SDK

本页代码框展示接口签名，不是可直接执行的脚本；完整用法见对应 Cookbook。

从 `iris.memory` 导入下列服务、模型与存储。`MemoryService` 是公开读写门面，`MemoryStore` 是存储实现者协议。

### 构造与资源

```text
MemoryService(
    store,
    *,
    mirror=None,
    overview_provider=None,
    overview_model=None,
    overview_config=None,
    generation_provider=None,
    generation_model=None,
    generation_config=None,
    prompt_source=None,
    observability=None,
    io_execution_mode=MemoryIOExecutionMode.INLINE,
)
```

- `store: MemoryStore` 是权威条目与生成状态存储；内置实现为 `SQLiteMemoryStore(db_path)`。
- `mirror: FileMemoryMirror | None` 是可选文件投影。`FileMemoryMirror(root, *, workspace_root=None)` 的 `initialize_layout()` 创建投影目录。
- 概览生成需 `overview_provider`、`overview_model`、mirror 及初始化后的 `PromptSource`；flush/dream 需 `generation_provider`、`generation_model` 及提示来源。普通读写不要求模型。
- 自定义 store 默认 `INLINE`；配置工厂构造的 SQLite 服务使用 `THREAD`，异步调用将一次完整同步操作交给后台 IO。`await wait_pending_io()` 等待已派发 IO 真正结束。SQLiteMemoryStore 每次操作管理自己的连接，没有服务级 `close()`。
- 服务借用 provider 与 observability，不替宿主关闭这些依赖。

配置工厂的完整调用面：

```text
build_memory_service_from_config(
    config,
    workspace_root,
    *,
    memory_service=None,
    overview_provider=None,
    overview_model=None,
    prompt_source=None,
    observability=None,
) -> MemoryService | None

resolve_memory_path(value: str, workspace_root: Path) -> Path
```

关闭时工厂返回 `None`；启用且传入服务时原样复用，不改写其策略与资源；否则构造 SQLite 服务，并将传入概览 provider/model 同时绑定为生成 provider/model。`resolve_memory_path` 返回 workspace 内的绝对路径。

### 写入模型与分类

`MemoryWriteInput` 必填 `text`、`reason`，两者不可为空白。其余字段：`namespace="project"`、`category="user"`、`kind="note"`、`source_type="sdk"`、`source_id=""`、`actor="sdk"`、`evidence=()`、`artifacts=[]`、`metadata={}`。

| 枚举 | 取值 |
| --- | --- |
| `MemoryCategory` | `user`、`feedback`、`reference`、`task`、`session` |
| `MemoryItemKind` | `fact`、`preference`、`note`、`summary`、`task_state`、`correction` |
| `MemoryItemStatus` | `active`、`deleted`、`superseded` |
| `MemorySourceType` | `message`、`tool_event`、`artifact`、`task`、`reference`、`sdk`、`generation` |
| `MemoryActor` | `sdk`、`agent`、`user`、`system` |

`MemoryItem` 包含稳定 `id`、上述内容/来源字段、`status`、`superseded_by`、`created_at`、`updated_at`、`deleted_at`。默认查询只返回 active 条目。

`MemoryItemPatch` 可修改 `text`、`category`、`kind`、`status`、`artifacts`、`evidence`、`metadata`；省略表示不变，显式 `null` 非法，集合是整体替换。`MemoryArtifactRef(path, mime_type="text/plain", metadata={})` 的 path 必须是相对路径。

`MemoryEvidenceRef` 两种形式：

- `kind="episode"`：`source_id` 指向 Episode，必需 `record_id` 和字符半开区间 `start`、`end`，满足 `end > start >= 0`。
- `kind="event"`：`source_id` 指向真实写入事件，不携带记录或字符区间。

### 读写方法

下表是同步签名。除 `observe()` 外，每个表中读写方法都有同签名的异步版本，在名称前加 `a`，例如 `aremember()`、`aget_item()`、`alist_events()`。

| 方法 | 返回与语义 |
| --- | --- |
| `observe(input: MemoryObserveInput)` | `MemoryEpisode`；保存原始材料，尚不能作为正式知识查询 |
| `remember(input: MemoryWriteInput)` | `MemoryItem`；创建正式条目并刷新分类投影 |
| `update(item_id, namespace, patch, *, actor=MemoryActor.SDK, reason, source_type=MemorySourceType.SDK, source_id="")` | 更新后的 `MemoryItem`，稳定 ID 不变 |
| `forget(item_id, namespace, *, actor=MemoryActor.SDK, reason, source_type=MemorySourceType.SDK, source_id="")` | `bool`；active 条目实际软删除为 true；未命中为 false |
| `get_item(item_id, namespaces)` | `MemoryItem \| None`；读取允许范围内的当前活跃条目 |
| `search(query: MemorySearchQuery, namespaces)` | `MemorySearchResponse`；本地检索，不自动调用 Decision |
| `list_items(namespaces, *, limit=50, categories=None, kinds=None)` | `list[MemoryItem]`；联合读取近期活跃条目，过滤后应用 limit；`None` 读取完整投影，整数为 1–100 |
| `list_events(namespace, *, item_id=None, limit=100)` | `list[MemoryEvent]`；操作历史，limit 为 1–100 |

`namespaces` 为 `Sequence[str]`，读取范围由调用方绑定，不由模型自行扩大。写入方法的 `namespace` 为单个字符串。不存在的更新、非法状态/来源等使用 `IrisMemoryError`；边界模型输入不合 schema 时由 Pydantic 报错。已经写入历史的旧概览或工具结果不会因 `forget()` 被追溯删除。

`MemoryObserveInput` 的字段为 `namespace="project"`、`text=""`、`source_type="sdk"`、`source_id=""`、`actor="sdk"`、`records=()`、`reason=""`、`artifacts=[]`、`metadata={}`。可提供 `MemoryRecord` 序列，或用 text 建立一条记录。`MemoryRecord` 包含自动 ID、`role="sdk"`、`text=""`、来源、发生时间、artifacts 和 metadata。一个 Episode 至少有一条记录且记录 ID 唯一。

### 搜索与工具

`MemorySearchQuery(query, required_terms=[], categories=[], kinds=[], limit=8)`：query 去除首尾空白后必须非空，limit 为 1–100。required_terms 中每一项都必须含可索引字符；正文必须同时匹配全部词组，英文大小写不敏感，按既有分词有序相邻匹配。它不是正则表达式或原始 FTS 语法。

本地分词使用英文/数字词和中文双字片段；普通查询词做 OR 检索，必要词组做 AND 约束，按 SQLite FTS/BM25 排序。返回 `MemorySearchResponse(items, has_more)`，items 内每个 `MemorySearchHit` 有 `item_id`、`namespace`、`category`、`kind`、`snippet`、`is_complete`。短文本返回全文，长文本返回命中位置附近连续 300 字符原文；`has_more` 只说明仍有候选，不强制继续查询。

| 注册方式 / YAML 名称 | 模型工具名 | 输入和结果 |
| --- | --- | --- |
| `memory.enabled` 自动注册 | `memory_search` | 输入 `MemorySearchQuery`；结果 `items`、`has_more`，还有候选时含 hint |
| `memory.enabled` 自动注册 | `memory_fetch` | `item_id`；结果 `item`，当前完整条目与元数据；缺失、非活跃、范围外均为读取错误 |
| `memory.remember` | `memory_remember` | `text`、`reason`、可选 category/kind；结果 `item` |
| `memory.update` | `memory_update` | `item_id`、`patch`、`reason`；结果 `item` |
| `memory.forget` | `memory_forget` | `item_id`、`reason`；结果 `deleted` |

写工具的 namespace 与调用来源由宿主绑定。正文投影未同步时结果可带 `warning`。不要在 YAML 手动声明 `memory.search`、`memory.fetch`；配置入口会拒绝重复声明。独立 SDK 工具装配使用以下入口，默认不注册工具：

```text
register_memory_tools(
    *,
    service: MemoryService,
    access_policy_factory: MemoryAccessPolicyFactory,
    registry: ToolRegistry | None = None,
    max_result_chars: int = 50000,
    tool_names: Sequence[str] = (),
    memory_decision_client: DecisionEvaluator | None = None,
    prompt_snapshot: PromptSnapshot | None = None,
) -> ToolRegistry
```

`tool_names` 使用表中的点号声明名。未提供 registry 时新建，提供时扩展该 registry。`MemoryAccessPolicy(read_namespaces=("project",), write_namespace="project")` 定义一次调用的范围；`MemoryAccessPolicyFactory` 是 `Callable[[ToolExecutionContext], MemoryAccessPolicy]`，可由 `default_memory_access_policy_factory(config)` 构建。

可选 Decision 只改变 `memory_search` 工具内部召回。它按当前范围、分类、类型读取完整 active 候选，再应用相同 required_terms，以一次 Score 请求评分全部剩余正文，保留分数至少为 2 的结果，按分数降序稳定排序并取 limit。返回完整正文且 `is_complete=true`；usage 位于工具 metadata 的 `decision`。本地 `MemoryService.search/asearch` 始终保持本地检索语义。Decision 连接配置见[工具参考](tools.md)。

### 生成、概览与查询结果

| 调用 | 返回 / 用途 |
| --- | --- |
| `await flush(namespace, *, scope=None)` | `GenerationResult`；提炼一批 Episode 为 Observation |
| `await dream(namespace, *, retry_blocked=False, scope=None)` | `GenerationResult`；整理观察与显式变化，不隐式 flush；可显式重试受阻输入 |
| `await refresh_overview(namespace)` | `MemoryOverviewGenerationResult`；生成并尝试发布概览，不在普通主 Run 中隐式执行 |
| `generation_state(namespace, *, scope=None)` / `await ageneration_state(...)` | `GenerationState`；积压、受阻输入、版本和最近阶段结果 |
| `load_overviews(namespaces)` / `await aload_overviews(namespaces)` | `tuple[MemoryOverviewDocument, ...]`；只读发布物，缺文件给缺产物说明，不调用模型 |
| `projection_warning(namespace)` | `str \| None`；分类投影的新鲜度说明 |
| `file_access(read_namespaces)` | `MemoryFileAccess \| None`；供通用文件工具使用的只读路径/版本能力 |
| `list_pending_sources(namespace)` / `await alist_pending_sources(...)` | `tuple[MemorySource, ...]`；待消费输入的 lifecycle 来源 |
| `await maintain_cycle(namespace, *, scope, cycle_id)` | `MemoryCycleResult`；一个有界周期的实际阶段结果、周期 ID 和剩余积压，调度器调用入口 |
| `list_episodes(namespace, *, after=None, limit=50)` / `await alist_episodes(...)` | `EpisodePage`；含已消费原文的完整 Episode 历史 |
| `list_generation_results(namespace, *, after=None, limit=50)` / `await alist_generation_results(...)` | `GenerationResultPage`；完整阶段历史，包含失败、冲突及旧轮次 |
| `get_observation(namespace, observation_id)` / `await aget_observation(...)` | `ObservationState \| None`；原始观察及处理去向 |
| `list_publications(namespace, *, after=None, limit=50)` / `await alist_publications(...)` | `MemoryPublicationPage`；分类与概览的发布记录 |
| `get_publication(namespace, publication_id)` / `await aget_publication(...)` | `MemoryPublicationRecord \| None`；当时的实际正文，按 namespace 隔离 |

上述分页结果提供 items 和 next_cursor。`MemoryHistoryCursor(created_at, id)` 按创建时间和
原始 ID 升序定位；limit 为 1–100，由 MemoryStore 唯一校验，Service 直接委托。
Memory SQLite 当前为 schema 7，旧 schema 在初始化时拒绝，不迁移或回填历史发布记录。

`MemoryPublicationRecord` 包含 publication_id、kind（projection/overview）、namespace、
item_revision、projection_revision、generation_result_id、created_at、status、documents 和 error。
每份 document 保存 path/text。status 为 published、failed、conflict 或 unconfirmed：
部分文件写入后失败只保留已写正文，已有新概览时为 conflict；文件完成而版本提交未确认时为
unconfirmed。只有完整发布完成才标 published。记录与文件不具有跨文件 ACID；旧 Item 前后值仍读
`MemoryEvent.before/after`，发布也不表示某次 Run 已采用它。

`GenerationResult` 有 `id`、`namespace`、`stage`（capture/flush/dream/overview）、`status`（completed/empty/failed/cancelled/conflict/blocked）、`usage`、`elapsed_seconds`、`error`、`input_ids`、`consumed_ranges`、`counts`、`item_revision`、`has_more`、`created_at`。这些成本独立于主 Run usage。

`GenerationState` 包含 `namespace`，`pending_episodes`、`pending_observations`、`blocked_observations`、`pending_changes`、`blocked_changes`，以及 `item_revision`、`projection_revision`、`overview_revision`、`latest_results`。`MemoryObservation` 则包含 text、applicability、category、kind、reason、evidence、target_item_ids、generation_model 和创建时间。

`MemoryOverviewGenerationResult` 包含 namespace/path、source_revision/current_revision/projection_revision、item_count、published/publication_reason、usage、elapsed_seconds。`published=false` 不等于数据库写入丢失，应查看 publication_reason。`MemoryOverviewDocument` 提供 namespace/path、source_revision、text、navigation、warning。

高级宿主可用 `MemoryMaintenanceScope`，显式传入 `allowed_sources`、`episode_sources` 和 `check`。来源集合均由 `(lifecycle_source_id, run_id)` 组成：`episode_sources` 限定本次可 flush 的原文；`allowed_sources` 保留给 Observation、显式 change 与 dream 的资格判断。缩小原文范围不会把已有下游材料一起排除；无 Run 来源的显式 observe 输入保留原处理语义。异步 `check(tuple[MemorySource, ...]) -> bool` 在消费前复查实际来源资格。常规集成应使用协调器，不必自行构造此范围。服务还提供 `add_change_listener(callback)`、`remove_change_listener(callback)`；回调在实际写入线程执行。

### 文件格式

`namespace_key(namespace)` 将名称编码为 `ns_` 加无填充 URL-safe Base64。默认 project 的目录是 `.iris/memory/namespaces/ns_cHJvamVjdA/`：

```text
Memory.md
User/user.md
User/preferences.md
Feedback/feedback.md
Feedback/corrections.md
Reference/notes.md
Tasks/task.md
Sessions/session_items.md
```

文件首行携带 `<!-- iris-memory source_revision: N -->`。`Memory.md` 有“核心事实”和“可查询的知识”，二者由 `<!-- iris-memory-knowledge-scope -->` 分隔。完整概览格式错误时读取报错；不要手工构造概览来替代生成入口。分类文件与概览都是派生投影，写入以 SQLite 为准。

人工检查路径可使用 `mirror.namespace_directory(namespace) -> Path`；`mirror.document_path(namespace, relative_path="Memory.md")` 在配置 workspace_root 时返回 workspace 相对路径。需要重新发布分类正文时，`mirror.rebuild_from_store(store, namespace) -> MemoryNamespaceState` 从当前数据库完整重建，不生成概览。概览仍通过 `refresh_overview()` 更新。

自定义 `MemoryStore` 应实现[完整存储协议](../../src/iris/memory/store.py)。关键事务不只是 Item CRUD：条目与事件一起提交，capture 按水位 CAS，flush 同时提交观察与消费区间，dream 同时比较条目版本、应用计划并消费输入，分类正文发布在短事务中绑定完整快照版本。`MemoryNamespaceState(namespace, item_revision=0, projection_revision=None)` 与 `MemoryNamespaceSnapshot(state, items)` 是供投影使用的只读结果；只实现查询和写入方法不足以支持自动生成。

## 维护协调器与 Evolution SDK

以下入口从 `iris.harness` 导入：

```text
MaintenanceCoordinator(*, idle_seconds=300, observability=None, live_publisher=None)
MemoryMaintenanceBinding(*, service, database_path: Path, namespace: str)
ProjectEvolutionBinding(*, workspace_root: Path, service)

build_project_evolution_binding(
    config,
    *,
    workspace_root: Path,
    prompt_source,
    provider,
    config_path: Path | None = None,
    observability=None,
) -> ProjectEvolutionBinding | None

runner.bind_maintenance(coordinator, *, memory=None, evolution=None) -> None
```

工厂关闭时返回 None，不启动后台工作；开启时创建服务绑定。`runner.bind_maintenance()` 在前台工作前执行，每个 runner 只能绑定一次；memory 服务必须就是该 runner 实际使用的对象，namespace 与 write_namespace 一致，Evolution workspace 与 runner workspace 一致。自动记忆或项目经验已开启但未绑定时，前台运行会报告 `IrisConfigError`。

| 协调器方法 | 结果与约定 |
| --- | --- |
| `await prepare()` | 绑定当前事件循环，启动必要监听与调度；runner 准备时会调用 |
| `snapshot()` | `MaintenanceSnapshot`；同步返回最近的不可变控制投影，不取锁、不读盘 |
| `await request_memory_cycle(binding)` | `MemoryCycleResult`；同资源请求合并，完成一个有界周期；跳过普通 idle，仍遵守前台、来源资格、锁和 worker 排空 |
| `await request_project_experience(binding)` | `EvolutionResult`；同项目请求合并，跳过普通 idle，仍检查前台/资格/锁 |
| `await request_revision(binding, request)` | `EvolutionResult`；保存请求并等这一项自己的结算 |
| `await unbind_memory(binding)` | 先关闭借用 runner；排空本资源维护，不关闭 service |
| `await unbind_evolution(binding)` | 同上；持久 pending 请求继续保留 |
| `await aclose()` | 停止派发、取消生成并排空 IO；不关闭宿主注入的服务、reader 和观测资源 |

同一协调器对同一 DB/namespace 或同一项目要求共享同一服务实例。Memory 与 Evolution 各有最多一个作业位置；所有前台工作共用空闲计数。来源 Run 要求 TERMINAL，来源 session 当前不能 WAITING。显式请求可不关联 session；有关联时也检查该 session 的真实等待状态。取消调用方等待不会取消已经保存的共享请求。

`MaintenanceSnapshot` 提供 coordinator_id、revision、foreground_count 和 resources。每个
`ResourceMaintenanceView` 包含稳定的 resource_ref、state、pending_request_id、cycle_id、
next_eligible_at 和 last_result_ref。状态为 idle、waiting_for_idle、waiting_for_foreground、
waiting_for_lock、running 或 closing，只投影现有调度事实。
`maintenance.changed` 只发送对应资源的视图到 resource scope；维护来源采用事实和 trace
共用 cycle_id，不归到任意前台 Run。

`MemoryCycleResult(cycle_id, results, has_more)` 保留原 `GenerationResult.status`。
空资源的 results 为空，不伪造生成成功。has_more 描述剩余积压；失败后的剩余输入仍可为 true，
协调器不会因此进入失败重试循环。宿主需要继续时可再次请求一轮。

从 `iris.evolution` 导入：

```text
RevisionTarget(kind="prompt" | "config", name: str)
EvolutionSession(lifecycle_source_id: str, session_id: str)
RevisionRequest(description: str, targets: tuple[RevisionTarget, ...], session=None)
```

description 和目标名称不可为空白；targets 至少一个，必须在配置开放范围内。无 session 的宿主请求不伪造经历或 Run；有 session 时传入真实 store.source_id 与 session ID。`RevisionEvidence(ref, quote)` 用于经历问题的原文证据，经验整理会检查 ref 和逐字片段确实来自本批材料。

`EvolutionResult` 字段：`stage` 为 experience/revision；`status` 为 updated/no_change/empty/failed/cancelled/conflict；还有 `reason`、`consumed_ranges`、`usage`、`has_more`、`effect`、`revision_id`、`publication_id`、`targets`。`effect` 描述后续采用时机，不是效果提升评分。

`ProjectEvolutionBinding.service` 提供 `await alist_pending_sources()`、`await alist_pending_sessions()`、`await enqueue_revision(request)`、`await maintain_cycle(scope=...)` 和 `await wait_pending_io()` 等领域入口。普通宿主用协调器请求，以保留资格与项目锁边界。材料、请求、进度和发布档案统一保存在 root workspace 的 `.iris/evolution/evolution.db`，由 `EvolutionMaterialStore` 管理，不依赖 Memory 数据库。当前 SQLite schema 为 6，只接受新空库或当前版本，旧版 SQLite 在初始化时拒绝；不读取、迁移或删除旧 JSON 数据。应用应通过 SDK 查询，不直接改写内部表。

高级宿主构造 `EvolutionMaintenanceScope` 时必须显式提供 `experience_sources`，用于 A 的原文选材和剩余材料判断。`allowed_sources`、`allowed_sessions` 继续用于 B 的请求资格，发布恢复也保持原有语义；不能为了限制新原文而缩小整个维护范围。两类领域存储分别持久保存原文准入与剩余有效内容事实，资格仍由宿主读取当前 lifecycle 状态决定，准入标记不会绕过前台或 WAITING 约束。

来源的连续捕获位置、消费位置与已观察终点持久保存；看到终态但捕获区间有缺口时仍需补采，
补齐才作为完整来源参与维护。过滤后的消息原文按来源和消息序号唯一保存，材料按合格来源和
未消费范围有界读取。消费推进水位，已消费且无档案引用的消息才局部回收；已回收的旧范围
不会因重复捕获重新保存正文。来源/session 调度概览与剩余工作检查只读取短状态，不加载材料正文或
请求证据；完整请求先按来源、session 和当前开放目标过滤，再应用数量上限。

| Evolution 历史入口 | 结果 |
| --- | --- |
| `list_publications(*, after=None, limit=50)` / `await alist_publications(...)` | `PublicationPage`；items 为 `PublicationSummary`，按 created_at/ID 升序 |
| `get_publication(publication_id)` / `await aget_publication(...)` | `PublicationHistoryEntry \| None`；None 仅表示 ID 不存在 |
| `list_revision_requests(*, after=None, limit=50)` / `await alist_revision_requests(...)` | `RevisionRequestPage`；items 为 `RevisionRequestSummary`，包括已结算请求 |
| `get_revision_request(revision_id)` / `await aget_revision_request(...)` | `RevisionItem \| None`；完整请求及证据 |

页包含 items/next_cursor，游标为 `EvolutionHistoryCursor(created_at, id)`，limit 为 1–100，由材料存储校验。
列表 SQL 只读取摘要列，不加载正文、材料或证据。`PublicationSummary` 包含 publication_id、
revision_id、created_at、stage、origin、description、targets、status、publication_state、reason、
published_at、settled、detail_status、proposed_revision_id；status 在尚无结果时为 None。
detail_status 为 available/expired；proposed_revision_id 仅在 A 的原子消费实际创建修订请求后赋值。
`RevisionRequestSummary` 包含 id、created_at、
description、targets、origin 和 status；origin 为 host/experience，status 在请求尚未最终结算时为 None。
`RevisionItem` 包含原 id、created_at、description、targets、evidence、origin；
指定请求的结算结果继续由 `store.revision_result(id)` 读取。

`PublicationHistoryEntry` 的字段：

| 字段 | 语义 |
| --- | --- |
| `summary` | `PublicationSummary`，已有记录的摘要始终可查 |
| `detail_status` | 与 summary.detail_status 相同的只读投影 |
| `detail` | available 时为完整 `PublicationRecord`，expired 时为 None |
| `evidence` | 保留的必要 `RevisionEvidence`；过期后不会填回 A 的整批原文 |
| `proposed_issue_summary` | A 有提案但尚未实际创建请求时的 `ProposedIssueSummary(description, targets)`；无提案或已创建请求时为 None |

已创建的提案请求通过 summary.proposed_revision_id 查询；未创建提案的必要 quote 放在 evidence，
不生成虚假的请求关联。旧调用方应改为读取 entry.detail，不能再把 get_publication 的结果直接
当作 PublicationRecord。Memory 的同名详情接口不受这个返回类型变更影响。

每个 workspace 保留 A/B 合计最新十条 `settled=true` 且 publication_state 不是 unconfirmed 的
完整详情，按 `(created_at,id)` 排序。updated/no_change/failed/cancelled/conflict 都占名额；
无待处理材料的 empty 调用不产生档案。未确认、未结算的详情额外保护，不占这十个名额。
过期记录保留摘要、必要证据与结算收据，移除 before/candidate/observed 全文及材料关联。
待处理请求本体与必要 quote 继续保留，其旧失败候选可正常过期。回收只检查本次解除关联触达
的消息，仍待消费或仍被其它完整/未完成档案引用的消息不删除。来源水位不受裁剪影响。
小摘要和请求可能持续增长，SQLite 文件也不保证随数据释放立即缩小；不自动执行 VACUUM。

`PublicationRecord` 详情包含 publication_id、revision_id、stage、created_at、outcome、
publication_state、origin、description、evidence_refs、consumed_ranges、targets，
以及 before_documents、candidate_documents、observed_documents、reason、usage、effect、published_at、settled。
文档保存 path/text，缺失基线用 text=None。A 档案持久保存本批实际选中的有序材料区间引用，
在同一读快照中从 Evolution 自有消息重组完整 materials；evidence_refs 从 `text.strip()` 非空的
record 投影，保持原 ref、完整 quote 与顺序。多次尝试引用同一份消息原文，不依赖外部 lifecycle
回读；完整窗口或恢复保护中的材料消费后仍可查询详情。B 档案只引用 revision_id，
在同一读快照中从不可变请求实体组装 request/evidence_refs，不重复持久化这些字段。
A 的 proposed_issue 仍在发布前保存，供确认后恢复结算；正式请求仅在 A 成功或 no_change
消费时创建。A 的固定 Skill 目标见文档 path。静态正文与状态分表保存，确认时不重写全文，
最终收尾按留存规则裁剪。
`after_documents` 是只读 Python 属性：仅 confirmed 时返回 candidate_documents，否则为空；
它不属于持久化字段，也不出现在 `model_dump()` 中。宿主可用此属性展示已确认正文，
或根据 publication_state 和 candidate_documents 构造自己的输出。

publication_state 为 not_published、confirmed 或 unconfirmed。只有原文件写入返回成功才记录
confirmed/published_at；no_change、conflict、failed 不冒充 updated。settled 表示本次尝试的收尾已完成，
失败或冲突的原请求仍可保持待处理，最终结算以 `store.revision_result(id)` 为准。
成功/no_change 时，材料消费或请求结算、无引用正文回收、档案收尾和本次留存裁剪在一个 SQLite
事务中完成，consumed_ranges 随材料消费提交。确定结束的失败/取消/冲突同样完成尝试收尾与
窗口裁剪，但不消费失败材料或结算仍待处理的请求。持久确认是此前独立的数据库提交，
确认成功而结算失败仍可恢复。已确认写入的结算重试不重跑模型或文件修改；旧 owner 重试已
结算记录时直接取得原持久结果，不撤销 settled、不重复消费，也不恢复 expired 详情。
实际目标文件写入与 SQLite 确认仍是两个操作，不构成跨资源事务。
重启后无法确认的发布保留 unconfirmed，返回 `publication_unconfirmed`，并在 observed_documents
保存读到的当前正文；不因正文等于候选就推断过去成功。未确认发布期间不自动重放项目修改。
该失败结果的 error_code 为 `publication_unconfirmed`。它阻止其他维护请求时，协调器以
`IrisEvolutionError` 结束这些等待，保留已保存的请求，并停止自动调度同一未确认记录。

## Goal SDK

`manager = SessionManager(runner, session_id)` 在启用 Goal 时提供 `manager.goal: GoalSession`，关闭时为 None。推荐用这个会话入口，让目标控制与普通输入共享同一准入 owner。

### GoalSession 方法

以下都是异步方法；除 `get()` 返回 GoalView，其余返回 GoalControlResult。

```text
await goal.create(objective: str, *, max_rounds=None, run_options=None)
await goal.get()
await goal.edit(*, objective=None, max_rounds=None, run_options=None)
await goal.pause(*, reason: str)
await goal.complete(*, reason: str)
await goal.resume(*, expected_activation_id: str | None = None)
await goal.clear()
```

- `create` 保存并允许调度，不等待模型完成；未完成的当前目标不能被覆盖，先 clear。省略 max_rounds 使用配置默认，省略 run_options 使用 `AgentRunOptions()`。
- `get` 完全只读，不启动运行、恢复或补结算。
- `edit` 至少提供一个字段，保留已用轮数并暂停。max_rounds 是总上限，不能小于 rounds_started。
- `pause`/`complete` 要求非空原因，停止后续推进但不取消已准入 Run。立即中断用 manager.interrupt。
- `resume` 优先接手已有 Run；需要接管脱离本进程的 ACTIVE 执行时显式提供 activation ID；WAITING 仍须 typed HITL response。
- `clear` 取消当前选择，保留历史 Goal/绑定/结果。completed 目标不能再编辑或恢复为进行中。

`GoalControlResult` 有 `view` 与 `disposition`，后者可为：scheduled、admitted、running、waiting、needs_recovery、occupied、stopped。它描述本次控制达到的阶段，不能作为最终目标完成证明。

### 视图与事件

`GoalView` 字段：

| 字段 | 含义 |
| --- | --- |
| `goal: GoalSnapshot \| None` | 当前选中的目标；clear 后可为 None |
| `armed: bool` | 本进程明确允许自动推进，且当前目标 active |
| `run: RunSnapshot \| None` | 当前 session lane 上的 Run |
| `run_goal_id: str \| None` | 该 Run 的目标归属；普通 Run 为 None |
| `interaction` | 当前 Run 的人工交互；可为 None |
| `settlement_pending: bool` | 有终态 Goal Run 尚待结算 |
| `driver_error` | 自动推进的进程错误；可为 None |

`GoalSnapshot` 包含 goal_id/session_id、revision、objective、status、reason、max_rounds、rounds_started、run_options、created_at、updated_at。`reason` 为 `GoalReason(code, text)`；`snapshot.ref` 是 `GoalRef(goal_id, revision)`。

`GoalChanged(session_id, view)` 是会话最新投影通知，可从 `manager.events()` 观察；不是逐操作持久重放日志。`GoalRunBinding` 记录 run_id、goal_id、round_no、admission_revision、settled_at、applied_report_call_id。自动轮准入增加轮数，恢复同一 Run 不增加。

### 模型工具与结算

`get_goal` 无参数。`report_goal` 输入为 `goal_id`、`revision >= 0`、`decision`（complete/blocked/continue）、非空 `reason`；身份来自真实当前 Run。两者都是普通可见工具，保留历史。

报告提交不改变目标、不结束 Run。正常结束后，最新成功提交报告才参与结算；complete/blocked 要求版本仍有效，且报告所在步骤及之后没有其它工作工具、后续用户输入或人工回答。新工作后重新申报；continue 可撤回旧结论。非正常 Run 终态使目标暂停，轮数用尽也暂停。清单是否全部完成不参与 Goal 状态结算。

### 自定义宿主与存储实现者

`iris.goal.GoalService(store, *, config=None, process_state_reader=None, run_options_validator=None)` 是同步领域 API，不拥有调度。只读方法为 `get_current(session_id)`、`get(goal_id)`、`get_view(session_id)`；控制方法如下：

```text
service.create(session_id, objective, *, max_rounds=None, run_options=None)  # GoalSnapshot
service.edit(expected: GoalRef, *, objective=None, max_rounds=None, run_options=None)
service.pause(expected: GoalRef, *, reason: GoalReason)
service.resume(expected: GoalRef)
service.complete(expected: GoalRef, *, reason: GoalReason)
service.clear(session_id, *, expected: GoalRef | None)  # GoalSnapshot | None
service.report(run_id, report: GoalReport)  # 核对并返回报告，不结算
service.settle_run(run_id, *, now)  # GoalSettlement
service.reconcile(session_id)  # tuple[GoalSettlement, ...]
```

edit/pause/resume/complete 返回 GoalSnapshot。mutations 使用 GoalRef 的精确版本；冲突报 `IrisGoalConflictError`，非法目标状态报 `IrisGoalStateError`。`GoalSettlement` 有 binding、goal、goal_changed。直接使用服务只改变领域事实，必须由自定义宿主另外拥有准入与执行；常规应用使用 SessionManager。

`GoalStore` 扩展同一 `LifecycleStore`：get_current_goal/get_goal/get_goal_run/list_unsettled_goal_runs 是读取面；create_goal/update_goal/admit_goal_run/settle_goal_run 是事务面。内置 InMemoryLifecycleStore 与 SQLiteStore 都实现它。目标轮数、Run 创建和绑定要原子准入，终态证据与目标结算要在同一权威存储完成，不能另放一个与 Run 松散同步的数据库。精确 command 类型供存储实现者查阅 [GoalStore](../../src/iris/goal/store.py)。

## Todo SDK 与文件格式

```text
snapshot = await runner.get_todo(session_id: str)  # TodoSnapshot
```

未启用时抛 `IrisTodoError`。session 不必已有文件；读操作不创建目录、缓存或文件。`TodoSnapshot` 有 `path: Path`、`items: tuple[TodoItem, ...]`、`error: str | None`；`TodoItem` 有 content/status，`TodoStatus` 为 pending/in_progress/completed。这些只读类型从 `iris.todo` 导入。

路径固定为 `<workspace>/.iris/todos/<session_id.encode("utf-8").hex()>.md`。文件允许 UTF-8 或 UTF-8 BOM：

```markdown
# 工作清单
- [ ] 尚未开始
- [-] 正在处理
- [x] 已经完成
```

允许空行、最多三个前导空格的 ATX 标题、无缩进单行条目。状态括号后至少一个空格或 tab，正文非空；`X` 也表示完成。普通段落、代码围栏、嵌套条目、条目续行均不属于格式。任一行非法或编码错误，返回空 items 和整份文件诊断；文件不存在返回空 items 且无 error；其它 OS 读取失败抛 IrisTodoError。

runtime 每个获准模型步骤读取一次，按 required contribution 注入。结束自查每 Run 最多安排一次，前提是有未完成项/诊断、有剩余模型步骤、未过期限。checkpoint 的 `todo_reminder_step` 只保存目标步骤编号；Todo 正文以文件为准，不随恢复或历史 fork 复制。新 session 使用新文件；子 Agent 使用自身 session 的文件。

CLI 的 `/todo` 只读，Goal 命令见[CLI 参考](cli.md)。更多取舍见[Goal 与 Todo 设计](../design/goals.md)。

维护本页时优先核对：[Memory 配置与导出](../../src/iris/memory/__init__.py)、[生成模型](../../src/iris/memory/generation_models.py)、[Evolution 配置](../../src/iris/evolution/config.py)、[维护协调器](../../src/iris/harness/maintenance.py)、[Goal 模型](../../src/iris/goal/models.py)、[Todo 解析](../../src/iris/todo/document.py)。
