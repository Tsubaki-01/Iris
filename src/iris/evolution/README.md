# `iris.evolution`

项目经验学习把已提交的真实任务经历整理为一个普通 Skill：
`<skills.root>/project-experience/SKILL.md`。Memory 可以关闭；该模块拥有独立材料和消费进度，
A 阶段整理经验，发现具体机制问题时可交给 B 修订项目明确开放的 prompt/config。
两阶段各自至多调用模型一次，不启动业务 Run，也不等待 Memory 内部生成。

## 启用与装配

```yaml
skills:
  enabled: true
evolution:
  enabled: true
  skill_max_chars: 8000
  input_budget_tokens: 32000
  output_budget_tokens: 8000
  prompt_targets: [compaction]
  config_targets: [compaction.summary_ratio]
maintenance:
  idle_seconds: 300
```

`evolution.enabled` 默认关闭，启用要求 `skills.enabled=true`。`policy_skill` 可指定策略文件，
相对 root workspace 解析；省略时每轮明确读取包内
[`self-evolution/SKILL.md`](self-evolution/SKILL.md)。Skill 统一规定承载位置、依据标准和适用范围；
`project_skill_update` prompt 负责经验合并与问题提炼，`evolution_review` prompt 负责当前目标检查
及各类字段的修订步骤。两份 prompt 不再重复整份策略；输出 schema、进度和调度由代码拥有。
策略 Skill 不被自动改写，输入/输出预算独立于业务 Run，必须为正数。
策略正文区分长期项目约定、临时要求和具体机制问题：偶发故障或笼统差评不直接触发参数调整，
明确的持续行为要求可以成为修订依据。修订保留未要求改变的角色、无关规则和适用条件；依据不足、
当前内容已满足要求或必要依赖无法确定时返回 no-change。它指导模型判断，不替代代码的目标、
schema 与发布校验，也不保证模型每次判断都正确。

两个目标列表默认空，此时只整理经验，不自动开放修订。prompt 支持 Memory 的
`memory_flush/memory_dream/memory_overview`、`project_skill_update` 和
`compaction/compaction_input`；config 支持 `context_policy.preserve_recent_tool_groups`、
`context_policy.old_result_preview_chars`、`compaction.input_budget_tokens`、
`compaction.keep_recent_ratio`、`compaction.summary_ratio`、`todo.enabled`、`system`。
声明只开放其中的子集，不支持任意路径或字段；`system` 仅更新已经使用简单模式的正文。

内置策略文件在每轮维护时读取；两个 A/B prompt 的默认内容只在项目初始化缺少对应文件时补齐。
升级内置模板不会覆盖已有项目定制。已有项目需要采用新版默认时，应先保留定制内容，再合并
`project_skill_update.j2` 与 `evolution_review.j2` 的对应修改；不要以删除全部项目模板的方式升级。

宿主先初始化项目 `PromptSource`，再构造并绑定服务；生成依赖沿用主 Agent 的 provider/model，
也可由宿主显式提供。`EvolutionService` 从 `iris.evolution.service` 导入，材料存储从
`iris.evolution.materials` 导入；包顶层只导出轻量配置与模型。服务构造器显式接收
`workspace_root、skill_path、store、provider、model、config、prompt_source`，不自行加载 Agent YAML。
完整宿主通常调用 `build_project_evolution_binding()`，由它绑定 prompt 领域说明及主 YAML 的
唯一解析入口；开放 config 修订时必须传入原 `config_path`。独立服务的 `prompt_targets` 与
`config_target` 同样由宿主绑定，不由模型决定。

工厂和 `EvolutionService` 构造器接受 `observability=...`，借用宿主的
[观测服务](../observability/README.md)。默认关闭且不读取全局配置；工厂只传 raw provider，
实际包装由构造器完成一次。宿主共享服务时，向 runner、Memory、Evolution 和维护协调器
传入同一个对象，并在维护排空、runner 关闭后统一 `await observability.aclose()`。
领域服务不创建或关闭 exporter，也不改变材料、发布和状态的权威来源。

实际 A/B 模型调用分别记录 `evolution_experience` / `evolution_revision` purpose。
在宿主已有 Evolution 维护 cycle 内，阶段结果通过 `iris.maintenance.result` 记录原有
stage、status 和可用 revision_id；`failed` 标记维护失败，`empty/no_change/conflict` 保持非错误。
独立 SDK 调用只保留模型 span 和原返回值，不向任意宿主 span 写维护结果，也不新建 cycle。
A 的异常或取消仍向外传播；B 仍返回原有失败/取消结果。提交后收到取消时，观测保留实际
已提交结果的原 status，不虚构 `committed` 状态，也不把模型成功后的发布失败回写为模型失败。

`MaintenanceCoordinator` 负责项目锁、空闲时机、来源资格与取消。服务的
`await maintain_cycle(scope=...)` 只在该锁内执行一轮；`scope.allowed_sources` 限定本次来源，
`scope.check(actual_sources)` 在模型前与发布前重查。宿主主动整理也走同一入口，
可以跳过 idle，但不能跳过前台、WAITING 或项目锁。绑定与关闭见 [harness](../harness/README.md)。

下面使用前述已开启 evolution 的 YAML 与宿主已有 `CompletionProvider`。一个 runner 的宿主
保留列表第一项即可；两个 runner 共用同一项目绑定与来源，贡献各自 session 的经历：

```python
from pathlib import Path

from iris.agents import load_agent_config
from iris.evolution import RevisionRequest, RevisionTarget
from iris.harness import (
    AgentRunRequest, AgentRunner, MaintenanceCoordinator, build_project_evolution_binding,
)
from iris.prompts import PromptSource
from iris.providers import CompletionProvider


async def run_project(config_path: Path, provider: CompletionProvider) -> None:
    config = load_agent_config(config_path)
    workspace = (config_path.parent / config.permissions.workspace).resolve()
    source = PromptSource.initialize(workspace, config.prompts.root)
    binding = build_project_evolution_binding(
        config, workspace_root=workspace, prompt_source=source,
        provider=provider, config_path=config_path,
    )
    assert binding is not None  # 示例 YAML 已开启 evolution。
    coordinator = MaintenanceCoordinator(idle_seconds=config.maintenance.idle_seconds)
    runners = [
        AgentRunner.from_config(config, config_path=config_path, provider=provider, prompt_source=source)
        for _ in range(2)
    ]
    try:
        for runner in runners:
            runner.bind_maintenance(coordinator, evolution=binding)
        for index, runner in enumerate(runners):
            await runner.start(AgentRunRequest(input="检查项目当前任务", session_id=f"work-{index}"))
        learned = await coordinator.request_project_experience(binding)
        print(learned.stage, learned.status)
        revised = await coordinator.request_revision(binding, RevisionRequest(
            description="检查摘要是否保留了明确的任务约束；当前已满足则保持原样。",
            targets=(RevisionTarget(kind="prompt", name="compaction"),),
        ))
        print(revised.status, revised.effect)
    finally:
        await coordinator.aclose()
        for runner in runners:
            await runner.aclose()
```

显式 B 请求不要求历史失败。可选的 `RevisionRequest.session` 使用
`EvolutionSession` 的 `lifecycle_source_id`、`session_id` 字段指定归属；宿主检查该 reader
和 session 当前是否 WAITING，不虚构 run_id。普通聊天纠正不会直接成为宿主命令，仍在终态
后由 A 理解。不新增 CLI 修订命令，`iris chat` 使用同一维护链。

## 材料与一次 A 操作

材料、请求、消费进度和发布档案统一位于 root workspace 的 `.iris/evolution/evolution.db`。
`EvolutionMaterialStore` 管理 schema 2，只接受新空库或当前版本；旧版 SQLite 在初始化时拒绝，
不读取、迁移或删除旧 JSON，不依赖 Memory 数据库。独立、完整发布的块保留
source/run/session、消息半开区间和原始引用；跨进程可以重复或重叠捕获同一区间，读取时同一
消息只出现一次。捕获位置、消费位置与已观察终点分别持久保存，每次捕获只合并当前来源的
连续区间。先收到带终态的后段但前面仍有缺口时，来源仍未封闭，harness 会继续补采；补齐后
才暴露终态与 outcome。正文清理后仍保留来源、封源与进度，查询不重建全部历史收据。

仅终态且捕获完整、其 session 当前不在 WAITING 的来源可生成；缺少 lifecycle reader 的
材料保持 pending。fork 继承前缀、child 内部轨迹，以及作为新事实的 Memory/Skill 读回正文
由捕获侧排除。服务不复制一套生命周期资格，也不读取 Memory 私有表。

`read_pending(allowed_sources=..., limit=128)` 先筛选合格来源，再按来源和消息顺序读取所需
捕获块，返回完整消息；过滤后 records 为空的消息仍占一个区间。`has_pending_materials()` 与
`has_pending_revisions()` 只检查短状态，供剩余工作判定使用，不加载材料正文或请求证据。

每轮重读当前经验并固定 `project_skill_update.j2` 的项目快照与策略 Skill。按完整请求
token 估算选择预算内的消息前缀，最多调用模型一次；未读范围不消费，第一条完整消息也
放不下时报告预算错误。空过滤区间不调用模型。项目模板仅表达可编辑策略，最终请求始终
追加领域固定说明与由响应模型生成的 JSON Schema。

模型返回 `body`、`reason` 和可选 `issue`。`body=null` 表示 no-change；否则返回完整 Markdown 正文，
由代码拼接稳定 name/description/frontmatter。正文受 `skill_max_chars` 限制，完整文件须
不超过 1000 行和 50000 字符，保证普通 `load_skill` 能完整读取。

`issue` 只保存问题描述、开放目标、真实记录引用和必要原文片段。程序核对引用及片段后绑定
实际来源；坏引用会使整个 A 响应失败，不先发布 Skill。问题与 A 消费进度同次保存，只清理
本次推进来源中已经完整消费的捕获块正文；迟到的已消费重复块不恢复正文副本。
A 可以在 Skill no-change 时产生问题；普通事实缺失或单次失败不强制触发 B。

发布前比较最初读取的 Skill 文件与当前文件；外部修改导致 conflict，保留用户文件与 pending。
成功/no-change 才确认实际处理范围，失败或取消不消费。原发布 owner 先把基线与候选存入
SQLite，再写目标文件并保存 confirmed 与 published_at；after_documents 由已确认候选投影，
不重复保存正文。确认后材料消费或请求结算、材料正文清理及档案收尾在一个 SQLite 事务中提交。
目标文件写入与数据库确认仍不属于同一个事务。若文件已发布而结算失败，
本进程保留实际收据，下一轮在同一项目锁内完成结算，不重新调用模型或重写目标。
已持久确认的发布重启后也只补结算；A 的消费与发布 ID 在同一数据库事务中确认，
重试不重复推进原文位置或重复创建问题。

重启后缺少确认的记录保留 `publication_state=unconfirmed`，返回
`publication_unconfirmed` 并保存 observed_documents；即使当前文件等于候选，也不填
after_documents 或 published_at。该未确认记录阻止自动重放本项目修改，材料继续保留。
失败结果通过 `error_code=publication_unconfirmed` 明示此状态；协调器结束因此受阻的手动等待，
保留持久请求并停止反复自动调度，后续显式调用仍可读取这个未确认事实。
档案的 settled 表示本次尝试收尾已完成；失败或冲突的原请求仍可待处理，最终结果用
store.revision_result(id) 查询。它不是另一份 Run 生命周期。
完整模型请求与响应仍由宿主观测记录，领域档案保存正文、证据及发布事实。

## B 修订与独立结算

每次 cycle 只做一个 A 或 B；通常先处理合格 B，没有合格 B 再做 A。显式
`request_project_experience()` 始终只请求 A，显式 B 按自身请求 ID 等待结果。A 完成后释放项目
锁，下一轮 B 重新获锁并读取问题、目标、策略和领域说明；另一进程已经结算的项不再收费。

B 返回 no-change，或一个 prompt 的完整候选正文，或主 YAML 的一组开放叶字段赋值。
prompt 在同一内存来源替换候选，再以领域代表变量渲染一次；实际固定 schema 仍由消费领域
追加。config 对原 YAML 声明应用赋值，再调用装配绑定的 `AgentConfig` 解析器，保留原路径
基准与未修改字段；不从规范化模型回写整份配置，不为修复候选而改动其它字段。PyYAML
序列化不保证保留注释或排版。

发布前再次检查来源/session 资格和文件基线。失败、取消或 conflict 保留 B，不重跑 A；
成功/no-change 独立结算本请求。A 已清理的正文不会删除问题中保留的必要片段。
重启后若收窄开放目标，旧请求保持 pending；只选择目标仍全部开放的项，不阻塞其他合格 A/B。
请求选择与来源/session 概览读取短调度字段，先判断来源、session 和当前开放目标，再应用
数量上限；只有选中请求才读取完整证据。指定请求 ID 是优先项，该项不存在、已结算或不合格
时仍继续选择普通合格候选，不因队首不合格就忽略后面的请求。
当前磁盘值不等于历史运行采用值；配置与来源采用事实由实际消费者发布，发布档案不能替代采用证明。

`list_publications(after=None, limit=50)` 返回 `PublicationSummary` 页；
`list_revision_requests(after=None, limit=50)` 返回 `RevisionRequestSummary` 页。
列表 SQL 不读取正文和证据；详情通过 `get_publication(publication_id)` 与
`get_revision_request(revision_id)` 获取。四个入口均提供同步及 `a` 前缀的 async 读取。
分页使用 `EvolutionHistoryCursor(created_at, id)`，按原创建时刻和 ID 升序，limit 为 1–100；
确认更新不移动历史位置。请求摘要 status 仅表示最终结算结果，未结算时为 None。
已完成请求的描述/evidence、选中材料与 before/candidate 正文保留在数据库中；静态档案正文
与状态分表，仅首次入库，后续确认和结算只更新状态表。`after_documents` 是只读 Python 属性，不进入
`model_dump()`；confirmed 时等于候选，否则为空。A 的固定 Skill 路径由文档 path 描述，
B 另保留原请求及有限 targets。`EvolutionResult.publication_id` 指向这份档案。

`EvolutionResult` 返回 `updated/no_change/empty/failed/cancelled/conflict` 状态、简短原因、
实际消费区间、usage、`has_more` 和生效说明；`stage` 区分 experience/revision，B 结果另有
`revision_id` 与目标。A 错误与取消记录后抛出；B 返回对应请求的 failed/cancelled 结果，
保持 pending，不误结束其它请求。协调器按既有规则等待外部活动、资格恢复或重启。
服务通过独立的 `BackgroundIO` 实例跟踪已派发短工作，与 Memory 共享等待和取消收尾算法，
不共享作业集合。`wait_pending_io()` 等待实际结束，协调器在此之前保留本类 worker/锁。

## 后续如何采用

首次生成由新 runner 发现，旧 runner 不热刷新 catalog；已经登记的 Skill 下一次
`load_skill` 读取当前正文，旧历史消息保持原样。生成成功表示文件与消费进度已更新，
不表示模型必定使用经验或学习收益已经得到验证。

prompt 在对应的下一次完整操作采用：正在生成的 Memory cycle 保持原快照，下一轮用新正文；
压缩同样在下一次操作采用。config 只有新建 runner 才采用，旧 runner 的新 session、
排队 follow-up 和 Goal 续跑继续使用旧配置；CLI 用户重启 `iris chat` 后采用新 YAML。
后台不会重建或关闭已有 runner，也不回写旧 Memory、摘要或消息。

生成、文件写入和 Memory 产物不会作为新的外部经历自触发；后续真实任务与用户纠正提供
反馈。候选合法、保存成功或单次表现符合预期均不代表经过验证的长期能力提升。

## 维护与验证

配置与材料模型见 [config.py](config.py)、[models.py](models.py)，材料持久化见
[materials.py](materials.py)，A/B 编排见 [service.py](service.py)，有限候选见 [revision.py](revision.py)。
针对性测试为 `tests/evolution/test_config.py`、`test_materials.py`、`test_project_skill.py`；
独立结算与有限候选见 `test_strategy_revision.py`、`test_prompt_revision.py`、`test_config_revision.py`；
并发、捕获与关闭由 `tests/harness/test_evolution_maintenance.py` 覆盖，后续采用由
`tests/harness/test_evolution_integration.py` 覆盖。这些使用受控 provider，不是模型学习收益评测。

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv run pytest tests/evolution -p no:cacheprovider --basetemp="$PWD\tmp\pytest-evolution"
uv run ruff check src/iris/evolution tests/evolution
```
