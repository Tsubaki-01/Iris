# `iris.evolution`

项目经验学习把已提交的真实任务经历整理为一个普通 Skill：
`<skills.root>/project-experience/SKILL.md`。Memory 可以关闭；该模块拥有独立材料和消费进度，
不启动业务 Run、不等待 Memory 内部生成，也不修改其它 Skill、prompt 或配置。

## 启用与装配

```yaml
skills:
  enabled: true
evolution:
  enabled: true
  skill_max_chars: 8000
  input_budget_tokens: 32000
  output_budget_tokens: 8000
maintenance:
  idle_seconds: 300
```

`evolution.enabled` 默认关闭，启用要求 `skills.enabled=true`。`policy_skill` 可指定策略文件，
相对 root workspace 解析；省略时每轮明确读取包内
[`self-evolution/SKILL.md`](self-evolution/SKILL.md)。策略指导经验整理方法，不拥有输出 schema、
进度或调度，也不被自动改写。输入/输出预算独立于业务 Run，必须为正数。

宿主先初始化项目 `PromptSource`，再构造并绑定服务；生成依赖沿用主 Agent 的 provider/model，
也可由宿主显式提供。`EvolutionService` 从 `iris.evolution.service` 导入，材料存储从
`iris.evolution.materials` 导入；包顶层只导出轻量配置与模型。服务构造器显式接收
`workspace_root、skill_path、store、provider、model、config、prompt_source`，不自行加载 Agent YAML。

`MaintenanceCoordinator` 负责项目锁、空闲时机、来源资格与取消。服务的
`await maintain_cycle(scope=...)` 只在该锁内执行一轮；`scope.allowed_sources` 限定本次来源，
`scope.check(actual_sources)` 在模型前与发布前重查。宿主主动整理也走同一入口，
可以跳过 idle，但不能跳过前台、WAITING 或项目锁。绑定与关闭见 [harness](../harness/README.md)。

## 材料与一次 A 操作

捕获材料位于 root workspace 的 `.iris/evolution/pending/`。独立、完整发布的块保留
source/run/session、消息半开区间和原始引用；跨进程可以重复捕获同一区间，项目锁内按
已消费范围去重。捕获位置与消费位置分开，正文清理后仍保留来源、封源与进度。

仅终态且捕获完整、其 session 当前不在 WAITING 的来源可生成；缺少 lifecycle reader 的
材料保持 pending。fork 继承前缀、child 内部轨迹，以及作为新事实的 Memory/Skill 读回正文
由捕获侧排除。服务不复制一套生命周期资格，也不读取 Memory 私有表。

每轮重读当前经验并固定 `project_skill_update.j2` 的项目快照与策略 Skill。按完整请求
token 估算选择预算内的消息前缀，最多调用模型一次；未读范围不消费，第一条完整消息也
放不下时报告预算错误。空过滤区间不调用模型。项目模板仅表达可编辑策略，最终请求始终
追加领域固定说明与由响应模型生成的 JSON Schema。

模型只返回 `body` 与 `reason`。`body=null` 表示 no-change；否则返回完整 Markdown 正文，
由代码拼接稳定 name/description/frontmatter。正文受 `skill_max_chars` 限制，完整文件须
不超过 1000 行和 50000 字符，保证普通 `load_skill` 能完整读取。

发布前比较最初读取的 Skill 文件与当前文件；外部修改导致 conflict，保留用户文件与 pending。
成功/no-change 才确认实际处理范围，失败或取消不消费。文件原子发布与进度提交不是多文件
事务；若文件已发布后进度失败，下次在项目锁内读取当前文件重新合并，不承诺 exactly-once。
只保留小型最新步骤记录，不保存模型请求、响应正文、历史版本或调用档案。

`EvolutionResult` 返回 `updated/no_change/empty/failed/cancelled/conflict` 状态、简短原因、
实际消费区间、usage、`has_more` 和生效说明。运行错误与取消记录后继续抛出；协调器按既有
规则等待下一次活动、资格恢复或重启。服务的 `wait_pending_io()` 用于等待已派发短工作真正
结束，协调器在此之前保留本类 worker/锁。

## 普通 Skill 如何采用

首次生成由新 runner 发现，旧 runner 不热刷新 catalog；已经登记的 Skill 下一次
`load_skill` 读取当前正文，旧历史消息保持原样。生成成功表示文件与消费进度已更新，
不表示模型必定使用经验或学习收益已经得到验证。

本阶段只有 A 项目经验整理，不接受 B 的 `prompt_targets/config_targets`，也不构建
历史配置现场、压缩/Memory 生成轨迹或独立评测平台。

## 维护与验证

配置与材料模型见 [config.py](config.py)、[models.py](models.py)，材料持久化见
[materials.py](materials.py)，一次生成与发布见 [service.py](service.py)。
针对性测试为 `tests/evolution/test_config.py`、`test_materials.py`、`test_project_skill.py`；
并发、捕获与关闭由 `tests/harness/test_evolution_maintenance.py` 覆盖。

```powershell
$env:UV_CACHE_DIR = "$PWD\tmp\uv-cache"
uv run pytest tests/evolution -p no:cacheprovider --basetemp="$PWD\tmp\pytest-evolution"
uv run ruff check src/iris/evolution tests/evolution
```
