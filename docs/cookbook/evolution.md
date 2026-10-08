# 使用项目经验与有限修订

Evolution 把真实工作经历整理成项目级 `project-experience` Skill，供后续 Agent 按需加载。还可以显式开放少量 prompt 或配置字段，让维护流程提出并发布修订。本文说明如何启用、查看产物，以及在 Python 宿主里绑定维护资源。

前提：已完成[快速开始](../getting-started/quickstart.md)，理解 [Skill 的发现与加载](skills-subagents.md)。开启这些机制会产生独立模型请求和本地文件写入；生成出文件表示机制完成了一次工作，不代表效果已经提升。

## 先只沉淀项目经验

在 workspace 保存 `agent.yaml`：

```yaml
name: project-worker
model: deepseek/deepseek-flash
system: |
  帮助用户完成项目工作，需要项目惯例时按需加载 Skill。
  区分实际结果与猜测，保留用户明确指出的修正。
permissions:
  workspace: .
skills:
  enabled: true
evolution:
  enabled: true
maintenance:
  idle_seconds: 300
  min_pending_runs: 10
```

```powershell
uv run iris chat agent.yaml
```

完成一些有具体反馈的工作后，保持宿主空闲。自动开始新一批经验整理默认要求同时空闲
300 秒、积累十个已终态且完整捕获有效原文的合格新 Run；同一 Run 多页材料只计一次。
每轮按预算读取获准材料，用一次独立模型请求决定是否更新 `.agents/skills/project-experience/SKILL.md`。
没有新材料时结果为 `empty`；有材料但没有值得改动的经验时可以是 `no_change`。

获准余料分轮继续，重启后不重新凑数，新到的 Run 另行累计。修订请求与发布恢复不等待十个
新 Run；空/全过滤来源可无模型收尾。不足数量可能长期等待，不会因超时自动放行；可以把
min_pending_runs 设为 1，或使用下文的手动整理入口。十个 Run 不等于固定 token 量或费用。

首次生成的 Skill 由新 runner 在构造时发现；已经发现这个 Skill 的 runner 在下一次 `load_skill` 时读取新正文。已有消息中的旧正文保持原样。查看文件、确认 catalog 出现名称、确认模型实际调用加载工具，是三种不同的观察结果。

## 开放明确的修订范围

以下是上面配置的 `evolution` 段替换示例：

```yaml
evolution:
  enabled: true
  prompt_targets: [project_skill_update]
  config_targets: [system]
```

`prompt_targets` 和 `config_targets` 默认都是空集合；只开启 Evolution 不会顺带开放 prompt/config 修改。经验整理发现有原文依据的机制问题时，可以提交开放目标的修订项；宿主也能明确提交一个请求。一次修订只发布一个 prompt 文件，或者在唯一主 YAML 中修改指定叶字段。

发布前会检查候选能否按对应领域契约使用：prompt 用代表变量渲染，配置用原 YAML 路径重新解析。这里验证的是结构与可用性，不包含自动效果评估。配置发布只影响新 runner；已存在 runner 的新会话、follow-up 和 Goal 续跑继续使用构造时配置。

## Python 宿主的完整装配

这个脚本使用上面的 `agent.yaml`，需要先开放 `config_targets: [system]`。它还兼容在同一配置中开启 `memory.enabled` 和 `memory.generation.enabled`：两类维护共享宿主空闲状态，使用各自的服务和工作位置。

在 `agent.yaml` 同目录保存为 `run_evolution.py`：

```python
import asyncio
from pathlib import Path

from iris import init_config
from iris.agents import load_agent_config
from iris.evolution import RevisionRequest, RevisionTarget
from iris.harness import (
    AgentRunner,
    MaintenanceCoordinator,
    MemoryMaintenanceBinding,
    build_project_evolution_binding,
)
from iris.lifecycle import AgentRunRequest
from iris.memory import build_memory_service_from_config, resolve_memory_path
from iris.prompts import PromptSource
from iris.providers import create_provider_client


async def main() -> None:
    init_config()
    config_path = Path("agent.yaml").resolve()
    config = load_agent_config(config_path)
    workspace = (config_path.parent / config.permissions.workspace).resolve()
    prompts = PromptSource.initialize(workspace, config.prompts.root)
    provider = create_provider_client(
        config.to_model_route(),
        api_style=config.model.api_style,
        base_url=config.model.base_url,
        timeout=config.model.timeout,
    )
    memory = build_memory_service_from_config(
        config.memory,
        workspace,
        prompt_source=prompts,
        overview_provider=provider,
        overview_model=config.model.name,
    )
    evolution = build_project_evolution_binding(
        config,
        workspace_root=workspace,
        prompt_source=prompts,
        provider=provider,
        config_path=config_path,
    )
    assert evolution is not None
    runner = AgentRunner.from_config(
        config,
        config_path=config_path,
        provider=provider,
        memory_service=memory,
        prompt_source=prompts,
    )
    coordinator = MaintenanceCoordinator(
        idle_seconds=config.maintenance.idle_seconds,
        min_pending_runs=config.maintenance.min_pending_runs,
    )
    memory_binding = (
        MemoryMaintenanceBinding(
            service=memory,
            database_path=resolve_memory_path(config.memory.path, workspace),
            namespace=config.memory.write_namespace,
        )
        if memory is not None and config.memory.generation.enabled
        else None
    )
    runner.bind_maintenance(coordinator, memory=memory_binding, evolution=evolution)
    try:
        result = await runner.start(
            AgentRunRequest(input="本项目的回答请先给结论，再给支持它的事实。")
        )
        print(result.run.stop_reason)
        experience = await coordinator.request_project_experience(evolution)
        print(experience.status, experience.reason, experience.effect)
        revision = await coordinator.request_revision(
            evolution,
            RevisionRequest(
                description="将 system 调整为先给结论，再说明事实依据，保留现有职责。",
                targets=(RevisionTarget(kind="config", name="system"),),
            ),
        )
        print(revision.status, revision.reason, revision.effect)
    finally:
        await coordinator.aclose()
        await runner.aclose()


asyncio.run(main())
```

```powershell
uv run python run_evolution.py
```

这个例子会把明确的修订请求交给模型，可能修改主 `agent.yaml` 的 `system` 字段；运行后检查打印的 `status`、`effect` 和文件实际差异。模型也可以选择 `no_change`，因此不能承诺固定改写内容。YAML 发布会重新序列化文件，原注释和排版不会保留。

`request_project_experience()` 跳过自动空闲和 Run 数量门槛，仍等待前台退出、合格来源和项目锁；
它只整理经验，不代替显式修订请求。`request_revision()` 同样跳过自动门槛，但等待本次持久请求
自己的结果，不会误把另一次维护完成当成本次完成。有关联 session 时仍遵守其 WAITING 状态。
本轮执行修订或恢复时，不会顺带将等待中的原文准入；新原文仍按经验整理自己的条件处理。

自动等待期间，`coordinator.snapshot()` 的资源视图可显示 waiting_for_materials 和
pending_new_runs/min_pending_runs，例如 7/10。数量未满足时 next_eligible_at 为 None；这些字段
来自最近一次异步检查，读取 snapshot 不会同步查询数据库。

## 查看发布历史

`evolution.service.list_publications()` 返回摘要页；按 publication_id 查询时，返回的是包含
摘要和详情状态的 `PublicationHistoryEntry`。以下片段可放在上例 `main()` 中取得 revision 后：

```python
if revision.publication_id is not None:
    entry = await evolution.service.aget_publication(revision.publication_id)
    if entry is None:
        print("发布记录不存在")
    elif entry.detail is None:
        print("完整详情已过期", entry.summary.status, entry.evidence)
    else:
        print(entry.detail.before_documents)
        print(entry.detail.candidate_documents)
        print(entry.detail.after_documents)
```

每个 workspace 保留经验整理与修订合计最近十次已收尾、非 unconfirmed 尝试的完整详情，
失败、取消和冲突也计数；未确认或未结算的恢复数据额外保留。更旧的记录仍有摘要和必要证据，
但 detail_status 为 expired，detail 为 None。待处理修订请求本身仍保留，不会因为某次失败
候选过期而消失。这里限制的是完整历史数量，不是数据库大小；详见[历史查询契约](../reference/memory-goals.md#维护协调器与-evolution-sdk)。

## 宿主何时可以关闭

一个宿主可以让多个 runner 绑定同一个协调器；同一数据库与 namespace 使用同一记忆服务，同一 workspace 使用同一经验服务。任一前台运行进入时都会撤销正在进行的后台生成；已开始的短 IO 要排空后才释放资源。

结束前先停止或等待前台运行；上例在所有调用返回后关闭协调器，再关闭 runner。协调器停止派发并等待真实 IO 收尾，runner 解除借用关系。协调器不替宿主关闭注入的 lifecycle reader、服务或观测资源。若只移除一项资源，先关闭借用它的 runner，再调用 `unbind_memory()` 或 `unbind_evolution()`。

经验材料、进度和发布历史位于 `.iris/evolution/evolution.db`；它们用于维护推进，不提供记忆 Search/Fetch。模型读回的记忆和 Skill 正文不会再次被复制为新事实，后续真实反馈仍能成为材料。对外使用的经验产物是 Skill，修订产物是明确开放的文件。历史列表返回摘要，查看正文时按 ID 调用详情接口，见[历史查询契约](../reference/memory-goals.md#维护协调器与-evolution-sdk)。

下一步：阅读[经验与修订设计](../design/evolution.md)，理解独立维护与采用时机；配置、请求和结果的完整规则见[长期能力参考](../reference/memory-goals.md)。

实现入口：[装配函数](../../src/iris/harness/evolution.py)、[维护协调器](../../src/iris/harness/maintenance.py)、[经验服务](../../src/iris/evolution/service.py)、[修订发布](../../src/iris/evolution/revision.py)。
