"""按宿主选定的主 Agent 配置构造项目学习资源。"""

from pathlib import Path
from typing import cast

from ..agents import AgentConfig, AgentSkillsConfig
from ..evolution.materials import EvolutionMaterialStore
from ..evolution.service import EvolutionService
from ..prompts import PromptSource
from ..providers import CompletionProvider
from ..skill import resolve_skills_root
from .maintenance import ProjectEvolutionBinding


def build_project_evolution_binding(
    config: AgentConfig,
    *,
    workspace_root: Path,
    prompt_source: PromptSource,
    provider: CompletionProvider,
) -> ProjectEvolutionBinding | None:
    """构造宿主持有的项目学习服务，不启动后台任务或业务 Run。

    Args:
        config: 宿主选定的主配置，其 Skill 依赖已在配置边界校验。
        workspace_root: 已解析的 root workspace。
        prompt_source: 宿主初始化后共享给实际消费者的来源。
        provider: 主配置已经解析的模型连接，也可由宿主明确注入。

    Returns:
        启用时返回待绑定资源；关闭时不创建项目学习存储。
    """
    if not config.evolution.enabled:
        return None
    skills = cast(AgentSkillsConfig, config.skills)
    skill_root = resolve_skills_root(skills.root, workspace_root=workspace_root)
    return ProjectEvolutionBinding(
        workspace_root=workspace_root,
        service=EvolutionService(
            workspace_root=workspace_root,
            skill_path=skill_root / "project-experience" / "SKILL.md",
            store=EvolutionMaterialStore(workspace_root),
            provider=provider,
            model=config.model.name,
            config=config.evolution,
            prompt_source=prompt_source,
        ),
    )
