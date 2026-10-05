"""按宿主选定的主 Agent 配置构造项目学习资源。"""

import json
from pathlib import Path
from typing import Any, cast

from ..agents import AgentConfig, AgentSkillsConfig, parse_agent_config
from ..evolution.materials import EvolutionMaterialStore
from ..evolution.revision import ConfigTarget, PromptTarget
from ..evolution.service import EvolutionService, project_skill_prompt_description
from ..exceptions import IrisConfigError
from ..memory import generation_prompt_descriptions, overview_prompt_description
from ..prompts import PromptSource
from ..providers import CompletionProvider
from ..runtime import compaction_prompt_descriptions
from ..skill import resolve_skills_root
from .maintenance import ProjectEvolutionBinding

_FIELD_DESCRIPTIONS = {
    "context_policy.preserve_recent_tool_groups": "近期工具组保留数；context policy 开启时生效。",
    "context_policy.old_result_preview_chars": "旧结果预览字符数；context policy 开启时生效。",
    "compaction.input_budget_tokens": "请求输入额度，已扣除输出预留；不能超出模型能力。",
    "compaction.keep_recent_ratio": "近期原文相对输入额度的软保留比例。",
    "compaction.summary_ratio": "摘要输出上限相对输入额度的比例。",
    "todo.enabled": "是否投影会话 Todo；开启要求 context_policy.enabled，不能代改其它字段。",
    "system": "只修改已采用简单模式的非空文本，不切换外部 context 模式。",
}


def _config_descriptions(config: AgentConfig) -> dict[str, str]:
    """字段解释与实际模型 JSON Schema 一起交给候选，不复制数值约束。"""
    schema = AgentConfig.model_json_schema()
    descriptions: dict[str, str] = {}
    for name in config.evolution.config_targets:
        section, *fields = name.split(".")
        field_schema = schema["properties"][section]
        if fields:
            definition = field_schema["$ref"].rsplit("/", 1)[1]
            field_schema = schema["$defs"][definition]["properties"][fields[0]]
        descriptions[name] = (
            f"{_FIELD_DESCRIPTIONS[name]}\n配置字段 schema："
            f"{json.dumps(field_schema, ensure_ascii=False)}"
        )
    return descriptions


def build_project_evolution_binding(
    config: AgentConfig,
    *,
    workspace_root: Path,
    prompt_source: PromptSource,
    provider: CompletionProvider,
    config_path: Path | None = None,
) -> ProjectEvolutionBinding | None:
    """构造宿主持有的项目学习服务，不启动后台任务或业务 Run。

    Args:
        config: 宿主选定的主配置，其 Skill 依赖已在配置边界校验。
        workspace_root: 已解析的 root workspace。
        prompt_source: 宿主初始化后共享给实际消费者的来源。
        provider: 主配置已经解析的模型连接，也可由宿主明确注入。
        config_path: 开放 config 修订时必须显式指定的主 YAML 路径。

    Returns:
        启用时返回待绑定资源；关闭时不创建项目学习存储。
    """
    if not config.evolution.enabled:
        return None
    if config.evolution.config_targets and config_path is None:
        raise IrisConfigError("开放 config_targets 需要显式 config_path")
    descriptions = {
        **generation_prompt_descriptions(),
        "memory_overview": overview_prompt_description(),
        **compaction_prompt_descriptions(),
        "project_skill_update": project_skill_prompt_description(),
    }
    prompt_targets = tuple(
        PromptTarget(name, *descriptions[name]) for name in config.evolution.prompt_targets
    )
    config_target = None
    if config.evolution.config_targets:
        path = cast(Path, config_path).resolve()

        def validate(raw: dict[str, Any]) -> None:
            parse_agent_config(raw, config_path=path)

        config_target = ConfigTarget(path, validate, _config_descriptions(config))
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
            prompt_targets=prompt_targets,
            config_target=config_target,
        ),
    )
