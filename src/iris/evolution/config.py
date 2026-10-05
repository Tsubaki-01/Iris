"""项目经验学习的纯声明配置，不装配服务或读取文件。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PromptTargetName = Literal[
    "memory_flush",
    "memory_dream",
    "memory_overview",
    "project_skill_update",
    "compaction",
    "compaction_input",
]
ConfigTargetName = Literal[
    "context_policy.preserve_recent_tool_groups",
    "context_policy.old_result_preview_chars",
    "compaction.input_budget_tokens",
    "compaction.keep_recent_ratio",
    "compaction.summary_ratio",
    "todo.enabled",
    "system",
]


class EvolutionConfig(BaseModel):
    """项目经验与有限策略修订的开关、目标、策略和独立请求预算。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    policy_skill: str | None = Field(default=None, min_length=1)
    skill_max_chars: int = Field(default=8000, gt=0)
    input_budget_tokens: int = Field(default=32000, gt=0)
    output_budget_tokens: int = Field(default=8000, gt=0)
    prompt_targets: tuple[PromptTargetName, ...] = ()
    config_targets: tuple[ConfigTargetName, ...] = ()
