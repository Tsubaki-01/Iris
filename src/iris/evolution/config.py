"""项目经验学习的纯声明配置，不装配服务或读取文件。"""

from pydantic import BaseModel, ConfigDict, Field


class EvolutionConfig(BaseModel):
    """A 阶段的开关、策略文件与独立请求预算。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    policy_skill: str | None = Field(default=None, min_length=1)
    skill_max_chars: int = Field(default=8000, gt=0)
    input_budget_tokens: int = Field(default=32000, gt=0)
    output_budget_tokens: int = Field(default=8000, gt=0)
