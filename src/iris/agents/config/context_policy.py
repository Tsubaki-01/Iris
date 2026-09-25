"""可回读上下文策略的配置入口。"""

from pydantic import BaseModel, ConfigDict, Field


class ContextPolicyConfig(BaseModel):
    """控制当前会话上下文的工具回读能力。"""

    enabled: bool = Field(default=True, strict=True)
    model_config = ConfigDict(extra="forbid", frozen=True)
