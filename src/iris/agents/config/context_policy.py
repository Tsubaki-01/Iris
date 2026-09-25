"""可回读上下文策略的配置入口。"""

from pydantic import BaseModel, ConfigDict, Field


class ContextPolicyConfig(BaseModel):
    """控制当前会话上下文的工具回读能力。"""

    enabled: bool = Field(default=True, strict=True)
    preserve_recent_tool_groups: int = Field(default=2, ge=0)
    old_result_preview_chars: int = Field(default=512, ge=0)
    model_config = ConfigDict(extra="forbid", frozen=True)
