"""可回读上下文策略的配置入口。"""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContextPolicyConfig(BaseModel):
    """控制当前会话上下文的工具回读能力。"""

    enabled: bool = Field(default=True, strict=True)
    preserve_recent_tool_groups: int = Field(default=2, ge=0)
    old_result_preview_chars: int = Field(default=512, ge=0)
    deferred_tools: bool = Field(default=False, strict=True)
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _require_enabled_for_deferred(self) -> "ContextPolicyConfig":
        if self.deferred_tools and not self.enabled:
            raise ValueError("deferred_tools 需要 context_policy.enabled=true")
        return self
