"""自动上下文压缩的输入预算与摘要配置。"""

from __future__ import annotations

from math import ceil, floor

from pydantic import BaseModel, ConfigDict, Field


class CompactionConfig(BaseModel):
    """压缩配置；输入预算已扣除模型输出预留。

    Attributes:
        input_budget_tokens (int): 完整模型请求的可用输入预算。
        keep_recent_ratio (float): 近期原文相对输入预算的保留目标比例。
        summary_ratio (float): 摘要生成上限相对输入预算的比例。
        timeout_seconds (float): 一次完整压缩操作的超时秒数。
    """

    input_budget_tokens: int = Field(default=96000, gt=0)
    keep_recent_ratio: float = Field(default=0.15, gt=0, lt=1, allow_inf_nan=False)
    summary_ratio: float = Field(default=0.05, gt=0, lt=1, allow_inf_nan=False)
    timeout_seconds: float = Field(default=300, gt=0, allow_inf_nan=False)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def trigger_tokens(self) -> int:
        """返回固定 80% 的触发与压缩后验收额度。"""
        return floor(self.input_budget_tokens * 0.8)

    @property
    def keep_recent_tokens(self) -> int:
        """返回近期原文的软保留目标。"""
        return ceil(self.input_budget_tokens * self.keep_recent_ratio)

    @property
    def summary_tokens(self) -> int:
        """返回摘要请求的最大输出额度。"""
        return ceil(self.input_budget_tokens * self.summary_ratio)


__all__ = ["CompactionConfig"]
