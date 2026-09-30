"""Goal 的声明式开关与新目标默认轮数。"""

from pydantic import BaseModel, ConfigDict, Field


class GoalConfig(BaseModel):
    """配置目标能力是否启用及创建时的默认轮数上限。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    max_rounds: int = Field(default=20, gt=0)


__all__ = ["GoalConfig"]
