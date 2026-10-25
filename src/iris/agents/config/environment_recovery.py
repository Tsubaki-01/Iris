"""显式环境恢复的配置声明。"""

from pydantic import BaseModel, ConfigDict


class EnvironmentRecoveryConfig(BaseModel):
    """控制 root 是否接纳环境保存与恢复操作。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False


__all__ = ["EnvironmentRecoveryConfig"]
