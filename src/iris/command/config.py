"""命令环境的原始配置边界；解析配置不连接执行环境。"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..sandbox import DockerConfig
from .models import CommandMode


class CommandConfig(BaseModel):
    """root 命令模式与默认期限，child 借用已选定的配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: CommandMode = CommandMode.NATIVE
    timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    docker: DockerConfig | None = None

    @model_validator(mode="after")
    def _mode_configuration(self) -> Self:
        """在唯一解析边界完成模式与可选配置的交叉约束。"""
        if self.mode is CommandMode.NATIVE and "docker" in self.model_fields_set:
            raise ValueError("native 模式不能声明 docker 配置")
        if self.mode is CommandMode.DOCKER and self.docker is None:
            object.__setattr__(self, "docker", DockerConfig())
        return self


__all__ = ["CommandConfig"]
