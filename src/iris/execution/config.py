"""命令环境的原始配置边界；解析配置不连接执行环境。"""

import re
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import ExecutionMode


class DockerConfig(BaseModel):
    """单个 root 共享的本地 Linux 容器配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    image: str = Field(default="python:3.12-slim", min_length=1)
    endpoint: str | None = None
    network: Literal["none", "bridge"] = "none"
    cpus: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    memory_mb: int = Field(default=1024, gt=0)
    pids_limit: int = Field(default=128, gt=0)
    environment: dict[str, str] = Field(default_factory=dict)

    @field_validator("endpoint")
    @classmethod
    def _local_endpoint(cls, value: str | None) -> str | None:
        """仅接受本机 socket/pipe，阻止配置切换到远程 daemon。"""
        if value is None:
            return None
        endpoint = urlsplit(value)
        if (
            value.startswith("unix:///")
            and len(endpoint.path) > 1
            and not endpoint.netloc
            and not endpoint.query
            and not endpoint.fragment
        ) or re.fullmatch(r"npipe:////\./pipe/[^/\\?#]+", value):
            return value
        raise ValueError("endpoint 必须为本地 unix:/// socket 或 npipe:////./pipe/ 管道")


class ExecutionConfig(BaseModel):
    """root 命令模式与默认期限，child 借用已选定的配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: ExecutionMode = ExecutionMode.NATIVE
    timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    docker: DockerConfig | None = None

    @model_validator(mode="after")
    def _mode_configuration(self) -> Self:
        """在唯一解析边界完成模式与可选配置的交叉约束。"""
        if self.mode is ExecutionMode.NATIVE and "docker" in self.model_fields_set:
            raise ValueError("native 模式不能声明 docker 配置")
        if self.mode is ExecutionMode.DOCKER and self.docker is None:
            object.__setattr__(self, "docker", DockerConfig())
        return self


__all__ = ["DockerConfig", "ExecutionConfig"]
