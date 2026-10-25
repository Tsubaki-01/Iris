"""Docker 环境的持久描述，不依赖 Docker driver。"""

from pydantic import BaseModel, ConfigDict

from .config import DockerConfig


class DockerSnapshot(BaseModel):
    """镜像和实际采用的容器创建参数。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    image_ref: str
    config: DockerConfig
    user: str
    working_dir: str
    entrypoint: tuple[str, ...]
    command: tuple[str, ...]
    workspace_writable: bool


__all__ = ["DockerSnapshot"]
