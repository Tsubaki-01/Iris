"""本地隔离环境配置；导入此包不加载可选 Docker 驱动。"""

from .config import DockerConfig

__all__ = ["DockerConfig"]
