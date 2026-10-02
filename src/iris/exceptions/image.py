"""图片解码、处理与本地副本保存异常。"""

from .base import IrisError


class IrisImageError(IrisError):
    """图片未能准备为可供模型使用的本地副本。"""

    runtime_error_code = "IMAGE_ERROR"


__all__ = ["IrisImageError"]
