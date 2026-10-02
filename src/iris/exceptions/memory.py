"""长期记忆服务与存储异常。"""

from .base import IrisError


class IrisMemoryError(IrisError):
    """记忆子系统错误的基类。"""

    runtime_error_source = "memory"
    runtime_error_code = "MEMORY_ERROR"
