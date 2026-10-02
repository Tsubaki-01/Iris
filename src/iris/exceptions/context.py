"""上下文构建与压缩异常。"""

from typing import Any

from .base import IrisError


class IrisContextError(IrisError, ValueError):
    """Context System 相关错误的基类。"""

    runtime_error_source = "context"
    runtime_error_code = "CONTEXT_ERROR"


class IrisContextCompactionError(IrisContextError):
    """摘要输入不可容纳、生成结果不可接受或整次操作超时。"""

    def __init__(self, message: str, *, code: str, **context: Any) -> None:
        super().__init__(message, **context)
        self.code = code

    @property
    def runtime_code(self) -> str:
        """返回压缩失败的稳定错误分类。"""
        return self.code
