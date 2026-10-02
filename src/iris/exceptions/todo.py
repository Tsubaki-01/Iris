"""Todo 查询、文件读取与提示构造的领域异常。"""

from .base import IrisError


class IrisTodoError(IrisError):
    """Todo 操作无法完成。"""

    runtime_error_code = "TODO_ERROR"


__all__ = ["IrisTodoError"]
