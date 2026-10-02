"""Hook 执行与公开返回协议的领域错误。"""

from .base import IrisError


class IrisHookError(IrisError):
    """Hook 的普通执行错误，按当前事件的既定策略处理。"""

    runtime_error_code = "HOOK_ERROR"


class IrisHookProtocolError(IrisHookError):
    """Hook 返回了当前事件不允许的值。"""

    runtime_error_code = "HOOK_PROTOCOL_ERROR"
