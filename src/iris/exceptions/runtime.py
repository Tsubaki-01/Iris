"""Runtime 协作式取消控制异常。"""

from .base import IrisError


class IrisCancellationRequestedError(IrisError):
    """Activation 已请求协作式取消时使用的内部控制流异常。"""

    runtime_error_code = "CANCELLATION_REQUESTED"
