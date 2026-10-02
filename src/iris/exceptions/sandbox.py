"""本地隔离环境资源异常。"""

from .base import IrisError


class IrisSandboxError(IrisError):
    """本地隔离环境资源操作失败，不携带命令或运行结算事实。"""

    runtime_error_code = "SANDBOX_ERROR"
