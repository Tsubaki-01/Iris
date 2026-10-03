"""Decision 服务请求、期限与响应解析异常。"""

from .provider import IrisProviderError


class IrisDecisionError(IrisProviderError):
    """判断服务失败；沿用 provider 来源，不接管外层取消。"""

    runtime_error_code = "DECISION_ERROR"
