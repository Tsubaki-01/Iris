"""Agent 与全局配置异常。"""

from .base import IrisError


class IrisConfigError(IrisError, ValueError):
    """配置出现问题时抛出，例如缺少必需的参数或值无效。"""

    runtime_error_source = "config"
    runtime_error_code = "CONFIG_ERROR"
