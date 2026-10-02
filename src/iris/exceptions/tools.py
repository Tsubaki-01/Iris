"""工具执行、校验与未知结果异常。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import IrisError

if TYPE_CHECKING:
    from ..command.models import CommandStopReceipt


class IrisToolError(IrisError):
    """工具相关错误的基类。"""

    runtime_error_source = "tool"
    runtime_error_code = "PROTOCOL_ERROR"


class IrisToolNotFoundError(IrisToolError):
    """请求调用的工具未找到时抛出。"""


class IrisToolExecutionError(IrisToolError):
    """工具执行失败时抛出。"""


class IrisToolValidationError(IrisToolError):
    """工具参数或状态无效时抛出。"""


class IrisToolOutcomeUnknownError(IrisToolError):
    """已 claim 的工具结果不明，必须由 runtime 结算而不能重放。"""

    runtime_error_code = "TOOL_OUTCOME_UNKNOWN"

    def __init__(
        self,
        message: str,
        *,
        stop_receipt: CommandStopReceipt | None = None,
        **context: Any,
    ) -> None:
        """分开保存执行结果未知的诊断与进程内停止事实。

        Args:
            message (str): 结果无法确认的说明。
            stop_receipt (CommandStopReceipt | None): 已确认的停止事实，不进入通用 context。
            **context (Any): 可用于普通错误详情的诊断字段。
        """
        super().__init__(message, **context)
        self.stop_receipt = stop_receipt
