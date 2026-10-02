"""命令执行与环境清理异常。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .tools import IrisToolError

if TYPE_CHECKING:
    from ..command.models import CommandOutcome


class IrisCommandError(IrisToolError):
    """命令执行环境准备或使用失败。"""

    runtime_error_code = "COMMAND_ERROR"


class IrisCommandCleanupError(IrisCommandError):
    """无法确认命令环境停止或必要收尾完成。"""

    runtime_error_code = "COMMAND_CLEANUP_FAILED"

    def __init__(
        self, message: str, *, command_outcome: CommandOutcome | None = None, **context: Any
    ) -> None:
        """保留已知前台事实，不把它混入清理错误的通用诊断。

        Args:
            message (str): 无法完成清理的说明。
            command_outcome (CommandOutcome | None): 已确认的命令结果，尚不代表环境已清理。
            **context (Any): 小型诊断；明确未启动时可包含 started=False。
        """
        super().__init__(message, **context)
        self.command_outcome = command_outcome
