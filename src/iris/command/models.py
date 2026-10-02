"""命令执行的进程内事实；不拥有工具、历史或 lifecycle 状态。"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from ..exceptions import IrisCommandCleanupError

type OutputTruncationReason = Literal[
    "byte_limit", "drain_timeout", "stream_error", "stream_closed"
]


class CommandMode(StrEnum):
    """由 root 实例选定的命令运行环境。"""

    NATIVE = "native"
    DOCKER = "docker"


class CommandStatus(StrEnum):
    """区分真实退出、局部期限和两种中断来源。"""

    EXITED = "exited"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    ENVIRONMENT_INTERRUPTED = "environment_interrupted"


@dataclass(frozen=True, slots=True)
class CommandEnvironment:
    """工具说明与 system 提示共同使用的启动期环境事实。"""

    host_os: str
    mode: CommandMode
    command_os: str
    command_shell: str


@dataclass(frozen=True, slots=True)
class CommandScope:
    """本次调用的归属，不决定容器分配。"""

    run_id: str
    session_id: str


@dataclass(frozen=True, slots=True)
class ShellCommand:
    """按当前环境 shell 语法执行的命令文本。"""

    command: str


@dataclass(frozen=True, slots=True)
class PythonCode:
    """由当前环境 Python 独立执行的完整源码。"""

    code: str


@dataclass(frozen=True, slots=True)
class CommandRequest:
    """已解析的命令、宿主目录、最终期限与可选的一次性 stdin 字节。"""

    call_id: str
    payload: ShellCommand | PythonCode
    cwd: Path
    timeout_seconds: float
    stdin: bytes | None = None


@dataclass(frozen=True, slots=True)
class CommandStopReceipt:
    """所属 live service 的一次已确认停止事实，不包含活动资源。"""

    service_id: str
    stop_id: str


@dataclass(slots=True)
class CommandStopSlot:
    """当前工具因果链共享的可写槽；投影 context 时保留对象 identity。"""

    receipt: CommandStopReceipt | None = None
    cleanup_error: IrisCommandCleanupError | None = None
    status: CommandStatus | None = None

    def record(
        self,
        *,
        status: CommandStatus | None = None,
        receipt: CommandStopReceipt | None = None,
        cleanup_error: IrisCommandCleanupError | None = None,
    ) -> None:
        """归并当前命令事实；后续空值不清除尚未消费的收据或清理失败。"""
        if status is not None:
            self.status = status
        if receipt is not None and self.receipt is None:
            self.receipt = receipt
        if cleanup_error is not None and self.cleanup_error is None:
            self.cleanup_error = cleanup_error


@dataclass(frozen=True, slots=True)
class CommandOutputStats:
    """已采集与已保留的原始字节；不完整采集时不推测后续输出量。"""

    stdout_bytes: int
    stderr_bytes: int
    stdout_retained_bytes: int
    stderr_retained_bytes: int
    truncation_reasons: frozenset[OutputTruncationReason]


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    """前台执行事实；取消及连带中断均不伪造退出码。"""

    mode: CommandMode
    status: CommandStatus
    exit_code: int | None
    stdout: str
    stderr: str
    output_stats: CommandOutputStats
    duration_seconds: float
    cwd: str
    stop_receipt: CommandStopReceipt | None = None

    @property
    def output_truncated(self) -> bool:
        """由输出事实派生是否存在额度截断或不完整采集。"""
        return bool(self.output_stats.truncation_reasons)


__all__ = [
    "CommandEnvironment",
    "CommandOutcome",
    "CommandOutputStats",
    "CommandRequest",
    "CommandStatus",
    "CommandStopSlot",
    "CommandMode",
    "CommandScope",
    "CommandStopReceipt",
    "OutputTruncationReason",
    "PythonCode",
    "ShellCommand",
]
