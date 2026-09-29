"""命令执行的进程内事实；不拥有工具、历史或 lifecycle 状态。"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ..exceptions import IrisCommandCleanupError


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
class CommandRequest:
    """工具边界已解析的命令、宿主工作目录与最终前台期限。"""

    call_id: str
    command: str
    cwd: Path
    timeout_seconds: float


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


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    """前台执行事实；取消及连带中断均不伪造退出码。"""

    mode: CommandMode
    status: CommandStatus
    exit_code: int | None
    stdout: str
    stderr: str
    output_truncated: bool
    duration_seconds: float
    cwd: str
    stop_receipt: CommandStopReceipt | None = None


__all__ = [
    "CommandEnvironment",
    "CommandOutcome",
    "CommandRequest",
    "CommandStatus",
    "CommandStopSlot",
    "CommandMode",
    "CommandScope",
    "CommandStopReceipt",
]
