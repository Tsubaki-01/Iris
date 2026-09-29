"""命令执行契约；导入此包不加载可选 Docker 驱动。"""

from .config import DockerConfig, ExecutionConfig
from .models import (
    CommandEnvironment,
    CommandOutcome,
    CommandRequest,
    CommandStatus,
    CommandStopSlot,
    ExecutionMode,
    ExecutionScope,
    ExecutionStopReceipt,
)
from .service import CommandService, ExecutionBinding, StopOperation

__all__ = [
    "CommandEnvironment",
    "CommandOutcome",
    "CommandRequest",
    "CommandService",
    "CommandStatus",
    "CommandStopSlot",
    "DockerConfig",
    "ExecutionBinding",
    "ExecutionConfig",
    "ExecutionMode",
    "ExecutionScope",
    "ExecutionStopReceipt",
    "StopOperation",
]
