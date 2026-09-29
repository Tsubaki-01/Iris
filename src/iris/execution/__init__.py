"""命令执行契约；导入此包不加载可选 Docker 驱动。"""

from .config import DockerConfig, ExecutionConfig
from .models import (
    CommandOutcome,
    CommandRequest,
    CommandStatus,
    CommandStopSlot,
    ExecutionMode,
    ExecutionScope,
    ExecutionStopReceipt,
)
from .service import CommandService, StopOperation

__all__ = [
    "CommandOutcome",
    "CommandRequest",
    "CommandService",
    "CommandStatus",
    "CommandStopSlot",
    "DockerConfig",
    "ExecutionConfig",
    "ExecutionMode",
    "ExecutionScope",
    "ExecutionStopReceipt",
    "StopOperation",
]
