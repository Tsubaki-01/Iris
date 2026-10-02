"""工具结果已知之后，向运行时交接尚未消费的控制事实。"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolExecutionControl:
    """保留原控制异常；命令收据仍由 CommandStopSlot 独占。"""

    error: BaseException


@dataclass(slots=True)
class ToolExecutionControlSlot:
    """同一次调用共享的控制槽，只记录最早观察到的原因。"""

    control: ToolExecutionControl | None = None

    def record(self, error: BaseException) -> None:
        """锁存首因，不让包装器恢复值或后续清理覆盖原控制。"""
        if self.control is None:
            self.control = ToolExecutionControl(error)
