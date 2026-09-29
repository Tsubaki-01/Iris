"""命令后端与工具、harness 之间的窄协议。"""

from typing import Protocol

from .models import CommandOutcome, CommandRequest, ExecutionScope, ExecutionStopReceipt


class StopOperation(Protocol):
    """一次停止的两个完成点，避免工具 body 等待自身排空。"""

    async def wait_stopped(self) -> ExecutionStopReceipt:
        """等待物理停止确认，不要求旧工具 body 全部返回。"""
        ...

    async def wait_drained(self) -> ExecutionStopReceipt:
        """等待物理停止与本次旧调用收尾，随后允许新的执行。"""
        ...


class CommandService(Protocol):
    """root 拥有的命令资源；不自行提交 run 或工具历史。"""

    async def prepare(self) -> None:
        """准备一次后端资源，Native 不探测 Docker。"""
        ...

    async def execute(self, scope: ExecutionScope, request: CommandRequest) -> CommandOutcome:
        """执行已解析的请求，返回确认事实或抛出 unknown。"""
        ...

    def stop(self, scope: ExecutionScope) -> StopOperation:
        """立即登记并调度/加入停止，返回可独立等待的操作。"""
        ...

    async def wait_drained(self, receipt: ExecutionStopReceipt) -> None:
        """等待既有停止操作排空，不能据此再次停止新环境。"""
        ...

    async def aclose(self) -> None:
        """清理本服务拥有的资源；失败后允许继续清理。"""
        ...


__all__ = ["CommandService", "StopOperation"]
