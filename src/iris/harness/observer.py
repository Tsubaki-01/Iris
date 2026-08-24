"""Committed lifecycle event 的 best-effort observer 边界。"""

from __future__ import annotations

from typing import Protocol

from ..lifecycle import RunEvent


class RunEventObserver(Protocol):
    """接收已经由权威 store 提交的 ordered run event。

    同一 observer 的调用由 runner 串行保序，不同 observer 的 lane 并行。单事件超时或普通异常
    仅记录日志并继续后续事件；runner task 被取消时 ``CancelledError`` 仍向上传播。
    """

    async def on_event(self, event: RunEvent) -> None:
        """处理一个不可变 committed event。"""


__all__ = ["RunEventObserver"]
