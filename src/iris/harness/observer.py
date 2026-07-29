"""Committed lifecycle event 的 best-effort observer 边界。"""

from __future__ import annotations

from typing import Protocol

from ..lifecycle import RunEvent


class RunEventObserver(Protocol):
    """接收已经由权威 store 提交的 ordered run event。"""

    async def on_event(self, event: RunEvent) -> None:
        """处理一个不可变 committed event。"""


__all__ = ["RunEventObserver"]
