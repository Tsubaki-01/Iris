"""后台同步 IO 的真实作业跟踪、取消后回执与排空。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypeVar

from .generation_worker import generation_worker

ResultT = TypeVar("ResultT")


class BackgroundIO:
    """每个服务独立持有的 IO 作业集合，不拥有线程池或维护调度。"""

    def __init__(self) -> None:
        """创建服务私有的真实作业跟踪集合。"""
        self._pending: set[asyncio.Task[object]] = set()

    async def run(
        self, operation: Callable[[], ResultT], *, complete_on_cancel: bool = False
    ) -> ResultT:
        """借用当前维护 worker 或默认线程池，短提交可在取消后收取回执。"""
        worker = generation_worker.get()
        task = asyncio.create_task(
            asyncio.to_thread(operation) if worker is None else worker.run(operation)
        )
        self._pending.add(task)
        task.add_done_callback(self._finished)
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if not complete_on_cancel or task.cancelled():
                    raise

    def _finished(self, task: asyncio.Task[object]) -> None:
        """回收已完成作业，包括已取消等待者不再接收的异常。"""
        self._pending.discard(task)
        if not task.cancelled():
            task.exception()

    async def wait_pending(self) -> None:
        """等待本服务已派发的同步作业真正结束。"""
        while self._pending:
            await asyncio.gather(*tuple(self._pending), return_exceptions=True)
