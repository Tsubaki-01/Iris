"""仅供后台维护使用的串行同步执行器，不改变前台 Memory IO 的执行位置。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from threading import Event
from typing import TypeVar

ResultT = TypeVar("ResultT")


class GenerationWorker:
    """维护任务拥有的单线程 worker；取消等待后继续跟踪真实作业。"""

    def __init__(self, *, on_idle: Callable[[], None]) -> None:
        """延迟启动线程，空闲回调在事件循环执行。"""
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="iris-memory")
        self._pending: set[asyncio.Future[object]] = set()
        self._cancelled = Event()
        self._on_idle = on_idle

    @property
    def busy(self) -> bool:
        """是否还有实际运行或已派发的同步工作。"""
        return bool(self._pending)

    @contextmanager
    def bind(self) -> Iterator[None]:
        """仅绑定当前维护 task；调用方等旧作业退出后才开始下一轮。"""
        self._cancelled.clear()
        token = generation_worker.set(self)
        try:
            yield
        finally:
            generation_worker.reset(token)

    def cancel(self) -> None:
        """通知当前同步计算在下一批边界退出，不强杀线程或中断提交。"""
        self._cancelled.set()

    def check_cancelled(self) -> None:
        """由可分批的纯计算在记录之间检查前台让位信号。"""
        if self._cancelled.is_set():
            raise asyncio.CancelledError

    async def run(self, operation: Callable[[], ResultT]) -> ResultT:
        """在专用线程执行完整作业，并隔离等待方的取消。"""
        future = asyncio.get_running_loop().run_in_executor(
            self._executor, copy_context().run, operation
        )
        self._pending.add(future)
        future.add_done_callback(self._finished)
        return await asyncio.shield(future)

    def _finished(self, future: asyncio.Future[object]) -> None:
        """回收真实作业；最后一个结束后允许维护 owner 重新安排空闲计时。"""
        self._pending.discard(future)
        if not future.cancelled():
            future.exception()
        if not self._pending:
            self._on_idle()

    async def aclose(self) -> None:
        """等待实际作业退出后释放线程，不占用默认线程池执行 shutdown。"""
        while self._pending:
            await asyncio.gather(
                *(asyncio.shield(future) for future in tuple(self._pending)),
                return_exceptions=True,
            )
        self._executor.shutdown(wait=False)


generation_worker: ContextVar[GenerationWorker | None] = ContextVar(
    "iris_memory_generation_worker", default=None
)


def check_generation_cancelled() -> None:
    """后台计算在记录边界协作退出，独立 SDK 调用保持原执行语义。"""
    worker = generation_worker.get()
    if worker is not None:
        worker.check_cancelled()
