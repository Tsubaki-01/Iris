"""后台专用 worker 与前台 IO 分离，并跟踪取消后的真实同步作业。"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from iris.memory import MemoryIOExecutionMode, MemoryService, SQLiteMemoryStore
from iris.utils.generation_worker import GenerationWorker


def test_wait_idle_drains_completed_future_before_its_callback_runs() -> None:
    """已完成作业的回收回调仍在队列时，排空不能占住事件循环。"""
    # 单独进程让旧实现的忙循环按超时失败，而不是卡住整个 pytest 事件循环。
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio
from iris.utils.generation_worker import GenerationWorker

async def main() -> None:
    idle = []
    worker = GenerationWorker(on_idle=lambda: idle.append(True))
    future = asyncio.get_running_loop().create_future()
    worker._pending.add(future)
    future.add_done_callback(worker._finished)
    future.set_result(42)
    await worker.wait_idle()
    assert not worker.busy
    assert idle == [True]
    await worker.aclose()

asyncio.run(main())
""",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_background_worker_does_not_use_saturated_default_executor(tmp_path: Path) -> None:
    """默认池被其他工作占满时，后台作业仍由专用线程执行。"""
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    started = asyncio.Event()
    release = threading.Event()
    service = MemoryService(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        io_execution_mode=MemoryIOExecutionMode.THREAD,
    )
    worker = GenerationWorker(on_idle=lambda: None)

    def occupy_default() -> int:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        return threading.get_ident()

    occupied = loop.run_in_executor(None, occupy_default)
    try:
        await asyncio.wait_for(started.wait(), 2)
        with worker.bind():
            background_thread = await asyncio.wait_for(service.run_async_io(threading.get_ident), 1)
        assert not occupied.done()
    finally:
        release.set()
        default_thread = await occupied
        await worker.aclose()
    assert background_thread != default_thread != threading.get_ident()


@pytest.mark.asyncio
async def test_cancelled_background_read_delays_next_cycle_but_not_foreground_io(
    tmp_path: Path,
) -> None:
    """旧同步读取仍运行时不排新维护；前台 IO 和关闭保持各自生命周期。"""
    service = MemoryService(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        io_execution_mode=MemoryIOExecutionMode.THREAD,
    )
    worker = GenerationWorker(on_idle=lambda: None)
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()

    def slow_state() -> int:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        return threading.get_ident()

    async def read() -> int:
        with worker.bind():
            return await service.run_async_io(slow_state)

    task = asyncio.create_task(read())
    try:
        await asyncio.wait_for(started.wait(), 2)
        worker.cancel()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert worker.busy
        foreground_thread = await asyncio.wait_for(service.run_async_io(threading.get_ident), 1)
        assert foreground_thread != threading.get_ident()
        closing = asyncio.create_task(worker.wait_idle())
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        await asyncio.wait_for(closing, 2)
        assert not worker.busy
        with worker.bind():
            assert await service.run_async_io(lambda: 42) == 42
    finally:
        release.set()
        await worker.aclose()


@pytest.mark.asyncio
async def test_inline_store_keeps_calling_thread_in_background_scope(tmp_path: Path) -> None:
    """后台作用域不会覆盖宿主为自定义 store 选择的 INLINE 契约。"""
    service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"))
    worker = GenerationWorker(on_idle=lambda: None)
    try:
        with worker.bind():
            assert await service.run_async_io(threading.get_ident) == threading.get_ident()
    finally:
        await worker.aclose()


@pytest.mark.asyncio
async def test_cancelled_idle_wait_keeps_tracking_real_work() -> None:
    """取消排空等待不会取消或遗失线程中尚未返回的作业。"""
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    worker = GenerationWorker(on_idle=lambda: None)

    def operation() -> int:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        return 42

    running = asyncio.create_task(worker.run(operation))
    try:
        await asyncio.wait_for(started.wait(), 2)
        draining = asyncio.create_task(worker.wait_idle())
        await asyncio.sleep(0)
        draining.cancel()
        with pytest.raises(asyncio.CancelledError):
            await draining
        assert worker.busy and not running.done()
        release.set()
        assert await running == 42
        await worker.wait_idle()
        assert not worker.busy
    finally:
        release.set()
        await worker.aclose()
