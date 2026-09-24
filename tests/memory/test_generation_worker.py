"""后台专用 worker 与前台 IO 分离，并跟踪取消后的真实同步作业。"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from iris.harness._memory_maintenance import MemoryMaintenance
from iris.memory import MemoryIOExecutionMode, MemoryService, SQLiteMemoryStore
from iris.memory._generation_worker import GenerationWorker
from iris.memory.generation_models import GenerationState
from iris.store import InMemoryLifecycleStore


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """旧同步读取仍运行时不排新维护；前台 IO 和关闭保持各自生命周期。"""
    service = MemoryService(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        io_execution_mode=MemoryIOExecutionMode.THREAD,
    )
    service.generation_config = service.generation_config.model_copy(update={"idle_seconds": 0})
    maintenance = MemoryMaintenance(
        service=service, namespace="project", lifecycle_store=InMemoryLifecycleStore()
    )
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    original = service.store.generation_state
    calls = 0

    def slow_state(namespace: str) -> GenerationState:
        nonlocal calls
        calls += 1
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        return original(namespace)

    monkeypatch.setattr(service.store, "generation_state", slow_state)
    try:
        await maintenance.prepare()
        await asyncio.wait_for(started.wait(), 2)
        maintenance.foreground_enter()
        task = maintenance._task
        assert task is not None
        await asyncio.gather(task, return_exceptions=True)
        foreground_thread = await asyncio.wait_for(service.run_async_io(threading.get_ident), 1)
        assert foreground_thread != threading.get_ident()
        maintenance.foreground_exit()
        await asyncio.sleep(0)
        assert calls == 1
        assert maintenance._task is None and maintenance._timer is None
        closing = asyncio.create_task(maintenance.aclose())
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        await asyncio.wait_for(closing, 2)
    finally:
        release.set()
        await maintenance.aclose()


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
