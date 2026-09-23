"""后台记忆维护在前台执行之外串行推进，取消与关闭遵守真实任务边界。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import TypeVar

import pytest

from iris.harness._memory_maintenance import MemoryMaintenance
from iris.store import InMemoryLifecycleStore

ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class GenerationState:
    """维护调度所需的当前工作量。"""

    pending_episodes: int = 0
    pending_observations: int = 0
    pending_changes: int = 0
    item_revision: int = 0
    projection_revision: int | None = 0
    overview_revision: int | None = 0


class MaintenanceService:
    """以同步点控制维护，不依赖真实模型或计时长等待。"""

    def __init__(self) -> None:
        self.generation_config = SimpleNamespace(idle_seconds=0.01, dream_input_budget_tokens=32000)
        self.store = SimpleNamespace(
            list_capture_sources=lambda *args: [], retry_blocked=lambda *args, **kwargs: 0
        )
        self.state = GenerationState(pending_episodes=1)
        self.mirror = SimpleNamespace(rebuild_from_store=self.rebuild_projection)
        self.listeners: list[Callable[[str], None]] = []
        self.calls: list[str] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.io_release = asyncio.Event()
        self.io_release.set()

    def add_change_listener(self, callback: Callable[[str], None]) -> None:
        self.listeners.append(callback)

    def remove_change_listener(self, callback: Callable[[str], None]) -> None:
        self.listeners.remove(callback)

    async def ageneration_state(self, namespace: str) -> GenerationState:
        return self.state

    def rebuild_projection(self, store: object, namespace: str) -> None:
        self.calls.append("projection")
        self.state = replace(self.state, projection_revision=self.state.item_revision)

    async def flush(self, namespace: str) -> SimpleNamespace:
        self.calls.append("flush")
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        self.state = GenerationState(pending_observations=1)
        return SimpleNamespace(status="completed")

    async def dream(self, namespace: str) -> SimpleNamespace:
        self.calls.append("dream")
        self.state = GenerationState(item_revision=1)
        return SimpleNamespace(status="completed")

    async def aload_overviews(self, namespaces: tuple[str, ...]) -> tuple[SimpleNamespace, ...]:
        return (SimpleNamespace(source_revision=self.state.overview_revision),)

    async def refresh_overview(self, namespace: str) -> None:
        self.calls.append("overview")
        self.state = GenerationState(item_revision=1, projection_revision=1, overview_revision=1)

    async def wait_pending_io(self) -> None:
        await self.io_release.wait()

    async def run_async_io(self, operation: Callable[[], ResultT]) -> ResultT:
        return operation()


@pytest.mark.asyncio
async def test_foreground_blocks_idle_and_cancels_generation_without_waiting() -> None:
    """持续前台即使超过 idle 也不生成；新入场立即打断后台模型。"""
    service = MaintenanceService()
    maintenance = MemoryMaintenance(
        service=service, namespace="project", lifecycle_store=InMemoryLifecycleStore()
    )
    maintenance.foreground_enter()
    await maintenance.prepare()
    await asyncio.sleep(0.03)
    assert service.calls == []
    maintenance.foreground_exit()
    await asyncio.wait_for(service.started.wait(), 1)
    maintenance.foreground_enter()
    await asyncio.wait_for(service.cancelled.wait(), 1)
    assert service.calls == ["flush"]
    service.release.set()
    maintenance.foreground_exit()
    await asyncio.sleep(0.06)
    assert service.calls == ["flush", "flush", "dream", "projection", "overview"]
    await maintenance.aclose()


@pytest.mark.asyncio
async def test_consumed_generation_repairs_projection_without_repeating_model_work() -> None:
    """知识已提交且输入已消费时只补投影，失败等待新活动再重试。"""
    service = MaintenanceService()
    service.state = GenerationState(item_revision=1, overview_revision=1)
    attempted = asyncio.Event()

    def fail_projection(store: object, namespace: str) -> None:
        service.calls.append("projection-failed")
        attempted.set()
        raise RuntimeError("projection temporarily unavailable")

    service.mirror.rebuild_from_store = fail_projection
    maintenance = MemoryMaintenance(
        service=service, namespace="project", lifecycle_store=InMemoryLifecycleStore()
    )
    await maintenance.prepare()
    await asyncio.wait_for(attempted.wait(), 1)
    await asyncio.sleep(0.04)
    assert service.calls == ["projection-failed"]
    service.mirror.rebuild_from_store = service.rebuild_projection
    service.listeners[0]("project")
    await asyncio.sleep(0.04)
    assert service.calls == ["projection-failed", "projection"]
    assert service.state.projection_revision == 1
    await maintenance.aclose()


@pytest.mark.asyncio
async def test_close_waits_dispatched_io_and_unsubscribes() -> None:
    """取消模型后关闭仍等待已派发数据库操作结束。"""
    service = MaintenanceService()
    maintenance = MemoryMaintenance(
        service=service, namespace="project", lifecycle_store=InMemoryLifecycleStore()
    )
    await maintenance.prepare()
    await asyncio.wait_for(service.started.wait(), 1)
    service.io_release.clear()
    closing = asyncio.create_task(maintenance.aclose())
    await asyncio.wait_for(service.cancelled.wait(), 1)
    assert not closing.done()
    service.io_release.set()
    await closing
    assert service.listeners == []


@pytest.mark.asyncio
async def test_no_work_does_not_generate_and_other_namespace_does_not_wake() -> None:
    """一次状态读取发现无工作后停止，变更通知只唤醒绑定空间。"""
    service = MaintenanceService()
    service.state = GenerationState()
    maintenance = MemoryMaintenance(
        service=service, namespace="project", lifecycle_store=InMemoryLifecycleStore()
    )
    await maintenance.prepare()
    await asyncio.sleep(0.03)
    service.state = GenerationState(pending_episodes=1)
    service.listeners[0]("other")
    await asyncio.sleep(0.03)
    assert service.calls == []
    service.listeners[0]("project")
    await asyncio.wait_for(service.started.wait(), 1)
    await maintenance.aclose()
