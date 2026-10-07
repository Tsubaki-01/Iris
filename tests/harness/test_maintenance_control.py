"""维护控制快照与主动 Memory 周期共享原有资源调度。"""

import asyncio
from pathlib import Path

import pytest
from filelock import FileLock

from iris.harness import MaintenanceChanged, MaintenanceCoordinator, MemoryMaintenanceBinding
from iris.memory import MemoryObserveInput
from iris.message import LLMRequest, LLMResponse
from iris.observability.facts import SourceAdopted
from iris.store import InMemoryLifecycleStore
from iris.streaming.projection import project_live_fact

from .fakes import RecordingPublisher, StaticProvider, text_response
from .test_maintenance_coordinator import memory_service
from .test_session_manager import _wait_until


@pytest.mark.asyncio
async def test_manual_memory_requests_wait_for_foreground_and_share_one_cycle(
    tmp_path: Path,
) -> None:
    """并发请求共享一轮，跳过 idle 但不能越过前台；状态只投影原 owner。"""
    entered, release = asyncio.Event(), asyncio.Event()

    class Provider(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            entered.set()
            await release.wait()
            return text_response('{"observations": []}')

    provider = Provider()
    service = memory_service(tmp_path / "memory.db", provider)
    service.observe(MemoryObserveInput(text="材料"))
    binding = MemoryMaintenanceBinding(
        service=service, database_path=tmp_path / "memory.db", namespace="project"
    )
    publisher = RecordingPublisher()
    coordinator = MaintenanceCoordinator(idle_seconds=3600, live_publisher=publisher)
    coordinator._attach(binding, InMemoryLifecycleStore())
    coordinator._foreground_enter()
    first = asyncio.create_task(coordinator.request_memory_cycle(binding))
    second = asyncio.create_task(coordinator.request_memory_cycle(binding))
    await _wait_until(lambda: coordinator.snapshot().resources[0].pending_request_id is not None)
    waiting = coordinator.snapshot()
    assert waiting.foreground_count == 1
    assert waiting.resources[0].state == "waiting_for_foreground"
    assert not entered.is_set()
    coordinator._foreground_exit()
    await asyncio.wait_for(entered.wait(), 1)
    running = coordinator.snapshot()
    assert running.resources[0].state == "running"
    assert running.resources[0].cycle_id is not None
    release.set()
    a, b = await asyncio.gather(first, second)
    assert a is b
    assert a.cycle_id == running.resources[0].cycle_id
    assert [result.stage for result in a.results] == ["flush"]
    assert len(provider.requests) == 1 and not a.has_more
    assert coordinator.snapshot().revision > waiting.revision
    assert coordinator.snapshot().resources[0].pending_request_id is None
    assert publisher.facts
    adoptions = [fact for fact in publisher.facts if isinstance(fact, SourceAdopted)]
    assert adoptions and all(fact.maintenance_cycle_id == a.cycle_id for fact in adoptions)
    assert all(fact.run_id is None for fact in adoptions)
    changed = next(fact for fact in publisher.facts if isinstance(fact, MaintenanceChanged))
    projected = project_live_fact(changed)
    assert len(projected) == 1 and projected[0].scope == "resource"
    assert projected[0].run_id is None and projected[0].session_id is None
    assert coordinator.snapshot() is coordinator.snapshot()
    await coordinator.aclose()


@pytest.mark.asyncio
async def test_cancelled_request_waiter_does_not_cancel_owned_memory_work(tmp_path: Path) -> None:
    """浏览器等待者断开不取消已接纳的周期；空资源不伪造生成结果。"""
    service = memory_service(tmp_path / "memory.db")
    binding = MemoryMaintenanceBinding(
        service=service, database_path=tmp_path / "memory.db", namespace="project"
    )
    coordinator = MaintenanceCoordinator(idle_seconds=3600)
    coordinator._attach(binding, InMemoryLifecycleStore())
    coordinator._foreground_enter()
    waiter = asyncio.create_task(coordinator.request_memory_cycle(binding))
    await _wait_until(lambda: coordinator.snapshot().resources[0].pending_request_id is not None)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    next_waiter = asyncio.create_task(coordinator.request_memory_cycle(binding))
    coordinator._foreground_exit()
    result = await asyncio.wait_for(next_waiter, 1)
    assert result.results == () and not result.has_more
    await coordinator.aclose()


@pytest.mark.asyncio
async def test_manual_memory_cycle_obeys_resource_lock(tmp_path: Path) -> None:
    """主动请求只能绕过 idle，跨进程资源锁依然决定 admission。"""
    service = memory_service(tmp_path / "memory.db")
    binding = MemoryMaintenanceBinding(
        service=service, database_path=tmp_path / "memory.db", namespace="project"
    )
    coordinator = MaintenanceCoordinator(idle_seconds=0)
    resource, _ = coordinator._attach(binding, InMemoryLifecycleStore())
    with FileLock(resource.lock_path, timeout=0):
        request = asyncio.create_task(coordinator.request_memory_cycle(binding))
        await _wait_until(lambda: coordinator.snapshot().resources[0].state == "waiting_for_lock")
        assert not request.done()
        assert coordinator.snapshot().resources[0].cycle_id is None
    result = await asyncio.wait_for(request, 2)
    assert result.results == ()
    await coordinator.aclose()


@pytest.mark.asyncio
async def test_failed_memory_cycle_reports_remaining_work_without_retry_loop(
    tmp_path: Path,
) -> None:
    """失败保留真实状态和积压，不自动无限重试，也不声称原文已消费。"""
    provider = StaticProvider(text_response("invalid-json"))
    service = memory_service(tmp_path / "memory.db", provider)
    service.observe(MemoryObserveInput(text="仍需处理"))
    binding = MemoryMaintenanceBinding(
        service=service, database_path=tmp_path / "memory.db", namespace="project"
    )
    coordinator = MaintenanceCoordinator(idle_seconds=0)
    coordinator._attach(binding, InMemoryLifecycleStore())
    result = await coordinator.request_memory_cycle(binding)
    assert [item.status for item in result.results] == ["failed"]
    assert result.has_more
    assert coordinator.snapshot().resources[0].last_result_ref == result.results[-1].id
    assert coordinator.snapshot().resources[0].state == "idle"
    assert len(provider.requests) == 1
    await coordinator.aclose()
