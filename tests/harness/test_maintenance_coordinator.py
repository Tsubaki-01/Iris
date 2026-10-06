"""宿主共享维护的资源身份、前台优先与来源资格。"""

from __future__ import annotations

import asyncio
import json
import threading
from contextvars import ContextVar
from pathlib import Path
from time import time_ns

import pytest
from filelock import FileLock, Timeout
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import iris.harness.maintenance as maintenance_module
from iris.exceptions import IrisConfigError, IrisRunStateError
from iris.harness import AgentRunner, MaintenanceCoordinator, MemoryMaintenanceBinding
from iris.lifecycle import AgentRunRequest, LifecycleStore
from iris.memory import MemoryObserveInput, MemoryService, MemoryWriteInput, SQLiteMemoryStore
from iris.memory.generation_models import MemoryGenerationConfig
from iris.memory.mirror import FileMemoryMirror
from iris.memory.service import MemoryIOExecutionMode
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.observability.service import Observability
from iris.prompts import PromptSource
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import ToolCapability, ToolRegistry
from iris.utils.generation_worker import generation_worker

from .fakes import StaticProvider, build_runtime, text_response, tool_response


def memory_service(
    path: Path,
    provider: StaticProvider | None = None,
    *,
    io_mode: MemoryIOExecutionMode = MemoryIOExecutionMode.INLINE,
    observability: Observability | None = None,
) -> MemoryService:
    """创建完整自动生成依赖。"""
    provider = provider or StaticProvider(text_response('{"observations": []}'))
    return MemoryService(
        SQLiteMemoryStore(path),
        mirror=FileMemoryMirror(path.parent / "mirror", workspace_root=path.parent),
        generation_provider=provider,
        generation_model="generation",
        prompt_source=PromptSource.initialize(path.parent),
        overview_provider=provider,
        overview_model="overview",
        io_execution_mode=io_mode,
        observability=observability,
    )


@pytest.mark.asyncio
async def test_memory_worker_resets_ended_trace_but_keeps_business_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observability: tuple[Observability, InMemorySpanExporter],
) -> None:
    observation, exporter = observability
    marker = ContextVar("memory_business_context", default="unset")
    generated = asyncio.Event()
    seen_markers: list[str] = []

    class Generation(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            seen_markers.append(marker.get())
            assert generation_worker.get() is coordinator._worker
            generated.set()
            return text_response('{"observations": []}')

    service = memory_service(
        tmp_path / "memory.db",
        Generation(),
        io_mode=MemoryIOExecutionMode.THREAD,
        observability=observation,
    )
    service.observe(MemoryObserveInput(text="待整理材料"))
    read_sources = service.list_pending_sources

    def read_in_worker(namespace: str) -> object:
        with observation.scope("memory-io"):
            assert generation_worker.get() is coordinator._worker
            seen_markers.append(marker.get())
            return read_sources(namespace)

    monkeypatch.setattr(service, "list_pending_sources", read_in_worker)
    coordinator = MaintenanceCoordinator(idle_seconds=0, observability=observation)
    coordinator._attach(
        MemoryMaintenanceBinding(
            service=service, database_path=tmp_path / "memory.db", namespace="project"
        ),
        InMemoryLifecycleStore(),
    )
    host = observation.start_span("ended-host")
    observation.end_span(host)
    token = marker.set("memory-host-value")
    try:
        try:
            with observation.use_span(host), observation.bind({"iris.run.id": "old-run"}):
                await coordinator.prepare()
                assert trace.get_current_span() is host
        finally:
            marker.reset(token)
        await asyncio.wait_for(generated.wait(), 2)
    finally:
        await coordinator.aclose()
    spans = [span for span in exporter.get_finished_spans() if span.name != "ended-host"]
    assert any(span.name == "memory-io" for span in spans)
    assert any(span.attributes.get("gen_ai.operation.name") == "chat" for span in spans)
    assert all(span.context.trace_id != host.get_span_context().trace_id for span in spans)
    assert all(span.attributes.get("iris.run.id") != "old-run" for span in spans)
    assert seen_markers and set(seen_markers) == {"memory-host-value"}
    cycle = next(span for span in spans if span.name == "iris.maintenance.cycle")
    assert cycle.parent is None
    assert cycle.attributes["iris.maintenance.kind"] == "memory"
    assert cycle.attributes["iris.maintenance.database"] == str(tmp_path / "memory.db")
    assert cycle.attributes["iris.maintenance.namespace"] == "project"
    assert "gen_ai.conversation.id" not in cycle.attributes
    assert all(span.parent.span_id == cycle.context.span_id for span in spans if span is not cycle)


def configured_runner(
    path: Path, service: MemoryService, *, store: LifecycleStore | None = None
) -> AgentRunner:
    """构造启用自动维护但尚未绑定宿主的 root runner。"""
    runtime = build_runtime(path, provider=StaticProvider(text_response()))
    config = runtime.environment.agent_config
    runtime.environment.agent_config = config.model_copy(
        update={
            "memory": config.memory.model_copy(
                update={"enabled": True, "generation": MemoryGenerationConfig(enabled=True)}
            )
        }
    )
    runtime.environment.memory_service = service
    return AgentRunner(runtime=runtime, store=store or InMemoryLifecycleStore())


@pytest.mark.asyncio
@pytest.mark.parametrize("foreground_only", [False, True])
async def test_enabled_runner_requires_explicit_binding_at_first_use(
    tmp_path: Path, foreground_only: bool
) -> None:
    """构造没有隐藏后台 owner；已经准备好的环境也必须检查绑定。"""
    runner = configured_runner(tmp_path, memory_service(tmp_path / "memory.db"))
    coordinator = MaintenanceCoordinator()
    if foreground_only:
        runner.bind_maintenance(coordinator)
    with pytest.raises(IrisConfigError, match="bind_maintenance"):
        await runner.start(AgentRunRequest(input="开始"))
    await runner.aclose()
    await coordinator.aclose()


@pytest.mark.asyncio
async def test_resource_identity_deduplicates_bindings_and_rejects_conflicting_service(
    tmp_path: Path,
) -> None:
    """同库同 namespace 只有一次注册，不能悄悄改变策略来源。"""
    service = memory_service(tmp_path / "memory.db")
    coordinator = MaintenanceCoordinator(idle_seconds=100)
    first = configured_runner(tmp_path, service)
    second = configured_runner(tmp_path, service)
    binding = MemoryMaintenanceBinding(
        service=service, database_path=tmp_path / "memory.db", namespace="project"
    )
    first.bind_maintenance(coordinator, memory=binding)
    second.bind_maintenance(
        coordinator,
        memory=MemoryMaintenanceBinding(
            service=service,
            database_path=tmp_path / "sub" / ".." / "memory.db",
            namespace="project",
        ),
    )
    await first.aprepare()
    await second.aprepare()
    assert len(service._change_listeners) == 1
    conflicting = configured_runner(tmp_path, memory_service(tmp_path / "memory.db"))
    with pytest.raises(IrisConfigError, match="同一 Memory 资源"):
        conflicting.bind_maintenance(
            coordinator,
            memory=MemoryMaintenanceBinding(
                service=conflicting.runtime.environment.memory_service,
                database_path=tmp_path / "memory.db",
                namespace="project",
            ),
        )
    await first.aclose()
    assert len(service._change_listeners) == 1
    await second.start(AgentRunRequest(input="仍能工作"))
    with pytest.raises(IrisRunStateError, match="首次"):
        second.bind_maintenance(coordinator, memory=binding)
    await second.aclose()
    await conflicting.aclose()
    await coordinator.aclose()
    assert service._change_listeners == []


@pytest.mark.asyncio
async def test_waiting_excludes_old_session_sources_after_runner_close(tmp_path: Path) -> None:
    """关闭 runner 保留 reader，A 的旧终态在 A WAITING 时排除，B 可学习。"""
    generated = asyncio.Event()

    class Generation(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            generated.set()
            return text_response('{"observations": []}')

    provider = Generation()
    service = memory_service(tmp_path / "memory.db", provider)
    coordinator = MaintenanceCoordinator(idle_seconds=0)
    first = configured_runner(tmp_path, service)
    second = configured_runner(tmp_path, service)
    first.runtime.environment.provider.responses.extend(
        [tool_response(ToolUseBlock(id="write", name="write"))]
    )
    first.runtime.environment.tool_bridge.tool_executor.registry.register_function(
        lambda: "written", name="write", description="写", capabilities={ToolCapability.WRITE}
    )
    binding = MemoryMaintenanceBinding(
        service=service, database_path=tmp_path / "memory.db", namespace="project"
    )
    first.bind_maintenance(coordinator, memory=binding)
    second.bind_maintenance(coordinator, memory=binding)
    coordinator._foreground_enter()
    await first.start(AgentRunRequest(input="A 的旧材料", session_id="a"))
    waiting = await first.start(AgentRunRequest(input="A 等待回答", session_id="a"))
    assert waiting.pending_interaction is not None
    await first.aclose()
    await second.start(AgentRunRequest(input="B 的完整材料", session_id="b"))
    coordinator._foreground_exit()
    await asyncio.wait_for(generated.wait(), 2)
    await asyncio.sleep(0.05)
    assert len(provider.requests) == 1
    records = json.loads(provider.requests[0].messages[1].text)["records"]
    assert any("B 的完整材料" in record["text"] for record in records)
    assert all("A " not in record["text"] for record in records)
    assert service.generation_state("project").pending_episodes > 0
    await second.aclose()
    await coordinator.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("remove_resource", [False, True])
async def test_real_worker_drain_retains_slot_and_os_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remove_resource: bool,
    observability: tuple[Observability, InMemorySpanExporter],
) -> None:
    """前台立即完成，已派发同步工作仍占锁；关闭不二次取消 drain。"""
    entered, release = threading.Event(), threading.Event()
    observation, exporter = observability
    released_at: list[int] = []

    class ObservedLock(FileLock):
        def release(self, force: bool = False) -> None:
            owned = self.is_locked
            super().release(force=force)
            if owned:
                released_at.append(time_ns())

    monkeypatch.setattr(maintenance_module, "FileLock", ObservedLock)
    service = memory_service(tmp_path / "memory.db", io_mode=MemoryIOExecutionMode.THREAD)
    service.observe(MemoryObserveInput(text="待整理"))
    read_sources = service.list_pending_sources

    def blocked_read(namespace: str) -> object:
        entered.set()
        assert release.wait(5)
        return read_sources(namespace)

    monkeypatch.setattr(service, "list_pending_sources", blocked_read)
    coordinator = MaintenanceCoordinator(idle_seconds=0, observability=observation)
    runner = configured_runner(tmp_path, service)
    binding = MemoryMaintenanceBinding(
        service=service, database_path=tmp_path / "memory.db", namespace="project"
    )
    runner.bind_maintenance(coordinator, memory=binding)
    closing: asyncio.Task[None] | None = None
    try:
        await runner.aprepare()
        assert await asyncio.to_thread(entered.wait, 2)
        task = coordinator._task
        result = await asyncio.wait_for(runner.start(AgentRunRequest(input="前台继续")), 2)
        assert result.assistant_message is not None
        assert coordinator._task is task and not task.done()
        resource = runner._maintenance.resource
        peer_lock = FileLock(resource.lock_path, timeout=0)
        with pytest.raises(Timeout):
            peer_lock.acquire()
        await runner.aclose()
        closing = asyncio.create_task(
            coordinator.unbind_memory(binding) if remove_resource else coordinator.aclose()
        )
        await asyncio.sleep(0)
        assert not closing.done()
        assert not any(
            span.name == "iris.maintenance.cycle" for span in exporter.get_finished_spans()
        )
        with pytest.raises(Timeout):
            peer_lock.acquire()
        release.set()
        await asyncio.wait_for(asyncio.shield(closing), 2)
        with peer_lock:
            assert not coordinator._worker.busy
        cycle = next(
            span for span in exporter.get_finished_spans() if span.name == "iris.maintenance.cycle"
        )
        assert len(released_at) == 1 and cycle.end_time >= released_at[0]
        assert cycle.attributes["iris.driver.outcome"] == "cancelled"
    finally:
        release.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await runner.aclose()
        await coordinator.aclose()


@pytest.mark.asyncio
async def test_cancellation_suppressing_provider_cannot_commit_and_retains_lock(
    tmp_path: Path,
) -> None:
    """模型吞取消后仍保留任务位置；迟到结果不能消费已经撤销的材料。"""
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class LateProvider(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
                return text_response('{"observations": []}')

    service = memory_service(tmp_path / "memory.db", LateProvider())
    service.observe(MemoryObserveInput(text="不能被迟到输出消费"))
    coordinator = MaintenanceCoordinator(idle_seconds=0)
    first = configured_runner(tmp_path, service)
    first.bind_maintenance(
        coordinator,
        memory=MemoryMaintenanceBinding(
            service=service, database_path=tmp_path / "memory.db", namespace="project"
        ),
    )
    second = AgentRunner(runtime=build_runtime(tmp_path), store=InMemoryLifecycleStore())
    second.bind_maintenance(coordinator)
    await first.aprepare()
    await asyncio.wait_for(entered.wait(), 2)
    task = coordinator._task
    await asyncio.wait_for(second.start(AgentRunRequest(input="前台立即入场")), 2)
    await asyncio.wait_for(cancelled.wait(), 2)
    assert coordinator._task is task and not task.done()
    lock = FileLock(first._maintenance.resource.lock_path, timeout=0)
    with pytest.raises(Timeout):
        lock.acquire()
    closing = asyncio.create_task(coordinator.aclose())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await asyncio.wait_for(closing, 2)
    assert service.generation_state("project").pending_episodes == 1
    with lock:
        assert coordinator._task is None
    await first.aclose()
    await second.aclose()


@pytest.mark.asyncio
async def test_unbind_requires_detach_and_keeps_other_resource(tmp_path: Path) -> None:
    """资源撤销不关闭服务；其他资源及其前台状态不阻碍本 runner 关闭。"""
    coordinator = MaintenanceCoordinator(idle_seconds=100)
    service = memory_service(tmp_path / "memory.db")
    other = memory_service(tmp_path / "other.db")
    first, second = configured_runner(tmp_path, service), configured_runner(tmp_path, other)
    binding = MemoryMaintenanceBinding(
        service=service, database_path=tmp_path / "memory.db", namespace="project"
    )
    first.bind_maintenance(coordinator, memory=binding)
    second.bind_maintenance(
        coordinator,
        memory=MemoryMaintenanceBinding(
            service=other, database_path=tmp_path / "other.db", namespace="project"
        ),
    )
    await first.aprepare()
    second._maintenance.foreground_enter()
    with pytest.raises(IrisRunStateError, match="仍有绑定 runner"):
        await coordinator.unbind_memory(binding)
    await first.aclose()
    await coordinator.unbind_memory(binding)
    assert not service._change_listeners
    assert len(other._change_listeners) == 1
    service.observe(MemoryObserveInput(text="仍可显式使用"))
    second._maintenance.foreground_exit()
    await second.aclose()
    await coordinator.aclose()


@pytest.mark.asyncio
async def test_external_waiting_change_reselects_other_sources(tmp_path: Path) -> None:
    """独立 SQLite reader 的状态变化使旧批次失效后，自动重选仍合格的 B。"""
    entered, release, reselected = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Generation(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                entered.set()
                await release.wait()
            else:
                reselected.set()
            return text_response('{"observations": []}')

    provider = Generation()
    service = memory_service(tmp_path / "memory.db", provider)
    lifecycle_path = tmp_path / "lifecycle.db"
    runner = configured_runner(tmp_path, service, store=SQLiteStore(lifecycle_path))
    runner.runtime.environment.provider.responses.append(text_response())
    coordinator = MaintenanceCoordinator(idle_seconds=0)
    runner.bind_maintenance(
        coordinator,
        memory=MemoryMaintenanceBinding(
            service=service, database_path=tmp_path / "memory.db", namespace="project"
        ),
    )
    coordinator._foreground_enter()
    await runner.start(AgentRunRequest(input="A 来源", session_id="a"))
    await runner.start(AgentRunRequest(input="B 来源", session_id="b"))
    coordinator._foreground_exit()
    await asyncio.wait_for(entered.wait(), 2)
    registry = ToolRegistry()
    registry.register_function(
        lambda: "written", name="write", description="写", capabilities={ToolCapability.WRITE}
    )
    peer = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            provider=StaticProvider(tool_response(ToolUseBlock(id="peer-write", name="write"))),
            registry=registry,
        ),
        store=SQLiteStore(lifecycle_path),
    )
    waiting = await peer.start(AgentRunRequest(input="另一个宿主使 A 等待", session_id="a"))
    assert waiting.pending_interaction is not None
    release.set()
    await asyncio.wait_for(reselected.wait(), 2)
    await asyncio.sleep(0.05)
    assert len(provider.requests) == 2
    records = json.loads(provider.requests[1].messages[1].text)["records"]
    assert any("B 来源" in record["text"] for record in records)
    assert all("A 来源" not in record["text"] for record in records)
    assert service.generation_state("project").pending_episodes == 1
    await peer.aclose()
    await runner.aclose()
    await coordinator.aclose()


@pytest.mark.asyncio
async def test_missing_lifecycle_reader_retains_pending_after_restart(tmp_path: Path) -> None:
    """纯内存来源重启后不猜测旧资格；没有自动重试或消费旧材料。"""
    service = memory_service(tmp_path / "memory.db")
    first = configured_runner(tmp_path, service)
    original = MaintenanceCoordinator(idle_seconds=100)
    binding = MemoryMaintenanceBinding(
        service=service, database_path=tmp_path / "memory.db", namespace="project"
    )
    first.bind_maintenance(original, memory=binding)
    await first.start(AgentRunRequest(input="只存在于原内存 lifecycle 的材料"))
    await first.aclose()
    await original.aclose()
    restarted = MaintenanceCoordinator(idle_seconds=0)
    second = configured_runner(tmp_path, service)
    second.bind_maintenance(restarted, memory=binding)
    await second.aprepare()
    await asyncio.sleep(0.05)
    assert service.generation_provider.requests == []
    assert service.generation_state("project").pending_episodes == 1
    assert restarted._task is None and restarted._timer is None
    await second.aclose()
    await restarted.aclose()


@pytest.mark.asyncio
async def test_lock_busy_retries_without_activity_and_yields_to_other_resource(
    tmp_path: Path,
    observability: tuple[Observability, InMemorySpanExporter],
) -> None:
    """idle=0 锁忙至少退让一秒，其他资源先行，无新输入也会重试。"""
    first_service = memory_service(tmp_path / "first.db")
    second_service = memory_service(tmp_path / "second.db")
    first_service.observe(MemoryObserveInput(text="first"))
    second_service.observe(MemoryObserveInput(text="second"))
    first, second = (
        configured_runner(tmp_path, first_service),
        configured_runner(tmp_path, second_service),
    )
    observation, exporter = observability
    coordinator = MaintenanceCoordinator(idle_seconds=0, observability=observation)
    for runner, name, service in (
        (first, "first", first_service),
        (second, "second", second_service),
    ):
        runner.bind_maintenance(
            coordinator,
            memory=MemoryMaintenanceBinding(
                service=service, database_path=tmp_path / f"{name}.db", namespace="project"
            ),
        )
    lock = FileLock(first._maintenance.resource.lock_path, timeout=0)
    lock.acquire()
    try:
        await first.aprepare()
        await second.aprepare()
        await asyncio.sleep(0.15)
        assert not first_service.generation_provider.requests
        assert len(second_service.generation_provider.requests) == 1
        cycles = [
            span for span in exporter.get_finished_spans() if span.name == "iris.maintenance.cycle"
        ]
        assert len(cycles) == 1
        assert cycles[0].attributes["iris.maintenance.database"] == str(tmp_path / "second.db")
        assert first._maintenance.resource.ready_at > asyncio.get_running_loop().time()
        assert coordinator._task is None
    finally:
        lock.release()
    deadline = asyncio.get_running_loop().time() + 2
    while not first_service.generation_provider.requests:
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.02)
    assert len(first_service.generation_provider.requests) == 1
    await first.aclose()
    await second.aclose()
    await coordinator.aclose()


@pytest.mark.asyncio
async def test_overview_failure_does_not_retry_after_own_dream_notification(tmp_path: Path) -> None:
    """dream 的自身写入不能被误认成新外部活动，失败 overview 只调用一次。"""
    failed = asyncio.Event()

    class Provider(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            if request.model == "overview":
                failed.set()
                raise RuntimeError("overview provider unavailable")
            source = json.loads(request.messages[1].text)
            return text_response(
                json.dumps(
                    {
                        "operations": [
                            {
                                "action": "update",
                                "target_id": source["items"][0]["id"],
                                "text": "提炼后的确认事实",
                                "reason": "整理用户指定的事实",
                                "evidence": [source["events"][0]["ref"]],
                            }
                        ],
                        "resolutions": [],
                    }
                )
            )

    provider = Provider()
    service = memory_service(tmp_path / "memory.db", provider)
    service.remember(MemoryWriteInput(text="已确认事实", reason="用户指定"))
    runner = configured_runner(tmp_path, service)
    coordinator = MaintenanceCoordinator(idle_seconds=0)
    runner.bind_maintenance(
        coordinator,
        memory=MemoryMaintenanceBinding(
            service=service, database_path=tmp_path / "memory.db", namespace="project"
        ),
    )
    await runner.aprepare()
    await asyncio.wait_for(failed.wait(), 2)
    await asyncio.sleep(0.05)
    assert [request.model for request in provider.requests] == ["generation", "overview"]
    assert service.generation_state("project").item_revision == 2
    assert coordinator._task is None and coordinator._timer is None
    await runner.aclose()
    await asyncio.sleep(0.05)
    assert [request.model for request in provider.requests] == ["generation", "overview"]
    await coordinator.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_memory", [False, True])
async def test_foreground_cancels_shared_generation_without_waiting(
    tmp_path: Path, with_memory: bool
) -> None:
    """任何绑定 runner 的前台及时入场，另一个 runner 的关闭不关闭宿主协调器。"""
    entered, cancelled = asyncio.Event(), asyncio.Event()

    class Background(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    service = memory_service(tmp_path / "memory.db", Background())
    service.observe(MemoryObserveInput(text="已有材料"))
    coordinator = MaintenanceCoordinator(idle_seconds=0)
    first = configured_runner(tmp_path, service)
    second = (
        configured_runner(tmp_path, service)
        if with_memory
        else AgentRunner(runtime=build_runtime(tmp_path), store=InMemoryLifecycleStore())
    )
    binding = MemoryMaintenanceBinding(
        service=service, database_path=tmp_path / "memory.db", namespace="project"
    )
    first.bind_maintenance(coordinator, memory=binding)
    second.bind_maintenance(coordinator, memory=binding if with_memory else None)
    await first.aprepare()
    await asyncio.wait_for(entered.wait(), 2)
    result = await asyncio.wait_for(second.start(AgentRunRequest(input="前台")), 2)
    assert result.assistant_message is not None
    await asyncio.wait_for(cancelled.wait(), 2)
    await first.aclose()
    assert not coordinator._closed
    await second.aclose()
    await coordinator.aclose()
