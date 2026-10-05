"""项目经验与 Memory 的独立维护位置及共同前台边界。"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from filelock import FileLock, Timeout

from iris.agents import AgentConfig
from iris.evolution.config import EvolutionConfig
from iris.evolution.materials import EvolutionMaterialStore
from iris.evolution.models import (
    EvolutionCaptureBlock,
    EvolutionMaintenanceScope,
    EvolutionResult,
    EvolutionSession,
    EvolutionSource,
    EvolutionSourceState,
)
from iris.evolution.service import EvolutionService
from iris.exceptions import IrisEvolutionError, IrisRunStateError
from iris.harness import (
    AgentRunner,
    MaintenanceCoordinator,
    MemoryMaintenanceBinding,
    ProjectEvolutionBinding,
)
from iris.harness._capture import RunCapture
from iris.harness._capture_records import capture_records
from iris.lifecycle import AgentRunRequest, FinishRun, RunStopReason
from iris.lifecycle.history import RunMessageSlice
from iris.memory import MemoryObserveInput
from iris.message import LLMRequest, LLMResponse, Msg, ToolUseBlock
from iris.prompts import PromptSource
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import ToolCapability

from ..store.test_run_input_commit import _input_command
from .fakes import StaticProvider, text_response, tool_response
from .test_maintenance_coordinator import memory_service


class ControlledEvolution:
    """固定资格入口，通过事件观察本类后台 task 的真实生命周期。"""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def alist_pending_sources(self) -> tuple[EvolutionSource, ...]:
        """无真实来源；本组只验证协调器独立调度与取消。"""
        return ()

    async def alist_pending_sessions(self) -> tuple[EvolutionSession, ...]:
        """调度测试没有宿主会话请求。"""
        return ()

    async def wait_pending_io(self) -> None:
        """此替身不执行同步 IO。"""

    async def maintain_cycle(self, *, scope: EvolutionMaintenanceScope) -> EvolutionResult:
        """保持当前调用，直到主动放行或本类收到取消。"""
        self.calls += 1
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return EvolutionResult(status="empty")


def project_service(tmp_path: Path, provider: StaticProvider | None = None) -> EvolutionService:
    """构造不依赖 Memory 的真实材料与单次 A 服务。"""
    return EvolutionService(
        workspace_root=tmp_path,
        skill_path=tmp_path / ".agents" / "skills" / "project-experience" / "SKILL.md",
        store=EvolutionMaterialStore(tmp_path),
        provider=provider or StaticProvider(text_response('{"body":null,"reason":"保持"}')),
        model="test-evolution",
        config=EvolutionConfig(enabled=True),
        prompt_source=PromptSource.initialize(tmp_path),
    )


def project_runner(tmp_path: Path, provider: StaticProvider | None = None) -> AgentRunner:
    """项目学习启用、Memory 关闭的 root runner。"""
    return AgentRunner.from_config(
        AgentConfig(
            name="project",
            model="openai/test",
            system="完成任务",
            permissions={"workspace": str(tmp_path)},
            context_policy={"enabled": False},
            skills={"enabled": True},
            evolution={"enabled": True},
        ),
        provider=provider or StaticProvider(text_response()),
    )


@pytest.mark.asyncio
async def test_memory_and_evolution_run_concurrently_and_cancel_independently(
    tmp_path: Path,
) -> None:
    """撤销一个项目只取消其 lane，Memory 仍可独立完成。"""
    memory_entered, memory_release = asyncio.Event(), asyncio.Event()

    class Provider(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            memory_entered.set()
            await memory_release.wait()
            return text_response('{"observations": []}')

    memory = memory_service(tmp_path / "memory.db", Provider())
    memory.observe(MemoryObserveInput(text="pending memory"))
    evolution = ControlledEvolution()
    coordinator = MaintenanceCoordinator(idle_seconds=0)
    binding = ProjectEvolutionBinding(
        workspace_root=tmp_path, service=cast(EvolutionService, evolution)
    )
    memory_resource, project = coordinator._attach(
        MemoryMaintenanceBinding(
            service=memory, database_path=tmp_path / "memory.db", namespace="project"
        ),
        InMemoryLifecycleStore(),
        evolution=binding,
    )
    try:
        await coordinator.prepare()
        await asyncio.wait_for(memory_entered.wait(), 2)
        await asyncio.wait_for(evolution.entered.wait(), 2)
        project.attachments -= 1
        await coordinator.unbind_evolution(binding)
        assert evolution.cancelled.is_set()
        assert coordinator._task is not None and not coordinator._task.done()
        memory_release.set()
        task = coordinator._task
        await asyncio.wait_for(asyncio.shield(task), 2)
        assert memory.generation_state("project").pending_episodes == 0
    finally:
        memory_release.set()
        await coordinator.aclose()


@pytest.mark.asyncio
async def test_foreground_cancels_both_lanes_without_waiting(tmp_path: Path) -> None:
    """共享前台入场分别撤销两项生成，不把两份取消状态混在一起。"""
    memory_entered, memory_cancelled = asyncio.Event(), asyncio.Event()

    class Provider(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            memory_entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                memory_cancelled.set()
                raise

    memory = memory_service(tmp_path / "memory.db", Provider())
    memory.observe(MemoryObserveInput(text="pending memory"))
    evolution = ControlledEvolution()
    coordinator = MaintenanceCoordinator(idle_seconds=0)
    coordinator._attach(
        MemoryMaintenanceBinding(
            service=memory, database_path=tmp_path / "memory.db", namespace="project"
        ),
        InMemoryLifecycleStore(),
        evolution=ProjectEvolutionBinding(
            workspace_root=tmp_path, service=cast(EvolutionService, evolution)
        ),
    )
    try:
        await coordinator.prepare()
        await asyncio.wait_for(memory_entered.wait(), 2)
        await asyncio.wait_for(evolution.entered.wait(), 2)
        coordinator._foreground_enter()
        await asyncio.wait_for(memory_cancelled.wait(), 2)
        await asyncio.wait_for(evolution.cancelled.wait(), 2)
        assert coordinator._worker is not coordinator._evolution_worker
    finally:
        await coordinator.aclose()


@pytest.mark.asyncio
async def test_explicit_project_requests_share_one_operation_and_skip_idle(tmp_path: Path) -> None:
    """等待者的取消不取消共享作业，同项目并发请求只执行一次 A。"""
    evolution = ControlledEvolution()
    coordinator = MaintenanceCoordinator(idle_seconds=300)
    binding = ProjectEvolutionBinding(
        workspace_root=tmp_path, service=cast(EvolutionService, evolution)
    )
    coordinator._attach(None, InMemoryLifecycleStore(), evolution=binding)
    first = asyncio.create_task(coordinator.request_project_experience(binding))
    second = asyncio.create_task(coordinator.request_project_experience(binding))
    try:
        await asyncio.wait_for(evolution.entered.wait(), 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not evolution.cancelled.is_set()
        evolution.release.set()
        result = await asyncio.wait_for(second, 2)
        assert result.status == "empty" and evolution.calls == 1
    finally:
        await coordinator.aclose()


@pytest.mark.asyncio
async def test_next_explicit_request_after_completion_gets_new_operation(tmp_path: Path) -> None:
    """上一轮结果可见后紧接着主动请求，不会挂在已经完成的任务位置。"""
    evolution = ControlledEvolution()
    evolution.release.set()
    coordinator = MaintenanceCoordinator(idle_seconds=300)
    binding = ProjectEvolutionBinding(
        workspace_root=tmp_path, service=cast(EvolutionService, evolution)
    )
    coordinator._attach(None, InMemoryLifecycleStore(), evolution=binding)
    try:
        assert (await coordinator.request_project_experience(binding)).status == "empty"
        assert (
            await asyncio.wait_for(coordinator.request_project_experience(binding), 1)
        ).status == "empty"
        assert evolution.calls == 2
    finally:
        await coordinator.aclose()


@pytest.mark.asyncio
async def test_memory_disabled_captures_and_publishes_project_skill(tmp_path: Path) -> None:
    """纯项目经验路径真实捕获原文、执行一次 A，并由新 runner 发现产物。"""
    generation = StaticProvider(
        text_response(
            json.dumps(
                {
                    "body": "# 项目经验\n使用 uv 运行测试。",
                    "reason": "记录实际约定",
                },
                ensure_ascii=False,
            )
        )
    )
    service = project_service(tmp_path, generation)
    runner = project_runner(tmp_path)
    coordinator = MaintenanceCoordinator(idle_seconds=300)
    binding = ProjectEvolutionBinding(workspace_root=tmp_path, service=service)
    runner.bind_maintenance(coordinator, evolution=binding)
    try:
        await runner.start(AgentRunRequest(input="本项目使用 uv 运行测试", run_id="learn"))
        assert runner.runtime.environment.memory_service is None
        assert runner.runtime.environment.capture_port is runner._capture
        assert len(service.store.list_pending_sources()) == 1
        result = await coordinator.request_project_experience(binding)
        assert result.status == "updated"
        assert "使用 uv" in service.skill_path.read_text(encoding="utf-8")
        assert len(generation.requests) == 1
        assert not (tmp_path / ".iris" / "memory").exists()
        assert service.store.list_pending_sources() == ()
        rebuilt = project_runner(tmp_path)
        assert rebuilt.runtime.environment.skill_registry.names() == ("project-experience",)
        await rebuilt.aclose()
    finally:
        await runner.aclose()
        await coordinator.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_evolution_terminal_capture_pages_seal_only_after_final_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    """260条终态原文分三页独立发布，无Memory时也不会提前开放生成资格。"""
    lifecycle = (
        SQLiteStore(tmp_path / "lifecycle.db") if backend == "sqlite" else InMemoryLifecycleStore()
    )
    command = replace(
        _input_command(lifecycle), message_delta=[Msg.user(f"source {i}") for i in range(260)]
    )
    lifecycle.commit_run_input(command)
    run = lifecycle.load_run(command.run_id)
    lifecycle.finish_run(
        FinishRun(
            run_id=run.run_id,
            expected_run_revision=run.revision,
            activation_id=command.activation_id,
            stop_reason=RunStopReason.CANCELLED,
            now=command.now,
        )
    )
    service = project_service(tmp_path)
    capture = RunCapture(evolution_service=service, lifecycle_store=lifecycle)
    commit = service.store.commit_capture
    observed: list[tuple[int, int | None, tuple[EvolutionSource, ...]]] = []

    def capture_page(block: EvolutionCaptureBlock) -> EvolutionSourceState:
        state = commit(block)
        observed.append(
            (
                state.captured_until,
                state.terminal_message_count,
                service.store.list_pending_sources(),
            )
        )
        return state

    monkeypatch.setattr(service.store, "commit_capture", capture_page)
    await capture.register_run(run)
    await capture.capture_pending()
    assert [(end, terminal) for end, terminal, _ in observed] == [
        (128, None),
        (256, None),
        (260, 260),
    ]
    assert observed[0][2] == observed[1][2] == ()
    assert len(observed[2][2]) == 1
    await capture.capture_pending()
    assert len(observed) == 3
    await capture.aclose()


@pytest.mark.asyncio
async def test_sqlite_restart_recovers_registered_terminal_tail(tmp_path: Path) -> None:
    """进程重启后同一SQLite source补齐终态尾部，后来的session消息不混入旧来源。"""
    database = tmp_path / "lifecycle.db"
    lifecycle = SQLiteStore(database)
    command = replace(_input_command(lifecycle), message_delta=[Msg.user("已提交但未捕获")])
    lifecycle.commit_run_input(command)
    run = lifecycle.load_run(command.run_id)
    service = project_service(tmp_path)
    source = EvolutionSource(
        lifecycle_source_id=lifecycle.source_id, run_id=run.run_id, session_id=run.session_id
    )
    service.store.register_source(source, run.initial_session_message_count)
    lifecycle.finish_run(
        FinishRun(
            run_id=run.run_id,
            expected_run_revision=run.revision,
            activation_id=command.activation_id,
            stop_reason=RunStopReason.CANCELLED,
            now=command.now,
        )
    )
    reopened = SQLiteStore(database)
    capture = RunCapture(evolution_service=service, lifecycle_store=reopened)
    await capture.prepare()
    assert reopened.source_id == lifecycle.source_id
    assert service.store.list_pending_sources() == (source,)
    pending = service.store.read_pending(
        allowed_sources=frozenset({(source.lifecycle_source_id, source.run_id)})
    )
    assert [record.text for item in pending.items for record in item.records] == ["已提交但未捕获"]
    await capture.aclose()


def test_shared_capture_keeps_skill_and_memory_references_without_readback_text() -> None:
    """读回旧经验不伪装成新事实，之后用户的实际纠正仍是新材料。"""
    messages = (
        Msg.assistant(
            [ToolUseBlock(id="skill", name="load_skill", input={"name": "project-experience"})]
        ),
        Msg.tool_result(
            tool_use_id="skill", name="load_skill", content="不要重新学习的旧Skill正文"
        ),
        Msg.assistant(
            [ToolUseBlock(id="memory", name="memory_fetch", input={"item_id": "old-item"})]
        ),
        Msg.tool_result(
            tool_use_id="memory",
            name="memory_fetch",
            content='{"item":{"id":"old-item","text":"旧记忆正文"}}',
        ),
        Msg.user("刚才的规则仅适用于Linux；Windows需另外处理"),
    )
    records = capture_records(
        RunMessageSlice(
            source_id="source",
            run_id="run",
            session_id="session",
            initial_message_count=4,
            start_message_count=4,
            end_message_count=9,
            terminal_message_count=9,
            outcome=RunStopReason.COMPLETED,
            messages=messages,
        ),
        end=9,
    )
    assert all(record.text == "" for record in records[:4])
    assert records[0].metadata["query"] == {"name": "project-experience"}
    assert records[3].metadata["memory_item_ids"] == ["old-item"]
    assert records[4].text.startswith("刚才的规则")
    assert [record.ref for record in records] == [
        f"source:run:{ordinal}:0" for ordinal in range(4, 9)
    ]


@pytest.mark.asyncio
async def test_waiting_and_missing_reader_keep_only_their_sources_pending(tmp_path: Path) -> None:
    """A会话WAITING和旧内存reader丢失都不能阻断仍合格的B会话。"""
    generation = StaticProvider(text_response('{"body":null,"reason":"无需修改"}'))
    service = project_service(tmp_path, generation)
    runner = project_runner(
        tmp_path,
        StaticProvider(
            text_response(),
            tool_response(ToolUseBlock(id="write", name="write")),
            text_response(),
        ),
    )
    runner.runtime.environment.tool_bridge.tool_executor.registry.register_function(
        lambda: "written",
        name="write",
        description="写",
        capabilities={ToolCapability.WRITE},
    )
    coordinator = MaintenanceCoordinator(idle_seconds=300)
    binding = ProjectEvolutionBinding(workspace_root=tmp_path, service=service)
    runner.bind_maintenance(coordinator, evolution=binding)
    try:
        await runner.start(AgentRunRequest(input="A 的旧终态", session_id="a"))
        waiting = await runner.start(AgentRunRequest(input="A 等待许可", session_id="a"))
        assert waiting.pending_interaction is not None
        await runner.start(AgentRunRequest(input="B 的完整材料", session_id="b"))
        result = await coordinator.request_project_experience(binding)
        assert result.status == "no_change"
        request_text = generation.requests[0].messages[1].text
        assert "B 的完整材料" in request_text and "A 的旧终态" not in request_text
    finally:
        await runner.aclose()
        await coordinator.aclose()
    restarted = MaintenanceCoordinator(idle_seconds=300)
    other = project_runner(tmp_path)
    other.bind_maintenance(restarted, evolution=binding)
    try:
        assert (await restarted.request_project_experience(binding)).status == "empty"
        assert len(generation.requests) == 1 and service.store.list_pending_sources()
    finally:
        await other.aclose()
        await restarted.aclose()


@pytest.mark.asyncio
async def test_project_worker_drain_keeps_lock_and_explicit_waiter_ends_on_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """已派发短IO不被关闭中断；任务位置和项目锁持有到真实收尾。"""
    entered, release = threading.Event(), threading.Event()
    service = project_service(tmp_path)
    original = service.store.list_pending_sources

    def blocked_sources() -> tuple[EvolutionSource, ...]:
        entered.set()
        assert release.wait(5)
        return original()

    monkeypatch.setattr(service.store, "list_pending_sources", blocked_sources)
    coordinator = MaintenanceCoordinator(idle_seconds=300)
    binding = ProjectEvolutionBinding(workspace_root=tmp_path, service=service)
    coordinator._attach(None, InMemoryLifecycleStore(), evolution=binding)
    requested = asyncio.create_task(coordinator.request_project_experience(binding))
    assert await asyncio.to_thread(entered.wait, 2)
    task = coordinator._evolution_task
    coordinator._foreground_enter()
    lock = FileLock(tmp_path / ".iris" / "evolution.lock", timeout=0)
    with pytest.raises(Timeout):
        lock.acquire()
    closing = asyncio.create_task(coordinator.aclose())
    await asyncio.sleep(0)
    assert not closing.done() and coordinator._evolution_task is task
    release.set()
    await asyncio.wait_for(closing, 2)
    with pytest.raises(asyncio.CancelledError):
        await requested
    with lock:
        assert not coordinator._evolution_worker.busy


@pytest.mark.asyncio
async def test_explicit_waiter_finishes_on_failure_and_queued_close(tmp_path: Path) -> None:
    """执行失败与前台等待中的宿主关闭都会结束主动等待者。"""

    class FailingEvolution(ControlledEvolution):
        async def maintain_cycle(self, *, scope: EvolutionMaintenanceScope) -> EvolutionResult:
            raise IrisEvolutionError("generation failed")

    service = FailingEvolution()
    coordinator = MaintenanceCoordinator(idle_seconds=300)
    binding = ProjectEvolutionBinding(
        workspace_root=tmp_path, service=cast(EvolutionService, service)
    )
    coordinator._attach(None, InMemoryLifecycleStore(), evolution=binding)
    with pytest.raises(IrisEvolutionError, match="generation failed"):
        await coordinator.request_project_experience(binding)
    coordinator._foreground_enter()
    requested = asyncio.create_task(coordinator.request_project_experience(binding))
    await asyncio.sleep(0)
    await coordinator.aclose()
    with pytest.raises(IrisRunStateError, match="关闭"):
        await requested


@pytest.mark.asyncio
async def test_other_runner_can_contribute_to_shared_project_without_local_enable(
    tmp_path: Path,
) -> None:
    """宿主选定主策略，其他root只贡献经历，不按其本地开关拒绝显式借用。"""
    service = project_service(tmp_path)
    coordinator = MaintenanceCoordinator(idle_seconds=300)
    binding = ProjectEvolutionBinding(workspace_root=tmp_path, service=service)
    first, second = project_runner(tmp_path), project_runner(tmp_path)
    config = second.runtime.environment.agent_config
    second.runtime.environment.agent_config = config.model_copy(
        update={"evolution": EvolutionConfig()}
    )
    first.bind_maintenance(coordinator, evolution=binding)
    second.bind_maintenance(coordinator, evolution=binding)
    try:
        await first.start(AgentRunRequest(input="第一条项目事实", run_id="same-id"))
        await second.start(AgentRunRequest(input="第二条项目事实", run_id="same-id"))
        sources = service.store.list_pending_sources()
        assert len(sources) == 2
        assert {source.lifecycle_source_id for source in sources} == {
            first.store.source_id,
            second.store.source_id,
        }
        assert (await coordinator.request_project_experience(binding)).status == "no_change"
        assert service.store.list_pending_sources() == ()
    finally:
        await first.aclose()
        await second.aclose()
        await coordinator.aclose()
