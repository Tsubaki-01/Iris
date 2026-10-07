"""显式 A/B 请求的独立结果与真实会话资格。"""

import asyncio
from pathlib import Path
from threading import Event

import pytest
from filelock import FileLock

from iris.evolution.config import EvolutionConfig
from iris.evolution.history import PublicationDocument, PublicationRecord
from iris.evolution.materials import EvolutionMaterialStore
from iris.evolution.models import (
    EvolutionResult,
    EvolutionSession,
    RevisionItem,
    RevisionRequest,
    RevisionTarget,
)
from iris.evolution.revision import PromptTarget
from iris.evolution.service import EvolutionService
from iris.exceptions import IrisEvolutionError, IrisRunStateError
from iris.harness import MaintenanceCoordinator, ProjectEvolutionBinding
from iris.lifecycle import AgentRunRequest
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.prompts import PromptSource
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolCapability

from .fakes import StaticProvider, text_response, tool_response
from .test_evolution_maintenance import project_runner


def revision_service(tmp_path: Path, provider: StaticProvider) -> EvolutionService:
    """开放一个真实 prompt 目标，不引入额外配置装配。"""
    return EvolutionService(
        workspace_root=tmp_path,
        skill_path=tmp_path / ".agents" / "skills" / "project-experience" / "SKILL.md",
        store=EvolutionMaterialStore(tmp_path),
        provider=provider,
        model="test-evolution",
        config=EvolutionConfig(enabled=True, prompt_targets=("project_skill_update",)),
        prompt_source=PromptSource.initialize(tmp_path),
        prompt_targets=(PromptTarget("project_skill_update", "项目经验策略", {}),),
    )


def request(description: str, session: EvolutionSession | None = None) -> RevisionRequest:
    return RevisionRequest(
        description=description,
        targets=(RevisionTarget(kind="prompt", name="project_skill_update"),),
        session=session,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["experience", "revision"])
async def test_unconfirmed_publication_finishes_new_request_without_spin(
    tmp_path: Path, command: str
) -> None:
    """新进程遇到旧未确认发布时，当前请求显式失败而非循环处理旧结果。"""
    provider = StaticProvider()
    service = revision_service(tmp_path, provider)
    old = await service.enqueue_revision(request("旧请求"))
    path = service.prompt_source.root / "project_skill_update.j2"
    service.store.save_publication(
        PublicationRecord(
            stage="revision",
            revision_id=old.id,
            origin="host_request",
            description=old.description,
            targets=old.targets,
            request=old,
            publication_state="unconfirmed",
            before_documents=(PublicationDocument(path=str(path), text="旧"),),
            candidate_documents=(PublicationDocument(path=str(path), text="新"),),
        )
    )
    coordinator = MaintenanceCoordinator(idle_seconds=3600)
    binding = ProjectEvolutionBinding(workspace_root=tmp_path, service=service)
    coordinator._attach(None, InMemoryLifecycleStore(), evolution=binding)
    try:
        operation = (
            coordinator.request_project_experience(binding)
            if command == "experience"
            else coordinator.request_revision(binding, request("新请求"))
        )
        with pytest.raises(IrisEvolutionError, match="publication_unconfirmed"):
            await asyncio.wait_for(operation, 1)
        assert coordinator._evolution_task is None and coordinator._timer is None
        assert coordinator.snapshot().resources[0].pending_request_id is None
        assert service.store.revision_result(old.id) is None
        assert service.list_publications().items[0].publication_state == "unconfirmed"
        assert provider.requests == []
    finally:
        await coordinator.aclose()


@pytest.mark.asyncio
async def test_settled_old_conflict_allows_new_revision_waiter_to_continue(tmp_path: Path) -> None:
    """旧结果收尾不等于本次请求完成；已无阻塞时继续合格的新请求。"""
    provider = StaticProvider(text_response('{"action":"no_change","reason":"维持"}'))
    service = revision_service(tmp_path, provider)
    old = await service.enqueue_revision(request("旧请求"))
    record = PublicationRecord(
        stage="revision",
        revision_id=old.id,
        origin="host_request",
        description=old.description,
        targets=old.targets,
        request=old,
    )
    record = record.model_copy(
        update={
            "outcome": EvolutionResult(
                stage="revision",
                revision_id=old.id,
                publication_id=record.publication_id,
                status="conflict",
                reason="旧文件基线冲突",
            )
        }
    )
    service.store.save_publication(record)
    coordinator = MaintenanceCoordinator(idle_seconds=3600)
    binding = ProjectEvolutionBinding(workspace_root=tmp_path, service=service)
    coordinator._attach(None, InMemoryLifecycleStore(), evolution=binding)
    try:
        result = await asyncio.wait_for(coordinator.request_revision(binding, request("新请求")), 1)
        assert result.revision_id != old.id and result.status == "no_change"
        assert service.get_publication(record.publication_id).settled
        assert coordinator.snapshot().resources[0].pending_request_id is None
    finally:
        await coordinator.aclose()


@pytest.mark.asyncio
async def test_explicit_a_skips_pending_b_and_returns_only_a(tmp_path: Path) -> None:
    provider = StaticProvider(text_response('{"action":"no_change","reason":"保持"}'))
    service = revision_service(tmp_path, provider)
    item = await service.enqueue_revision(request("已有的 B"))
    coordinator = MaintenanceCoordinator(idle_seconds=300)
    binding = ProjectEvolutionBinding(workspace_root=tmp_path, service=service)
    coordinator._attach(None, InMemoryLifecycleStore(), evolution=binding)
    try:
        result = await asyncio.wait_for(coordinator.request_project_experience(binding), 2)
        assert result.stage == "experience" and result.status == "empty"
        assert provider.requests == []
        assert service.store.revision_result(item.id) is None
    finally:
        await coordinator.aclose()


@pytest.mark.asyncio
async def test_queued_b_failure_does_not_block_next_explicit_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = StaticProvider(
        text_response("invalid response"),
        text_response('{"action":"no_change","reason":"第二项完成"}'),
    )
    service = revision_service(tmp_path, provider)
    enqueued: list[RevisionItem] = []
    both_enqueued = asyncio.Event()
    enqueue = service.enqueue_revision

    async def observed_enqueue(value: RevisionRequest) -> RevisionItem:
        item = await enqueue(value)
        enqueued.append(item)
        if len(enqueued) == 2:
            both_enqueued.set()
        return item

    monkeypatch.setattr(service, "enqueue_revision", observed_enqueue)
    coordinator = MaintenanceCoordinator(idle_seconds=300)
    binding = ProjectEvolutionBinding(workspace_root=tmp_path, service=service)
    coordinator._attach(None, InMemoryLifecycleStore(), evolution=binding)
    coordinator._foreground_enter()
    first = asyncio.create_task(coordinator.request_revision(binding, request("第一项")))
    second = asyncio.create_task(coordinator.request_revision(binding, request("第二项")))
    try:
        await asyncio.wait_for(both_enqueued.wait(), 2)
        assert provider.requests == []
        coordinator._foreground_exit()
        results = await asyncio.wait_for(asyncio.gather(first, second), 2)
        by_id = {result.revision_id: result for result in results}
        assert by_id[enqueued[0].id].status == "failed"
        assert by_id[enqueued[1].id].status == "no_change"
        assert service.store.revision_result(enqueued[0].id) is None
        assert service.store.revision_result(enqueued[1].id) is not None
        assert len(provider.requests) == 2
    finally:
        await coordinator.aclose()
        await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_reader", [False, True])
async def test_ineligible_host_session_does_not_block_unrelated_request(
    tmp_path: Path, missing_reader: bool
) -> None:
    provider = StaticProvider(text_response('{"action":"no_change","reason":"宿主请求完成"}'))
    service = revision_service(tmp_path, provider)
    runner = project_runner(
        tmp_path, StaticProvider(tool_response(ToolUseBlock(id="write", name="write")))
    )
    runner.runtime.environment.tool_bridge.tool_executor.registry.register_function(
        lambda: "written", name="write", description="写", capabilities={ToolCapability.WRITE}
    )
    coordinator = MaintenanceCoordinator(idle_seconds=300)
    binding = ProjectEvolutionBinding(workspace_root=tmp_path, service=service)
    runner.bind_maintenance(coordinator, evolution=binding)
    waiting_task: asyncio.Task | None = None
    try:
        waiting = await runner.start(AgentRunRequest(input="等待许可", session_id="waiting"))
        assert waiting.pending_interaction is not None
        waiting_task = asyncio.create_task(
            coordinator.request_revision(
                binding,
                request(
                    "属于等待会话",
                    EvolutionSession(
                        lifecycle_source_id="lost-reader"
                        if missing_reader
                        else runner.store.source_id,
                        session_id="waiting",
                    ),
                ),
            )
        )
        result = await asyncio.wait_for(
            coordinator.request_revision(binding, request("独立宿主请求")), 2
        )
        assert result.stage == "revision" and result.status == "no_change"
        assert not waiting_task.done()
        await asyncio.sleep(0.05)
        assert len(provider.requests) == 1
        assert coordinator._evolution_task is None and coordinator._timer is None
    finally:
        await runner.aclose()
        await coordinator.aclose()
    with pytest.raises(IrisRunStateError, match="关闭"):
        await waiting_task


@pytest.mark.asyncio
async def test_other_process_settlement_completes_exact_waiter_without_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = StaticProvider()
    service = revision_service(tmp_path, provider)
    published: asyncio.Future[RevisionItem] = asyncio.get_running_loop().create_future()
    enqueue = service.enqueue_revision

    async def observed_enqueue(value: RevisionRequest) -> RevisionItem:
        item = await enqueue(value)
        published.set_result(item)
        return item

    monkeypatch.setattr(service, "enqueue_revision", observed_enqueue)
    coordinator = MaintenanceCoordinator(idle_seconds=300)
    binding = ProjectEvolutionBinding(workspace_root=tmp_path, service=service)
    coordinator._attach(None, InMemoryLifecycleStore(), evolution=binding)
    coordinator._foreground_enter()
    requested = asyncio.create_task(coordinator.request_revision(binding, request("另一进程处理")))
    try:
        item = await asyncio.wait_for(published, 2)
        settled = EvolutionResult(
            stage="revision", revision_id=item.id, status="no_change", reason="已由另一进程完成"
        )
        with FileLock(tmp_path / ".iris" / "evolution.lock", timeout=0):
            service.store.settle_revision(item.id, settled)
        coordinator._foreground_exit()
        assert await asyncio.wait_for(requested, 2) == settled
        assert provider.requests == []
    finally:
        await coordinator.aclose()


@pytest.mark.asyncio
async def test_a_requested_during_b_waits_for_its_own_cycle(tmp_path: Path) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    class Provider(StaticProvider):
        async def complete(self, value: LLMRequest) -> LLMResponse:
            entered.set()
            await release.wait()
            return await super().complete(value)

    provider = Provider(text_response('{"action":"no_change","reason":"B 完成"}'))
    service = revision_service(tmp_path, provider)
    coordinator = MaintenanceCoordinator(idle_seconds=300)
    binding = ProjectEvolutionBinding(workspace_root=tmp_path, service=service)
    coordinator._attach(None, InMemoryLifecycleStore(), evolution=binding)
    revision = asyncio.create_task(coordinator.request_revision(binding, request("先处理 B")))
    experience: asyncio.Task | None = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        experience = asyncio.create_task(coordinator.request_project_experience(binding))
        await asyncio.sleep(0)
        assert not experience.done()
        release.set()
        b_result, a_result = await asyncio.wait_for(asyncio.gather(revision, experience), 2)
        assert b_result.stage == "revision" and b_result.status == "no_change"
        assert a_result.stage == "experience" and a_result.status == "empty"
        assert len(provider.requests) == 1
    finally:
        release.set()
        await coordinator.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("unbind", [False, True])
async def test_request_publication_drains_before_close_or_unbind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unbind: bool
) -> None:
    """提交已经派发后，关闭等待短 IO，调用结束且持久请求保留。"""
    entered, release = Event(), Event()
    service = revision_service(tmp_path, StaticProvider())
    original = service.store.enqueue_revision

    def blocked_enqueue(item: RevisionItem) -> None:
        entered.set()
        assert release.wait(5)
        original(item)

    monkeypatch.setattr(service.store, "enqueue_revision", blocked_enqueue)
    coordinator = MaintenanceCoordinator(idle_seconds=300)
    binding = ProjectEvolutionBinding(workspace_root=tmp_path, service=service)
    _, project = coordinator._attach(None, InMemoryLifecycleStore(), evolution=binding)
    requested = asyncio.create_task(coordinator.request_revision(binding, request("保存中的 B")))
    assert await asyncio.to_thread(entered.wait, 2)
    if unbind:
        project.attachments -= 1
    closing = asyncio.create_task(
        coordinator.unbind_evolution(binding) if unbind else coordinator.aclose()
    )
    try:
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        await asyncio.wait_for(closing, 2)
        with pytest.raises(IrisRunStateError, match="关闭|撤销"):
            await asyncio.wait_for(requested, 2)
        assert (
            len(
                service.store.read_pending_revisions(
                    allowed_sources=frozenset(),
                    allowed_sessions=frozenset(),
                    allowed_targets=frozenset({("prompt", "project_skill_update")}),
                )
            )
            == 1
        )
    finally:
        release.set()
        await coordinator.aclose()
        requested.cancel()
        await asyncio.gather(requested, closing, return_exceptions=True)
