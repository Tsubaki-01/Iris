"""真实 Run 捕获到自动学习的时间与数量双门槛。"""

import asyncio
import json
import multiprocessing
from collections.abc import Callable
from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.evolution.config import EvolutionConfig
from iris.evolution.materials import EvolutionMaterialStore
from iris.evolution.models import EvolutionMaintenanceScope, EvolutionResult
from iris.evolution.revision import PromptTarget
from iris.evolution.service import EvolutionService
from iris.exceptions import IrisMemoryError
from iris.harness import (
    AgentRunner,
    MaintenanceChanged,
    MaintenanceCoordinator,
    MemoryMaintenanceBinding,
    ProjectEvolutionBinding,
)
from iris.lifecycle import AgentRunRequest
from iris.memory import MemoryObserveInput, MemoryService
from iris.memory.generation_models import (
    MemoryCycleResult,
    MemoryLearningReadiness,
    MemoryMaintenanceScope,
    MemorySource,
)
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.prompts import PromptSource
from iris.store import SQLiteStore
from iris.streaming.projection import project_live_fact
from iris.tools import ToolCapability, ToolRegistry

from .fakes import RecordingPublisher, StaticProvider, build_runtime, text_response, tool_response
from .test_evolution_maintenance import project_runner, project_service
from .test_maintenance_coordinator import configured_runner, memory_service


async def _until(predicate: Callable[[], bool], timeout: float = 5) -> None:
    """等待可观察状态而非固定运行速度，涵盖一秒短探测下限。"""
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


def _memory_setup(
    path: Path,
    *,
    idle_seconds: float = 0,
    provider: StaticProvider | None = None,
    publisher: RecordingPublisher | None = None,
) -> tuple[
    MaintenanceCoordinator, AgentRunner, MemoryService, MemoryMaintenanceBinding, StaticProvider
]:
    """独立固定主模型和生成模型，使用真实 SQLite lifecycle 与原文捕获。"""
    generated = provider or StaticProvider(
        *[text_response('{"observations": []}') for _ in range(30)]
    )
    memory = memory_service(path / "memory.db", generated)
    runner = configured_runner(path, memory, store=SQLiteStore(path / "lifecycle.db"))
    runner.runtime.environment.provider = StaticProvider(*[text_response() for _ in range(30)])
    coordinator = MaintenanceCoordinator(idle_seconds=idle_seconds, live_publisher=publisher)
    binding = MemoryMaintenanceBinding(
        service=memory, database_path=path / "memory.db", namespace="project"
    )
    runner.bind_maintenance(coordinator, memory=binding)
    coordinator._foreground_enter()
    return coordinator, runner, memory, binding, generated


async def _runs(runner: AgentRunner, count: int) -> None:
    for index in range(count):
        await runner.start(AgentRunRequest(input=f"第 {index} 次真实经历"))


@pytest.mark.asyncio
async def test_memory_waits_for_ten_runs_and_projects_waiting_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publisher = RecordingPublisher()
    coordinator, runner, memory, _, generated = _memory_setup(tmp_path, publisher=publisher)
    try:
        await _runs(runner, 9)
        coordinator._foreground_exit()
        await _until(
            lambda: (
                bool(generated.requests)
                or coordinator.snapshot().resources[0].state == "waiting_for_materials"
            )
        )
        assert generated.requests == []
        view = coordinator.snapshot().resources[0]
        assert view.pending_new_runs == 9 and view.min_pending_runs == 10
        assert view.next_eligible_at is None
        assert all(
            not item.admitted for item in memory.store.read_learning_readiness("project").sources
        )
        changed = next(
            fact
            for fact in publisher.facts
            if isinstance(fact, MaintenanceChanged)
            and fact.resource.state == "waiting_for_materials"
        )
        wire_resource = project_live_fact(changed)[0].payload["resource"]
        assert isinstance(wire_resource, dict)
        assert wire_resource["state"] == "waiting_for_materials"
        assert wire_resource["pending_new_runs"] == 9
        assert wire_resource["min_pending_runs"] == 10
        assert wire_resource["next_eligible_at"] is None
        facts_before_probe = len(publisher.facts)
        probes = 0
        read = memory.aread_learning_readiness

        async def count_probe(namespace: str) -> MemoryLearningReadiness:
            nonlocal probes
            probes += 1
            return await read(namespace)

        async def no_body_preparation(namespace: str) -> tuple[MemorySource, ...]:
            raise AssertionError("普通未达门槛探测不应进入解析正文的下游准备")

        with monkeypatch.context() as check:
            check.setattr(memory, "aread_learning_readiness", count_probe)
            check.setattr(memory, "alist_pending_sources", no_body_preparation)
            assert coordinator.snapshot() is coordinator.snapshot()
            assert probes == 0
            await asyncio.sleep(1.2)
            assert 1 <= probes <= 2
            assert coordinator.snapshot().resources[0].state == "waiting_for_materials"
            assert memory.list_generation_results("project").items == ()
            assert generated.requests == []
            assert all(
                fact.resource.cycle_id is None
                for fact in publisher.facts[facts_before_probe:]
                if isinstance(fact, MaintenanceChanged)
            )
        coordinator._foreground_enter()
        await _runs(runner, 1)
        coordinator._foreground_exit()
        await _until(
            lambda: (
                bool(generated.requests)
                and memory.generation_state("project").pending_episodes == 0
            )
        )
        assert len(generated.requests) == 1
    finally:
        await runner.aclose()
        await coordinator.aclose()


@pytest.mark.asyncio
async def test_explicit_unsourced_material_does_not_admit_nine_new_runs(tmp_path: Path) -> None:
    coordinator, runner, memory, _, generated = _memory_setup(tmp_path)
    try:
        await _runs(runner, 9)
        memory.observe(MemoryObserveInput(text="显式整理这条材料", namespace="project"))
        coordinator._foreground_exit()
        await _until(
            lambda: (
                bool(generated.requests)
                and coordinator.snapshot().resources[0].state == "waiting_for_materials"
            )
        )
        assert len(generated.requests) == 1
        payload = json.loads(generated.requests[0].messages[-1].text)
        assert [record["text"] for record in payload["records"]] == ["显式整理这条材料"]
        remaining = memory.store.read_learning_readiness("project")
        assert len(remaining.sources) == 9 and not remaining.has_unsourced
        assert all(not item.admitted for item in remaining.sources)
    finally:
        await runner.aclose()
        await coordinator.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [3, 10])
async def test_idle_and_count_hold_automatic_work_but_manual_request_bypasses_both(
    tmp_path: Path, count: int
) -> None:
    coordinator, runner, memory, binding, generated = _memory_setup(tmp_path, idle_seconds=3600)
    try:
        await _runs(runner, count)
        coordinator._foreground_exit()
        await asyncio.sleep(0)
        assert generated.requests == []
        assert all(
            not item.admitted for item in memory.store.read_learning_readiness("project").sources
        )
        result = await asyncio.wait_for(coordinator.request_memory_cycle(binding), 5)
        assert [item.stage for item in result.results] == ["flush"]
        assert memory.generation_state("project").pending_episodes == 0
        assert len(generated.requests) == 1
    finally:
        await runner.aclose()
        await coordinator.aclose()


@pytest.mark.asyncio
async def test_admitted_batch_continues_with_nine_runs_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class OneRunProvider(StaticProvider):
        def estimate_input_tokens(self, request: LLMRequest) -> int:
            return len(json.loads(request.messages[-1].text).get("records", [])) or 1

    provider = OneRunProvider(*[text_response('{"observations": []}') for _ in range(30)])
    coordinator, runner, memory, _, _ = _memory_setup(tmp_path, provider=provider)
    memory.generation_config = memory.generation_config.model_copy(
        update={"flush_input_budget_tokens": 2}
    )
    finished = asyncio.Event()
    maintain = memory.maintain_cycle

    async def one_cycle(
        namespace: str, *, scope: MemoryMaintenanceScope, cycle_id: str
    ) -> MemoryCycleResult:
        result = await maintain(namespace, scope=scope, cycle_id=cycle_id)
        coordinator.idle_seconds = 3600
        finished.set()
        return result

    monkeypatch.setattr(memory, "maintain_cycle", one_cycle)
    try:
        await _runs(runner, 10)
        coordinator._foreground_exit()
        await asyncio.wait_for(finished.wait(), 5)
        remaining = memory.store.read_learning_readiness("project").sources
        assert len(remaining) == 9 and all(item.admitted for item in remaining)
    finally:
        await runner.aclose()
        await coordinator.aclose()
    reopened = memory_service(tmp_path / "memory.db", provider)
    reopened.generation_config = reopened.generation_config.model_copy(
        update={"flush_input_budget_tokens": 2}
    )
    successor = MaintenanceCoordinator(idle_seconds=0)
    successor._attach(
        MemoryMaintenanceBinding(
            service=reopened, database_path=tmp_path / "memory.db", namespace="project"
        ),
        SQLiteStore(tmp_path / "lifecycle.db"),
    )
    try:
        await successor.prepare()
        await _until(lambda: reopened.generation_state("project").pending_episodes == 0)
        assert len(provider.requests) == 10
    finally:
        await successor.aclose()


@pytest.mark.asyncio
async def test_evolution_has_its_own_ten_run_gate(tmp_path: Path) -> None:
    generated = StaticProvider(*[text_response('{"body":null,"reason":"保持"}') for _ in range(5)])
    service = project_service(tmp_path, generated)
    runner = project_runner(tmp_path, StaticProvider(*[text_response() for _ in range(15)]))
    coordinator = MaintenanceCoordinator(idle_seconds=0)
    binding = ProjectEvolutionBinding(workspace_root=tmp_path, service=service)
    runner.bind_maintenance(coordinator, evolution=binding)
    coordinator._foreground_enter()
    try:
        await _runs(runner, 9)
        coordinator._foreground_exit()
        await _until(
            lambda: (
                bool(generated.requests)
                or coordinator.snapshot().resources[0].state == "waiting_for_materials"
            )
        )
        assert generated.requests == []
        assert service.list_publications().items == ()
        assert coordinator.snapshot().resources[0].pending_new_runs == 9
        coordinator._foreground_enter()
        await _runs(runner, 1)
        coordinator._foreground_exit()
        await _until(
            lambda: (
                bool(service.list_publications().items)
                and service.list_publications().items[-1].settled
            )
        )
        assert len(generated.requests) == 1
        assert service.store.read_learning_readiness().sources == ()
    finally:
        await runner.aclose()
        await coordinator.aclose()


@pytest.mark.asyncio
async def test_failed_learning_does_not_retry_on_material_probe(tmp_path: Path) -> None:
    class Failing(StaticProvider):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            raise IrisMemoryError("暂时无法生成")

    coordinator, runner, _, _, generated = _memory_setup(tmp_path, provider=Failing())
    try:
        await _runs(runner, 10)
        coordinator._foreground_exit()
        await _until(lambda: bool(generated.requests) and coordinator._task is None)
        await asyncio.sleep(1.1)
        assert len(generated.requests) == 1
        assert coordinator.snapshot().resources[0].state != "waiting_for_materials"
    finally:
        await runner.aclose()
        await coordinator.aclose()


def _remote_capture(directory: str) -> None:
    """独立宿主完成第十个 Run，只捕获，不向父进程发送 wake。"""

    async def run() -> None:
        coordinator, runner, _, _, _ = _memory_setup(Path(directory), idle_seconds=3600)
        try:
            await _runs(runner, 1)
        finally:
            await runner.aclose()
            await coordinator.aclose()

    asyncio.run(run())


@pytest.mark.asyncio
async def test_probe_discovers_tenth_run_from_another_process(tmp_path: Path) -> None:
    coordinator, runner, memory, _, generated = _memory_setup(tmp_path)
    process = multiprocessing.get_context("spawn").Process(
        target=_remote_capture, args=(str(tmp_path),)
    )
    try:
        await _runs(runner, 9)
        coordinator._foreground_exit()
        await _until(lambda: coordinator.snapshot().resources[0].state == "waiting_for_materials")
        process.start()
        await asyncio.to_thread(process.join, 15)
        assert process.exitcode == 0
        await _until(
            lambda: (
                bool(generated.requests)
                and memory.generation_state("project").pending_episodes == 0
            )
        )
        assert len(generated.requests) == 1
    finally:
        if process.is_alive():
            process.terminate()
            await asyncio.to_thread(process.join, 5)
        await runner.aclose()
        await coordinator.aclose()


@pytest.mark.asyncio
async def test_probe_rechecks_lifecycle_without_memory_change(tmp_path: Path) -> None:
    coordinator, runner, memory, _, generated = _memory_setup(tmp_path)
    registry = ToolRegistry()
    registry.register_function(
        lambda: "written", name="write", description="写", capabilities={ToolCapability.WRITE}
    )
    peer = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            provider=StaticProvider(tool_response(ToolUseBlock(id="write", name="write"))),
            registry=registry,
        ),
        store=SQLiteStore(tmp_path / "lifecycle.db"),
    )
    try:
        await runner.start(
            AgentRunRequest(input="暂时被同会话 WAITING 阻止的旧经历", session_id="a")
        )
        await _runs(runner, 9)
        waiting = await peer.start(AgentRunRequest(input="等待许可", session_id="a"))
        assert waiting.pending_interaction is not None
        coordinator._foreground_exit()
        await _until(lambda: coordinator.snapshot().resources[0].state == "waiting_for_materials")
        assert coordinator.snapshot().resources[0].pending_new_runs == 9
        before = memory.store.read_learning_readiness("project")
        await peer.cancel(waiting.run.run_id)
        assert memory.store.read_learning_readiness("project") == before
        await _until(
            lambda: (
                bool(generated.requests)
                and memory.generation_state("project").pending_episodes == 0
            )
        )
        assert len(generated.requests) == 1
    finally:
        await peer.aclose()
        await runner.aclose()
        await coordinator.aclose()


class _BatchEvolutionProvider(StaticProvider):
    """每轮仅容纳一条消息；首轮从实际输入提出 Q，随后 B 修订明确目标。"""

    def __init__(self) -> None:
        super().__init__()
        self.learned_refs: list[str] = []

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        return len(json.loads(request.messages[-1].text).get("materials", [])) or 1

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        payload = json.loads(request.messages[-1].text)
        if "materials" not in payload:
            return text_response(
                '{"action":"prompt","target":"compaction",'
                '"body":"保留明确的项目工具约定。","reason":"采用原文约定"}'
            )
        records = [record for item in payload["materials"] for record in item["records"]]
        first = not self.learned_refs
        self.learned_refs.extend(record["ref"] for record in records)
        output: dict[str, object] = {"body": None, "reason": "经验正文不变"}
        if first:
            output["issue"] = {
                "description": "摘要保留项目工具约定",
                "targets": [{"kind": "prompt", "name": "compaction"}],
                "evidence": [{"ref": records[0]["ref"], "quote": records[0]["text"]}],
            }
        return text_response(json.dumps(output, ensure_ascii=False))


def _batch_project(
    path: Path, provider: _BatchEvolutionProvider
) -> tuple[EvolutionService, AgentRunner, MaintenanceCoordinator, ProjectEvolutionBinding]:
    """以相同配置重建真实领域服务、持久 lifecycle 与宿主。"""
    service = EvolutionService(
        workspace_root=path,
        skill_path=path / ".agents" / "skills" / "project-experience" / "SKILL.md",
        store=EvolutionMaterialStore(path),
        provider=provider,
        model="test-evolution",
        config=EvolutionConfig(enabled=True, input_budget_tokens=1, prompt_targets=("compaction",)),
        prompt_source=PromptSource.initialize(path),
        prompt_targets=(PromptTarget("compaction", "保留工具约定", {}),),
    )
    runner = AgentRunner.from_config(
        AgentConfig(
            name="project",
            model="openai/test",
            system="完成任务",
            permissions={"workspace": str(path)},
            context_policy={"enabled": False},
            skills={"enabled": True},
            evolution={"enabled": True},
        ),
        provider=StaticProvider(*[text_response() for _ in range(12)]),
        store=SQLiteStore(path / "lifecycle.db"),
    )
    coordinator = MaintenanceCoordinator(idle_seconds=0)
    binding = ProjectEvolutionBinding(workspace_root=path, service=service)
    runner.bind_maintenance(coordinator, evolution=binding)
    return service, runner, coordinator, binding


@pytest.mark.asyncio
async def test_batch_restart_revision_retention_and_next_batch_are_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """X1/X2：十 Run 分预算续处理，Q/B 结算，十一条裁剪后余料与新批各自推进。"""
    provider = _BatchEvolutionProvider()
    service, runner, coordinator, _ = _batch_project(tmp_path, provider)
    first_done = asyncio.Event()
    maintain = service.maintain_cycle

    async def first_cycle(*, scope: EvolutionMaintenanceScope) -> EvolutionResult:
        result = await maintain(scope=scope)
        coordinator.idle_seconds = 3600
        first_done.set()
        return result

    monkeypatch.setattr(service, "maintain_cycle", first_cycle)
    coordinator._foreground_enter()
    try:
        await _runs(runner, 10)
        coordinator._foreground_exit()
        await asyncio.wait_for(first_done.wait(), 5)
        (first,) = service.list_publications().items
        assert first.proposed_revision_id is not None
        assert len(service.store.read_learning_readiness().sources) == 10
        assert all(item.admitted for item in service.store.read_learning_readiness().sources)
    finally:
        await runner.aclose()
        await coordinator.aclose()

    service, runner, coordinator, _ = _batch_project(tmp_path, provider)
    eleventh_done = asyncio.Event()
    maintain_reopened = service.maintain_cycle

    async def stop_at_eleven(*, scope: EvolutionMaintenanceScope) -> EvolutionResult:
        result = await maintain_reopened(scope=scope)
        if len(service.list_publications().items) == 11:
            coordinator.idle_seconds = 3600
            eleventh_done.set()
        return result

    monkeypatch.setattr(service, "maintain_cycle", stop_at_eleven)
    try:
        await runner.aprepare()
        await asyncio.wait_for(eleventh_done.wait(), 10)
        archived = service.get_publication(first.publication_id)
        assert archived.detail_status == "expired" and archived.detail is None
        assert archived.summary.proposed_revision_id == first.proposed_revision_id
        assert service.store.revision_result(first.proposed_revision_id).status == "updated"
        assert (service.prompt_source.root / "compaction.j2").read_text(encoding="utf-8") == (
            "保留明确的项目工具约定。"
        )
        remainder = service.store.read_learning_readiness().sources
        assert 0 < len(remainder) < 10 and all(item.admitted for item in remainder)
        coordinator._foreground_enter()
        fresh = await runner.start(AgentRunRequest(input="下一批的新经历"))
        coordinator.idle_seconds = 0
        coordinator._foreground_exit()
        await _until(lambda: coordinator.snapshot().resources[0].state == "waiting_for_materials")
        (last,) = service.store.read_learning_readiness().sources
        assert last.source.run_id == fresh.run.run_id and not last.admitted
        assert coordinator.snapshot().resources[0].pending_new_runs == 1
        assert len(provider.learned_refs) == len(set(provider.learned_refs)) == 20
        assert len(provider.requests) == 21
        assert (
            sum(item.detail_status == "available" for item in service.list_publications().items)
            == 10
        )
    finally:
        await runner.aclose()
        await coordinator.aclose()
