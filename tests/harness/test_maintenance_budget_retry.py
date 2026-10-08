"""自动维护须保留调大预算后恢复已有受阻材料的正常路径。"""

import asyncio
from pathlib import Path

import pytest

from iris.harness import AgentRunner, MaintenanceCoordinator, MemoryMaintenanceBinding
from iris.lifecycle import AgentRunRequest
from iris.memory.generation_models import (
    MemoryCycleResult,
    MemoryLearningReadiness,
    MemoryMaintenanceScope,
)
from iris.message import ToolUseBlock
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import ToolCapability, ToolRegistry

from ..memory.test_generation import Provider, flush_one, service
from ..memory.test_generation_store import _episode, _flush
from .fakes import StaticProvider, build_runtime, tool_response
from .test_maintenance_coordinator import configured_runner, memory_service


def _ignore_observation(payload: dict[str, object]) -> dict[str, object]:
    """按实际待处理标识完成观察，不额外写入新知识。"""
    return {
        "operations": [],
        "resolutions": [
            {
                "observation_id": payload["observations"][0]["id"],
                "target_id": None,
                "reason": "无新增知识",
            }
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [1, 32000])
async def test_reopened_host_retries_blocked_observation_only_after_budget_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: int
) -> None:
    """仅有 blocked 的库重新装配后应识别新预算，同预算不产生空维护周期。"""
    provider = Provider(_ignore_observation)
    original = service(tmp_path, provider, dream_input_budget_tokens=1)
    _flush(original.store, _episode(original.store))
    assert (await original.dream("project")).status == "blocked"
    before = original.generation_state("project")
    assert before.pending_episodes == before.pending_observations == before.pending_changes == 0
    assert before.blocked_observations == 1 and provider.requests == []

    reopened = service(tmp_path, provider, dream_input_budget_tokens=budget)
    first_probe = asyncio.Event()
    read = reopened.aread_learning_readiness
    maintain = reopened.maintain_cycle
    cycles: list[MemoryCycleResult] = []

    async def observed_read(namespace: str) -> MemoryLearningReadiness:
        readiness = await read(namespace)
        first_probe.set()
        return readiness

    async def observed_cycle(
        namespace: str, *, scope: MemoryMaintenanceScope, cycle_id: str
    ) -> MemoryCycleResult:
        result = await maintain(namespace, scope=scope, cycle_id=cycle_id)
        cycles.append(result)
        return result

    monkeypatch.setattr(reopened, "aread_learning_readiness", observed_read)
    monkeypatch.setattr(reopened, "maintain_cycle", observed_cycle)
    coordinator = MaintenanceCoordinator(idle_seconds=0)
    coordinator._attach(
        MemoryMaintenanceBinding(
            service=reopened, database_path=tmp_path / "memory.db", namespace="project"
        ),
        InMemoryLifecycleStore(),
    )
    try:
        await coordinator.prepare()
        await asyncio.wait_for(first_probe.wait(), 2)
        async with asyncio.timeout(2):
            while coordinator._task is not None:
                await asyncio.sleep(0)
        if budget == 1:
            assert cycles == [] and provider.requests == []
            assert reopened.generation_state("project").blocked_observations == 1
        else:
            assert len(cycles) == len(provider.requests) == 1
            assert [(result.stage, result.status) for result in cycles[0].results] == [
                ("dream", "completed")
            ]
            assert not cycles[0].has_more
            after = reopened.generation_state("project")
            assert after.blocked_observations == after.pending_observations == 0
    finally:
        await coordinator.aclose()


@pytest.mark.asyncio
async def test_changed_budget_retries_after_lifecycle_only_waiting_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """新预算先受 WAITING 阻止，只有生命周期恢复时仍能重新判断受阻材料。"""
    original = memory_service(tmp_path / "memory.db", Provider(flush_one))
    original.generation_config = original.generation_config.model_copy(
        update={"dream_input_budget_tokens": 1}
    )
    lifecycle_path = tmp_path / "lifecycle.db"
    runner = configured_runner(tmp_path, original, store=SQLiteStore(lifecycle_path))
    previous = MaintenanceCoordinator(idle_seconds=3600)
    runner.bind_maintenance(
        previous,
        memory=MemoryMaintenanceBinding(
            service=original, database_path=tmp_path / "memory.db", namespace="project"
        ),
    )
    try:
        await runner.start(AgentRunRequest(input="本项目使用 uv", session_id="shared"))
        assert (await original.flush("project")).status == "completed"
        assert (await original.dream("project")).status == "blocked"
        assert original.generation_state("project").pending_episodes == 0
        original.mirror.rebuild_from_store(original.store, "project")
    finally:
        await runner.aclose()
        await previous.aclose()

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
        store=SQLiteStore(lifecycle_path),
    )
    provider = Provider(_ignore_observation)
    reopened = memory_service(tmp_path / "memory.db", provider)
    coordinator = MaintenanceCoordinator(idle_seconds=0)
    first_probe = asyncio.Event()
    read = reopened.aread_learning_readiness

    async def observed_read(namespace: str) -> MemoryLearningReadiness:
        readiness = await read(namespace)
        first_probe.set()
        return readiness

    monkeypatch.setattr(reopened, "aread_learning_readiness", observed_read)
    coordinator._attach(
        MemoryMaintenanceBinding(
            service=reopened, database_path=tmp_path / "memory.db", namespace="project"
        ),
        SQLiteStore(lifecycle_path),
    )
    try:
        waiting = await peer.start(AgentRunRequest(input="等待许可", session_id="shared"))
        assert waiting.pending_interaction is not None
        await coordinator.prepare()
        await asyncio.wait_for(first_probe.wait(), 2)
        async with asyncio.timeout(2):
            while coordinator._task is not None:
                await asyncio.sleep(0)
        assert provider.requests == []
        before = reopened.store.read_learning_readiness("project")
        assert before.sources == () and not before.has_pending_derived
        await peer.cancel(waiting.run.run_id)
        assert reopened.store.read_learning_readiness("project") == before
        async with asyncio.timeout(3):
            while reopened.generation_state("project").blocked_observations:
                await asyncio.sleep(0.01)
        assert len(provider.requests) == 1
        assert reopened.generation_state("project").pending_observations == 0
    finally:
        await peer.aclose()
        await coordinator.aclose()
