"""Memory 拥有一轮有界阶段顺序，宿主只负责调度。"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import iris.memory.service as memory_service
from iris.exceptions import IrisMemoryError
from iris.memory import MemoryMaintenanceScope, MemoryObserveInput, MemoryService, SQLiteMemoryStore
from iris.memory.generation_models import GenerationResult, GenerationState, MemorySource
from iris.prompts import PromptSnapshot, PromptSource

from .test_generation import Provider
from .test_generation import service as generation_service
from .test_generation_store import _episode, _flush


async def _eligible(sources: tuple[MemorySource, ...]) -> bool:
    """本测试不含生命周期来源。"""
    return True


@pytest.mark.asyncio
async def test_cycle_returns_completed_dream_revision(tmp_path: Path) -> None:
    """周期结果采用真实完成返回值，不能保留生成前快照的版本。"""
    provider = Provider(
        lambda source: {
            "operations": [
                {
                    "action": "add",
                    "new_key": "new-item",
                    "text": "use uv",
                    "category": "reference",
                    "kind": "fact",
                    "reason": "record",
                    "evidence": source["observations"][0]["evidence"],
                }
            ],
            "resolutions": [
                {
                    "observation_id": source["observations"][0]["id"],
                    "target_id": "new-item",
                    "reason": "stored",
                }
            ],
        }
    )
    service = generation_service(tmp_path, provider)
    _flush(service.store, _episode(service.store))
    cycle = await service.maintain_cycle(
        "project",
        scope=MemoryMaintenanceScope(frozenset(), _eligible, frozenset()),
        cycle_id="cycle",
    )
    (result,) = cycle.results
    stored = next(
        item for item in service.list_generation_results("project").items if item.id == result.id
    )
    assert result.status == "completed" and result.stage == "dream"
    assert (
        result.item_revision
        == stored.item_revision
        == service.generation_state("project").item_revision
        == 1
    )
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_cycle_preserves_flush_remaining_work(tmp_path: Path) -> None:
    """长原文只处理一个有界批次，阶段 has_more 使用消费后返回值。"""
    provider = Provider(lambda _: {"observations": []})
    service = generation_service(tmp_path, provider, flush_input_budget_tokens=1400)
    service.observe(MemoryObserveInput(text="a" * 4000))
    cycle = await service.maintain_cycle(
        "project",
        scope=MemoryMaintenanceScope(frozenset(), _eligible, frozenset()),
        cycle_id="partial",
    )
    (result,) = cycle.results
    assert result.stage == "flush" and result.has_more and cycle.has_more
    assert 0 < result.consumed_ranges[0].end < 4000
    assert len(provider.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_observation", [False, True])
async def test_cycle_dreams_existing_input_first_then_repairs_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_observation: bool
) -> None:
    service = MemoryService(
        SQLiteMemoryStore(tmp_path / "memory.db"), prompt_source=PromptSource.initialize(tmp_path)
    )
    scope = MemoryMaintenanceScope(frozenset(), _eligible, frozenset())
    state = replace(
        service.generation_state("project"),
        pending_episodes=1,
        pending_observations=int(existing_observation),
    )
    calls = []

    async def read(namespace: str, *, scope: MemoryMaintenanceScope) -> GenerationState:
        return state

    async def flush(
        service: MemoryService,
        namespace: str,
        *,
        prompt_snapshot: PromptSnapshot,
        scope: MemoryMaintenanceScope,
    ) -> GenerationResult:
        nonlocal state
        calls.append("flush")
        state = replace(state, pending_episodes=0, pending_observations=1)
        return GenerationResult(namespace=namespace, stage="flush", status="completed")

    async def dream(
        service: MemoryService,
        namespace: str,
        *,
        prompt_snapshot: PromptSnapshot,
        scope: MemoryMaintenanceScope,
    ) -> GenerationResult:
        nonlocal state
        calls.append("dream")
        state = replace(state, pending_observations=0, item_revision=1)
        return GenerationResult(namespace=namespace, stage="dream", status="completed")

    def projection(store: SQLiteMemoryStore, namespace: str) -> None:
        nonlocal state
        calls.append("projection")
        state = replace(state, projection_revision=1)

    async def overview(
        namespace: str, prompt_snapshot: PromptSnapshot
    ) -> tuple[object, GenerationResult]:
        nonlocal state
        calls.append("overview")
        state = replace(state, overview_revision=1)
        return object(), GenerationResult(namespace=namespace, stage="overview", status="completed")

    monkeypatch.setattr(service, "ageneration_state", read)
    monkeypatch.setattr(memory_service, "flush", flush)
    monkeypatch.setattr(memory_service, "dream", dream)
    monkeypatch.setattr(service, "_refresh_overview", overview)
    service.mirror = SimpleNamespace(rebuild_from_store=projection)
    result = await service.maintain_cycle("project", scope=scope, cycle_id="test-cycle")
    assert calls == (["dream"] if existing_observation else ["flush", "dream"]) + [
        "projection",
        "overview",
    ]
    assert result.has_more is existing_observation
    assert [item.stage for item in result.results] == (
        ["dream"] if existing_observation else ["flush", "dream"]
    ) + ["overview"]


@pytest.mark.asyncio
async def test_projection_only_failure_propagates_without_starting_learning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = MemoryService(
        SQLiteMemoryStore(tmp_path / "memory.db"), prompt_source=PromptSource.initialize(tmp_path)
    )
    scope = MemoryMaintenanceScope(frozenset(), _eligible, frozenset())
    state = replace(service.generation_state("project"), item_revision=1, overview_revision=1)
    calls = []

    async def read(namespace: str, *, scope: MemoryMaintenanceScope) -> GenerationState:
        return state

    def projection(store: SQLiteMemoryStore, namespace: str) -> None:
        calls.append("projection")
        raise IrisMemoryError("投影写入失败")

    monkeypatch.setattr(service, "ageneration_state", read)
    service.mirror = SimpleNamespace(rebuild_from_store=projection)
    with pytest.raises(IrisMemoryError, match="投影写入失败"):
        await service.maintain_cycle("project", scope=scope, cycle_id="test-cycle")
    assert calls == ["projection"]
