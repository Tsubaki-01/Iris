"""自动维护只消费已封源且宿主允许的材料。"""

import asyncio
import json
from pathlib import Path

import pytest

from iris.memory import MemoryGenerationConfig, MemoryMaintenanceScope, MemoryService
from iris.memory.generation_models import (
    EpisodeSlice,
    FlushCommit,
    GenerationResult,
    MemoryCaptureSource,
    MemorySource,
)
from iris.memory.models import (
    MemoryEpisode,
    MemoryEvent,
    MemoryEventType,
    MemoryEvidenceRef,
    MemoryItem,
    MemoryObservation,
    MemoryRecord,
    MemorySourceType,
)
from iris.memory.sqlite import SQLiteMemoryStore
from iris.message import LLMRequest, LLMResponse, TextBlock
from iris.prompts import PromptSource


def capture(
    store: SQLiteMemoryStore,
    run_id: str,
    *,
    terminal: bool = True,
    call_id: str = "",
    lifecycle_source_id: str = "host",
) -> MemoryEpisode:
    """写入一个保留真实来源关系的捕获片段。"""
    source = MemoryCaptureSource(
        lifecycle_source_id=lifecycle_source_id,
        run_id=run_id,
        session_id=run_id,
        namespace="project",
        initial_message_count=0,
        captured_until=0,
    )
    store.register_source(source)
    episode = MemoryEpisode(
        source_id=run_id,
        records=(MemoryRecord(text="使用 uv", metadata={"call_id": call_id}),),
        metadata={
            "lifecycle_source_id": lifecycle_source_id,
            "run_id": run_id,
            "session_id": run_id,
        },
    )
    sealed = source.model_copy(
        update={
            "captured_until": 1,
            "terminal_message_count": 1 if terminal else None,
            "outcome": "failed" if terminal else None,
        }
    )
    assert store.commit_capture(sealed, expected_captured_until=0, episode=episode)
    return episode


def tool_source(run_id: str, call_id: str, lifecycle_source_id: str = "host") -> str:
    """构造新契约的稳定工具来源三元组。"""
    return json.dumps(
        [lifecycle_source_id, run_id, call_id], ensure_ascii=False, separators=(",", ":")
    )


def test_scope_filters_before_limit_and_counts_only_eligible(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    capture(store, "waiting")
    capture(store, "unsealed", terminal=False)
    eligible = capture(store, "ready")
    allowed = frozenset({("host", "ready"), ("host", "unsealed")})
    progresses = store.list_pending_episodes("project", limit=1, allowed_sources=allowed)
    assert [p.episode.id for p in progresses] == [eligible.id]
    assert progresses[0].source_outcome == "failed"
    assert store.generation_state("project", allowed_sources=allowed).pending_episodes == 1
    assert {s.run_id for s in store.list_pending_sources("project")} == {
        "waiting",
        "unsealed",
        "ready",
    }


def test_unmatched_tool_change_stays_pending_but_sdk_change_is_eligible(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    tool_event = MemoryEvent(
        event_type=MemoryEventType.ADD,
        source_type=MemorySourceType.TOOL_EVENT,
        source_id=tool_source("ready", "call-1"),
    )
    store.add_item(MemoryItem(text="工具知识"), event=tool_event)
    sdk_event = MemoryEvent(event_type=MemoryEventType.ADD)
    store.add_item(MemoryItem(text="SDK知识"), event=sdk_event)
    empty = frozenset()
    snapshot = store.read_dream_snapshot("project", allowed_sources=empty)
    assert [c.event_id for c in snapshot.changes] == [sdk_event.id]
    capture(store, "ready", call_id="call-1")
    assert [
        c.event_id for c in store.read_dream_snapshot("project", allowed_sources=empty).changes
    ] == [sdk_event.id]
    allowed = frozenset({("host", "ready")})
    snapshot = store.read_dream_snapshot("project", allowed_sources=allowed)
    assert len(snapshot.changes) == 2
    assert [source.run_id for source in snapshot.sources] == ["ready"]


@pytest.mark.parametrize("other_source,other_run", [("host", "other"), ("another-host", "ready")])
def test_reused_call_id_cannot_borrow_another_run_eligibility(
    tmp_path: Path,
    other_source: str,
    other_run: str,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    ready = MemoryEvent(
        event_type=MemoryEventType.ADD,
        source_type=MemorySourceType.TOOL_EVENT,
        source_id=tool_source("ready", "call_1"),
    )
    waiting = MemoryEvent(
        event_type=MemoryEventType.ADD,
        source_type=MemorySourceType.TOOL_EVENT,
        source_id=tool_source(other_run, "call_1", other_source),
    )
    for event in (ready, waiting):
        store.add_item(MemoryItem(text="待整理知识"), event=event)
    capture(store, "ready", call_id="call_1")
    allowed = frozenset({("host", "ready")})
    snapshot = store.read_dream_snapshot("project", allowed_sources=allowed)
    assert [change.event_id for change in snapshot.changes] == [ready.id]
    capture(store, other_run, call_id="call_1", lifecycle_source_id=other_source)
    snapshot = store.read_dream_snapshot("project", allowed_sources=allowed)
    assert [change.event_id for change in snapshot.changes] == [ready.id]
    assert snapshot.sources == (MemorySource("host", "ready", "ready"),)
    assert store.generation_state("project", allowed_sources=allowed).pending_changes == 1


def test_tool_source_without_full_identity_stays_pending(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    event = MemoryEvent(
        event_type=MemoryEventType.ADD,
        source_type=MemorySourceType.TOOL_EVENT,
        source_id="call_1",
    )
    store.add_item(MemoryItem(text="来源不完整"), event=event)
    capture(store, "ready", call_id="call_1")
    snapshot = store.read_dream_snapshot("project", allowed_sources=frozenset({("host", "ready")}))
    assert snapshot.changes == ()


def test_mixed_observation_requires_all_sources_and_scoped_retry(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    episodes = [capture(store, run) for run in ("ready", "waiting")]
    mixed = MemoryObservation(
        text="共同约定",
        reason="两次经历",
        evidence=tuple(
            MemoryEvidenceRef(kind="episode", source_id=e.id, record_id=e.records[0].id, end=5)
            for e in episodes
        ),
    )
    assert store.commit_flush(
        FlushCommit(
            "project",
            tuple(EpisodeSlice(e.id, e.records[0].id, 0, 5, "使用 uv") for e in episodes),
            (mixed,),
            GenerationResult(namespace="project", stage="flush", status="completed"),
        )
    )
    ready = frozenset({("host", "ready")})
    both = frozenset({("host", "ready"), ("host", "waiting")})
    assert store.read_dream_snapshot("project", allowed_sources=ready).observations == ()
    assert {source.run_id for source in store.list_pending_sources("project")} == {
        "ready",
        "waiting",
    }
    snapshot = store.read_dream_snapshot("project", allowed_sources=both)
    assert [o.id for o in snapshot.observations] == [mixed.id]
    assert store.block_dream(snapshot, reason="容量", budget=1, dependency_item_ids=())
    assert store.retry_blocked("project", budget=2, allowed_sources=ready) == 0
    assert store.generation_state("project", allowed_sources=ready).blocked_observations == 0
    assert store.retry_blocked("project", budget=2, allowed_sources=both) == 1


class ScopeProvider:
    """按批次返回空提取或无修改计划，保留真实请求以验证选材。"""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []
        self.after_model = lambda: None

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        return len(request.messages[1].text)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        data = json.loads(request.messages[1].text)
        result = (
            {"observations": []}
            if "records" in data
            else {
                "operations": [],
                "resolutions": [
                    {"observation_id": o["id"], "reason": "无需保留"} for o in data["observations"]
                ],
            }
        )
        self.after_model()
        return LLMResponse(
            provider="test", finish_reason="stop", content=[TextBlock(text=json.dumps(result))]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("has_evidence", [True, False])
async def test_flush_rechecks_actual_batch_before_consuming(
    tmp_path: Path, has_evidence: bool
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    episode = capture(store, "ready")
    if not has_evidence:
        # 原始记录在读取边界声明不可作新证据，仍需消费资格。
        with store._connection() as connection:
            connection.execute(
                "UPDATE memory_episodes SET payload=json_set(payload,"
                "'$.records[0].metadata.evidence_allowed',json('false')) WHERE id=?",
                (episode.id,),
            )
    provider = ScopeProvider()
    memory = MemoryService(
        store,
        generation_provider=provider,
        generation_model="test",
        prompt_source=PromptSource.initialize(tmp_path),
    )
    checks: list[tuple[MemorySource, ...]] = []

    async def check(sources: tuple[MemorySource, ...]) -> bool:
        checks.append(sources)
        return len(checks) == 1

    scope = MemoryMaintenanceScope(frozenset({("host", "ready")}), check)
    with pytest.raises(asyncio.CancelledError):
        await memory.flush("project", scope=scope)
    assert len(checks) == 2 and checks[0] == checks[1]
    assert checks[0][0].run_id == "ready"
    assert len(provider.requests) == int(has_evidence)
    assert memory.generation_state("project").pending_episodes == 1


@pytest.mark.asyncio
async def test_dream_reselection_and_blocking_keep_scope(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    provider = ScopeProvider()
    memory = MemoryService(
        store,
        generation_provider=provider,
        generation_model="test",
        generation_config=MemoryGenerationConfig(dream_input_budget_tokens=5000),
        prompt_source=PromptSource.initialize(tmp_path),
    )
    events = []
    for run, text in (
        ("waiting", "等待中的知识"),
        ("large", "large fact " * 2000),
        ("small", "独立的小事实"),
    ):
        event = MemoryEvent(
            event_type=MemoryEventType.ADD,
            source_type=MemorySourceType.TOOL_EVENT,
            source_id=tool_source(run, run),
        )
        store.add_item(MemoryItem(text=text), event=event)
        capture(store, run, call_id=run)
        events.append(event)
    checked = []

    async def check(sources: tuple[MemorySource, ...]) -> bool:
        checked.append({source.run_id for source in sources})
        return True

    scope = MemoryMaintenanceScope(frozenset({("host", "large"), ("host", "small")}), check)
    result = await memory.dream("project", scope=scope)
    assert result.status == "completed" and result.counts["blocked"] == 1
    assert not result.has_more
    assert checked == [{"large"}, {"small"}, {"small"}]
    assert len(provider.requests) == 1
    assert "等待中的知识" not in provider.requests[0].messages[1].text
    assert [c.event_id for c in store.read_dream_snapshot("project").changes] == [events[0].id]


@pytest.mark.asyncio
async def test_dream_checks_sources_after_model_and_before_block(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    event = MemoryEvent(
        event_type=MemoryEventType.ADD,
        source_type=MemorySourceType.TOOL_EVENT,
        source_id=tool_source("ready", "call"),
    )
    store.add_item(MemoryItem(text="待整理"), event=event)
    capture(store, "ready", call_id="call")
    provider = ScopeProvider()
    memory = MemoryService(
        store,
        generation_provider=provider,
        generation_model="test",
        prompt_source=PromptSource.initialize(tmp_path),
    )
    eligible = True

    def expire() -> None:
        nonlocal eligible
        eligible = False

    provider.after_model = expire

    async def check(sources: tuple[MemorySource, ...]) -> bool:
        assert [source.run_id for source in sources] == ["ready"]
        return eligible

    scope = MemoryMaintenanceScope(frozenset({("host", "ready")}), check)
    with pytest.raises(asyncio.CancelledError):
        await memory.dream("project", scope=scope)
    assert memory.generation_state("project").pending_changes == 1
    memory.generation_config = MemoryGenerationConfig(dream_input_budget_tokens=1)
    with pytest.raises(asyncio.CancelledError):
        await memory.dream("project", scope=scope)
    assert memory.generation_state("project").blocked_changes == 0


@pytest.mark.asyncio
async def test_capacity_blocking_yields_after_a_bounded_batch(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    for index in range(17):
        store.add_item(
            MemoryItem(text=f"事实 {index}"), event=MemoryEvent(event_type=MemoryEventType.ADD)
        )
    provider = ScopeProvider()
    memory = MemoryService(
        store,
        generation_provider=provider,
        generation_model="test",
        generation_config=MemoryGenerationConfig(dream_input_budget_tokens=1),
        prompt_source=PromptSource.initialize(tmp_path),
    )
    result = await memory.dream("project")
    assert result.status == "blocked" and result.counts["blocked"] == 16
    assert result.has_more and not provider.requests
    assert memory.generation_state("project").pending_changes == 1
