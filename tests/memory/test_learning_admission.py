"""原文准入只计独立 Run，短事实与消费游标保持一致。"""

import sqlite3
from pathlib import Path

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory import MemoryGenerationConfig, MemoryMaintenanceScope, MemoryService
from iris.memory.generation_models import (
    EpisodeSlice,
    FlushCommit,
    GenerationResult,
    MemoryCaptureSource,
    MemorySource,
)
from iris.memory.models import MemoryEpisode, MemoryObserveInput, MemoryRecord, MemoryWriteInput
from iris.memory.sqlite import SQLiteMemoryStore
from iris.prompts import PromptSource

from .test_generation_store import _flush
from .test_maintenance_scope import ScopeProvider


def capture(
    store: SQLiteMemoryStore,
    run_id: str,
    *,
    namespace: str = "project",
    records: tuple[MemoryRecord, ...] | None = None,
    terminal: bool = True,
) -> tuple[MemoryCaptureSource, MemoryEpisode]:
    """给同一来源追加一页完整材料，可先捕获再封源。"""
    source = store.register_source(
        MemoryCaptureSource(
            lifecycle_source_id="host",
            run_id=run_id,
            session_id=run_id,
            namespace=namespace,
            initial_message_count=0,
            captured_until=0,
        )
    )
    episode = MemoryEpisode(
        namespace=namespace,
        source_id=run_id,
        records=records or (MemoryRecord(id="message-1", text="使用 uv"),),
        metadata={"lifecycle_source_id": "host", "session_id": run_id},
    )
    updated = source.model_copy(
        update={
            "captured_until": source.captured_until + 1,
            "terminal_message_count": source.captured_until + 1 if terminal else None,
            "outcome": "failed" if terminal else None,
        }
    )
    assert store.commit_capture(
        updated, expected_captured_until=source.captured_until, episode=episode
    )
    return updated, episode


def consume(store: SQLiteMemoryStore, episode: MemoryEpisode, start: int, end: int) -> bool:
    """消费首条记录的指定连续区间。"""
    record = episode.records[0]
    return store.commit_flush(
        FlushCommit(
            episode.namespace,
            (EpisodeSlice(episode.id, record.id, start, end, record.text[start:end]),),
            (),
            GenerationResult(namespace=episode.namespace, stage="flush", status="completed"),
        )
    )


async def eligible(sources: tuple[MemorySource, ...]) -> bool:
    """测试宿主已确认传入来源的生命周期资格。"""
    return True


def service(
    path: Path, store: SQLiteMemoryStore, provider: ScopeProvider, *, budget: int = 32000
) -> MemoryService:
    """使用实际存储与无工具模型替身，不启用镜像。"""
    return MemoryService(
        store,
        generation_provider=provider,
        generation_model="test",
        generation_config=MemoryGenerationConfig(flush_input_budget_tokens=budget),
        prompt_source=PromptSource.initialize(path),
    )


def test_readiness_and_admission_do_not_read_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    capture(store, "ready")
    capture(store, "unsealed", terminal=False)
    capture(store, "empty", records=(MemoryRecord(text="", metadata={"evidence_allowed": False}),))
    MemoryService(store).observe(MemoryObserveInput(text="显式材料"))
    connect = store._connect

    def metadata_only() -> sqlite3.Connection:
        connection = connect()

        def authorize(action: int, table: str, column: str, database: str, source: str) -> int:
            if action == sqlite3.SQLITE_READ and column == "payload":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)
        return connection

    monkeypatch.setattr(store, "_connect", metadata_only)
    before = store.read_learning_readiness("project")
    assert before.has_unsourced
    assert before.item_revision == 0 and before.projection_revision is None
    assert not before.has_pending_derived
    rows = {item.source.run_id: item for item in before.sources}
    assert rows["ready"].complete and rows["ready"].has_content
    assert not rows["unsealed"].complete
    assert not rows["empty"].has_content
    after = store.admit_learning_sources(
        "project", allowed_sources=frozenset({("host", "ready"), ("host", "empty")}), threshold=1
    )
    assert {item.source.run_id for item in after.sources if item.admitted} == {"ready"}


def test_admission_counts_runs_freshly_and_isolates_namespaces(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    capture(store, "r0", terminal=False)
    capture(store, "r0")
    for index in range(1, 9):
        capture(store, f"r{index}")
    capture(store, "r9", terminal=False)
    capture(store, "empty", records=(MemoryRecord(text=""),))
    capture(store, "r0", namespace="other")
    allowed = frozenset(("host", f"r{index}") for index in range(10))
    first = store.admit_learning_sources("project", allowed_sources=allowed, threshold=10)
    assert not any(item.admitted for item in first.sources)
    capture(store, "r9")
    second = store.admit_learning_sources("project", allowed_sources=allowed, threshold=10)
    assert len([item for item in second.sources if item.admitted]) == 10
    capture(store, "r10")
    third = store.admit_learning_sources(
        "project", allowed_sources=allowed | {("host", "r10")}, threshold=10
    )
    assert len([item for item in third.sources if item.admitted]) == 10
    assert not store.read_learning_readiness("other").sources[0].admitted


@pytest.mark.asyncio
async def test_partial_consumption_reopens_admitted_and_drains_empty_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    source, episode = capture(
        store,
        "run",
        records=(
            MemoryRecord(id="first", text="uv"),
            MemoryRecord(id="tail", text="ignored" * 100, metadata={"evidence_allowed": False}),
        ),
    )
    allowed = frozenset({("host", "run")})
    store.admit_learning_sources("project", allowed_sources=allowed, threshold=1)
    assert consume(store, episode, 0, 1)
    reopened = SQLiteMemoryStore(store.path)
    state = reopened.read_learning_readiness("project").sources[0]
    assert state.admitted and state.has_content
    assert not reopened.commit_capture(source, expected_captured_until=0, episode=episode)
    assert not consume(reopened, episode, 0, 1)
    assert reopened.read_learning_readiness("project").sources[0] == state
    assert reopened.commit_capture(source, expected_captured_until=1, episode=None)
    assert consume(reopened, episode, 1, 2)
    state = reopened.read_learning_readiness("project").sources[0]
    assert state.admitted and not state.has_content
    provider = ScopeProvider()
    monkeypatch.setattr(
        provider, "estimate_input_tokens", lambda _: pytest.fail("无证据尾部不应估算模型预算")
    )
    memory = service(tmp_path, reopened, provider, budget=1)
    cycle = await memory.maintain_cycle(
        "project",
        scope=MemoryMaintenanceScope(allowed, eligible, episode_sources=allowed),
        cycle_id="empty-tail",
    )
    assert not provider.requests and not cycle.has_more
    assert reopened.read_learning_readiness("project").sources == ()


@pytest.mark.asyncio
async def test_episode_scope_excludes_raw_but_preserves_dream_and_unsourced(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    _, old = capture(store, "old")
    _flush(store, old)
    capture(store, "new")
    provider = ScopeProvider()
    memory = service(tmp_path, store, provider)
    memory.remember(MemoryWriteInput(text="SDK显式记忆", reason="用户要求"))
    scope = MemoryMaintenanceScope(
        frozenset({("host", "old"), ("host", "new")}), eligible, episode_sources=frozenset()
    )
    first = await memory.maintain_cycle("project", scope=scope, cycle_id="downstream")
    assert [result.stage for result in first.results] == ["dream"]
    assert not first.has_more
    assert store.generation_state("project").pending_episodes == 1
    memory.observe(MemoryObserveInput(text="无Run的显式原文"))
    second = await memory.maintain_cycle("project", scope=scope, cycle_id="unsourced")
    assert [result.stage for result in second.results] == ["flush"]
    assert not second.has_more
    assert store.generation_state("project").pending_episodes == 1


@pytest.mark.asyncio
async def test_unadmitted_empty_source_drains_without_reading_other_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    capture(store, "empty", records=(MemoryRecord(text="", metadata={"evidence_allowed": False}),))
    _, blocked = capture(store, "blocked")
    provider = ScopeProvider()
    monkeypatch.setattr(
        provider, "estimate_input_tokens", lambda _: pytest.fail("全空来源不应估算模型预算")
    )
    memory = service(tmp_path, store, provider, budget=1)
    scope = MemoryMaintenanceScope(
        frozenset({("host", "empty"), ("host", "blocked")}),
        eligible,
        episode_sources=frozenset({("host", "empty")}),
    )
    result = await memory.maintain_cycle("project", scope=scope, cycle_id="empty")
    assert not provider.requests and not result.has_more
    assert [item.episode.id for item in store.list_pending_episodes("project")] == [blocked.id]


@pytest.mark.asyncio
async def test_readiness_reports_derived_hint_and_async_admission(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    capture(store, "run")
    memory = MemoryService(store)
    memory.remember(MemoryWriteInput(text="显式事实", reason="用户要求"))
    ready = await memory.aread_learning_readiness("project")
    assert ready.item_revision == 1 and ready.has_pending_derived
    admitted = await memory.aadmit_learning_sources(
        "project", allowed_sources=frozenset({("host", "run")}), threshold=1
    )
    assert admitted.sources[0].admitted and admitted.item_revision == 1
    assert admitted.has_pending_derived
    snapshot = store.read_dream_snapshot("project")
    assert store.block_dream(snapshot, reason="预算不足", budget=1, dependency_item_ids=())
    assert not (await memory.aread_learning_readiness("project")).has_pending_derived


def test_failed_admission_rolls_back_all_sources(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    for index in range(10):
        capture(store, f"r{index}")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_admit AFTER UPDATE OF admitted ON memory_capture_sources "
            "WHEN NEW.run_id='r5' BEGIN SELECT RAISE(ABORT,'admission interrupted'); END"
        )
    with pytest.raises(IrisMemoryError):
        store.admit_learning_sources(
            "project",
            allowed_sources=frozenset(("host", f"r{index}") for index in range(10)),
            threshold=10,
        )
    assert not any(item.admitted for item in store.read_learning_readiness("project").sources)
