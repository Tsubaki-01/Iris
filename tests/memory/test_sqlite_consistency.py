"""验证记忆公开存储接口的并发写入与完整搜索结果。"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from threading import Event

import pytest

from iris.memory import (
    MemoryCandidate,
    MemoryCandidateStatus,
    MemoryConfig,
    MemoryEvent,
    MemoryEventType,
    MemoryItem,
    MemoryItemKind,
    MemoryItemPatch,
    MemoryItemStatus,
    MemoryObserveInput,
    MemoryQuery,
    MemoryScope,
    MemoryService,
    MemoryWriteInput,
    SQLiteMemoryStore,
    build_memory_service_from_config,
)


def test_concurrent_candidate_promotion_creates_one_item_and_event_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一候选的并发晋升返回相同条目且只记录一次晋升。"""
    first_store = SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False)
    second_store = SQLiteMemoryStore(first_store.path, use_fts=False)
    first_service = MemoryService(first_store)
    second_service = MemoryService(second_store)
    scope = MemoryScope(workspace_id="workspace", agent_id="agent")
    episode = first_service.observe(MemoryObserveInput(scope=scope, text="用户偏好简洁回答"))
    candidate = first_service.add_candidate(
        MemoryCandidate(
            scope=scope,
            episode_ids=[episode.id],
            text=episode.text,
            reason="明确偏好",
        )
    )
    first_read = Event()
    release_first = Event()
    fetch_candidate = first_store._fetch_candidate

    def pause_after_read(
        connection: sqlite3.Connection,
        candidate_id: str,
        candidate_scope: MemoryScope,
    ) -> MemoryCandidate | None:
        """让另一连接有机会在首次读取后进入晋升流程。"""
        result = fetch_candidate(connection, candidate_id, candidate_scope)
        first_read.set()
        assert release_first.wait(timeout=5)
        return result

    monkeypatch.setattr(first_store, "_fetch_candidate", pause_after_read)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            first_service.promote_candidate,
            candidate.id,
            scope,
            kind=MemoryItemKind.NOTE,
            reason="确认偏好",
        )
        assert first_read.wait(timeout=5)
        second = executor.submit(
            second_service.promote_candidate,
            candidate.id,
            scope,
            kind=MemoryItemKind.NOTE,
            reason="再次确认偏好",
        )
        try:
            # 正确的事务会让第二个连接等待首个连接提交。
            second.result(timeout=0.25)
        except TimeoutError:
            pass
        finally:
            release_first.set()
        first_item = first.result(timeout=5)
        second_item = second.result(timeout=5)

    assert first_item.id == second_item.id
    assert [item.id for item in first_service.list_items(scope)] == [first_item.id]
    assert first_service.list_candidates(scope)[0].status == MemoryCandidateStatus.ACCEPTED
    event_types = [event.event_type for event in first_service.list_events(scope)]
    assert event_types.count(MemoryEventType.ADD) == 1
    assert event_types.count(MemoryEventType.CANDIDATE_ACCEPT) == 1


def test_concurrent_disjoint_item_patches_preserve_both_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不同连接更新同一条目的不同字段时保留两次合法修改。"""
    first_store = SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False)
    second_store = SQLiteMemoryStore(first_store.path, use_fts=False)
    service = MemoryService(first_store)
    scope = MemoryScope(workspace_id="workspace", agent_id="agent")
    item = service.remember(MemoryWriteInput(scope=scope, text="原始内容", reason="记录偏好"))
    first_read = Event()
    release_first = Event()
    fetch_item = first_store._fetch_item

    def pause_after_read(
        connection: sqlite3.Connection,
        item_id: str,
        item_scope: MemoryScope,
        *,
        include_deleted: bool,
    ) -> MemoryItem | None:
        """固定首次读改写与另一次字段更新的重叠顺序。"""
        result = fetch_item(connection, item_id, item_scope, include_deleted=include_deleted)
        first_read.set()
        assert release_first.wait(timeout=5)
        return result

    monkeypatch.setattr(first_store, "_fetch_item", pause_after_read)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            first_store.update_item,
            item.id,
            scope,
            MemoryItemPatch(text="更新后的内容"),
            event=MemoryEvent(scope=scope, event_type=MemoryEventType.UPDATE, item_id=item.id),
        )
        assert first_read.wait(timeout=5)
        second = executor.submit(
            second_store.update_item,
            item.id,
            scope,
            MemoryItemPatch(importance=0.8),
            event=MemoryEvent(scope=scope, event_type=MemoryEventType.UPDATE, item_id=item.id),
        )
        try:
            second.result(timeout=0.25)
        except TimeoutError:
            pass
        finally:
            release_first.set()
        first.result(timeout=5)
        second.result(timeout=5)

    updated = second_store.get_item(item.id, scope)
    assert updated is not None
    assert updated.text == "更新后的内容"
    assert updated.importance == 0.8
    assert len(second_store.list_events(scope, item_id=item.id)) == 3


def test_enabling_fts_indexes_items_written_without_fts(tmp_path: Path) -> None:
    """首次启用全文搜索后能同时搜到既有条目与新条目。"""
    path = tmp_path / "memory.db"
    service = MemoryService(SQLiteMemoryStore(path, use_fts=False))
    scope = MemoryScope(workspace_id="workspace", agent_id="agent")
    old_item = service.remember(
        MemoryWriteInput(scope=scope, text="shared preference old", reason="记录既有偏好")
    )
    indexed_store = SQLiteMemoryStore(path, use_fts=True)
    assert indexed_store.fts_enabled
    indexed_service = MemoryService(indexed_store)
    new_item = indexed_service.remember(
        MemoryWriteInput(scope=scope, text="shared preference new", reason="记录新增偏好")
    )

    results = indexed_service.recall(MemoryQuery(scope=scope, text="shared"))

    assert {result.item.id for result in results} == {old_item.id, new_item.id}
    assert {result.source for result in results} == {"sqlite_fts"}


def test_reenabling_fts_rebuilds_additions_updates_and_deletions(tmp_path: Path) -> None:
    """重新启用索引后，关闭期间的新增、改写与删除都反映到召回结果。"""
    path = tmp_path / "memory.db"
    service = MemoryService(SQLiteMemoryStore(path, use_fts=True))
    scope = MemoryScope(workspace_id="workspace", agent_id="agent")
    retained = service.remember(
        MemoryWriteInput(scope=scope, text="shared retained preference", reason="记录偏好")
    )
    updated = service.remember(
        MemoryWriteInput(scope=scope, text="obsolete preference", reason="记录偏好")
    )
    deleted = service.remember(
        MemoryWriteInput(scope=scope, text="shared removed preference", reason="记录偏好")
    )
    unindexed_store = SQLiteMemoryStore(path, use_fts=False)
    unindexed_service = MemoryService(unindexed_store)
    added = unindexed_service.remember(
        MemoryWriteInput(scope=scope, text="shared added preference", reason="记录新增偏好")
    )
    unindexed_store.update_item(
        updated.id,
        scope,
        MemoryItemPatch(text="shared updated preference"),
        event=MemoryEvent(scope=scope, event_type=MemoryEventType.UPDATE, item_id=updated.id),
    )
    unindexed_service.forget(deleted.id, scope, reason="撤销偏好")
    indexed_store = SQLiteMemoryStore(path, use_fts=True)
    assert indexed_store.fts_enabled
    indexed_service = MemoryService(indexed_store)

    results = indexed_service.recall(MemoryQuery(scope=scope, text="shared"))

    assert {result.item.id for result in results} == {retained.id, updated.id, added.id}
    assert {result.source for result in results} == {"sqlite_fts"}
    assert indexed_service.recall(MemoryQuery(scope=scope, text="obsolete")) == []
    all_results = indexed_service.recall(
        MemoryQuery(scope=scope, text="shared", include_deleted=True)
    )
    assert {result.item.id for result in all_results} == {
        retained.id,
        updated.id,
        added.id,
        deleted.id,
    }


def test_fts_search_including_deleted_returns_active_and_deleted_matches(tmp_path: Path) -> None:
    """包含删除项的召回不会因全文索引中的活跃命中而丢掉删除项。"""
    store = SQLiteMemoryStore(tmp_path / "memory.db", use_fts=True)
    assert store.fts_enabled
    service = MemoryService(store)
    scope = MemoryScope(workspace_id="workspace", agent_id="agent")
    active = service.remember(
        MemoryWriteInput(scope=scope, text="shared active preference", reason="记录偏好")
    )
    deleted = service.remember(
        MemoryWriteInput(scope=scope, text="shared deleted preference", reason="记录偏好")
    )
    service.forget(deleted.id, scope, reason="撤销偏好")

    results = service.recall(MemoryQuery(scope=scope, text="shared", include_deleted=True))

    assert {result.item.id: result.item.status for result in results} == {
        active.id: MemoryItemStatus.ACTIVE,
        deleted.id: MemoryItemStatus.DELETED,
    }
    active_results = service.recall(MemoryQuery(scope=scope, text="shared"))
    assert [result.item.id for result in active_results] == [active.id]


def test_memory_search_config_has_only_index_option_and_query_owns_limit(tmp_path: Path) -> None:
    """配置只选择全文索引，实际返回数量由每次查询声明。"""
    config = MemoryConfig(backend="sqlite", search={"use_fts": False}, mirror={"enabled": False})
    assert config.search.model_dump() == {"use_fts": False}
    service = build_memory_service_from_config(config, tmp_path)
    assert service is not None
    scope = MemoryScope(workspace_id="workspace", agent_id="agent")
    for number in range(3):
        service.remember(
            MemoryWriteInput(scope=scope, text=f"preference {number}", reason="记录偏好")
        )

    assert len(service.recall(MemoryQuery(scope=scope, text="preference", limit=1))) == 1
    assert len(service.recall(MemoryQuery(scope=scope, text="preference", limit=2))) == 2
