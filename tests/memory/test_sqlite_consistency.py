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
    MemoryEvent,
    MemoryEventType,
    MemoryItem,
    MemoryItemKind,
    MemoryItemPatch,
    MemoryItemStatus,
    MemoryObserveInput,
    MemoryQuery,
    MemoryService,
    MemoryWriteInput,
    SQLiteMemoryStore,
)


def test_concurrent_candidate_promotion_creates_one_item_and_event_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一候选的并发晋升返回相同条目且只记录一次晋升。"""
    first_store = SQLiteMemoryStore(tmp_path / "memory.db")
    second_store = SQLiteMemoryStore(first_store.path)
    first_service = MemoryService(first_store)
    second_service = MemoryService(second_store)
    namespace = "project"
    episode = first_service.observe(
        MemoryObserveInput(namespace=namespace, text="用户偏好简洁回答")
    )
    candidate = first_service.add_candidate(
        MemoryCandidate(
            namespace=namespace,
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
        candidate_namespace: str,
    ) -> MemoryCandidate | None:
        """让另一连接有机会在首次读取后进入晋升流程。"""
        result = fetch_candidate(connection, candidate_id, candidate_namespace)
        first_read.set()
        assert release_first.wait(timeout=5)
        return result

    monkeypatch.setattr(first_store, "_fetch_candidate", pause_after_read)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            first_service.promote_candidate,
            candidate.id,
            namespace,
            kind=MemoryItemKind.NOTE,
            reason="确认偏好",
        )
        assert first_read.wait(timeout=5)
        second = executor.submit(
            second_service.promote_candidate,
            candidate.id,
            namespace,
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
    assert [item.id for item in first_service.list_items([namespace])] == [first_item.id]
    assert first_service.list_candidates(namespace)[0].status == MemoryCandidateStatus.ACCEPTED
    event_types = [event.event_type for event in first_service.list_events(namespace)]
    assert event_types.count(MemoryEventType.ADD) == 1
    assert event_types.count(MemoryEventType.CANDIDATE_ACCEPT) == 1


def test_concurrent_disjoint_item_patches_preserve_both_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不同连接更新同一条目的不同字段时保留两次合法修改。"""
    first_store = SQLiteMemoryStore(tmp_path / "memory.db")
    second_store = SQLiteMemoryStore(first_store.path)
    service = MemoryService(first_store)
    namespace = "project"
    item = service.remember(
        MemoryWriteInput(namespace=namespace, text="原始内容", reason="记录偏好")
    )
    first_read = Event()
    release_first = Event()
    fetch_item = first_store._fetch_item

    def pause_after_read(
        connection: sqlite3.Connection,
        item_id: str,
        item_namespace: str,
        *,
        include_deleted: bool,
    ) -> MemoryItem | None:
        """固定首次读改写与另一次字段更新的重叠顺序。"""
        result = fetch_item(connection, item_id, item_namespace, include_deleted=include_deleted)
        first_read.set()
        assert release_first.wait(timeout=5)
        return result

    monkeypatch.setattr(first_store, "_fetch_item", pause_after_read)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            first_store.update_item,
            item.id,
            namespace,
            MemoryItemPatch(text="更新后的内容"),
            event=MemoryEvent(
                namespace=namespace, event_type=MemoryEventType.UPDATE, item_id=item.id
            ),
        )
        assert first_read.wait(timeout=5)
        second = executor.submit(
            second_store.update_item,
            item.id,
            namespace,
            MemoryItemPatch(importance=0.8),
            event=MemoryEvent(
                namespace=namespace, event_type=MemoryEventType.UPDATE, item_id=item.id
            ),
        )
        try:
            second.result(timeout=0.25)
        except TimeoutError:
            pass
        finally:
            release_first.set()
        first.result(timeout=5)
        second.result(timeout=5)

    updated = second_store.get_item(item.id, [namespace])
    assert updated is not None
    assert updated.text == "更新后的内容"
    assert updated.importance == 0.8
    assert len(second_store.list_events(namespace, item_id=item.id)) == 3


def test_reopening_fts_preserves_existing_and_new_items(tmp_path: Path) -> None:
    """重开数据库后能同时搜到既有条目与新条目。"""
    path = tmp_path / "memory.db"
    service = MemoryService(SQLiteMemoryStore(path))
    namespace = "project"
    old_item = service.remember(
        MemoryWriteInput(namespace=namespace, text="shared preference old", reason="记录既有偏好")
    )
    indexed_store = SQLiteMemoryStore(path)
    indexed_service = MemoryService(indexed_store)
    new_item = indexed_service.remember(
        MemoryWriteInput(namespace=namespace, text="shared preference new", reason="记录新增偏好")
    )

    results = indexed_service.recall(MemoryQuery(namespaces=[namespace], text="shared"))

    assert {result.item.id for result in results} == {old_item.id, new_item.id}
    assert {result.source for result in results} == {"sqlite_fts"}


def test_rebuild_fts_tracks_additions_updates_and_deletions(tmp_path: Path) -> None:
    """显式重建索引后，新增、改写与软删除都反映到召回结果。"""
    path = tmp_path / "memory.db"
    service = MemoryService(SQLiteMemoryStore(path))
    namespace = "project"
    retained = service.remember(
        MemoryWriteInput(namespace=namespace, text="shared retained preference", reason="记录偏好")
    )
    updated = service.remember(
        MemoryWriteInput(namespace=namespace, text="obsolete preference", reason="记录偏好")
    )
    deleted = service.remember(
        MemoryWriteInput(namespace=namespace, text="shared removed preference", reason="记录偏好")
    )
    writer_store = SQLiteMemoryStore(path)
    writer_service = MemoryService(writer_store)
    added = writer_service.remember(
        MemoryWriteInput(namespace=namespace, text="shared added preference", reason="记录新增偏好")
    )
    writer_store.update_item(
        updated.id,
        namespace,
        MemoryItemPatch(text="shared updated preference"),
        event=MemoryEvent(
            namespace=namespace, event_type=MemoryEventType.UPDATE, item_id=updated.id
        ),
    )
    writer_service.forget(deleted.id, namespace, reason="撤销偏好")
    writer_store.rebuild_index()
    indexed_store = SQLiteMemoryStore(path)
    indexed_service = MemoryService(indexed_store)

    results = indexed_service.recall(MemoryQuery(namespaces=[namespace], text="shared"))

    assert {result.item.id for result in results} == {retained.id, updated.id, added.id}
    assert {result.source for result in results} == {"sqlite_fts"}
    assert indexed_service.recall(MemoryQuery(namespaces=[namespace], text="obsolete")) == []
    all_results = indexed_service.recall(
        MemoryQuery(namespaces=[namespace], text="shared", include_deleted=True)
    )
    assert {result.item.id for result in all_results} == {
        retained.id,
        updated.id,
        added.id,
        deleted.id,
    }


def test_fts_search_including_deleted_returns_active_and_deleted_matches(tmp_path: Path) -> None:
    """包含删除项的召回不会因全文索引中的活跃命中而丢掉删除项。"""
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    service = MemoryService(store)
    namespace = "project"
    active = service.remember(
        MemoryWriteInput(namespace=namespace, text="shared active preference", reason="记录偏好")
    )
    deleted = service.remember(
        MemoryWriteInput(namespace=namespace, text="shared deleted preference", reason="记录偏好")
    )
    service.forget(deleted.id, namespace, reason="撤销偏好")

    results = service.recall(
        MemoryQuery(namespaces=[namespace], text="shared", include_deleted=True)
    )

    assert {result.item.id: result.item.status for result in results} == {
        active.id: MemoryItemStatus.ACTIVE,
        deleted.id: MemoryItemStatus.DELETED,
    }
    active_results = service.recall(MemoryQuery(namespaces=[namespace], text="shared"))
    assert [result.item.id for result in active_results] == [active.id]


def test_memory_query_owns_result_limit(tmp_path: Path) -> None:
    """结果条数由每次查询声明。"""
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    for number in range(3):
        store.add_item(
            MemoryItem(text=f"preference {number}"),
            event=MemoryEvent(event_type=MemoryEventType.ADD),
        )

    assert len(store.search(MemoryQuery(text="preference", limit=1))) == 1
    assert len(store.search(MemoryQuery(text="preference", limit=2))) == 2
