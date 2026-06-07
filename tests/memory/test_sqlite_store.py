from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory import (
    MemoryActor,
    MemoryCandidate,
    MemoryCandidateStatus,
    MemoryCategory,
    MemoryEvent,
    MemoryEventType,
    MemoryItem,
    MemoryItemKind,
    MemoryItemStatus,
    MemoryQuery,
    MemoryScope,
    SQLiteMemoryStore,
)


def test_sqlite_store_initializes_repeatedly(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"

    first = SQLiteMemoryStore(path, use_fts=False)
    second = SQLiteMemoryStore(path, use_fts=False)

    assert first.list_items(_scope()) == []
    assert second.list_items(_scope()) == []


def test_sqlite_store_searches_with_fallback_when_fts_disabled(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False)
    scope = _scope()
    item = MemoryItem(scope=scope, text="用户喜欢简洁的技术回答")
    event = MemoryEvent(
        scope=scope,
        event_type=MemoryEventType.ADD,
        actor=MemoryActor.SDK,
        item_id=item.id,
        reason="test seed",
    )

    store.add_item(item, event=event)
    results = store.search(MemoryQuery(scope=scope, text="简洁", limit=5))

    assert [result.item.id for result in results] == [item.id]


def test_sqlite_store_falls_back_when_fts_returns_no_unicode_matches(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    scope = _scope()
    item = MemoryItem(scope=scope, text="用户喜欢简洁的技术回答")
    event = MemoryEvent(
        scope=scope,
        event_type=MemoryEventType.ADD,
        actor=MemoryActor.SDK,
        item_id=item.id,
        reason="test seed",
    )

    store.add_item(item, event=event)
    results = store.search(MemoryQuery(scope=scope, text="简洁", limit=5))

    assert [result.item.id for result in results] == [item.id]


def test_sqlite_store_delete_tombstones_and_hides_item_by_default(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False)
    scope = _scope()
    item = MemoryItem(scope=scope, text="需要被删除的记忆")
    store.add_item(
        item,
        event=MemoryEvent(
            scope=scope,
            event_type=MemoryEventType.ADD,
            item_id=item.id,
            reason="test seed",
        ),
    )

    deleted = store.delete_item(
        item.id,
        scope,
        event=MemoryEvent(
            scope=scope,
            event_type=MemoryEventType.DELETE,
            item_id=item.id,
            reason="user requested deletion",
        ),
    )

    assert deleted is True
    assert store.get_item(item.id, scope) is None
    assert store.list_items(scope) == []
    deleted_items = store.list_items(scope, include_deleted=True)
    assert deleted_items[0].status == MemoryItemStatus.DELETED
    assert MemoryEventType.DELETE in {
        event.event_type for event in store.list_events(scope, item_id=item.id)
    }


def test_sqlite_store_delete_returns_false_when_active_item_is_missing(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False)
    scope = _scope()

    deleted = store.delete_item(
        "missing",
        scope,
        event=MemoryEvent(
            scope=scope,
            event_type=MemoryEventType.DELETE,
            item_id="missing",
            reason="not found",
        ),
    )

    assert deleted is False
    assert store.list_events(scope) == []


def test_sqlite_store_keeps_full_scope_isolation(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False)
    owner_scope = _scope(agent_id="agent-a")
    other_scope = _scope(agent_id="agent-b")
    item = MemoryItem(scope=owner_scope, text="只有 agent-a 能看到")

    store.add_item(
        item,
        event=MemoryEvent(
            scope=owner_scope,
            event_type=MemoryEventType.ADD,
            item_id=item.id,
            reason="test seed",
        ),
    )

    assert store.search(MemoryQuery(scope=other_scope, text="agent-a")) == []


def test_sqlite_store_filters_items_before_limit(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False)
    scope = _scope()
    user_item = MemoryItem(
        scope=scope,
        text="较早的用户记忆",
        category=MemoryCategory.USER,
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
    )
    task_item = MemoryItem(
        scope=scope,
        text="较新的任务记忆",
        category=MemoryCategory.TASK,
        created_at="2026-01-02T00:00:00",
        updated_at="2026-01-02T00:00:00",
    )
    for item in (user_item, task_item):
        store.add_item(
            item,
            event=MemoryEvent(
                scope=scope,
                event_type=MemoryEventType.ADD,
                item_id=item.id,
                reason="test seed",
            ),
        )

    items = store.list_items(scope, limit=1, categories=[MemoryCategory.USER])

    assert [item.id for item in items] == [user_item.id]


def test_sqlite_store_rejects_duplicate_item_id_across_scopes(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False)
    owner_scope = _scope(agent_id="agent-a")
    other_scope = _scope(agent_id="agent-b")
    item = MemoryItem(id="shared-item-id", scope=owner_scope, text="owner text")

    store.add_item(
        item,
        event=MemoryEvent(
            scope=owner_scope,
            event_type=MemoryEventType.ADD,
            item_id=item.id,
            reason="test seed",
        ),
    )

    with pytest.raises(IrisMemoryError):
        store.add_item(
            MemoryItem(id=item.id, scope=other_scope, text="other text"),
            event=MemoryEvent(
                scope=other_scope,
                event_type=MemoryEventType.ADD,
                item_id=item.id,
                reason="duplicate id",
            ),
        )

    stored = store.get_item(item.id, owner_scope)
    assert stored is not None
    assert stored.text == "owner text"
    assert store.get_item(item.id, other_scope) is None


def test_sqlite_store_promotes_candidate_in_single_transaction(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False)
    scope = _scope()
    candidate = MemoryCandidate(
        scope=scope,
        episode_ids=["episode-a"],
        category=MemoryCategory.USER,
        text="用户偏好简洁中文回答",
        confidence=0.9,
        importance=0.8,
        reason="candidate reason",
        metadata={"memory_kind": "preference"},
    )
    store.add_candidate(
        candidate,
        event=MemoryEvent(
            scope=scope,
            event_type=MemoryEventType.CANDIDATE_ADD,
            episode_id="episode-a",
            reason="test seed",
        ),
    )

    item = store.promote_candidate(
        candidate.id,
        scope,
        kind=MemoryItemKind.PREFERENCE,
        actor=MemoryActor.SDK,
        reason="policy accepted",
    )

    assert item.text == candidate.text
    assert item.episode_id == "episode-a"
    assert item.source_id == candidate.id
    assert item.kind == MemoryItemKind.PREFERENCE
    assert item.metadata["candidate_id"] == candidate.id
    assert store.list_candidates(scope)[0].status == MemoryCandidateStatus.ACCEPTED
    events = store.list_events(scope)
    assert {event.event_type for event in events} >= {
        MemoryEventType.ADD,
        MemoryEventType.CANDIDATE_ACCEPT,
    }
    assert item.id in {event.item_id for event in events}
    assert candidate.id in {
        event.metadata.get("candidate_id") for event in events if event.metadata
    }


def test_sqlite_store_promote_candidate_is_idempotent_after_accept(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False)
    scope = _scope()
    candidate = MemoryCandidate(
        scope=scope,
        episode_ids=["episode-a"],
        text="可重试晋升候选",
        reason="candidate reason",
    )
    store.add_candidate(
        candidate,
        event=MemoryEvent(
            scope=scope,
            event_type=MemoryEventType.CANDIDATE_ADD,
            episode_id="episode-a",
            reason="test seed",
        ),
    )

    first = store.promote_candidate(
        candidate.id,
        scope,
        kind=MemoryItemKind.NOTE,
        actor=MemoryActor.SDK,
        reason="policy accepted",
    )
    second = store.promote_candidate(
        candidate.id,
        scope,
        kind=MemoryItemKind.NOTE,
        actor=MemoryActor.SDK,
        reason="retry after accept",
    )

    assert second.id == first.id
    assert [item.id for item in store.list_items(scope)] == [first.id]
    event_types = [event.event_type for event in store.list_events(scope)]
    assert event_types.count(MemoryEventType.ADD) == 1
    assert event_types.count(MemoryEventType.CANDIDATE_ACCEPT) == 1


def test_sqlite_store_promote_candidate_rolls_back_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False)
    scope = _scope()
    candidate = MemoryCandidate(
        scope=scope,
        episode_ids=["episode-a"],
        text="失败时不能留下 item",
        reason="candidate reason",
    )
    store.add_candidate(
        candidate,
        event=MemoryEvent(
            scope=scope,
            event_type=MemoryEventType.CANDIDATE_ADD,
            episode_id="episode-a",
            reason="test seed",
        ),
    )

    def _fail_candidate_update(*_args: object, **_kwargs: object) -> None:
        raise sqlite3.OperationalError("simulated candidate update failure")

    monkeypatch.setattr(store, "_upsert_candidate", _fail_candidate_update)

    with pytest.raises(IrisMemoryError):
        store.promote_candidate(
            candidate.id,
            scope,
            kind=MemoryItemKind.NOTE,
            actor=MemoryActor.SDK,
            reason="policy accepted",
        )

    assert store.list_items(scope) == []
    assert store.list_candidates(scope)[0].status == MemoryCandidateStatus.PENDING
    assert MemoryEventType.ADD not in {event.event_type for event in store.list_events(scope)}
    assert MemoryEventType.CANDIDATE_ACCEPT not in {
        event.event_type for event in store.list_events(scope)
    }


def test_sqlite_store_rejects_duplicate_candidate_id_across_scopes(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False)
    owner_scope = _scope(agent_id="agent-a")
    other_scope = _scope(agent_id="agent-b")
    candidate = MemoryCandidate(
        id="shared-candidate-id",
        scope=owner_scope,
        episode_ids=["episode-a"],
        text="owner candidate",
        reason="test seed",
    )

    store.add_candidate(
        candidate,
        event=MemoryEvent(
            scope=owner_scope,
            event_type=MemoryEventType.CANDIDATE_ADD,
            episode_id="episode-a",
            reason="test seed",
        ),
    )

    with pytest.raises(IrisMemoryError):
        store.add_candidate(
            MemoryCandidate(
                id=candidate.id,
                scope=other_scope,
                episode_ids=["episode-b"],
                text="other candidate",
                reason="duplicate id",
            ),
            event=MemoryEvent(
                scope=other_scope,
                event_type=MemoryEventType.CANDIDATE_ADD,
                episode_id="episode-b",
                reason="duplicate id",
            ),
        )

    assert store.list_candidates(owner_scope)[0].text == "owner candidate"
    assert store.list_candidates(other_scope) == []


def test_sqlite_store_closes_connections_after_operations(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(path, use_fts=False)
    scope = _scope()
    item = MemoryItem(scope=scope, text="connection cleanup")

    store.add_item(
        item,
        event=MemoryEvent(
            scope=scope,
            event_type=MemoryEventType.ADD,
            item_id=item.id,
            reason="test seed",
        ),
    )
    assert store.list_items(scope)[0].id == item.id

    path.unlink()
    assert not path.exists()


def test_sqlite_store_wraps_unserializable_metadata(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False)
    scope = _scope()
    item = MemoryItem(scope=scope, text="metadata error", metadata={"bad": object()})

    with pytest.raises(IrisMemoryError):
        store.add_item(
            item,
            event=MemoryEvent(
                scope=scope,
                event_type=MemoryEventType.ADD,
                item_id=item.id,
                reason="test seed",
            ),
        )


def _scope(*, agent_id: str = "agent") -> MemoryScope:
    return MemoryScope(workspace_id="workspace", agent_id=agent_id, collection="default")
