from __future__ import annotations

from pathlib import Path

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory import (
    MemoryActor,
    MemoryEvent,
    MemoryEventType,
    MemoryItem,
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

    store.delete_item(
        item.id,
        scope,
        event=MemoryEvent(
            scope=scope,
            event_type=MemoryEventType.DELETE,
            item_id=item.id,
            reason="user requested deletion",
        ),
    )

    assert store.get_item(item.id, scope) is None
    assert store.list_items(scope) == []
    deleted_items = store.list_items(scope, include_deleted=True)
    assert deleted_items[0].status == MemoryItemStatus.DELETED
    assert MemoryEventType.DELETE in {
        event.event_type for event in store.list_events(scope, item_id=item.id)
    }


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
