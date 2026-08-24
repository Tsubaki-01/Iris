from __future__ import annotations

from pathlib import Path

from iris.memory import (
    MemoryEvent,
    MemoryEventType,
    MemoryItem,
    MemoryItemStatus,
    MemoryScope,
    SQLiteMemoryStore,
)


def test_scope_order_queries_use_covering_order_indexes(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "query-plan.db", use_fts=False)
    scope = MemoryScope(workspace_id="workspace", agent_id="agent")
    scope_params = _scope_params(scope)

    with store._connection() as connection:
        item_plan = connection.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT * FROM memory_items
            WHERE scope_workspace_id = ?
              AND scope_agent_id = ?
              AND scope_collection = ?
              AND scope_visibility = ?
              AND scope_session_id = ?
              AND status = ?
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            [*scope_params, MemoryItemStatus.ACTIVE.value, 10],
        ).fetchall()
        event_plan = connection.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT * FROM memory_events
            WHERE scope_workspace_id = ?
              AND scope_agent_id = ?
              AND scope_collection = ?
              AND scope_visibility = ?
              AND scope_session_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            [*scope_params, 10],
        ).fetchall()

    item_details = " ".join(str(row[3]) for row in item_plan)
    event_details = " ".join(str(row[3]) for row in event_plan)
    assert "idx_memory_items_scope_status_updated" in item_details
    assert "idx_memory_events_scope_created" in event_details
    assert "TEMP B-TREE" not in item_details
    assert "TEMP B-TREE" not in event_details


def test_scope_order_indexes_preserve_item_and_event_order(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "ordering.db", use_fts=False)
    scope = MemoryScope(workspace_id="workspace", agent_id="agent")
    items = [
        MemoryItem(
            id="item-a",
            scope=scope,
            text="first",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-02T00:00:00Z",
        ),
        MemoryItem(
            id="item-c",
            scope=scope,
            text="second",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-03T00:00:00Z",
        ),
        MemoryItem(
            id="item-b",
            scope=scope,
            text="third",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-03T00:00:00Z",
        ),
    ]
    events = [
        MemoryEvent(
            id=f"event-{suffix}",
            scope=scope,
            event_type=MemoryEventType.ADD,
            item_id=item.id,
            created_at=created_at,
        )
        for suffix, item, created_at in (
            ("a", items[0], "2026-01-02T00:00:00Z"),
            ("c", items[1], "2026-01-03T00:00:00Z"),
            ("b", items[2], "2026-01-03T00:00:00Z"),
        )
    ]
    for item, event in zip(items, events, strict=True):
        store.add_item(item, event=event)

    assert [item.id for item in store.list_items(scope)] == [
        "item-c",
        "item-b",
        "item-a",
    ]
    assert [event.id for event in store.list_events(scope)] == [
        "event-c",
        "event-b",
        "event-a",
    ]


def _scope_params(scope: MemoryScope) -> list[str]:
    return [
        scope.workspace_id,
        scope.agent_id,
        scope.collection,
        scope.visibility.value,
        scope.session_id or "",
    ]
