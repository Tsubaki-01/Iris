from __future__ import annotations

from pathlib import Path

from iris.memory import (
    MemoryEventType,
    MemoryObserveInput,
    MemoryQuery,
    MemoryScope,
    MemoryService,
    MemorySourceType,
    MemoryWriteInput,
    SQLiteMemoryStore,
)


def test_observe_writes_episode_and_event_only(tmp_path: Path) -> None:
    service = _service(tmp_path)
    scope = _scope()

    episode = service.observe(
        MemoryObserveInput(
            scope=scope,
            text="用户说希望回答更短",
            source_type=MemorySourceType.MESSAGE,
            source_id="msg_1",
        )
    )

    assert episode.source_id == "msg_1"
    assert service.list_items(scope) == []
    events = service.list_events(scope)
    assert [(event.event_type, event.episode_id) for event in events] == [
        (MemoryEventType.OBSERVE, episode.id)
    ]


def test_remember_recall_and_build_context(tmp_path: Path) -> None:
    service = _service(tmp_path)
    scope = _scope()

    item = service.remember(
        MemoryWriteInput(
            scope=scope,
            text="用户偏好简洁的中文回答",
            reason="explicit user preference",
        )
    )
    results = service.recall(MemoryQuery(scope=scope, text="简洁", limit=5))
    bundle = service.build_context(MemoryQuery(scope=scope, text="简洁", limit=5), max_chars=100)

    assert [result.item.id for result in results] == [item.id]
    assert bundle.fragments[0].item_id == item.id
    assert bundle.omitted_count == 0


def test_forget_tombstones_without_leaking_cross_scope_existence(tmp_path: Path) -> None:
    service = _service(tmp_path)
    owner_scope = _scope(agent_id="agent-a")
    other_scope = _scope(agent_id="agent-b")
    item = service.remember(
        MemoryWriteInput(
            scope=owner_scope,
            text="只能由 owner scope 删除",
            reason="test seed",
        )
    )

    service.forget(item.id, other_scope, reason="wrong scope request")

    assert service.get_item(item.id, owner_scope) is not None
    service.forget(item.id, owner_scope, reason="owner deletion request")
    assert service.get_item(item.id, owner_scope) is None


def _service(tmp_path: Path) -> MemoryService:
    return MemoryService(SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False))


def _scope(*, agent_id: str = "agent") -> MemoryScope:
    return MemoryScope(workspace_id="workspace", agent_id=agent_id, collection="default")
