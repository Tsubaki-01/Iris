from __future__ import annotations

from pathlib import Path

from iris.memory import (
    FileMemoryMirror,
    MemoryCandidate,
    MemoryCandidateStatus,
    MemoryEventType,
    MemoryItemKind,
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

    assert service.forget(item.id, other_scope, reason="wrong scope request") is False

    assert service.get_item(item.id, owner_scope) is not None
    assert service.forget(item.id, owner_scope, reason="owner deletion request") is True
    assert service.get_item(item.id, owner_scope) is None


def test_forget_rebuilds_mirror_only_when_item_was_deleted(tmp_path: Path) -> None:
    mirror = _RecordingMirror(tmp_path / ".iris" / "memory")
    service = MemoryService(
        SQLiteMemoryStore(tmp_path / ".iris" / "memory" / "memory.db", use_fts=False),
        mirror=mirror,
    )
    owner_scope = _scope(agent_id="agent-a")
    other_scope = _scope(agent_id="agent-b")
    item = service.remember(
        MemoryWriteInput(
            scope=owner_scope,
            text="只删除一次",
            reason="test seed",
        )
    )

    assert service.forget(item.id, other_scope, reason="wrong scope request") is False
    assert mirror.rebuilt_scopes == []

    assert service.forget(item.id, owner_scope, reason="owner deletion request") is True
    assert mirror.rebuilt_scopes == [owner_scope]


def test_promote_candidate_returns_item_and_syncs_mirror(tmp_path: Path) -> None:
    root = tmp_path / ".iris" / "memory"
    service = MemoryService(
        SQLiteMemoryStore(root / "memory.db", use_fts=False),
        mirror=FileMemoryMirror(root),
    )
    scope = _scope()
    candidate = service.add_candidate(
        MemoryCandidate(
            scope=scope,
            episode_ids=["episode-a"],
            text="用户偏好简洁中文回答",
            reason="candidate reason",
        )
    )

    item = service.promote_candidate(
        candidate.id,
        scope,
        kind=MemoryItemKind.PREFERENCE,
        reason="policy accepted",
    )

    assert item.source_id == candidate.id
    assert service.list_candidates(scope)[0].status == MemoryCandidateStatus.ACCEPTED
    mirror_content = (root / "User" / "preferences.md").read_text(encoding="utf-8")
    events_content = (root / "Sessions" / "recent_events.md").read_text(encoding="utf-8")
    assert item.id in mirror_content
    assert "event_type: add" in events_content
    assert "event_type: candidate_accept" in events_content


class _RecordingMirror(FileMemoryMirror):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.rebuilt_scopes: list[MemoryScope] = []

    def rebuild_from_store(self, store: object, scope: MemoryScope) -> None:
        self.rebuilt_scopes.append(scope)


def _service(tmp_path: Path) -> MemoryService:
    return MemoryService(SQLiteMemoryStore(tmp_path / "memory.db", use_fts=False))


def _scope(*, agent_id: str = "agent") -> MemoryScope:
    return MemoryScope(workspace_id="workspace", agent_id=agent_id, collection="default")
