from __future__ import annotations

from pathlib import Path

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory import (
    FileMemoryMirror,
    MemoryCategory,
    MemoryEvent,
    MemoryEventType,
    MemoryItem,
    MemoryItemPatch,
    MemoryObserveInput,
    MemorySearchQuery,
    MemoryService,
    MemorySourceType,
    MemoryWriteInput,
    SQLiteMemoryStore,
)


def test_observe_writes_episode_and_event_only(tmp_path: Path) -> None:
    mirror = FileMemoryMirror(tmp_path / "mirror")
    service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"), mirror=mirror)
    namespace = "project"

    episode = service.observe(
        MemoryObserveInput(
            namespace=namespace,
            text="用户说希望回答更短",
            source_type=MemorySourceType.MESSAGE,
            source_id="msg_1",
        )
    )

    assert episode.source_id == "msg_1"
    assert service.list_items([namespace]) == []
    events = service.list_events(namespace)
    assert [(event.event_type, event.episode_id) for event in events] == [
        (MemoryEventType.OBSERVE, episode.id)
    ]
    assert not mirror.root.exists()


def test_remember_and_search_return_a_complete_short_snippet(tmp_path: Path) -> None:
    service = _service(tmp_path)
    namespace = "project"

    item = service.remember(
        MemoryWriteInput(
            namespace=namespace,
            text="用户偏好简洁的中文回答",
            reason="explicit user preference",
        )
    )
    results = service.search(MemorySearchQuery(query="简洁", limit=5), [namespace]).items
    assert [result.item_id for result in results] == [item.id]
    assert results[0].namespace == namespace
    assert results[0].snippet == item.text
    assert results[0].is_complete


def test_forget_tombstones_without_leaking_cross_namespace_existence(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    owner_namespace = "private-a"
    other_namespace = "private-b"
    item = service.remember(
        MemoryWriteInput(
            namespace=owner_namespace,
            text="只能由 owner namespace 删除",
            reason="test seed",
        )
    )

    assert service.forget(item.id, other_namespace, reason="wrong namespace request") is False
    assert service.get_item(item.id, [owner_namespace]) is not None
    assert service.forget(item.id, owner_namespace, reason="owner deletion request") is True
    assert service.get_item(item.id, [owner_namespace]) is None


def test_sqlite_search_keeps_full_namespace_isolation(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    owner_namespace = "private-a"
    other_namespace = "private-b"
    item = MemoryItem(namespace=owner_namespace, text="只有 agent-a 能看到")
    store.add_item(
        item,
        event=MemoryEvent(
            namespace=owner_namespace,
            event_type=MemoryEventType.ADD,
            item_id=item.id,
            reason="test seed",
        ),
    )

    assert store.search(MemorySearchQuery(query="agent-a"), [other_namespace]).items == ()


def _service(tmp_path: Path) -> MemoryService:
    return MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"))


def test_update_keeps_identity_refreshes_search_and_relocates_mirror(tmp_path: Path) -> None:
    mirror = FileMemoryMirror(tmp_path / "mirror")
    service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"), mirror=mirror)
    item = service.remember(MemoryWriteInput(text="initial bananas", reason="seed"))
    updated = service.update(
        item.id,
        "project",
        MemoryItemPatch(text="updated oranges", category=MemoryCategory.REFERENCE),
        reason="correct facts",
    )

    assert updated.id == item.id
    assert service.get_item(item.id, ["project"]) == updated
    assert service.search(MemorySearchQuery(query="bananas"), ["project"]).items == ()
    assert [
        result.item_id
        for result in service.search(MemorySearchQuery(query="oranges"), ["project"]).items
    ] == [item.id]
    assert (
        service.search(MemorySearchQuery(query="oranges", categories=["user"]), ["project"]).items
        == ()
    )
    assert (
        service.search(MemorySearchQuery(query="oranges", categories=["reference"]), ["project"])
        .items[0]
        .item_id
        == item.id
    )
    directory = mirror.namespace_directory("project")
    assert item.id not in (directory / "User/user.md").read_text(encoding="utf-8")
    assert "updated oranges" in (directory / "Reference/notes.md").read_text(encoding="utf-8")
    assert service.forget(item.id, "project", reason="finished") is True
    assert service.search(MemorySearchQuery(query="oranges"), ["project"]).items == ()
    assert item.id not in (directory / "Reference/notes.md").read_text(encoding="utf-8")


def test_committed_writes_survive_automatic_mirror_failure_but_explicit_rebuild_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    mirror = FileMemoryMirror(tmp_path / "mirror")
    service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"), mirror=mirror)

    def fail(*args: object, **kwargs: object) -> None:
        raise IrisMemoryError("mirror unavailable")

    monkeypatch.setattr(mirror, "rebuild_from_store", fail)
    with caplog.at_level("WARNING", logger="iris.memory.service"):
        item = service.remember(MemoryWriteInput(text="original text", reason="seed"))
        updated = service.update(
            item.id, "project", MemoryItemPatch(text="new text"), reason="edit"
        )
        assert service.get_item(item.id, ["project"]) == updated
        assert service.forget(item.id, "project", reason="remove") is True
    assert service.get_item(item.id, ["project"]) is None
    assert len(caplog.records) == 3
    assert all(
        record.levelname == "WARNING" and "mirror" in record.message for record in caplog.records
    )
    with pytest.raises(IrisMemoryError, match="mirror unavailable"):
        mirror.rebuild_from_store(service.store, "project")
