"""验证条目版本、完整快照与跨实例正文发布。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory import (
    FileMemoryMirror,
    MemoryCandidate,
    MemoryEvent,
    MemoryEventType,
    MemoryItem,
    MemoryItemPatch,
    MemoryItemStatus,
    MemoryLevel,
    MemoryObserveInput,
    MemorySearchQuery,
    MemoryService,
    MemoryWriteInput,
    SQLiteMemoryStore,
)


def test_namespace_revision_tracks_effective_l2_changes_only(tmp_path: Path) -> None:
    """实质 L2 变更推进版本，观察、候选、未变化和未命中保持原版本。"""
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    service = MemoryService(store)
    assert store.read_namespace_state("project").item_revision == 0
    episode = service.observe(MemoryObserveInput(text="observation"))
    service.add_candidate(
        MemoryCandidate(episode_ids=[episode.id], text="candidate", reason="review")
    )
    assert store.read_namespace_state("project").item_revision == 0

    item = service.remember(MemoryWriteInput(text="first", reason="seed"))
    before = store.read_namespace_snapshot("project")
    assert before.state.item_revision == 1
    assert [entry.id for entry in before.items] == [item.id]
    assert store.read_namespace_state("other").item_revision == 0
    assert service.update(item.id, "project", MemoryItemPatch(), reason="empty") == item
    assert service.update(item.id, "project", MemoryItemPatch(text="first"), reason="same") == item
    assert store.read_namespace_state("project").item_revision == 1
    assert len(service.list_events("project", item_id=item.id)) == 1

    service.update(item.id, "project", MemoryItemPatch(text="second"), reason="change")
    assert store.read_namespace_snapshot("project").state.item_revision == 2
    assert before.items[0].text == "first"
    assert not service.forget("missing", "project", reason="miss")
    assert store.read_namespace_state("project").item_revision == 2
    assert service.forget(item.id, "project", reason="done")
    assert store.read_namespace_snapshot("project").items == ()
    assert store.read_namespace_state("project").item_revision == 3
    assert not service.forget(item.id, "project", reason="already done")
    assert store.read_namespace_state("project").item_revision == 3


def test_namespace_snapshot_excludes_inactive_and_non_l2_items(tmp_path: Path) -> None:
    """正文快照仅含 active L2，搜索也能读取 active L1 item。"""
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    episodic = MemoryItem(text="episodic marker", level=MemoryLevel.EPISODIC)
    for item in (
        episodic,
        MemoryItem(text="deleted marker", status=MemoryItemStatus.DELETED),
        MemoryItem(text="superseded marker", status=MemoryItemStatus.SUPERSEDED),
    ):
        store.add_item(item, event=MemoryEvent(event_type=MemoryEventType.ADD, item_id=item.id))
    snapshot = store.read_namespace_snapshot("project")
    assert snapshot.items == ()
    assert snapshot.state.item_revision == 0
    assert [
        result.item_id
        for result in store.search(MemorySearchQuery(query="marker"), ["project"]).items
    ] == [episodic.id]


def test_partial_publication_leaves_database_committed_and_projection_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """分类发布失败不回滚已保存正文，完整投影版本保持上次成功值。"""
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    mirror = FileMemoryMirror(tmp_path / "mirror")
    service = MemoryService(store, mirror=mirror)
    item = service.remember(MemoryWriteInput(text="beforetoken", reason="seed"))
    assert store.read_namespace_state("project").projection_revision == 1
    original_replace = mirror._atomic_replace
    count = 0

    def partially_replace(relative_path: str, content: str) -> None:
        nonlocal count
        count += 1
        if count == 2:
            raise IrisMemoryError("publication interrupted")
        original_replace(relative_path, content)

    monkeypatch.setattr(mirror, "_atomic_replace", partially_replace)
    updated = service.update(item.id, "project", MemoryItemPatch(text="aftertoken"), reason="edit")
    state = store.read_namespace_state("project")
    assert service.get_item(item.id, ["project"]) == updated
    assert [
        result.item_id
        for result in service.search(MemorySearchQuery(query="aftertoken"), ["project"]).items
    ] == [item.id]
    assert service.search(MemorySearchQuery(query="beforetoken"), ["project"]).items == ()
    assert (state.item_revision, state.projection_revision) == (2, 1)
    view = service.file_access(["project"])
    assert view is not None
    assert "未同步" in view.warning("project", 2)
    monkeypatch.setattr(mirror, "_atomic_replace", original_replace)
    mirror.rebuild_from_store(store, "project")
    assert store.read_namespace_state("project").projection_revision == 2
    assert view.warning("project", 1) is not None
    assert view.warning("project", 2) is None


def test_late_publisher_reads_latest_snapshot_inside_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """迟到的发布操作在取得锁后读取最新快照，不覆盖新条目。"""
    first_store = SQLiteMemoryStore(tmp_path / "memory.db")
    second_store = SQLiteMemoryStore(first_store.path)
    first_mirror = FileMemoryMirror(tmp_path / "mirror")
    second_mirror = FileMemoryMirror(first_mirror.root)
    first = MemoryService(first_store, mirror=first_mirror)
    second = MemoryService(second_store, mirror=second_mirror)
    entered = Event()
    resume = Event()
    rebuild = first_mirror.rebuild_from_store

    def delayed_rebuild(store: SQLiteMemoryStore, namespace: str) -> None:
        entered.set()
        assert resume.wait(timeout=5)
        rebuild(store, namespace)

    monkeypatch.setattr(first_mirror, "rebuild_from_store", delayed_rebuild)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(first.remember, MemoryWriteInput(text="first", reason="seed"))
        assert entered.wait(timeout=5)
        second.remember(MemoryWriteInput(text="second", reason="seed"))
        resume.set()
        pending.result(timeout=5)
    path = first_mirror.namespace_directory("project") / "User/user.md"
    text = path.read_text(encoding="utf-8")
    assert "first" in text and "second" in text
    assert first_store.read_namespace_state("project").projection_revision == 2
    view = first.file_access(["project"])
    assert view is not None
    assert view.source_revision(text.splitlines()[0]) == 2
