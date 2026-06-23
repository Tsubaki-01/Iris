from __future__ import annotations

import json
from pathlib import Path

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory import (
    FileMemoryMirror,
    MemoryCategory,
    MemoryEventType,
    MemoryItemKind,
    MemoryObserveInput,
    MemoryScope,
    MemoryService,
    MemorySourceType,
    MemoryWriteInput,
    SQLiteMemoryStore,
)


def test_mirror_initializes_fixed_layout_without_overwriting_existing_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / ".iris" / "memory"
    existing = root / "User" / "user.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("人工备注\n", encoding="utf-8")

    FileMemoryMirror(root).initialize_layout()

    assert (root / "memory.db").exists() is False
    assert (root / "Memory.md").exists()
    assert (root / "User" / "profile.json").exists()
    assert (root / "Feedback" / "corrections.md").exists()
    assert (root / "Reference" / "docs").is_dir()
    assert (root / "Tasks" / "plans").is_dir()
    assert (root / "Sessions" / "session_items.md").exists()
    assert (root / "Sessions" / "session_summaries").is_dir()
    assert existing.read_text(encoding="utf-8") == "人工备注\n"


def test_mirror_rejects_relative_path_escape(tmp_path: Path) -> None:
    mirror = FileMemoryMirror(tmp_path / ".iris" / "memory")

    with pytest.raises(IrisMemoryError):
        mirror._resolve_relative("../outside.md")


def test_mirror_item_writes_expected_category_file(tmp_path: Path) -> None:
    mirror = FileMemoryMirror(tmp_path / ".iris" / "memory")
    scope = _scope()
    service = MemoryService(
        SQLiteMemoryStore(tmp_path / ".iris" / "memory" / "memory.db", use_fts=False),
        mirror=mirror,
    )

    item = service.remember(
        MemoryWriteInput(
            scope=scope,
            text="用户偏好简洁回答",
            reason="explicit user preference",
            kind=MemoryItemKind.PREFERENCE,
            confidence=0.8,
            importance=0.7,
        )
    )

    content = (tmp_path / ".iris" / "memory" / "User" / "preferences.md").read_text(
        encoding="utf-8"
    )
    assert item.id in content
    assert "category: user" in content
    assert "kind: preference" in content
    assert "用户偏好简洁回答" in content
    assert "confidence: 0.8" in content
    assert "importance: 0.7" in content


def test_mirror_replaces_block_with_literal_backslashes(tmp_path: Path) -> None:
    mirror = FileMemoryMirror(tmp_path / ".iris" / "memory")
    scope = _scope()
    service = MemoryService(
        SQLiteMemoryStore(tmp_path / ".iris" / "memory" / "memory.db", use_fts=False),
        mirror=mirror,
    )
    text = "Windows 路径 C:\\1_project\\new_folder，正则字面量 \\g<name>"

    item = service.remember(MemoryWriteInput(scope=scope, text=text, reason="seed"))
    mirror.mirror_item(item)

    content = (tmp_path / ".iris" / "memory" / "User" / "user.md").read_text(encoding="utf-8")
    assert text in content


def test_remember_mirrors_add_event_summary(tmp_path: Path) -> None:
    mirror = FileMemoryMirror(tmp_path / ".iris" / "memory")
    scope = _scope()
    service = MemoryService(
        SQLiteMemoryStore(tmp_path / ".iris" / "memory" / "memory.db", use_fts=False),
        mirror=mirror,
    )

    item = service.remember(
        MemoryWriteInput(
            scope=scope,
            text="用户偏好简洁回答",
            reason="explicit user preference",
        )
    )

    content = (tmp_path / ".iris" / "memory" / "Sessions" / "recent_events.md").read_text(
        encoding="utf-8"
    )
    assert item.id in content
    assert "event_type: add" in content
    assert f"item_id: {item.id}" in content


def test_session_items_do_not_mix_with_recent_events(tmp_path: Path) -> None:
    mirror = FileMemoryMirror(tmp_path / ".iris" / "memory")
    scope = _scope()
    service = MemoryService(
        SQLiteMemoryStore(tmp_path / ".iris" / "memory" / "memory.db", use_fts=False),
        mirror=mirror,
    )

    item = service.remember(
        MemoryWriteInput(
            scope=scope,
            text="当前会话摘要",
            reason="session summary",
            category=MemoryCategory.SESSION,
            kind=MemoryItemKind.SUMMARY,
        )
    )

    session_items = (tmp_path / ".iris" / "memory" / "Sessions" / "session_items.md").read_text(
        encoding="utf-8"
    )
    recent_events = (tmp_path / ".iris" / "memory" / "Sessions" / "recent_events.md").read_text(
        encoding="utf-8"
    )
    assert item.id in session_items
    assert "### Memory Item" in session_items
    assert "### Memory Item" not in recent_events
    assert "### Memory Event" in recent_events


def test_task_state_updates_task_json(tmp_path: Path) -> None:
    mirror = FileMemoryMirror(tmp_path / ".iris" / "memory")
    scope = _scope()
    service = MemoryService(
        SQLiteMemoryStore(tmp_path / ".iris" / "memory" / "memory.db", use_fts=False),
        mirror=mirror,
    )

    item = service.remember(
        MemoryWriteInput(
            scope=scope,
            text="阶段二实现 mirror",
            reason="task state",
            category=MemoryCategory.TASK,
            kind=MemoryItemKind.TASK_STATE,
            metadata={"stage": 2, "status": "in_progress"},
        )
    )

    data = json.loads(
        (tmp_path / ".iris" / "memory" / "Tasks" / "task.json").read_text(encoding="utf-8")
    )
    assert data["items"][0]["id"] == item.id
    assert data["items"][0]["metadata"] == {"stage": 2, "status": "in_progress"}


def test_observe_mirrors_recent_event_summary(tmp_path: Path) -> None:
    mirror = FileMemoryMirror(tmp_path / ".iris" / "memory")
    scope = _scope()
    service = MemoryService(
        SQLiteMemoryStore(tmp_path / ".iris" / "memory" / "memory.db", use_fts=False),
        mirror=mirror,
    )

    episode = service.observe(
        MemoryObserveInput(
            scope=scope,
            text="用户说希望保持本地优先",
            source_type=MemorySourceType.MESSAGE,
            source_id="msg_1",
        )
    )

    content = (tmp_path / ".iris" / "memory" / "Sessions" / "recent_events.md").read_text(
        encoding="utf-8"
    )
    assert episode.id in content
    assert "event_type: observe" in content


def test_rebuild_from_store_is_deterministic_and_omits_deleted_items(tmp_path: Path) -> None:
    root = tmp_path / ".iris" / "memory"
    store = SQLiteMemoryStore(root / "memory.db", use_fts=False)
    mirror = FileMemoryMirror(root)
    service = MemoryService(store, mirror=mirror)
    scope = _scope()
    active = service.remember(
        MemoryWriteInput(scope=scope, text="保留的记忆", reason="active seed")
    )
    deleted = service.remember(
        MemoryWriteInput(scope=scope, text="删除的记忆", reason="delete seed")
    )
    service.forget(deleted.id, scope, reason="cleanup deleted item")

    mirror.rebuild_from_store(store, scope)
    first = (root / "User" / "user.md").read_text(encoding="utf-8")
    mirror.rebuild_from_store(store, scope)
    second = (root / "User" / "user.md").read_text(encoding="utf-8")

    assert first == second
    assert active.id in first
    assert "保留的记忆" in first
    assert deleted.id not in first
    assert "删除的记忆" not in first


def test_rebuild_from_store_keeps_other_scope_mirror_blocks(tmp_path: Path) -> None:
    root = tmp_path / ".iris" / "memory"
    store = SQLiteMemoryStore(root / "memory.db", use_fts=False)
    mirror = FileMemoryMirror(root)
    service = MemoryService(store, mirror=mirror)
    scope_a = _scope(agent_id="agent-a")
    scope_b = _scope(agent_id="agent-b")
    item_a = service.remember(MemoryWriteInput(scope=scope_a, text="agent-a 的记忆", reason="seed"))
    item_b = service.remember(MemoryWriteInput(scope=scope_b, text="agent-b 的记忆", reason="seed"))

    service.forget(item_b.id, scope_b, reason="remove scope b")

    content = (root / "User" / "user.md").read_text(encoding="utf-8")
    assert item_a.id in content
    assert "agent-a 的记忆" in content
    assert item_b.id not in content
    assert "agent-b 的记忆" not in content


def test_rebuild_from_store_keeps_other_scope_task_json_entries(tmp_path: Path) -> None:
    root = tmp_path / ".iris" / "memory"
    store = SQLiteMemoryStore(root / "memory.db", use_fts=False)
    mirror = FileMemoryMirror(root)
    service = MemoryService(store, mirror=mirror)
    scope_a = _scope(agent_id="agent-a")
    scope_b = _scope(agent_id="agent-b")
    task_a = service.remember(
        MemoryWriteInput(
            scope=scope_a,
            text="agent-a 任务状态",
            reason="seed",
            category=MemoryCategory.TASK,
            kind=MemoryItemKind.TASK_STATE,
        )
    )
    service.remember(MemoryWriteInput(scope=scope_b, text="agent-b 普通记忆", reason="seed"))

    mirror.rebuild_from_store(store, scope_b)

    data = json.loads((root / "Tasks" / "task.json").read_text(encoding="utf-8"))
    assert [entry["id"] for entry in data["items"]] == [task_a.id]


def test_rebuild_from_store_projects_all_active_items(tmp_path: Path) -> None:
    root = tmp_path / ".iris" / "memory"
    store = SQLiteMemoryStore(root / "memory.db", use_fts=False)
    mirror = FileMemoryMirror(root)
    service = MemoryService(store, mirror=mirror)
    scope = _scope()
    items = [
        service.remember(MemoryWriteInput(scope=scope, text=f"全量投影记忆 {index}", reason="seed"))
        for index in range(101)
    ]
    (root / "User" / "user.md").write_text("", encoding="utf-8")

    mirror.rebuild_from_store(store, scope)

    content = (root / "User" / "user.md").read_text(encoding="utf-8")
    assert all(item.id in content for item in items)


def test_recent_events_mirror_keeps_recent_limit_and_header(tmp_path: Path) -> None:
    root = tmp_path / ".iris" / "memory"
    store = SQLiteMemoryStore(root / "memory.db", use_fts=False)
    mirror = FileMemoryMirror(root)
    service = MemoryService(store, mirror=mirror)
    scope = _scope()
    items = [
        service.remember(MemoryWriteInput(scope=scope, text=f"事件投影记忆 {index}", reason="seed"))
        for index in range(105)
    ]

    content = (root / "Sessions" / "recent_events.md").read_text(encoding="utf-8")
    assert "recent projection" in content
    assert "The complete audit logs shall be subject to SQLite memory_events." in content
    assert content.count("### Memory Event") == 100
    assert items[0].id not in content
    assert items[-1].id in content


def test_rebuild_from_store_reconstructs_recent_events(tmp_path: Path) -> None:
    root = tmp_path / ".iris" / "memory"
    store = SQLiteMemoryStore(root / "memory.db", use_fts=False)
    mirror = FileMemoryMirror(root)
    service = MemoryService(store, mirror=mirror)
    scope = _scope()

    item = service.remember(MemoryWriteInput(scope=scope, text="需要事件重建的记忆", reason="seed"))
    (root / "Sessions" / "recent_events.md").write_text("", encoding="utf-8")

    mirror.rebuild_from_store(store, scope)

    content = (root / "Sessions" / "recent_events.md").read_text(encoding="utf-8")
    assert "event_type: add" in content
    assert f"item_id: {item.id}" in content


def test_forget_rebuilds_active_mirror(tmp_path: Path) -> None:
    root = tmp_path / ".iris" / "memory"
    store = SQLiteMemoryStore(root / "memory.db", use_fts=False)
    mirror = FileMemoryMirror(root)
    service = MemoryService(store, mirror=mirror)
    scope = _scope()
    item = service.remember(MemoryWriteInput(scope=scope, text="临时记忆", reason="seed"))

    service.forget(item.id, scope, reason="remove from mirror")

    content = (root / "User" / "user.md").read_text(encoding="utf-8")
    assert item.id not in content
    assert service.list_events(scope)[0].event_type == MemoryEventType.DELETE


def _scope(*, agent_id: str = "agent") -> MemoryScope:
    return MemoryScope(workspace_id="workspace", agent_id=agent_id, collection="default")
