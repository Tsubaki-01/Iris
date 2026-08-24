from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory import (
    FileMemoryMirror,
    MemoryCategory,
    MemoryEvent,
    MemoryEventType,
    MemoryItem,
    MemoryItemKind,
    MemoryScope,
)


def test_project_batch_reads_renders_and_replaces_each_target_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mirror = FileMemoryMirror(tmp_path / "mirror")
    mirror.initialize_layout()
    target = mirror.root / "User/user.md"
    target.write_text("  manual note\n", encoding="utf-8")
    items = [_item("item-a"), _item("item-b")]
    reads = 0
    renders = 0
    replaces = 0
    original_read = Path.read_text
    original_render = mirror._render_target
    original_replace = os.replace

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        nonlocal reads
        if path == target:
            reads += 1
        return original_read(path, *args, **kwargs)

    def render_target(*args: object, **kwargs: object) -> str:
        nonlocal renders
        renders += 1
        return original_render(*args, **kwargs)

    def replace(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None:
        nonlocal replaces
        if Path(destination) == target:
            replaces += 1
        original_replace(source, destination)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(mirror, "_render_target", render_target)
    monkeypatch.setattr(os, "replace", replace)

    mirror.project_batch(items=items)

    content = original_read(target, encoding="utf-8")
    assert reads == 1
    assert renders == 1
    assert replaces == 1
    assert content.startswith("  manual note\n")
    assert "item-a" in content
    assert "item-b" in content


def test_single_projection_methods_match_single_element_batches(tmp_path: Path) -> None:
    scope = _scope()
    item = _item("item-a", scope=scope)
    event = _event("event-a", scope=scope, item_id=item.id)
    singles = FileMemoryMirror(tmp_path / "singles")
    batches = FileMemoryMirror(tmp_path / "batches")

    singles.mirror_item(item)
    singles.mirror_event(event)
    batches.project_batch(items=[item], events=[event])

    assert (singles.root / "User/user.md").read_text(encoding="utf-8") == (
        batches.root / "User/user.md"
    ).read_text(encoding="utf-8")
    assert (singles.root / "Sessions/recent_events.md").read_text(encoding="utf-8") == (
        batches.root / "Sessions/recent_events.md"
    ).read_text(encoding="utf-8")


def test_batch_updates_task_json_once_and_keeps_stable_order(tmp_path: Path) -> None:
    mirror = FileMemoryMirror(tmp_path / "mirror")
    first = _item(
        "task-z",
        category=MemoryCategory.TASK,
        kind=MemoryItemKind.TASK_STATE,
    )
    second = _item(
        "task-a",
        category=MemoryCategory.TASK,
        kind=MemoryItemKind.TASK_STATE,
    )

    mirror.project_batch(items=[first, second])

    payload = json.loads((mirror.root / "Tasks/task.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in payload["items"]] == ["task-a", "task-z"]


def test_rebuild_reads_store_once_and_preserves_manual_and_other_scope(
    tmp_path: Path,
) -> None:
    mirror = FileMemoryMirror(tmp_path / "mirror")
    current_scope = _scope("agent-a")
    other_scope = _scope("agent-b")
    stale = _item("stale", scope=current_scope)
    other = _item("other", scope=other_scope)
    mirror.project_batch(items=[stale, other])
    target = mirror.root / "User/user.md"
    target.write_text(
        f"manual preface\n\n{target.read_text(encoding='utf-8')}",
        encoding="utf-8",
    )
    replacement = _item("replacement", scope=current_scope)
    event = _event("replacement-event", scope=current_scope, item_id=replacement.id)
    store = _RebuildStore(items=[replacement], events=[event])

    mirror.rebuild_from_store(store, current_scope)  # type: ignore[arg-type]

    content = target.read_text(encoding="utf-8")
    assert store.item_reads == 1
    assert store.event_reads == 1
    assert "manual preface" in content
    assert "replacement" in content
    assert "other" in content
    assert "stale" not in content


def test_initialize_retries_after_failure_and_skips_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "mirror"
    mirror = FileMemoryMirror(root)
    original_mkdir = Path.mkdir
    root_calls = 0
    fail = True

    def mkdir(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal root_calls, fail
        if path == root:
            root_calls += 1
            if fail:
                fail = False
                raise OSError("injected init failure")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)

    with pytest.raises(IrisMemoryError, match="初始化失败"):
        mirror.initialize_layout()
    mirror.initialize_layout()
    successful_calls = root_calls
    mirror.initialize_layout()

    assert root_calls == successful_calls


def test_atomic_replace_failure_keeps_target_and_removes_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mirror = FileMemoryMirror(tmp_path / "mirror")
    mirror.initialize_layout()
    target = mirror.root / "User/user.md"
    target.write_text("manual note\n", encoding="utf-8")

    def fail_replace(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(IrisMemoryError, match="写入失败"):
        mirror.project_batch(items=[_item("item-a")])

    assert target.read_text(encoding="utf-8") == "manual note\n"
    assert list(target.parent.glob(f".{target.name}.*.tmp")) == []


class _RebuildStore:
    def __init__(self, *, items: list[MemoryItem], events: list[MemoryEvent]) -> None:
        self.items = items
        self.events = events
        self.item_reads = 0
        self.event_reads = 0

    def list_items(self, scope: MemoryScope, **kwargs: object) -> list[MemoryItem]:
        del scope, kwargs
        self.item_reads += 1
        return list(self.items)

    def list_events(self, scope: MemoryScope, **kwargs: object) -> list[MemoryEvent]:
        del scope, kwargs
        self.event_reads += 1
        return list(self.events)


def _scope(agent_id: str = "agent") -> MemoryScope:
    return MemoryScope(workspace_id="workspace", agent_id=agent_id)


def _item(
    item_id: str,
    *,
    scope: MemoryScope | None = None,
    category: MemoryCategory = MemoryCategory.USER,
    kind: MemoryItemKind = MemoryItemKind.NOTE,
) -> MemoryItem:
    return MemoryItem(
        id=item_id,
        scope=scope or _scope(),
        text=f"text for {item_id}",
        category=category,
        kind=kind,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )


def _event(
    event_id: str,
    *,
    scope: MemoryScope,
    item_id: str,
) -> MemoryEvent:
    return MemoryEvent(
        id=event_id,
        scope=scope,
        event_type=MemoryEventType.ADD,
        item_id=item_id,
        created_at="2026-01-01T00:00:00Z",
    )
