"""验证 namespace 独立正文、原子文件替换和受控发布。"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory import (
    FileMemoryMirror,
    MemoryArtifactRef,
    MemoryCategory,
    MemoryItemKind,
    MemoryItemPatch,
    MemoryService,
    MemoryWriteInput,
    SQLiteMemoryStore,
    namespace_key,
)
from iris.memory.files import BODY_PATHS


def test_body_preserves_artifact_references_and_saved_business_metadata(tmp_path: Path) -> None:
    mirror = FileMemoryMirror(tmp_path / "mirror")
    service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"), mirror=mirror)
    item = service.remember(
        MemoryWriteInput(
            text="需求文档已整理",
            reason="保留交接依据",
            category=MemoryCategory.TASK,
            kind=MemoryItemKind.TASK_STATE,
            artifacts=[
                MemoryArtifactRef(
                    path="docs/project-handoff.md",
                    mime_type="text/markdown",
                    metadata={"version": 3},
                )
            ],
            metadata={"owner": "maintainer", "stage": "ready"},
        )
    )
    body = (mirror.namespace_directory("project") / "Tasks/task.md").read_text(encoding="utf-8")
    assert item.text in body
    assert "docs/project-handoff.md" in body and "text/markdown" in body
    assert '"version": 3' in body
    assert '"owner": "maintainer"' in body and '"stage": "ready"' in body
    assert item.created_at in body
    assert item.evidence[0].source_id in body
    assert "evidence:" in body
    assert body.index(item.text) < body.index("<details>") < body.index(item.id)
    assert "### Memory Item" not in body
    assert "</details>\n\n---" in body


def test_body_keeps_raw_markdown_and_escapes_folded_metadata(tmp_path: Path) -> None:
    """原文完整保留，元数据的 Markdown/HTML 字符不改变折叠结构。"""
    mirror = FileMemoryMirror(tmp_path / "mirror")
    service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"), mirror=mirror)
    text = "# 原标题\n\n- 第一项\n\n```python\nprint('原文')\n```\n"
    item = service.remember(
        MemoryWriteInput(
            text=text,
            reason="依据 `原文` | <约定>\n下一行",
            source_id="message<1>",
            metadata={"key": "````\n</details>"},
        )
    )
    body = (mirror.namespace_directory("project") / "User/user.md").read_text(encoding="utf-8")
    assert text in body
    assert body.index(text) < body.index("<details>")
    assert item.evidence[0].source_id in body
    assert "message&lt;1&gt;" in body
    assert "&lt;约定&gt;" in body
    assert body.count("</details>") == 1
    assert item.id in body


def test_namespace_layout_keeps_complete_items_and_does_not_touch_legacy_files(
    tmp_path: Path,
) -> None:
    mirror = FileMemoryMirror(tmp_path / "mirror")
    legacy = mirror.root / "User/user.md"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("existing user notes", encoding="utf-8")
    service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"), mirror=mirror)
    namespace = "研究:中文 / alpha"
    first = service.remember(
        MemoryWriteInput(namespace=namespace, text="完整正文\n第二行", reason="seed")
    )
    second = service.remember(
        MemoryWriteInput(namespace="other", text="another namespace", reason="seed")
    )
    directory = mirror.namespace_directory(namespace)
    assert directory != mirror.namespace_directory("other")
    key = namespace_key(namespace)[3:]
    assert base64.urlsafe_b64decode(key + "=" * (-len(key) % 4)).decode("utf-8") == namespace
    assert all((directory / path).is_file() for path in BODY_PATHS)
    body = (directory / "User/user.md").read_text(encoding="utf-8")
    assert body.startswith("<!-- iris-memory source_revision: 1 -->")
    assert first.id in body and first.text in body and namespace in body
    assert second.id not in body
    assert legacy.read_text(encoding="utf-8") == "existing user notes"
    assert not (directory / "Memory.md").exists()
    view = service.file_access([namespace])
    assert view is not None
    assert set(view.document_paths) == {directory / path for path in ("Memory.md", *BODY_PATHS)}


def test_reclassification_and_forget_remove_only_derived_item_text(tmp_path: Path) -> None:
    mirror = FileMemoryMirror(tmp_path / "mirror")
    service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"), mirror=mirror)
    item = service.remember(MemoryWriteInput(text="exact value 42", reason="seed"))
    directory = mirror.namespace_directory("project")
    service.update(
        item.id,
        "project",
        MemoryItemPatch(category=MemoryCategory.USER, kind=MemoryItemKind.PREFERENCE),
        reason="classify",
    )
    assert item.id not in (directory / "User/user.md").read_text(encoding="utf-8")
    assert item.id in (directory / "User/preferences.md").read_text(encoding="utf-8")
    assert service.forget(item.id, "project", reason="done")
    assert item.id not in (directory / "User/preferences.md").read_text(encoding="utf-8")
    state = service.store.read_namespace_state("project")
    assert state.item_revision == state.projection_revision == 3


def test_initial_body_creation_failure_does_not_claim_synced_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mirror = FileMemoryMirror(tmp_path / "mirror")
    service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"), mirror=mirror)
    original_replace = mirror._atomic_replace
    calls = 0

    def replace(relative_path: str, content: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise IrisMemoryError("initial creation failed")
        original_replace(relative_path, content)

    monkeypatch.setattr(mirror, "_atomic_replace", replace)
    item = service.remember(MemoryWriteInput(text="saved in SQLite", reason="seed"))
    state = service.store.read_namespace_state("project")
    assert service.get_item(item.id, ["project"]) == item
    assert state.item_revision == 1 and state.projection_revision is None
    assert "未同步" in service.projection_warning("project")


def test_atomic_replace_failure_keeps_target_and_cleans_own_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mirror = FileMemoryMirror(tmp_path / "mirror")
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    service = MemoryService(store, mirror=mirror)
    service.remember(MemoryWriteInput(text="original", reason="seed"))
    target = mirror.namespace_directory("project") / "User/user.md"
    before = target.read_text(encoding="utf-8")

    def fail_replace(*args: object, **kwargs: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(IrisMemoryError, match="写入失败"):
        mirror.rebuild_from_store(store, "project")
    assert target.read_text(encoding="utf-8") == before
    assert list(target.parent.glob(f".{target.name}.*.tmp")) == []
