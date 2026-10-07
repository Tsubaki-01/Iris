"""完整 Memory 历史与发布正文由领域存储持久保存。"""

import sqlite3
from pathlib import Path

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory import (
    FileMemoryMirror,
    GenerationResult,
    MemoryItemPatch,
    MemoryOverviewContent,
    MemoryService,
    MemoryWriteInput,
    SQLiteMemoryStore,
)

from .test_generation_store import _episode, _flush
from .test_overview import _Provider, _service


@pytest.mark.asyncio
async def test_history_includes_consumed_episodes_and_all_stage_results(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    consumed, pending = _episode(store), _episode(store)
    observation = _flush(store, consumed)
    service = MemoryService(store)
    first = await service.alist_episodes("project", limit=1)
    second = service.list_episodes("project", after=first.next_cursor, limit=1)
    assert [first.items[0].id, second.items[0].id] == [consumed.id, pending.id]
    assert second.next_cursor is None
    assert (await service.aget_observation("project", observation.id)).observation == observation
    assert service.get_observation("other", observation.id) is None
    for item_id in ("a", "b", "c"):
        store.record_generation_result(
            GenerationResult(
                id=item_id,
                namespace="history",
                stage="flush",
                status="failed",
                error="真实失败",
                created_at="2026-10-07T00:00:00+00:00",
            )
        )
    results = service.list_generation_results("history", limit=2)
    rest = await service.alist_generation_results("history", after=results.next_cursor, limit=2)
    assert [item.id for item in (*results.items, *rest.items)] == ["a", "b", "c"]
    assert rest.next_cursor is None
    with pytest.raises(IrisMemoryError):
        service.list_episodes("project", limit=0)


@pytest.mark.asyncio
async def test_published_bodies_survive_overwrite_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    mirror = FileMemoryMirror(tmp_path / "mirror")
    service = MemoryService(SQLiteMemoryStore(path), mirror=mirror)
    item = service.remember(MemoryWriteInput(text="第一版", reason="原始要求"))
    first = service.list_publications("project").items[0]
    assert first.status == "published" and first.item_revision == first.projection_revision == 1
    body = next(document for document in first.documents if Path(document.path).name == "user.md")
    assert "第一版" in body.text
    service.update(item.id, "project", MemoryItemPatch(text="第二版"), reason="修正")
    reopened = MemoryService(SQLiteMemoryStore(path))
    assert await reopened.aget_publication("project", first.publication_id) == first
    assert reopened.get_publication("other", first.publication_id) is None
    second = (
        await reopened.alist_publications(
            "project", after=reopened.list_publications("project", limit=1).next_cursor
        )
    ).items[0]
    assert second.item_revision == 2
    assert "第二版" in next(
        document.text for document in second.documents if Path(document.path).name == "user.md"
    )
    assert "第二版" not in body.text
    assert service.list_events("project", item_id=item.id)[0].before["text"] == "第一版"


def test_partial_projection_records_only_written_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mirror = FileMemoryMirror(tmp_path / "mirror")
    service = MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"), mirror=mirror)
    replace = mirror._atomic_replace
    calls = 0

    def fail_second(path: str, text: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise IrisMemoryError("第二份文件写入失败")
        replace(path, text)

    monkeypatch.setattr(mirror, "_atomic_replace", fail_second)
    service.remember(MemoryWriteInput(text="已保存条目", reason="写入"))
    publication = service.list_publications("project").items[0]
    assert publication.status == "failed"
    assert len(publication.documents) == 1 and publication.projection_revision is None
    assert "第二份文件" in publication.error
    document = publication.documents[0]
    assert Path(document.path).read_text(encoding="utf-8") == document.text


def test_overview_conflict_does_not_claim_candidate_was_published(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    mirror = FileMemoryMirror(tmp_path / "mirror")
    service = MemoryService(store, mirror=mirror)
    item = service.remember(MemoryWriteInput(text="第一版", reason="开始"))
    old = store.read_namespace_snapshot("project")
    service.update(item.id, "project", MemoryItemPatch(text="第二版"), reason="更新")
    fresh = store.read_namespace_snapshot("project")
    mirror.publish_overview(
        store,
        fresh,
        MemoryOverviewContent(core_facts="第二版", knowledge_scope="项目"),
        generation_result_id="result-2",
    )
    published, _ = mirror.publish_overview(
        store,
        old,
        MemoryOverviewContent(core_facts="旧候选", knowledge_scope="旧"),
        generation_result_id="result-1",
    )
    assert not published
    records = [
        record for record in service.list_publications("project").items if record.kind == "overview"
    ]
    assert records[0].generation_result_id == "result-2"
    assert records[0].status == "published" and "第二版" in records[0].documents[0].text
    assert records[1].status == "conflict" and records[1].documents == ()


def test_written_projection_with_failed_state_commit_is_unconfirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """文件完成而 SQLite 版本提交失败时，不伪造跨文件原子成功。"""
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    mirror = FileMemoryMirror(tmp_path / "mirror")
    service = MemoryService(store, mirror=mirror)
    service.remember(MemoryWriteInput(text="原文", reason="开始"))
    failed = False

    class Connection(sqlite3.Connection):
        def execute(self, sql: str, parameters: object = ()) -> sqlite3.Cursor:
            nonlocal failed
            if "projection_revision = excluded.projection_revision" in sql and not failed:
                failed = True
                raise sqlite3.OperationalError("版本提交失败")
            return super().execute(sql, parameters)

    def connect() -> sqlite3.Connection:
        connection = sqlite3.connect(store.path, factory=Connection)
        connection.row_factory = sqlite3.Row
        return connection

    monkeypatch.setattr(store, "_connect", connect)
    with pytest.raises(IrisMemoryError):
        mirror.rebuild_from_store(store, "project")
    publication = service.list_publications("project").items[-1]
    assert publication.status == "unconfirmed" and publication.documents


@pytest.mark.asyncio
async def test_overview_publication_points_to_actual_generation_result(tmp_path: Path) -> None:
    service = _service(tmp_path, _Provider())
    service.remember(MemoryWriteInput(text="原文", reason="开始"))
    result = await service.refresh_overview("project")
    assert result.published
    publication = next(
        item for item in service.list_publications("project").items if item.kind == "overview"
    )
    generated = service.list_generation_results("project").items[-1]
    assert publication.generation_result_id == generated.id
    assert generated.stage == "overview" and generated.status == "completed"
