"""唯一消息正文、精确发布关联和消费后的局部回收。"""

import json
import sqlite3
from pathlib import Path

import pytest

from iris.evolution.history import PublicationRecord
from iris.evolution.materials import EvolutionMaterialStore
from iris.exceptions import IrisEvolutionError

from .test_materials import _block, _result, _source, body_count
from .test_project_skill import prepare
from .test_publications import reopen


@pytest.mark.asyncio
async def test_failed_attempts_share_one_original_message(tmp_path: Path) -> None:
    """十次真实失败尝试分别归档，但输入消息只保存一次。"""
    service, provider, scope = prepare(tmp_path)
    provider.output = {"body": "", "reason": "无效候选"}
    selected = service.store.read_pending(allowed_sources=scope.allowed_sources).items
    for _ in range(10):
        with pytest.raises(IrisEvolutionError):
            await service.maintain_cycle(scope=scope)
    summaries = service.list_publications().items
    assert len(summaries) == len(provider.requests) == 10
    assert all(item.status == "failed" and item.settled for item in summaries)
    with sqlite3.connect(service.store.path) as database:
        assert database.execute("SELECT count(*) FROM messages").fetchone()[0] == 1
        assert database.execute("SELECT count(*) FROM publication_materials").fetchone()[0] == 10
        assert (
            database.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='captures'"
            ).fetchall()
            == []
        )
        for state, detail in database.execute(
            "SELECT state_json,detail_json FROM publications "
            "JOIN publication_details ON publication_id=id"
        ):
            assert "materials" not in json.loads(state)
            assert "materials" not in json.loads(detail)
    restarted = EvolutionMaterialStore(tmp_path)
    for summary in summaries:
        entry = restarted.get_publication(summary.publication_id)
        assert entry is not None and entry.detail is not None
        assert entry.detail.materials == selected
    assert restarted.read_pending(allowed_sources=scope.allowed_sources).items == selected


def test_out_of_order_overlap_keeps_one_row_for_each_message_including_empty(
    tmp_path: Path,
) -> None:
    """空消息也参与连续水位，重叠块不会复制或覆盖已捕获消息。"""
    store = EvolutionMaterialStore(tmp_path)
    source = _source()
    store.register_source(source, 2)
    partial = store.commit_capture(_block(source, 4, 6, terminal=6, empty=(5,)))
    assert partial.captured_until == 2 and partial.terminal_message_count is None
    complete = store.commit_capture(_block(source, 2, 5, empty=(3,)))
    assert complete.captured_until == complete.terminal_message_count == 6
    store.commit_capture(_block(source, 2, 6, terminal=6, empty=(3, 5)))
    with sqlite3.connect(store.path) as database:
        rows = database.execute(
            "SELECT message_ordinal,records_json FROM messages ORDER BY message_ordinal"
        ).fetchall()
    assert [row[0] for row in rows] == [2, 3, 4, 5]
    assert [bool(json.loads(row[1])) for row in rows] == [True, False, True, False]
    pending = store.read_pending(allowed_sources=frozenset({("lifecycle", "run")}))
    assert [item.start_message_count for item in pending.items] == [2, 3, 4, 5]
    assert [bool(item.records) for item in pending.items] == [True, False, True, False]


@pytest.mark.asyncio
async def test_consumed_message_remains_available_through_publication_reference(
    tmp_path: Path,
) -> None:
    """源材料已消费，重建 service 不需要 lifecycle reader 就能还原档案。"""
    service, provider, scope = prepare(tmp_path)
    selected = service.store.read_pending(allowed_sources=scope.allowed_sources).items
    result = await service.maintain_cycle(scope=scope)
    assert result.status == "updated" and result.publication_id is not None
    assert body_count(service.store) == 0
    with sqlite3.connect(service.store.path) as database:
        assert database.execute("SELECT count(*) FROM messages").fetchone()[0] == 1
        assert database.execute(
            "SELECT position,start_count,end_count FROM publication_materials "
            "WHERE publication_id=? ORDER BY position",
            (result.publication_id,),
        ).fetchall() == [(0, 0, 1)]
    entry = reopen(service).get_publication(result.publication_id)
    assert entry is not None and entry.detail is not None
    record = entry.detail
    assert record.materials == selected
    assert record.evidence_refs[0].quote == selected[0].records[0].text
    assert len(provider.requests) == 1


def test_repeated_consumed_prefix_does_not_resurrect_deleted_messages(tmp_path: Path) -> None:
    """重复捕获不复活已回收前缀，未消费后缀仍完整保留。"""
    store = EvolutionMaterialStore(tmp_path)
    source = _source()
    store.register_source(source, 2)
    store.commit_capture(_block(source, 2, 6, terminal=6))
    allowed = frozenset({("lifecycle", "run")})
    selected = store.read_pending(allowed_sources=allowed, limit=2).items
    store.consume(selected, _result(selected))
    restarted = EvolutionMaterialStore(tmp_path)
    restarted.commit_capture(_block(source, 3, 6, terminal=6))
    with sqlite3.connect(store.path) as database:
        assert database.execute(
            "SELECT message_ordinal FROM messages ORDER BY message_ordinal"
        ).fetchall() == [(4,), (5,)]
    remaining = restarted.read_pending(allowed_sources=allowed)
    assert [item.start_message_count for item in remaining.items] == [4, 5]
    restarted.consume(remaining.items, _result(remaining.items))
    restarted.commit_capture(_block(source, 2, 6, terminal=6))
    with sqlite3.connect(store.path) as database:
        assert database.execute("SELECT count(*) FROM messages").fetchone()[0] == 0
    assert restarted.register_source(source, 2).consumed_until == 6


def test_candidate_reference_failure_rolls_back_archive_and_preserves_pending(
    tmp_path: Path,
) -> None:
    """关联写入失败时整份候选归档回滚，待处理原文不受影响。"""
    store = EvolutionMaterialStore(tmp_path)
    source = _source()
    store.register_source(source, 2)
    store.commit_capture(_block(source, 2, 4, terminal=4))
    allowed = frozenset({("lifecycle", "run")})
    selected = store.read_pending(allowed_sources=allowed).items
    record = PublicationRecord(
        stage="experience", origin="experience", description="候选归档", materials=selected
    )
    with sqlite3.connect(store.path) as database:
        database.execute(
            "CREATE TRIGGER fail_reference AFTER INSERT ON publication_materials "
            "BEGIN SELECT RAISE(ABORT, 'reference interrupted'); END"
        )
    with pytest.raises(IrisEvolutionError, match="reference interrupted"):
        store.save_publication(record)
    restarted = EvolutionMaterialStore(tmp_path)
    assert restarted.get_publication(record.publication_id) is None
    assert restarted.read_pending(allowed_sources=allowed).items == selected
    with sqlite3.connect(store.path) as database:
        assert database.execute("SELECT count(*) FROM publication_materials").fetchone()[0] == 0
        assert database.execute("SELECT count(*) FROM publication_details").fetchone()[0] == 0
        assert database.execute("SELECT count(*) FROM messages").fetchone()[0] == 2
