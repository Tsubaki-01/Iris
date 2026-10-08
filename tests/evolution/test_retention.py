"""十次完整历史窗口与发布恢复、原文引用的原子边界。"""

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris.evolution._publication import PublicationJournal
from iris.evolution.history import PublicationDocument, PublicationRecord
from iris.evolution.materials import EvolutionMaterialStore
from iris.evolution.models import EvolutionMaterial, EvolutionResult, EvolutionStatus, RevisionItem
from iris.exceptions import IrisEvolutionError

from .test_materials import _block, _issue, _source


def _candidate(
    store: EvolutionMaterialStore,
    number: int,
    *,
    materials: tuple[EvolutionMaterial, ...] | None = None,
    issue: RevisionItem | None = None,
) -> PublicationRecord:
    """创建真实已捕获来源的候选，固定创建顺序以避免时钟影响。"""
    if materials is None:
        source = _source(f"run-{number}")
        store.register_source(source, 2)
        store.commit_capture(_block(source, 2, 3, terminal=3))
        materials = store.read_pending(
            allowed_sources=frozenset({("lifecycle", source.run_id)})
        ).items
    return PublicationRecord(
        publication_id=f"p{number:03d}",
        created_at=datetime(2026, 10, 8, tzinfo=UTC) + timedelta(seconds=number),
        stage="experience",
        origin="experience",
        description="整理本批经验",
        materials=materials,
        proposed_issue=issue,
        before_documents=(PublicationDocument(path="SKILL.md", text=f"before-{number}"),),
        candidate_documents=(PublicationDocument(path="SKILL.md", text=f"candidate-{number}"),),
        observed_documents=(PublicationDocument(path="SKILL.md", text=f"observed-{number}"),),
    )


def _finish(
    store: EvolutionMaterialStore, record: PublicationRecord, status: EvolutionStatus = "no_change"
) -> EvolutionResult:
    """通过原发布 owner 完成实际尝试，而不直接改数据库结算字段。"""
    journal = PublicationJournal(store)
    result = EvolutionResult(status=status, publication_id=record.publication_id, reason="测试结果")
    if status in {"failed", "cancelled"}:
        journal.failure(record, result)
        return result
    journal.begin(record)
    return journal.complete(record, result)


def _fill(store: EvolutionMaterialStore, start: int, stop: int) -> None:
    for number in range(start, stop):
        _finish(store, _candidate(store, number))


@pytest.mark.parametrize("last_status", ["updated", "no_change", "failed", "cancelled", "conflict"])
def test_eleventh_finished_attempt_expires_only_oldest_details(
    tmp_path: Path, last_status: EvolutionStatus
) -> None:
    store = EvolutionMaterialStore(tmp_path)
    _fill(store, 0, 10)
    assert store.get_publication("p000").detail_status == "available"
    _finish(store, _candidate(store, 10), last_status)
    expired = store.get_publication("p000")
    assert expired.detail_status == expired.summary.detail_status == "expired"
    assert expired.detail is None and expired.summary.settled
    assert store.get_publication("missing") is None
    assert [item.detail_status for item in store.list_publications().items] == ["expired"] + [
        "available"
    ] * 10
    with sqlite3.connect(store.path) as database:
        assert (
            database.execute(
                "SELECT count(*) FROM publication_details WHERE publication_id='p000'"
            ).fetchone()[0]
            == 0
        )
        assert (
            database.execute(
                "SELECT count(*) FROM publication_materials WHERE publication_id='p000'"
            ).fetchone()[0]
            == 0
        )
        state = json.loads(
            database.execute("SELECT state_json FROM publications WHERE id='p000'").fetchone()[0]
        )
        assert not state.get("observed_documents")
        assert (
            database.execute(
                "SELECT count(*) FROM messages WHERE source_key=?",
                (json.dumps(["lifecycle", "run-0"], separators=(",", ":")),),
            ).fetchone()[0]
            == 0
        )


def test_unconfirmed_and_confirmed_unsettled_are_protected_outside_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = EvolutionMaterialStore(tmp_path)
    journal = PublicationJournal(store)
    unknown, confirmed = _candidate(store, 0), _candidate(store, 1)
    journal.begin(unknown)
    journal.begin(confirmed)
    settle = store.settle_publication

    def fail_settlement(record: PublicationRecord) -> EvolutionResult:
        raise IrisEvolutionError("结算暂时失败")

    monkeypatch.setattr(store, "settle_publication", fail_settlement)
    with pytest.raises(IrisEvolutionError, match="结算暂时失败"):
        journal.complete(confirmed, EvolutionResult(status="updated", publication_id="p001"))
    monkeypatch.setattr(store, "settle_publication", settle)
    _fill(store, 2, 13)
    for identity, state in (("p000", "unconfirmed"), ("p001", "confirmed")):
        entry = EvolutionMaterialStore(tmp_path).get_publication(identity)
        assert entry.detail_status == "available" and not entry.summary.settled
        assert entry.detail.publication_state == state
        assert entry.detail.materials
    assert sum(item.detail_status == "available" for item in store.list_publications().items) == 12


def test_shared_consumed_messages_survive_until_last_complete_reference_expires(
    tmp_path: Path,
) -> None:
    store = EvolutionMaterialStore(tmp_path)
    first = _candidate(store, 0)
    _finish(store, first, "failed")
    _finish(store, _candidate(store, 1, materials=first.materials))
    key = json.dumps(["lifecycle", "run-0"], separators=(",", ":"))
    _fill(store, 2, 11)
    assert store.get_publication("p000").detail is None
    assert store.get_publication("p001").detail.materials == first.materials
    with sqlite3.connect(store.path) as database:
        assert (
            database.execute("SELECT count(*) FROM messages WHERE source_key=?", (key,)).fetchone()[
                0
            ]
            == 1
        )
    _finish(store, _candidate(store, 11))
    with sqlite3.connect(store.path) as database:
        assert (
            database.execute("SELECT count(*) FROM messages WHERE source_key=?", (key,)).fetchone()[
                0
            ]
            == 0
        )
    source = first.materials[0].source
    store.commit_capture(_block(source, 2, 3, terminal=3))
    assert not store.read_pending(allowed_sources=frozenset({("lifecycle", source.run_id)})).items
    with sqlite3.connect(store.path) as database:
        assert (
            database.execute("SELECT count(*) FROM messages WHERE source_key=?", (key,)).fetchone()[
                0
            ]
            == 0
        )


def test_stale_owner_receipt_cannot_revive_expired_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_store = EvolutionMaterialStore(tmp_path)
    first_owner = PublicationJournal(first_store)
    record = _candidate(first_store, 0)
    first_owner.begin(record)
    settle = first_store.settle_publication

    def fail_settlement(record: PublicationRecord) -> EvolutionResult:
        raise IrisEvolutionError("保留旧收据")

    monkeypatch.setattr(first_store, "settle_publication", fail_settlement)
    with pytest.raises(IrisEvolutionError, match="保留旧收据"):
        first_owner.complete(record, EvolutionResult(status="updated", publication_id="p000"))
    monkeypatch.setattr(first_store, "settle_publication", settle)
    second_store = EvolutionMaterialStore(tmp_path)
    committed = PublicationJournal(second_store).resume()
    assert committed.status == "updated" and committed.consumed_ranges
    _fill(second_store, 1, 11)
    assert second_store.get_publication("p000").detail_status == "expired"
    assert first_owner.resume() == committed
    assert first_owner.resume() is None
    entry = first_store.get_publication("p000")
    assert entry.detail_status == "expired" and entry.detail is None and entry.summary.settled
    with sqlite3.connect(first_store.path) as database:
        assert (
            database.execute(
                "SELECT count(*) FROM consumed_publications WHERE publication_id='p000'"
            ).fetchone()[0]
            == 1
        )


@pytest.mark.parametrize("status", ["conflict", "no_change"])
def test_expired_proposal_keeps_real_summary_or_actual_request_link(
    tmp_path: Path, status: EvolutionStatus
) -> None:
    store = EvolutionMaterialStore(tmp_path)
    record = _candidate(store, 0)
    issue = _issue(record.materials[0].source)
    record = record.model_copy(update={"proposed_issue": issue})
    _finish(store, record, status)
    _fill(store, 1, 11)
    entry = store.get_publication("p000")
    assert entry.detail_status == "expired" and entry.evidence == issue.evidence
    if status == "conflict":
        assert entry.summary.proposed_revision_id is None
        assert store.get_revision_request(issue.id) is None
        assert entry.proposed_issue_summary.description == issue.description
        assert entry.proposed_issue_summary.targets == issue.targets
    else:
        assert entry.summary.proposed_revision_id == issue.id
        assert entry.proposed_issue_summary is None
        assert store.get_revision_request(issue.id) == issue
        assert store.revision_result(issue.id) is None


def test_retention_failure_rolls_back_consumption_and_old_detail_expiration(tmp_path: Path) -> None:
    store = EvolutionMaterialStore(tmp_path)
    _fill(store, 0, 10)
    record = _candidate(store, 10)
    with sqlite3.connect(store.path) as database:
        for event in ("DELETE", "UPDATE"):
            database.execute(
                f"CREATE TRIGGER fail_retention_{event} BEFORE {event} ON publication_details "
                "WHEN OLD.publication_id='p000' "
                "BEGIN SELECT RAISE(ABORT, 'retention interrupted'); END"
            )
    journal = PublicationJournal(store)
    journal.begin(record)
    with pytest.raises(IrisEvolutionError, match="retention interrupted"):
        journal.complete(record, EvolutionResult(status="updated", publication_id="p010"))
    assert store.get_publication("p000").detail_status == "available"
    pending = store.get_publication("p010")
    assert pending.detail.publication_state == "confirmed" and not pending.summary.settled
    assert store.register_source(record.materials[0].source, 2).consumed_until == 2
    with sqlite3.connect(store.path) as database:
        database.execute("DROP TRIGGER fail_retention_DELETE")
        database.execute("DROP TRIGGER fail_retention_UPDATE")
    assert PublicationJournal(EvolutionMaterialStore(tmp_path)).resume().status == "updated"
    assert store.get_publication("p000").detail_status == "expired"
    assert store.register_source(record.materials[0].source, 2).consumed_until == 3


def test_pending_revision_attempts_share_window_without_pinning_full_materials(
    tmp_path: Path,
) -> None:
    store = EvolutionMaterialStore(tmp_path)
    first = _candidate(store, 0)
    issue = _issue(first.materials[0].source)
    _finish(store, first.model_copy(update={"proposed_issue": issue}))
    for number in range(1, 12):
        record = PublicationRecord(
            publication_id=f"p{number:03d}",
            created_at=first.created_at + timedelta(seconds=number),
            stage="revision",
            origin="experience",
            revision_id=issue.id,
            description=issue.description,
            targets=issue.targets,
        )
        _finish(store, record, "failed")
    assert store.get_publication("p000").detail_status == "expired"
    assert store.get_publication("p001").detail_status == "expired"
    assert store.get_publication("p001").evidence == issue.evidence
    assert store.get_revision_request(issue.id) == issue
    assert store.revision_result(issue.id) is None
    summaries = store.list_publications().items
    assert sum(item.detail_status == "available" for item in summaries) == 10
    with sqlite3.connect(store.path) as database:
        assert database.execute("SELECT count(*) FROM messages").fetchone()[0] == 0


def test_equal_creation_times_use_id_not_completion_order(tmp_path: Path) -> None:
    store = EvolutionMaterialStore(tmp_path)
    created_at = datetime(2026, 10, 8, tzinfo=UTC)
    for number in reversed(range(11)):
        record = _candidate(store, number).model_copy(update={"created_at": created_at})
        _finish(store, record)
    assert store.get_publication("p000").detail_status == "expired"
    assert all(
        store.get_publication(f"p{i:03d}").detail_status == "available" for i in range(1, 11)
    )
