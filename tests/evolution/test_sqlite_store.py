"""Evolution 的正文去重、摘要读取与 SQLite 事务契约。"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris.evolution.history import PublicationDocument, PublicationRecord
from iris.evolution.materials import EvolutionMaterialStore
from iris.evolution.models import EvolutionResult, HostOrigin, RevisionItem, RevisionTarget
from iris.exceptions import IrisEvolutionError


def publication() -> PublicationRecord:
    """一份已准备好写入的完整档案。"""
    return PublicationRecord(
        publication_id="publication",
        stage="experience",
        origin="experience",
        description="保留未完成事项",
        publication_state="unconfirmed",
        before_documents=(PublicationDocument(path="compaction.j2", text="旧正文"),),
        candidate_documents=(PublicationDocument(path="compaction.j2", text="新正文"),),
    )


def test_after_body_is_derived_only_from_confirmation() -> None:
    record = publication()
    assert record.after_documents == ()
    confirmed = record.model_copy(update={"publication_state": "confirmed"})
    assert confirmed.after_documents == confirmed.candidate_documents
    assert "after_documents" not in confirmed.model_dump()


def test_state_update_does_not_rewrite_publication_body(tmp_path: Path) -> None:
    store = EvolutionMaterialStore(tmp_path)
    record = publication()
    store.save_publication(record)
    assert store.path == tmp_path / ".iris" / "evolution" / "evolution.db"
    with sqlite3.connect(store.path) as connection:
        body = connection.execute("SELECT detail_json FROM publication_details").fetchone()[0]
        assert "after_documents" not in body
        connection.execute(
            "CREATE TRIGGER immutable_body BEFORE UPDATE ON publication_details "
            "BEGIN SELECT RAISE(ABORT, 'body rewritten'); END"
        )
    confirmed = record.model_copy(
        update={
            "publication_state": "confirmed",
            "published_at": datetime.now(UTC),
            "outcome": EvolutionResult(stage="experience", status="updated"),
        }
    )
    store.save_publication(confirmed)
    assert (
        EvolutionMaterialStore(tmp_path).get_publication(record.publication_id).detail == confirmed
    )


def test_history_lists_do_not_read_detail_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EvolutionMaterialStore(tmp_path)
    record = publication()
    store.save_publication(record)
    request = RevisionItem(
        description="保留约束",
        targets=(RevisionTarget(kind="prompt", name="compaction"),),
        origin=HostOrigin(),
    )
    store.enqueue_revision(request)
    connect = sqlite3.connect

    def connect_without_bodies(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = connect(*args, **kwargs)

        def authorize(action: int, table: str, column: str, database: str, source: str) -> int:
            if action == sqlite3.SQLITE_READ and (
                (table == "publications" and column == "state_json")
                or table == "publication_details"
                or (table == "requests" and column == "payload")
            ):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect_without_bodies)
    summary = store.list_publications().items[0]
    assert summary.publication_id == record.publication_id
    assert "candidate_documents" not in summary.model_dump()
    request_summary = store.list_revision_requests().items[0]
    assert request_summary.id == request.id
    assert "evidence" not in request_summary.model_dump()


def test_request_detail_survives_settlement(tmp_path: Path) -> None:
    store = EvolutionMaterialStore(tmp_path)
    item = RevisionItem(
        description="原始修订请求",
        targets=(RevisionTarget(kind="config", name="system"),),
        origin=HostOrigin(),
    )
    store.enqueue_revision(item)
    result = EvolutionResult(stage="revision", status="no_change", revision_id=item.id)
    store.settle_revision(item.id, result)
    restarted = EvolutionMaterialStore(tmp_path)
    assert restarted.get_revision_request(item.id) == item
    assert restarted.list_revision_requests().items[0].status == "no_change"
    assert restarted.revision_result(item.id) == result


def test_current_schema_is_created_and_reopened(tmp_path: Path) -> None:
    """当前结构版本与重新打开的同一份状态一致。"""
    store = EvolutionMaterialStore(tmp_path)
    record = publication()
    store.save_publication(record)
    with sqlite3.connect(store.path) as database:
        assert database.execute("PRAGMA user_version").fetchone()[0] == 5
    assert EvolutionMaterialStore(tmp_path).get_publication(record.publication_id).detail == record


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_old_schema_is_rejected_without_rewriting_it(tmp_path: Path, version: int) -> None:
    """旧持久契约明确拒绝，原表和版本都不被自动迁移。"""
    path = tmp_path / ".iris" / "evolution" / "evolution.db"
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as database:
        database.execute("CREATE TABLE old_schema (value TEXT)")
        database.execute("INSERT INTO old_schema VALUES ('preserved')")
        database.execute(f"PRAGMA user_version={version}")
    with pytest.raises(IrisEvolutionError, match="schema 5"):
        EvolutionMaterialStore(tmp_path)
    with sqlite3.connect(path) as database:
        assert database.execute("PRAGMA user_version").fetchone()[0] == version
        assert database.execute("SELECT value FROM old_schema").fetchone()[0] == "preserved"
        assert database.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [
            ("old_schema",)
        ]
