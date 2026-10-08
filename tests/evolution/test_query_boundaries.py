"""Evolution 增量捕获与短元数据查询的读取范围。"""

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from iris.evolution.materials import EvolutionMaterialStore
from iris.evolution.models import (
    EvolutionResult,
    EvolutionSession,
    HostOrigin,
    RevisionItem,
    RevisionTarget,
)

from .test_materials import _block, _issue, _result, _source


def _deny_large_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """只限制 SQL 读取，不限制捕获或请求正常写入。"""
    connect = sqlite3.connect

    def connect_without_bodies(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        database = connect(*args, **kwargs)

        def authorize(
            action: int,
            table: str | None,
            column: str | None,
            database_name: str | None,
            trigger_name: str | None,
        ) -> int:
            if action == sqlite3.SQLITE_READ and (
                (table == "messages" and column == "records_json")
                or (table == "requests" and column == "payload")
                or table == "publication_details"
            ):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        database.set_authorizer(authorize)
        return database

    monkeypatch.setattr(sqlite3, "connect", connect_without_bodies)


def test_capture_uses_short_watermarks_without_loading_message_bodies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """历史正文不可读时仍能登记新来源、捕获和重读封源水位。"""
    store = EvolutionMaterialStore(tmp_path)
    history = _source("history")
    store.register_source(history, 2)
    store.commit_capture(_block(history, 2, 4, terminal=4))
    current = _source("current")
    _deny_large_columns(monkeypatch)

    registered = store.register_source(current, 2)
    assert registered.captured_until == 2
    partial = store.commit_capture(_block(current, 2, 3))
    assert partial.captured_until == 3 and partial.terminal_message_count is None
    complete = store.commit_capture(_block(current, 3, 4, terminal=4))
    assert complete.captured_until == complete.terminal_message_count == 4
    assert complete.outcome == "completed"
    assert store.register_source(current, 2) == complete
    assert store.list_capture_sources("lifecycle") == ()


@pytest.mark.parametrize("operation", ["materials", "revisions", "sources", "sessions"])
def test_scheduling_reads_short_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """存在性和来源概览不读取消息正文或请求证据。"""
    store = EvolutionMaterialStore(tmp_path)
    pending = _source("pending")
    store.register_source(pending, 2)
    store.commit_capture(_block(pending, 2, 3, terminal=3))
    revision_source = _source("revision-only")
    store.enqueue_revision(_issue(revision_source))
    session = EvolutionSession(lifecycle_source_id="reader", session_id="host")
    request = RevisionItem(
        description="宿主请求",
        targets=(RevisionTarget(kind="prompt", name="compaction"),),
        origin=HostOrigin(session=session),
    )
    store.enqueue_revision(request)
    _deny_large_columns(monkeypatch)

    if operation == "materials":
        assert store.has_pending_materials(allowed_sources=frozenset({("lifecycle", "pending")}))
        assert not store.has_pending_materials(allowed_sources=frozenset())
    elif operation == "revisions":
        assert store.has_pending_revisions(
            allowed_sources=frozenset({("lifecycle", "revision-only")}),
            allowed_sessions=frozenset(),
            allowed_targets=frozenset({("prompt", "compaction")}),
        )
        assert store.has_pending_revisions(
            allowed_sources=frozenset(),
            allowed_sessions=frozenset({("reader", "host")}),
            allowed_targets=frozenset({("prompt", "compaction")}),
        )
        assert not store.has_pending_revisions(
            allowed_sources=frozenset(),
            allowed_sessions=frozenset(),
            allowed_targets=frozenset({("prompt", "compaction")}),
        )
    elif operation == "sources":
        assert set(store.list_pending_sources()) == {pending, revision_source}
    else:
        assert store.list_pending_sessions() == (session,)


@pytest.mark.parametrize("preferred", ["last", "excluded", "missing", "settled"])
def test_revision_priority_fills_limit_after_eligibility_filtering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preferred: str
) -> None:
    """优先项不排他，未选请求的 payload 不参加反序列化。"""
    store = EvolutionMaterialStore(tmp_path)
    target = (RevisionTarget(kind="prompt", name="compaction"),)
    created = datetime(2026, 10, 8, tzinfo=UTC)
    for index, identity in enumerate(("excluded", "first", "second", "last", "settled")):
        origin = (
            HostOrigin(session=EvolutionSession(lifecycle_source_id="reader", session_id="waiting"))
            if identity == "excluded"
            else HostOrigin()
        )
        store.enqueue_revision(
            RevisionItem(
                id=identity,
                created_at=created + timedelta(seconds=index),
                description=identity,
                targets=target,
                origin=origin,
            )
        )
    store.settle_revision(
        "settled", EvolutionResult(stage="revision", status="no_change", revision_id="settled")
    )
    parsed: list[str] = []
    parse = RevisionItem.model_validate_json

    def tracked_parse(value: str | bytes | bytearray, **kwargs: Any) -> RevisionItem:
        result = parse(value, **kwargs)
        parsed.append(result.id)
        return result

    monkeypatch.setattr(RevisionItem, "model_validate_json", staticmethod(tracked_parse))
    items = store.read_pending_revisions(
        allowed_sources=frozenset(),
        allowed_sessions=frozenset(),
        allowed_targets=frozenset({("prompt", "compaction")}),
        limit=2,
        requested_revision_id=preferred,
    )
    expected = ["last", "first"] if preferred == "last" else ["first", "second"]
    assert [item.id for item in items] == expected
    assert sorted(parsed) == sorted(expected)


def test_consumption_does_not_delete_an_unrelated_source_body(tmp_path: Path) -> None:
    """本轮消费只能清理所选来源，其他来源仍可按原文学习。"""
    store = EvolutionMaterialStore(tmp_path)
    current, unrelated = _source("current"), _source("unrelated")
    for source in (current, unrelated):
        store.register_source(source, 2)
        store.commit_capture(_block(source, 2, 4, terminal=4))
    allowed = frozenset({("lifecycle", "current")})
    selected = store.read_pending(allowed_sources=allowed).items
    unrelated_key = json.dumps(["lifecycle", "unrelated"], separators=(",", ":"))
    with sqlite3.connect(store.path) as database:
        unrelated_body = database.execute(
            "SELECT message_ordinal,records_json FROM messages WHERE source_key=? "
            "ORDER BY message_ordinal",
            (unrelated_key,),
        ).fetchall()
        database.execute(
            "CREATE TRIGGER preserve_unrelated BEFORE DELETE ON messages "
            'WHEN OLD.source_key=\'["lifecycle","unrelated"]\' '
            "BEGIN SELECT RAISE(ABORT, 'unrelated body deleted'); END"
        )
    store.consume(selected, _result(selected))
    with sqlite3.connect(store.path) as database:
        assert (
            database.execute(
                "SELECT message_ordinal,records_json FROM messages WHERE source_key=? "
                "ORDER BY message_ordinal",
                (unrelated_key,),
            ).fetchall()
            == unrelated_body
        )
    remaining = store.read_pending(allowed_sources=frozenset({("lifecycle", "unrelated")}))
    assert [item.start_message_count for item in remaining.items] == [2, 3]
    assert not store.read_pending(allowed_sources=allowed).items
