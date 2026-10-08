"""项目经验材料的连续捕获、跨进程重复与持久消费。"""

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from iris.evolution.materials import EvolutionMaterialStore
from iris.evolution.models import (
    EvolutionCaptureBlock,
    EvolutionMaterial,
    EvolutionRange,
    EvolutionRecord,
    EvolutionResult,
    EvolutionSession,
    EvolutionSource,
    ExperienceOrigin,
    HostOrigin,
    RevisionEvidence,
    RevisionItem,
    RevisionTarget,
)
from iris.exceptions import IrisEvolutionError


def _source(run_id: str = "run", session_id: str = "session") -> EvolutionSource:
    return EvolutionSource(lifecycle_source_id="lifecycle", run_id=run_id, session_id=session_id)


def _block(
    source: EvolutionSource,
    start: int,
    end: int,
    *,
    terminal: int | None = None,
    empty: tuple[int, ...] = (),
) -> EvolutionCaptureBlock:
    return EvolutionCaptureBlock(
        source=source,
        initial_message_count=2,
        start_message_count=start,
        end_message_count=end,
        terminal_message_count=terminal,
        outcome="completed" if terminal is not None else None,
        records=tuple(
            EvolutionRecord(
                ref=f"{source.run_id}:{ordinal}:0",
                message_ordinal=ordinal,
                block_index=0,
                role="user",
                text=f"message {ordinal}",
                occurred_at="2026-10-05T00:00:00+00:00",
            )
            for ordinal in range(start, end)
            if ordinal not in empty
        ),
    )


def _result(items: tuple[EvolutionMaterial, ...]) -> EvolutionResult:
    return EvolutionResult(
        status="no_change",
        reason="本批无需调整经验",
        consumed_ranges=tuple(
            EvolutionRange(
                source=item.source,
                start_message_count=item.start_message_count,
                end_message_count=item.end_message_count,
            )
            for item in items
        ),
    )


def _issue(source: EvolutionSource, *, item_id: str = "issue") -> RevisionItem:
    """只携带B需要的稳定引用与小片段。"""
    return RevisionItem(
        id=item_id,
        description="压缩需要保留关键标识符",
        targets=(RevisionTarget(kind="prompt", name="compaction"),),
        evidence=(RevisionEvidence(ref="run:2:0", quote="message 2"),),
        origin=ExperienceOrigin(sources=(source,)),
    )


def body_count(store: EvolutionMaterialStore) -> int:
    """读取仍待消费的材料正文数量。"""
    with sqlite3.connect(store.path) as database:
        return database.execute(
            "SELECT count(*) FROM captures WHERE body_json IS NOT NULL"
        ).fetchone()[0]


def test_a_progress_preserves_issue_before_body_cleanup_and_b_settles_independently(
    tmp_path: Path,
) -> None:
    store = EvolutionMaterialStore(tmp_path)
    source = _source()
    store.register_source(source, 2)
    store.commit_capture(_block(source, 2, 3, terminal=3))
    allowed = frozenset({("lifecycle", "run")})
    selected = store.read_pending(allowed_sources=allowed).items
    issue = _issue(source)
    store.consume(selected, _result(selected), issue=issue)
    assert body_count(store) == 0
    restarted = EvolutionMaterialStore(tmp_path)
    assert restarted.read_pending(allowed_sources=allowed).items == ()
    assert restarted.read_pending_revisions(
        allowed_sources=allowed,
        allowed_sessions=frozenset(),
        allowed_targets=frozenset({("prompt", "compaction")}),
    ) == (issue,)
    assert restarted.list_pending_sources() == (source,)
    restarted.record_step(EvolutionResult(status="failed", stage="revision", revision_id=issue.id))
    assert restarted.read_pending_revisions(
        allowed_sources=allowed,
        allowed_sessions=frozenset(),
        allowed_targets=frozenset({("prompt", "compaction")}),
    ) == (issue,)
    settled = EvolutionResult(
        status="no_change", stage="revision", revision_id=issue.id, reason="当前规则已满足"
    )
    restarted.settle_revision(issue.id, settled)
    final = EvolutionMaterialStore(tmp_path)
    assert final.revision_result(issue.id) == settled
    assert (
        final.read_pending_revisions(
            allowed_sources=allowed,
            allowed_sessions=frozenset(),
            allowed_targets=frozenset({("prompt", "compaction")}),
        )
        == ()
    )
    assert final.list_pending_sources() == ()


def test_host_request_is_independent_and_origin_filter_precedes_limit(tmp_path: Path) -> None:
    store = EvolutionMaterialStore(tmp_path)
    waiting = RevisionItem(
        id="waiting",
        description="暂时不可处理",
        targets=(RevisionTarget(kind="config", name="system"),),
        origin=HostOrigin(
            session=EvolutionSession(lifecycle_source_id="reader", session_id="waiting")
        ),
    )
    independent = RevisionItem(
        id="independent",
        description="没有历史失败的明确请求",
        targets=waiting.targets,
        origin=HostOrigin(),
    )
    store.enqueue_revision(waiting)
    store.enqueue_revision(independent)
    assert store.list_pending_sources() == ()
    assert store.list_pending_sessions() == (waiting.origin.session,)
    assert store.read_pending_revisions(
        allowed_sources=frozenset(),
        allowed_sessions=frozenset(),
        allowed_targets=frozenset({("config", "system")}),
        limit=1,
    ) == (independent,)
    assert store.revision_result(waiting.id) is None


def test_only_contiguous_terminal_sources_become_pending(tmp_path: Path) -> None:
    store = EvolutionMaterialStore(tmp_path)
    source = _source()
    registered = store.register_source(source, 2)
    assert registered.captured_until == registered.consumed_until == 2
    assert store.list_pending_sources() == ()
    incomplete = store.commit_capture(_block(source, 4, 6, terminal=6))
    assert incomplete.captured_until == 2
    assert incomplete.terminal_message_count is None
    assert store.list_capture_sources("lifecycle") == (incomplete,)
    assert store.list_pending_sources() == ()

    complete = store.commit_capture(_block(source, 2, 4))
    assert complete.captured_until == complete.terminal_message_count == 6
    assert complete.outcome == "completed"
    assert store.list_pending_sources() == (source,)
    restarted = EvolutionMaterialStore(tmp_path)
    assert restarted.list_capture_sources("lifecycle") == ()
    assert restarted.register_source(source, 2) == complete


def test_pending_deduplicates_overlaps_and_filters_before_limit(tmp_path: Path) -> None:
    store = EvolutionMaterialStore(tmp_path)
    excluded = _source("a")
    allowed = _source("b")
    for source in (excluded, allowed):
        store.register_source(source, 2)
        store.commit_capture(_block(source, 2, 5, terminal=5, empty=(3,)))
        store.commit_capture(_block(source, 2, 4, empty=(3,)))
    pending = store.read_pending(allowed_sources=frozenset({("lifecycle", "b")}), limit=2)
    assert pending.has_more
    assert [item.start_message_count for item in pending.items] == [2, 3]
    assert all(item.source == allowed for item in pending.items)
    assert pending.items[0].records[0].ref == "b:2:0"
    assert pending.items[1].records == ()
    assert store.read_pending(allowed_sources=frozenset(), limit=1).items == ()
    assert set(item.run_id for item in store.list_pending_sources()) == {"a", "b"}


def test_successful_ranges_survive_body_cleanup_and_restart(tmp_path: Path) -> None:
    store = EvolutionMaterialStore(tmp_path)
    source = _source()
    store.register_source(source, 2)
    block = _block(source, 2, 5, terminal=5)
    store.commit_capture(block)
    allowed = frozenset({("lifecycle", "run")})
    first = store.read_pending(allowed_sources=allowed, limit=1).items
    store.consume(first, _result(first))

    restarted = EvolutionMaterialStore(tmp_path)
    state = restarted.register_source(source, 2)
    assert state.consumed_until == 3
    assert state.captured_until == state.terminal_message_count == 5
    remaining = restarted.read_pending(allowed_sources=allowed).items
    assert [item.start_message_count for item in remaining] == [3, 4]
    restarted.consume(remaining, _result(remaining))
    assert body_count(restarted) == 0

    after_cleanup = EvolutionMaterialStore(tmp_path)
    assert after_cleanup.list_pending_sources() == ()
    final = after_cleanup.register_source(source, 2)
    assert final.captured_until == final.consumed_until == final.terminal_message_count == 5
    assert after_cleanup.commit_capture(block).consumed_until == 5
    assert after_cleanup.read_pending(allowed_sources=allowed).items == ()


def test_two_processes_publish_complete_duplicate_ranges(tmp_path: Path) -> None:
    """独立进程可重复捕获，同一项目消费时只得到每条消息一次。"""
    source = _source()
    block = _block(source, 2, 6, terminal=6)
    script = """
import sys
from pathlib import Path
from iris.evolution.materials import EvolutionMaterialStore
from iris.evolution.models import EvolutionCaptureBlock
store = EvolutionMaterialStore(Path(sys.argv[1]))
block = EvolutionCaptureBlock.model_validate_json(sys.argv[2])
store.register_source(block.source, block.initial_message_count)
store.commit_capture(block)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), block.model_dump_json()],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 0, stdout + stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate()
    store = EvolutionMaterialStore(tmp_path)
    pending = store.read_pending(allowed_sources=frozenset({("lifecycle", "run")}))
    assert [item.start_message_count for item in pending.items] == [2, 3, 4, 5]
    assert body_count(store) == 2
    assert store.register_source(source, 2).captured_until == 6


@pytest.mark.parametrize("failure", ["progress", "cleanup"])
def test_consume_rolls_back_progress_issue_and_cleanup(tmp_path: Path, failure: str) -> None:
    store = EvolutionMaterialStore(tmp_path)
    source = _source()
    store.register_source(source, 2)
    store.commit_capture(_block(source, 2, 4, terminal=4))
    allowed = frozenset({("lifecycle", "run")})
    selected = store.read_pending(allowed_sources=allowed).items
    issue = _issue(source)
    event = (
        "BEFORE INSERT ON progress"
        if failure == "progress"
        else "BEFORE UPDATE OF body_json ON captures"
    )
    with sqlite3.connect(store.path) as database:
        database.execute(
            f"CREATE TRIGGER fail_consume {event} "
            "BEGIN SELECT RAISE(ABORT, 'consume interrupted'); END"
        )
    with pytest.raises(IrisEvolutionError, match="consume interrupted"):
        store.consume(selected, _result(selected), issue=issue)
    restarted = EvolutionMaterialStore(tmp_path)
    assert restarted.register_source(source, 2).consumed_until == 2
    assert restarted.read_pending(allowed_sources=allowed).items == selected
    assert restarted.get_revision_request(issue.id) is None
    assert body_count(restarted) == 1


def test_failed_capture_does_not_advance_source(tmp_path: Path) -> None:
    store = EvolutionMaterialStore(tmp_path)
    source = _source()
    store.register_source(source, 2)
    with sqlite3.connect(store.path) as database:
        database.execute(
            "CREATE TRIGGER fail_capture AFTER INSERT ON captures "
            "BEGIN SELECT RAISE(ABORT, 'capture interrupted'); END"
        )
    with pytest.raises(IrisEvolutionError, match="capture interrupted"):
        store.commit_capture(_block(source, 2, 4, terminal=4))
    assert store.list_capture_sources("lifecycle")[0].captured_until == 2
    assert store.list_pending_sources() == ()
    assert body_count(store) == 0


def test_failed_request_settlement_keeps_request_pending(tmp_path: Path) -> None:
    store = EvolutionMaterialStore(tmp_path)
    item = RevisionItem(
        description="保留约束",
        targets=(RevisionTarget(kind="prompt", name="compaction"),),
        origin=HostOrigin(),
    )
    store.enqueue_revision(item)
    with sqlite3.connect(store.path) as database:
        database.execute(
            "CREATE TRIGGER fail_settle BEFORE INSERT ON progress "
            "BEGIN SELECT RAISE(ABORT, 'settlement interrupted'); END"
        )
    with pytest.raises(IrisEvolutionError, match="settlement interrupted"):
        store.settle_revision(item.id, EvolutionResult(status="no_change", stage="revision"))
    assert store.revision_result(item.id) is None
    assert store.list_revision_requests().items[0].status is None
    assert store.get_revision_request(item.id) == item


def test_failed_step_retains_pending_and_corrupt_progress_fails_at_load(tmp_path: Path) -> None:
    store = EvolutionMaterialStore(tmp_path)
    source = _source()
    store.register_source(source, 2)
    store.commit_capture(_block(source, 2, 3, terminal=3))
    store.record_step(EvolutionResult(status="failed", reason="provider failed"))
    assert store.list_pending_sources() == (source,)
    with sqlite3.connect(store.path) as database:
        result = EvolutionResult.model_validate_json(
            database.execute("SELECT latest_step FROM progress").fetchone()[0]
        )
        assert result.status == "failed"
        database.execute("UPDATE sources SET consumed_until='invalid'")
    with pytest.raises(IrisEvolutionError):
        EvolutionMaterialStore(tmp_path).register_source(source, 2)
