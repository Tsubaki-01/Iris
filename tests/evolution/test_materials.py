"""项目经验材料的连续捕获、跨进程重复与持久消费。"""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from pydantic import BaseModel

import iris.evolution.materials as material_module
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
    assert list((store.root / "blocks").glob("*.json")) == []
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


@pytest.mark.parametrize("settled", [False, True])
def test_host_request_cleanup_during_pending_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settled: bool
) -> None:
    """枚举后的正文消失只有在已有完成收据时才可跳过。"""
    store = EvolutionMaterialStore(tmp_path)
    item = RevisionItem(
        description="明确宿主请求",
        targets=(RevisionTarget(kind="prompt", name="compaction"),),
        origin=HostOrigin(),
    )
    store.enqueue_revision(item)
    reached, release = Event(), Event()
    original_read = material_module._read

    def read(path: Path, schema: type[BaseModel]) -> BaseModel:
        if path.parent.name == "requests":
            reached.set()
            assert release.wait(timeout=10)
        return original_read(path, schema)

    monkeypatch.setattr(material_module, "_read", read)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(
            store.read_pending_revisions,
            allowed_sources=frozenset(),
            allowed_sessions=frozenset(),
            allowed_targets=frozenset({("prompt", "compaction")}),
        )
        try:
            assert reached.wait(timeout=10)
            if settled:
                store.settle_revision(
                    item.id,
                    EvolutionResult(status="no_change", stage="revision", revision_id=item.id),
                )
                assert list((store.root / "requests").glob("*.json")) == []
            else:
                next((store.root / "requests").glob("*.json")).unlink()
            release.set()
            if settled:
                assert pending.result(timeout=10) == ()
            else:
                with pytest.raises(IrisEvolutionError, match="读取失败"):
                    pending.result(timeout=10)
        finally:
            release.set()


def test_failed_a_progress_does_not_publish_issue_or_consume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = EvolutionMaterialStore(tmp_path)
    source = _source()
    store.register_source(source, 2)
    store.commit_capture(_block(source, 2, 3, terminal=3))
    allowed = frozenset({("lifecycle", "run")})
    selected = store.read_pending(allowed_sources=allowed).items
    original = material_module.atomic_write_text

    def fail_progress(path: Path, content: str) -> None:
        if path.name == "progress.json":
            raise OSError("not published")
        original(path, content)

    monkeypatch.setattr(material_module, "atomic_write_text", fail_progress)
    with pytest.raises(IrisEvolutionError):
        store.consume(selected, _result(selected), issue=_issue(source))
    assert store.read_pending(allowed_sources=allowed).items == selected
    assert (
        store.read_pending_revisions(
            allowed_sources=allowed,
            allowed_sessions=frozenset(),
            allowed_targets=frozenset({("prompt", "compaction")}),
        )
        == ()
    )


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
    assert list((restarted.root / "blocks").glob("*.json")) == []

    after_cleanup = EvolutionMaterialStore(tmp_path)
    assert after_cleanup.list_pending_sources() == ()
    final = after_cleanup.register_source(source, 2)
    assert final.captured_until == final.consumed_until == final.terminal_message_count == 5
    assert after_cleanup.commit_capture(block).consumed_until == 5
    assert after_cleanup.read_pending(allowed_sources=allowed).items == ()


@pytest.mark.parametrize("failed_directory", ["blocks", "captures"])
def test_failed_block_publication_does_not_advance_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_directory: str
) -> None:
    store = EvolutionMaterialStore(tmp_path)
    source = _source()
    store.register_source(source, 2)
    original_write = material_module.atomic_write_text

    def fail_body(path: Path, content: str) -> None:
        if path.parent.name == failed_directory:
            raise OSError("正文未发布")
        original_write(path, content)

    monkeypatch.setattr(material_module, "atomic_write_text", fail_body)
    with pytest.raises(IrisEvolutionError, match="发布"):
        store.commit_capture(_block(source, 2, 4, terminal=4))
    assert store.list_capture_sources("lifecycle")[0].captured_until == 2
    assert store.list_pending_sources() == ()


def test_failed_progress_publication_keeps_bodies_and_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = EvolutionMaterialStore(tmp_path)
    source = _source()
    store.register_source(source, 2)
    store.commit_capture(_block(source, 2, 4, terminal=4))
    selected = store.read_pending(allowed_sources=frozenset({("lifecycle", "run")})).items
    original_write = material_module.atomic_write_text

    def fail_progress(path: Path, content: str) -> None:
        if path.name == "progress.json":
            raise OSError("进度未发布")
        original_write(path, content)

    monkeypatch.setattr(material_module, "atomic_write_text", fail_progress)
    with pytest.raises(IrisEvolutionError):
        store.consume(selected, _result(selected))
    assert store.register_source(source, 2).consumed_until == 2
    assert store.read_pending(allowed_sources=frozenset({("lifecycle", "run")})).items == selected


def test_capture_progress_remains_readable_during_body_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """捕获线程已经开始读收据时，消费方清理正文也不删除它依赖的进度来源。"""
    store = EvolutionMaterialStore(tmp_path)
    source = _source()
    store.register_source(source, 2)
    store.commit_capture(_block(source, 2, 4, terminal=4))
    selected = store.read_pending(allowed_sources=frozenset({("lifecycle", "run")})).items
    reached = Event()
    release = Event()
    original_read = material_module._read

    def read(path: Path, schema: type[BaseModel]) -> BaseModel:
        if path.parent.name == "captures" and not reached.is_set():
            reached.set()
            assert release.wait(timeout=10)
        return original_read(path, schema)

    monkeypatch.setattr(material_module, "_read", read)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(store.register_source, source, 2)
        try:
            assert reached.wait(timeout=10)
            store.consume(selected, _result(selected))
            assert list((store.root / "blocks").glob("*.json")) == []
            release.set()
            assert pending.result(timeout=10).captured_until == 4
        finally:
            release.set()
    assert store.register_source(source, 2).consumed_until == 4
    assert store.list_capture_sources("lifecycle") == ()


def test_failed_step_retains_pending_and_corrupt_progress_fails_at_load(tmp_path: Path) -> None:
    store = EvolutionMaterialStore(tmp_path)
    source = _source()
    store.register_source(source, 2)
    store.commit_capture(_block(source, 2, 3, terminal=3))
    store.record_step(EvolutionResult(status="failed", reason="provider failed"))
    assert store.list_pending_sources() == (source,)
    progress = store.root / "progress.json"
    payload = json.loads(progress.read_text(encoding="utf-8"))
    assert payload["latest_step"]["status"] == "failed"
    payload["consumed"] = {'["lifecycle","run"]': "invalid"}
    progress.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IrisEvolutionError):
        EvolutionMaterialStore(tmp_path).list_pending_sources()


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
    assert len(list((store.root / "blocks").glob("*.json"))) == 2
    assert store.register_source(source, 2).captured_until == 6
