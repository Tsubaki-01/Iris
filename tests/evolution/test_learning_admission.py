"""原文就绪短状态、持久准入和 A/B 范围分离。"""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from iris.evolution.materials import EvolutionMaterialStore
from iris.evolution.models import EvolutionSource, RevisionRequest, RevisionTarget
from iris.evolution.revision import PromptTarget
from iris.exceptions import IrisEvolutionError

from .test_materials import _block, _result, _source
from .test_project_skill import prepare
from .test_publications import reopen
from .test_query_boundaries import _deny_large_columns


def _capture(
    store: EvolutionMaterialStore, identity: str, *, empty: bool = False
) -> EvolutionSource:
    """登记一个两条消息的终态来源。"""
    source = _source(identity)
    store.register_source(source, 2)
    store.commit_capture(_block(source, 2, 4, terminal=4, empty=(2, 3) if empty else ()))
    return source


def test_admission_counts_only_complete_eligible_new_runs(tmp_path: Path) -> None:
    """九个有效来源不准入；补齐第十个后全部准入，不计空或不合格来源。"""
    store = EvolutionMaterialStore(tmp_path)
    sources = [_capture(store, f"run-{index}") for index in range(9)]
    empty = _capture(store, "empty", empty=True)
    excluded = _capture(store, "excluded")
    incomplete = _source("incomplete")
    store.register_source(incomplete, 2)
    store.commit_capture(_block(incomplete, 2, 3))
    allowed = frozenset(
        (item.lifecycle_source_id, item.run_id) for item in (*sources, empty, incomplete)
    )
    first = store.admit_learning_sources(allowed_sources=allowed, threshold=10)
    assert not any(item.admitted for item in first.sources)
    assert next(item for item in first.sources if item.source == incomplete).complete is False
    assert next(item for item in first.sources if item.source == empty).has_content is False

    store.commit_capture(_block(incomplete, 3, 4, terminal=4))
    extra = _capture(store, "extra")
    allowed = allowed | {(extra.lifecycle_source_id, extra.run_id)}
    admitted = store.admit_learning_sources(allowed_sources=allowed, threshold=10)
    expected = set((*sources, incomplete, extra))
    assert {item.source for item in admitted.sources if item.admitted} == expected
    assert next(item for item in admitted.sources if item.source == excluded).admitted is False
    assert store.admit_learning_sources(allowed_sources=allowed, threshold=10) == admitted


def test_readiness_and_admission_do_not_load_raw_or_request_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """短查询与准入均可在消息正文不可读时完成。"""
    store = EvolutionMaterialStore(tmp_path)
    sources = [_capture(store, f"run-{index}") for index in range(10)]
    _deny_large_columns(monkeypatch)
    ready = store.read_learning_readiness()
    assert len(ready.sources) == 10
    assert all(item.complete and item.has_content and not item.admitted for item in ready.sources)
    result = store.admit_learning_sources(
        allowed_sources=frozenset((item.lifecycle_source_id, item.run_id) for item in sources),
        threshold=10,
    )
    assert all(item.admitted for item in result.sources)


def test_admitted_remainder_survives_reopen_and_late_capture(tmp_path: Path) -> None:
    """部分消费后标记仍在，空尾不再凑数，新 Run 也不借旧批准入。"""
    store = EvolutionMaterialStore(tmp_path)
    source = _source("run-0")
    store.register_source(source, 2)
    block = _block(source, 2, 5, terminal=5, empty=(4,))
    store.commit_capture(block)
    sources = [source, *[_capture(store, f"run-{index}") for index in range(1, 10)]]
    allowed = frozenset((item.lifecycle_source_id, item.run_id) for item in sources)
    store.admit_learning_sources(allowed_sources=allowed, threshold=10)
    selected = store.read_pending(
        allowed_sources=frozenset({("lifecycle", "run-0")}), limit=2
    ).items
    store.consume(selected, _result(selected))
    restarted = EvolutionMaterialStore(tmp_path)
    restarted.commit_capture(block)
    state = next(
        item for item in restarted.read_learning_readiness().sources if item.source == source
    )
    assert state.admitted and state.complete and not state.has_content
    fresh = _capture(restarted, "fresh")
    ready = restarted.admit_learning_sources(
        allowed_sources=allowed | {(fresh.lifecycle_source_id, fresh.run_id)}, threshold=10
    )
    assert sum(item.admitted for item in ready.sources) == 10
    assert not next(item for item in ready.sources if item.source == fresh).admitted
    tail = restarted.read_pending(allowed_sources=frozenset({("lifecycle", "run-0")})).items
    restarted.consume(tail, _result(tail))
    assert source not in {item.source for item in restarted.read_learning_readiness().sources}


def test_concurrent_admission_uses_persisted_source_facts(tmp_path: Path) -> None:
    """两个独立连接重复准入同一批，最终事实不被覆盖或重复计数。"""
    store = EvolutionMaterialStore(tmp_path)
    sources = [_capture(store, f"run-{index}") for index in range(10)]
    allowed = frozenset((item.lifecycle_source_id, item.run_id) for item in sources)

    def admit() -> None:
        result = EvolutionMaterialStore(tmp_path).admit_learning_sources(
            allowed_sources=allowed, threshold=10
        )
        assert len(result.sources) == 10 and all(item.admitted for item in result.sources)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(admit) for _ in range(2)]
        for future in futures:
            future.result(timeout=10)
    assert all(item.admitted for item in store.read_learning_readiness().sources)


@pytest.mark.asyncio
@pytest.mark.parametrize("include_valid", [False, True])
async def test_empty_sources_settle_without_provider_budget_or_skill_reads(
    tmp_path: Path, include_valid: bool
) -> None:
    """全空来源独立无模型收尾，不被其他有效输入或损坏 Skill 阻挡。"""
    service, provider, scope = prepare(tmp_path, input_budget_tokens=1)
    empty = _source("empty")
    service.store.register_source(empty, 2)
    block = _block(empty, 2, 4, terminal=4)
    service.store.commit_capture(
        block.model_copy(
            update={
                "records": tuple(item.model_copy(update={"text": " \n"}) for item in block.records)
            }
        )
    )
    service.skill_path.parent.mkdir(parents=True)
    service.skill_path.write_bytes(b"\xff")
    empty_scope = frozenset({("lifecycle", "empty")})
    scope = replace(
        scope,
        allowed_sources=scope.allowed_sources | empty_scope,
        experience_sources=scope.allowed_sources | empty_scope if include_valid else empty_scope,
    )
    result = await service.maintain_cycle(scope=scope)
    assert result.status == "no_change" and result.has_more is include_valid
    assert provider.requests == []
    assert service.store.read_pending(allowed_sources=empty_scope).items == ()
    assert service.store.read_pending(allowed_sources=frozenset({("host", "run")})).items
    entry = service.get_publication(result.publication_id)
    assert entry is not None and entry.detail is not None
    assert entry.detail.before_documents == entry.detail.candidate_documents == ()
    assert entry.detail.publication_state == "not_published"
    assert service.skill_path.read_bytes() == b"\xff"


@pytest.mark.asyncio
async def test_revision_runs_with_empty_experience_scope(tmp_path: Path) -> None:
    """A 的准入范围为空不妨碍 B，未准入原文也不进入 has_more。"""
    service, provider, scope = prepare(tmp_path, prompt_targets=["compaction"])
    service.prompt_targets = (PromptTarget("compaction", "摘要", {}),)
    request = await service.enqueue_revision(
        RevisionRequest(
            description="保留目标", targets=(RevisionTarget(kind="prompt", name="compaction"),)
        )
    )
    scope = replace(scope, experience_sources=frozenset())
    provider.output = {"action": "no_change", "reason": "已满足"}
    result = await service.maintain_cycle(scope=scope)
    assert result.stage == "revision" and result.revision_id == request.id
    assert not result.has_more and len(provider.requests) == 1
    assert (await service.maintain_cycle(scope=scope)).status == "empty"


@pytest.mark.asyncio
async def test_confirmed_recovery_ignores_new_experience_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """已发布未结算的旧工作只补结算，不要求新的 A 准入来源。"""
    service, provider, scope = prepare(tmp_path)

    def fail_settlement(record: object) -> None:
        raise IrisEvolutionError("暂时无法结算")

    monkeypatch.setattr(service.store, "settle_publication", fail_settlement)
    with pytest.raises(IrisEvolutionError, match="暂时无法结算"):
        await service.maintain_cycle(scope=scope)
    restarted = reopen(service)
    result = await restarted.maintain_cycle(scope=replace(scope, experience_sources=frozenset()))
    assert result.status == "updated" and not result.has_more
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_failure_and_consume_rollback_preserve_admission_and_remaining_content(
    tmp_path: Path,
) -> None:
    """模型失败和消费事务回滚均不撤销既有准入或剩余有效正文。"""
    service, provider, scope = prepare(tmp_path)
    sources = [_capture(service.store, f"extra-{index}") for index in range(9)]
    allowed = scope.allowed_sources | frozenset(
        (item.lifecycle_source_id, item.run_id) for item in sources
    )
    ready = await service.aadmit_learning_sources(allowed_sources=allowed, threshold=10)
    assert all(item.admitted for item in ready.sources)
    provider.output = {"body": "", "reason": "无效候选"}
    with pytest.raises(IrisEvolutionError):
        await service.maintain_cycle(scope=scope)
    selected = service.store.read_pending(allowed_sources=scope.experience_sources).items
    with sqlite3.connect(service.store.path) as database:
        database.execute(
            "CREATE TRIGGER fail_consume BEFORE INSERT ON progress "
            "BEGIN SELECT RAISE(ABORT, 'consume interrupted'); END"
        )
    with pytest.raises(IrisEvolutionError, match="consume interrupted"):
        service.store.consume(selected, _result(selected))
    ready = await service.aread_learning_readiness()
    assert len(ready.sources) == 10
    assert all(item.admitted and item.has_content for item in ready.sources)
