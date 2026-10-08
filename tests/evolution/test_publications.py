"""发布档案保存原正文，并从真实确认事实续接未完成结算。"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from iris.evolution.history import PublicationRecord
from iris.evolution.materials import EvolutionMaterialStore
from iris.evolution.models import RevisionRequest, RevisionTarget
from iris.evolution.revision import PromptTarget
from iris.evolution.service import EvolutionService
from iris.exceptions import IrisEvolutionError

from .test_materials import body_count
from .test_project_skill import prepare

if TYPE_CHECKING:
    from iris.evolution.revision import PreparedRevision


def reopen(service: EvolutionService) -> EvolutionService:
    """模拟新进程，不继承发布 owner 的内存收据。"""
    return EvolutionService(
        workspace_root=service.workspace_root,
        skill_path=service.skill_path,
        store=EvolutionMaterialStore(service.workspace_root),
        provider=service.provider,
        model=service.model,
        config=service.config,
        prompt_source=service.prompt_source,
        prompt_targets=service.prompt_targets,
        config_target=service.config_target,
    )


@pytest.mark.asyncio
async def test_experience_archive_survives_consumed_body_cleanup(tmp_path: Path) -> None:
    service, provider, scope = prepare(tmp_path)
    result = await service.maintain_cycle(scope=scope)
    assert result.status == "updated"
    record = (await service.aget_publication(result.publication_id)).detail
    assert record.publication_state == "confirmed" and record.settled
    assert record.before_documents[0].text is None
    assert record.after_documents[0].text == service.skill_path.read_text(encoding="utf-8")
    assert (
        record.published_at is not None and record.materials[0].records[0].text == "本项目使用 uv"
    )
    assert body_count(service.store) == 0
    assert reopen(service).get_publication(record.publication_id).detail == record
    assert len(provider.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["updated", "no_change", "failed", "conflict"])
async def test_revision_archives_request_and_distinguishes_outcomes(
    tmp_path: Path, mode: str
) -> None:
    service, provider, scope = prepare(tmp_path, prompt_targets=["compaction"])
    service.prompt_targets = (PromptTarget("compaction", "摘要", {}),)
    path = service.prompt_source.root / "compaction.j2"
    before = path.read_text(encoding="utf-8")
    request = await service.enqueue_revision(
        RevisionRequest(
            description="保留明确工具约定",
            targets=(RevisionTarget(kind="prompt", name="compaction"),),
        )
    )
    provider.output = (
        {"action": "no_change", "reason": "维持"}
        if mode == "no_change"
        else {
            "action": "prompt",
            "target": "compaction",
            "body": "{{ missing }}" if mode == "failed" else "新策略",
            "reason": "调整",
        }
    )
    if mode == "conflict":
        provider.on_complete = lambda: path.write_text("人工修订", encoding="utf-8")
    result = await service.maintain_cycle(scope=scope)
    assert result.status == mode
    record = service.get_publication(result.publication_id).detail
    assert record.revision_id == request.id and record.description == request.description
    assert record.before_documents[0].text == before
    assert record.outcome.status == mode
    if mode == "updated":
        assert (
            record.publication_state == "confirmed" and record.after_documents[0].text == "新策略"
        )
        assert record.published_at is not None
    else:
        assert record.publication_state == "not_published"
        assert record.after_documents == () and record.published_at is None
    assert (await service.alist_revision_requests()).items[0].id == request.id
    assert await service.aget_revision_request(request.id) == request
    assert reopen(service).get_publication(record.publication_id).detail == record


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [False, True])
async def test_confirmation_gap_uses_receipt_only_in_original_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restart: bool
) -> None:
    service, provider, scope = prepare(tmp_path, prompt_targets=["compaction"])
    service.prompt_targets = (PromptTarget("compaction", "摘要", {}),)
    request = await service.enqueue_revision(
        RevisionRequest(
            description="精简摘要", targets=(RevisionTarget(kind="prompt", name="compaction"),)
        )
    )
    provider.output = {
        "action": "prompt",
        "target": "compaction",
        "body": "新策略",
        "reason": "精简",
    }
    save = service.store.save_publication
    failed_once = False

    def fail_confirmation(record: PublicationRecord) -> None:
        nonlocal failed_once
        if record.publication_state == "confirmed" and not failed_once:
            failed_once = True
            raise IrisEvolutionError("确认写入失败")
        save(record)

    monkeypatch.setattr(service.store, "save_publication", fail_confirmation)
    first = await service.maintain_cycle(scope=scope)
    assert first.status == "failed"
    record = service.get_publication(first.publication_id).detail
    assert record.publication_state == "unconfirmed" and record.after_documents == ()
    assert (service.prompt_source.root / "compaction.j2").read_text(encoding="utf-8") == "新策略"
    resumed = reopen(service) if restart else service
    second = await resumed.maintain_cycle(scope=scope)
    record = resumed.get_publication(first.publication_id).detail
    assert len(provider.requests) == 1
    if restart:
        assert second.status == "failed" and "publication_unconfirmed" in second.reason
        assert record.publication_state == "unconfirmed" and record.published_at is None
        assert record.after_documents == () and record.observed_documents[0].text == "新策略"
        assert resumed.store.revision_result(request.id) is None
    else:
        assert (
            second.status == "updated"
            and record.publication_state == "confirmed"
            and record.settled
        )
        assert resumed.store.revision_result(request.id).publication_id == record.publication_id


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["progress", "archive"])
async def test_confirmed_experience_resumes_settlement_without_generation(
    tmp_path: Path, failure: str
) -> None:
    service, provider, scope = prepare(tmp_path)
    event = (
        "BEFORE INSERT ON progress"
        if failure == "progress"
        else "BEFORE UPDATE ON publications WHEN NEW.settled=1"
    )
    with sqlite3.connect(service.store.path) as database:
        database.execute(
            f"CREATE TRIGGER fail_settlement {event} "
            "BEGIN SELECT RAISE(ABORT, 'settlement interrupted'); END"
        )
    with pytest.raises(IrisEvolutionError, match="settlement interrupted"):
        await service.maintain_cycle(scope=scope)
    record = service.get_publication(service.list_publications().items[0].publication_id).detail
    assert record.publication_state == "confirmed" and not record.settled
    assert record.consumed_ranges == () and record.outcome.consumed_ranges == ()
    assert body_count(service.store) > 0
    with sqlite3.connect(service.store.path) as database:
        database.execute("DROP TRIGGER fail_settlement")
    resumed = reopen(service)
    result = await resumed.maintain_cycle(scope=scope)
    assert result.status == "updated" and result.publication_id == record.publication_id
    assert len(provider.requests) == 1
    assert resumed.get_publication(record.publication_id).summary.settled
    assert (
        resumed.get_publication(record.publication_id).detail.consumed_ranges
        == result.consumed_ranges
    )
    assert not resumed.store.list_pending_sources()


@pytest.mark.asyncio
async def test_candidate_archive_precedes_original_file_publisher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import iris.evolution.service as service_module

    service, provider, scope = prepare(tmp_path, prompt_targets=["compaction"])
    service.prompt_targets = (PromptTarget("compaction", "摘要", {}),)
    await service.enqueue_revision(
        RevisionRequest(
            description="精简", targets=(RevisionTarget(kind="prompt", name="compaction"),)
        )
    )
    provider.output = {
        "action": "prompt",
        "target": "compaction",
        "body": "新策略",
        "reason": "精简",
    }
    publish = service_module.publish_revision

    def checked(candidate: PreparedRevision) -> bool:
        record = service.get_publication(service.list_publications().items[0].publication_id).detail
        assert record.publication_state == "unconfirmed" and record.outcome is None
        assert record.candidate_documents[0].text == "新策略"
        assert record.after_documents == () and record.published_at is None
        assert candidate.path.read_text(encoding="utf-8") == record.before_documents[0].text
        return publish(candidate)

    monkeypatch.setattr(service_module, "publish_revision", checked)
    assert (await service.maintain_cycle(scope=scope)).status == "updated"


def test_publication_and_request_pages_use_stable_order(tmp_path: Path) -> None:
    from iris.evolution.models import HostOrigin, RevisionItem

    store = EvolutionMaterialStore(tmp_path)
    timestamp = datetime(2026, 10, 7, tzinfo=UTC)
    for identity in ("c", "a", "b"):
        store.save_publication(
            PublicationRecord(
                publication_id=identity,
                created_at=timestamp,
                stage="experience",
                origin="experience",
                description=identity,
            )
        )
        store.enqueue_revision(
            RevisionItem(
                id=identity,
                created_at=timestamp,
                description=identity,
                origin=HostOrigin(),
                targets=(RevisionTarget(kind="prompt", name="compaction"),),
            )
        )
    first = store.list_publications(limit=2)
    # 状态更新仍处于原创建位置，不影响后续游标页。
    store.save_publication(
        store.get_publication("a").detail.model_copy(update={"reason": "已更新"})
    )
    rest = store.list_publications(after=first.next_cursor, limit=2)
    assert [record.publication_id for record in (*first.items, *rest.items)] == ["a", "b", "c"]
    assert rest.next_cursor is None
    requests = store.list_revision_requests(limit=2)
    last = store.list_revision_requests(after=requests.next_cursor)
    assert [item.id for item in (*requests.items, *last.items)] == ["a", "b", "c"]
    with pytest.raises(IrisEvolutionError):
        store.list_publications(limit=0)


@pytest.mark.asyncio
async def test_confirmed_settlement_retry_does_not_duplicate_issue(tmp_path: Path) -> None:
    """结算已提交而调用方重试时，不重复消费或创建 A 提出的修订请求。"""
    from .test_strategy_revision import issue_output

    service, provider, scope = prepare(tmp_path, prompt_targets=["compaction"])
    provider.output = issue_output()
    first = await service.maintain_cycle(scope=scope)
    record = service.get_publication(first.publication_id).detail
    assert service.store.settle_publication(record) == first
    assert len(service.list_revision_requests().items) == 1
    assert body_count(service.store) == 0
    assert len(provider.requests) == 1
