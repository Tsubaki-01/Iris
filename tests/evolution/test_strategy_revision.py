"""项目经验 A 与有限策略 B 分轮结算，失败不重复学习已消费材料。"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from iris.agents import parse_agent_config
from iris.evolution.config import EvolutionConfig
from iris.evolution.materials import EvolutionMaterialStore
from iris.evolution.models import (
    EvolutionMaintenanceScope,
    EvolutionResult,
    EvolutionSession,
    RevisionItem,
    RevisionRequest,
    RevisionTarget,
)
from iris.evolution.revision import ConfigTarget, PreparedRevision, PromptTarget
from iris.evolution.service import EvolutionService
from iris.exceptions import IrisEvolutionError
from iris.observability.service import Observability

from .test_materials import body_count
from .test_project_skill import eligible_session, prepare

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from iris.evolution.history import PublicationRecord


@pytest.mark.asyncio
async def test_invalid_issue_reference_keeps_a_file_and_materials(tmp_path: Path) -> None:
    service, provider, scope = prepare(tmp_path, prompt_targets=["compaction"])
    provider.output = {
        "body": "# 经验\n使用 uv。",
        "reason": "提炼",
        "issue": {
            "description": "压缩遗漏状态",
            "targets": [{"kind": "prompt", "name": "compaction"}],
            "evidence": [{"ref": "missing", "quote": "不存在"}],
        },
    }
    with pytest.raises(IrisEvolutionError):
        await service.maintain_cycle(scope=scope)
    assert not service.skill_path.exists()
    assert service.store.list_pending_sources()


@pytest.mark.asyncio
async def test_plain_fact_without_issue_does_not_trigger_revision(tmp_path: Path) -> None:
    service, provider, scope = prepare(tmp_path)
    result = await service.maintain_cycle(scope=scope)
    assert result.stage == "experience" and not result.has_more
    assert (await service.maintain_cycle(scope=scope)).status == "empty"
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_a_contract_is_available_from_the_actual_request(tmp_path: Path) -> None:
    from iris.evolution.service import project_skill_prompt_description

    service, provider, scope = prepare(tmp_path)
    await service.maintain_cycle(scope=scope)
    description, variables = project_skill_prompt_description()
    schema = json.loads(provider.requests[0].messages[0].text.rsplit("\n", 1)[1])
    assert json.dumps(schema, ensure_ascii=False) in description
    assert variables == {}


def issue_output() -> dict[str, object]:
    """问题引用 fixture 的真实原文，不提供模型自造来源身份。"""
    return {
        "body": None,
        "reason": "经验不变但摘要需保留工具约定",
        "issue": {
            "description": "摘要应保留明确的项目依赖工具约定",
            "targets": [{"kind": "prompt", "name": "compaction"}],
            "evidence": [{"ref": "host:run:0:0", "quote": "本项目使用 uv"}],
        },
    }


@pytest.mark.asyncio
async def test_a_issue_survives_cleanup_and_failed_b_does_not_repeat_a(tmp_path: Path) -> None:
    service, provider, scope = prepare(tmp_path, prompt_targets=["compaction"])
    service.prompt_targets = (PromptTarget("compaction", "自然语言摘要，无模板变量", {}),)
    provider.output = issue_output()
    a = await service.maintain_cycle(scope=scope)
    assert a.stage == "experience" and a.status == "no_change" and a.has_more
    assert len(provider.requests) == 1
    assert service.store.read_pending(allowed_sources=scope.allowed_sources).items == ()
    assert body_count(service.store) == 0
    (pending,) = service.store.read_pending_revisions(
        allowed_sources=scope.allowed_sources,
        allowed_sessions=frozenset(),
        allowed_targets=frozenset({("prompt", "compaction")}),
    )
    assert pending.evidence[0].quote == "本项目使用 uv"
    provider.output = {
        "action": "prompt",
        "target": "compaction",
        "body": "{{ missing }}",
        "reason": "bad",
    }
    failed = await service.maintain_cycle(scope=scope)
    assert failed.stage == "revision" and failed.status == "failed"
    assert failed.revision_id == pending.id and not failed.has_more
    assert service.store.revision_result(pending.id) is None
    provider.output = {
        "action": "prompt",
        "target": "compaction",
        "body": "保留明确的工具约定。",
        "reason": "具体纠正",
    }
    updated = await service.maintain_cycle(scope=scope)
    assert updated.stage == "revision" and updated.status == "updated"
    assert (service.prompt_source.root / "compaction.j2").read_text(
        encoding="utf-8"
    ) == "保留明确的工具约定。"
    assert len(provider.requests) == 3
    assert "历史" in provider.requests[1].messages[1].text
    assert (await service.maintain_cycle(scope=scope)).status == "empty"


@pytest.mark.asyncio
async def test_unknown_quote_cannot_create_issue_or_consume_a(tmp_path: Path) -> None:
    service, provider, scope = prepare(tmp_path, prompt_targets=["compaction"])
    provider.output = issue_output()
    provider.output["issue"]["evidence"][0]["quote"] = "用户其实使用 pip"
    with pytest.raises(IrisEvolutionError, match="原文"):
        await service.maintain_cycle(scope=scope)
    assert service.store.read_pending(allowed_sources=scope.allowed_sources).items
    assert (
        service.store.read_pending_revisions(
            allowed_sources=scope.allowed_sources,
            allowed_sessions=frozenset(),
            allowed_targets=frozenset({("prompt", "compaction")}),
        )
        == ()
    )


@pytest.mark.asyncio
async def test_host_request_needs_no_historical_failure_and_no_change_settles(
    tmp_path: Path,
) -> None:
    service, provider, _ = prepare(tmp_path, prompt_targets=["compaction"])
    service.prompt_targets = (PromptTarget("compaction", "自然语言摘要", {}),)
    item = await service.enqueue_revision(
        RevisionRequest(
            description="请检查当前摘要规则是否重复",
            targets=(RevisionTarget(kind="prompt", name="compaction"),),
        )
    )

    async def no_sources(sources: tuple[object, ...]) -> bool:
        raise AssertionError("宿主无 Run 请求不走伪造来源资格")

    scope = EvolutionMaintenanceScope(
        frozenset(), no_sources, frozenset(), eligible_session, experience_sources=frozenset()
    )
    provider.output = {"action": "no_change", "reason": "当前规则已满足要求"}
    result = await service.maintain_cycle(scope=scope)
    assert result.status == "no_change" and result.revision_id == item.id
    assert service.store.revision_result(item.id) is not None
    payload = json.loads(provider.requests[0].messages[1].text)
    assert payload["issue"]["evidence"] == []
    assert payload["issue"]["origin"] == {"kind": "host", "session": None}
    assert (await service.maintain_cycle(scope=scope)).status == "empty"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,original,remaining",
    [
        ("config", "compaction.input_budget_tokens", "system"),
        ("prompt", "compaction", "memory_dream"),
    ],
)
async def test_reopened_pending_respects_narrowed_current_targets(
    tmp_path: Path, kind: str, original: str, remaining: str
) -> None:
    service, provider, scope = prepare(tmp_path, **{f"{kind}_targets": [original]})
    item = await service.enqueue_revision(
        RevisionRequest(
            description="检查原开放目标", targets=(RevisionTarget(kind=kind, name=original),)
        )
    )
    path = tmp_path / "agent.yaml"
    path.write_text("name: project\nmodel: openai/test\nsystem: 原规则\n", encoding="utf-8")
    target = path if kind == "config" else service.prompt_source.root / f"{original}.j2"
    baseline = target.read_text(encoding="utf-8")
    reopened = EvolutionService(
        workspace_root=tmp_path,
        skill_path=service.skill_path,
        store=EvolutionMaterialStore(tmp_path),
        provider=provider,
        model=service.model,
        config=EvolutionConfig(enabled=True, **{f"{kind}_targets": [remaining]}),
        prompt_source=service.prompt_source,
        prompt_targets=(PromptTarget(remaining, "自然语言策略", {}),) if kind == "prompt" else (),
        config_target=ConfigTarget(
            path, lambda raw: parse_agent_config(raw, config_path=path), {remaining: "当前目标"}
        )
        if kind == "config"
        else None,
    )
    provider.output = (
        {"action": "config", "assignments": {original: 20000}, "reason": "旧请求"}
        if kind == "config"
        else {"action": "prompt", "target": original, "body": "旧请求候选", "reason": "旧请求"}
    )

    empty_scope = replace(scope, allowed_sources=frozenset(), experience_sources=frozenset())
    result = await reopened.maintain_cycle(scope=empty_scope)

    assert result.status == "empty" and not result.has_more
    assert provider.requests == []
    assert target.read_text(encoding="utf-8") == baseline
    assert reopened.store.revision_result(item.id) is None
    assert reopened.store.read_pending_revisions(
        allowed_sources=scope.allowed_sources,
        allowed_sessions=scope.allowed_sessions,
        allowed_targets=frozenset({(kind, original)}),
    ) == (item,)
    current = await reopened.enqueue_revision(
        RevisionRequest(
            description="当前开放目标", targets=(RevisionTarget(kind=kind, name=remaining),)
        )
    )
    provider.output = {"action": "no_change", "reason": "当前目标无需修改"}
    revised = await reopened.maintain_cycle(scope=empty_scope)
    assert revised.status == "no_change" and revised.revision_id == current.id
    assert not revised.has_more
    provider.output = {"body": None, "reason": "经历无需新增规则"}
    learned = await reopened.maintain_cycle(scope=scope)
    assert learned.stage == "experience" and learned.status == "no_change"
    assert not learned.has_more and len(provider.requests) == 2
    assert reopened.store.revision_result(item.id) is None


@pytest.mark.asyncio
async def test_cancelled_b_keeps_request_pending_with_its_identity(
    tmp_path: Path, observability: tuple[Observability, InMemorySpanExporter]
) -> None:
    from opentelemetry.trace import StatusCode

    observation, exporter = observability
    service, provider, scope = prepare(
        tmp_path, observability=observation, prompt_targets=["compaction"]
    )
    service.prompt_targets = (PromptTarget("compaction", "自然语言摘要", {}),)
    item = await service.enqueue_revision(
        RevisionRequest(
            description="修订摘要", targets=(RevisionTarget(kind="prompt", name="compaction"),)
        )
    )
    started = asyncio.Event()

    async def wait(request: object) -> object:
        started.set()
        await asyncio.Event().wait()

    provider.complete = wait
    with observation.bind({"iris.maintenance.kind": "evolution"}), observation.scope("cycle"):
        task = asyncio.create_task(service.maintain_cycle(scope=scope))
        await started.wait()
        task.cancel()
        result = await task
    assert result.status == "cancelled" and result.revision_id == item.id
    assert service.store.revision_result(item.id) is None
    model, cycle = exporter.get_finished_spans()
    assert model.attributes["iris.model.purpose"] == "evolution_revision"
    assert model.attributes["iris.model.outcome"] == "cancelled"
    assert cycle.status.status_code is StatusCode.UNSET
    [event] = cycle.events
    assert event.attributes["iris.maintenance.status"] == "cancelled"
    assert event.attributes["iris.maintenance.revision_id"] == item.id


@pytest.mark.asyncio
async def test_requested_a_does_not_consume_queued_b_and_waiting_session_is_excluded(
    tmp_path: Path,
) -> None:
    service, provider, scope = prepare(tmp_path, prompt_targets=["compaction"])
    service.prompt_targets = (PromptTarget("compaction", "自然语言摘要", {}),)
    item = await service.enqueue_revision(
        RevisionRequest(
            description="明确修订",
            targets=(RevisionTarget(kind="prompt", name="compaction"),),
            session=EvolutionSession(lifecycle_source_id="host", session_id="waiting"),
        )
    )
    a = await service.maintain_cycle(scope=replace(scope, experience_only=True))
    assert a.stage == "experience" and a.status == "updated"
    assert service.store.revision_result(item.id) is None
    assert (await service.maintain_cycle(scope=scope)).status == "empty"
    assert len(provider.requests) == 1
    provider.output = {"action": "no_change", "reason": "无需修改"}
    b = await service.maintain_cycle(
        scope=replace(scope, allowed_sessions=frozenset({("host", "waiting")}))
    )
    assert b.stage == "revision" and b.revision_id == item.id


@pytest.mark.asyncio
async def test_b_checks_session_again_before_publish_and_keeps_manual_edits(tmp_path: Path) -> None:
    service, provider, scope = prepare(tmp_path, prompt_targets=["compaction"])
    service.prompt_targets = (PromptTarget("compaction", "自然语言摘要", {}),)
    item = await service.enqueue_revision(
        RevisionRequest(
            description="修订", targets=(RevisionTarget(kind="prompt", name="compaction"),)
        )
    )
    target = service.prompt_source.root / "compaction.j2"
    provider.output = {"action": "prompt", "target": "compaction", "body": "候选", "reason": "修改"}
    provider.on_complete = lambda: target.write_text("人工编辑", encoding="utf-8")
    conflict = await service.maintain_cycle(scope=scope)
    assert conflict.status == "conflict" and target.read_text(encoding="utf-8") == "人工编辑"
    calls = 0

    async def expire(session: EvolutionSession | None) -> bool:
        nonlocal calls
        calls += 1
        return calls == 1

    provider.on_complete = lambda: None
    cancelled = await service.maintain_cycle(scope=replace(scope, check_session=expire))
    assert cancelled.status == "cancelled" and cancelled.revision_id == item.id
    assert service.store.revision_result(item.id) is None
    assert target.read_text(encoding="utf-8") == "人工编辑"


@pytest.mark.asyncio
async def test_another_process_settlement_is_observed_without_second_model(tmp_path: Path) -> None:
    from iris.evolution.materials import EvolutionMaterialStore
    from iris.evolution.models import EvolutionResult

    service, provider, scope = prepare(tmp_path, prompt_targets=["compaction"])
    item = await service.enqueue_revision(
        RevisionRequest(
            description="明确检查", targets=(RevisionTarget(kind="prompt", name="compaction"),)
        )
    )
    other = EvolutionMaterialStore(tmp_path)
    other.settle_revision(
        item.id, EvolutionResult(stage="revision", status="no_change", revision_id=item.id)
    )
    result = await service.maintain_cycle(
        scope=replace(scope, allowed_sources=frozenset(), experience_sources=frozenset())
    )
    assert result.status == "empty" and provider.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "no_change", "conflict"])
@pytest.mark.parametrize("in_cycle", [False, True])
async def test_observed_revision_reports_returned_domain_result(
    tmp_path: Path,
    status: str,
    in_cycle: bool,
    observability: tuple[Observability, InMemorySpanExporter],
) -> None:
    from opentelemetry.trace import StatusCode

    observation, exporter = observability
    service, provider, scope = prepare(
        tmp_path, observability=observation, prompt_targets=["compaction"]
    )
    service.prompt_targets = (PromptTarget("compaction", "自然语言摘要", {}),)
    item = await service.enqueue_revision(
        RevisionRequest(
            description="修订摘要", targets=(RevisionTarget(kind="prompt", name="compaction"),)
        )
    )
    provider.output = (
        {"action": "no_change", "reason": "当前内容已足够"}
        if status == "no_change"
        else {
            "action": "prompt",
            "target": "compaction",
            "body": "{{ missing }}" if status == "failed" else "新摘要规则",
            "reason": "修改",
        }
    )
    if status == "conflict":
        target = service.prompt_source.root / "compaction.j2"
        provider.on_complete = lambda: target.write_text("人工修改", encoding="utf-8")
    with (
        observation.bind({"iris.maintenance.kind": "evolution"} if in_cycle else {}),
        observation.scope("cycle" if in_cycle else "host"),
    ):
        result = await service.maintain_cycle(scope=scope)
    assert result.status == status
    model, owner = exporter.get_finished_spans()
    assert model.attributes["iris.model.purpose"] == "evolution_revision"
    assert model.attributes["iris.model.outcome"] == "completed"
    assert model.status.status_code is StatusCode.UNSET
    if in_cycle:
        [event] = owner.events
        assert event.name == "iris.maintenance.result"
        assert dict(event.attributes) == {
            "iris.maintenance.kind": "evolution",
            "iris.maintenance.stage": "revision",
            "iris.maintenance.status": status,
            "iris.maintenance.revision_id": item.id,
        }
    else:
        assert not owner.events
    assert owner.status.status_code is (
        StatusCode.ERROR if in_cycle and status == "failed" else StatusCode.UNSET
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["updated", "no_change"])
async def test_revision_cancelled_after_commit_reports_original_result_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    observability: tuple[Observability, InMemorySpanExporter],
) -> None:
    observation, exporter = observability
    service, provider, scope = prepare(
        tmp_path, observability=observation, prompt_targets=["compaction"]
    )
    service.prompt_targets = (PromptTarget("compaction", "自然语言摘要", {}),)
    item = await service.enqueue_revision(
        RevisionRequest(
            description="检查摘要", targets=(RevisionTarget(kind="prompt", name="compaction"),)
        )
    )
    provider.output = (
        {"action": "no_change", "reason": "当前规则已满足"}
        if status == "no_change"
        else {"action": "prompt", "target": "compaction", "body": "新摘要规则", "reason": "修改"}
    )
    committed: list[EvolutionResult] = []
    entered = threading.Event()
    release = threading.Event()
    original = service._commit_revision

    def commit(
        revision: RevisionItem,
        candidate: PreparedRevision,
        usage: dict[str, int],
        publication: PublicationRecord,
    ) -> EvolutionResult:
        result = original(revision, candidate, usage, publication)
        committed.append(result)
        entered.set()
        release.wait()
        return result

    monkeypatch.setattr(service, "_commit_revision", commit)
    with observation.bind({"iris.maintenance.kind": "evolution"}), observation.scope("cycle"):
        task = asyncio.create_task(service.maintain_cycle(scope=scope))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            release.set()
            result = await task
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await service.wait_pending_io()
    assert result is committed[0]
    assert result.status == status and result.revision_id == item.id
    assert service.store.revision_result(item.id) is not None
    model, cycle = exporter.get_finished_spans()
    assert model.attributes["iris.model.outcome"] == "completed"
    [event] = cycle.events
    assert event.attributes["iris.maintenance.status"] == status
    assert event.attributes["iris.maintenance.revision_id"] == item.id
