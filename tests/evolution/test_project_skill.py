"""A 阶段有界整理、固定协议、发布与消费边界。"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import iris.evolution.service as evolution_service
from iris.evolution.config import EvolutionConfig
from iris.evolution.materials import EvolutionMaterialStore
from iris.evolution.models import (
    EvolutionCaptureBlock,
    EvolutionMaintenanceScope,
    EvolutionRecord,
    EvolutionSession,
    EvolutionSource,
)
from iris.evolution.service import EvolutionService
from iris.exceptions import IrisEvolutionError
from iris.message import LLMRequest, LLMResponse, TextBlock
from iris.observability.service import Observability
from iris.prompts import PromptSource
from iris.skill.discovery import discover_skills
from iris.skill.frontmatter import parse_frontmatter, split_frontmatter
from iris.skill.models import SkillDiscoveryOptions, SkillScope
from iris.skill.registry import SkillRegistry
from iris.skill.tool import LoadSkillInput, LoadSkillTool
from iris.tools import ToolExecutionContext

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from iris.evolution.history import PublicationRecord
    from iris.evolution.models import EvolutionResult


class Provider:
    """只返回受控输出，保留真实构造请求。"""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []
        self.output = {"body": "# 项目经验\n\n使用 uv 管理依赖。", "reason": "保留项目约定"}
        self.tokens = 20
        self.on_complete: Callable[[], None] = lambda: None

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        return self.tokens

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        self.on_complete()
        return LLMResponse(
            provider="test",
            content=[TextBlock(text=json.dumps(self.output))],
            finish_reason="stop",
            input_tokens=20,
            output_tokens=5,
            total_tokens=25,
        )


async def eligible(sources: tuple[EvolutionSource, ...]) -> bool:
    return True


async def eligible_session(session: EvolutionSession | None) -> bool:
    return True


def prepare(
    tmp_path: Path, *, observability: Observability | None = None, **config: object
) -> tuple[EvolutionService, Provider, EvolutionMaintenanceScope]:
    provider = Provider()
    store = EvolutionMaterialStore(tmp_path)
    source = EvolutionSource(lifecycle_source_id="host", run_id="run", session_id="session")
    store.register_source(source, initial_message_count=0)
    store.commit_capture(
        EvolutionCaptureBlock(
            source=source,
            initial_message_count=0,
            start_message_count=0,
            end_message_count=1,
            terminal_message_count=1,
            outcome="completed",
            records=(
                EvolutionRecord(
                    ref="host:run:0:0",
                    message_ordinal=0,
                    block_index=0,
                    role="user",
                    text="本项目使用 uv",
                    occurred_at="2026-10-05",
                ),
            ),
        )
    )
    service = EvolutionService(
        workspace_root=tmp_path,
        skill_path=tmp_path / ".agents/skills/project-experience/SKILL.md",
        store=store,
        provider=provider,
        model="test",
        config=EvolutionConfig(**config),
        prompt_source=PromptSource.initialize(tmp_path),
        observability=observability,
    )
    return (
        service,
        provider,
        EvolutionMaintenanceScope(
            frozenset({("host", "run")}), eligible, frozenset(), eligible_session
        ),
    )


def append_source(service: EvolutionService, run_id: str) -> EvolutionMaintenanceScope:
    """提交另一个完整来源，供下一轮读取。"""
    source = EvolutionSource(lifecycle_source_id="host", run_id=run_id, session_id=run_id)
    service.store.register_source(source, initial_message_count=0)
    service.store.commit_capture(
        EvolutionCaptureBlock(
            source=source,
            initial_message_count=0,
            start_message_count=0,
            end_message_count=1,
            terminal_message_count=1,
            outcome="completed",
            records=(
                EvolutionRecord(
                    ref=f"host:{run_id}:0:0",
                    message_ordinal=0,
                    block_index=0,
                    role="user",
                    text="继续使用 uv",
                    occurred_at="2026-10-05",
                ),
            ),
        )
    )
    return EvolutionMaintenanceScope(
        frozenset({("host", "run"), ("host", run_id)}), eligible, frozenset(), eligible_session
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("in_cycle", [False, True])
async def test_update_publishes_stable_skill_and_consumes_only_once(
    tmp_path: Path,
    in_cycle: bool,
    observability: tuple[Observability, InMemorySpanExporter],
) -> None:
    observation, exporter = observability
    service, provider, scope = prepare(tmp_path, observability=observation)
    with (
        observation.bind({"iris.maintenance.kind": "evolution"} if in_cycle else {}),
        observation.scope("cycle" if in_cycle else "host"),
    ):
        result = await service.maintain_cycle(scope=scope)
        empty = await service.maintain_cycle(scope=scope)
    assert result.status == "updated" and not result.has_more
    frontmatter, body = split_frontmatter(service.skill_path.read_text(encoding="utf-8"))
    assert parse_frontmatter(frontmatter)["name"] == "project-experience"
    assert body.strip() == provider.output["body"]
    assert result.usage["total_tokens"] == 25
    assert empty.status == "empty"
    assert len(provider.requests) == 1
    assert not provider.requests[0].tools
    prompt = provider.requests[0].messages[0].text
    assert "body" in prompt and "reason" in prompt and "项目" in prompt
    spans = exporter.get_finished_spans()
    [model] = [span for span in spans if span.name.startswith("chat ")]
    assert model.attributes["iris.model.purpose"] == "evolution_experience"
    events = [event for span in spans for event in span.events]
    assert [event.attributes["iris.maintenance.status"] for event in events] == (
        ["updated", "empty"] if in_cycle else []
    )
    assert all(event.name == "iris.maintenance.result" for event in events)


@pytest.mark.asyncio
async def test_no_change_confirms_material_without_creating_skill(tmp_path: Path) -> None:
    service, provider, scope = prepare(tmp_path)
    provider.output = {"body": None, "reason": "没有可复用的新经验"}
    result = await service.maintain_cycle(scope=scope)
    assert result.status == "no_change" and not service.skill_path.exists()
    assert service.store.list_pending_sources() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    [
        {"body": "bad"},
        {"body": "x" * 8001, "reason": "too large"},
        {"body": "line\n" * 1001, "reason": "too many lines"},
    ],
)
async def test_invalid_output_preserves_file_and_pending(
    tmp_path: Path, output: dict[str, object]
) -> None:
    service, provider, scope = prepare(tmp_path)
    service.skill_path.parent.mkdir(parents=True)
    service.skill_path.write_text("old content", encoding="utf-8")
    provider.output = output
    with pytest.raises(IrisEvolutionError):
        await service.maintain_cycle(scope=scope)
    assert service.skill_path.read_text(encoding="utf-8") == "old content"
    assert len(service.store.list_pending_sources()) == 1


@pytest.mark.asyncio
async def test_request_budget_rejects_without_consuming_or_calling_model(tmp_path: Path) -> None:
    service, provider, scope = prepare(tmp_path, input_budget_tokens=10)
    with pytest.raises(IrisEvolutionError, match="预算"):
        await service.maintain_cycle(scope=scope)
    assert provider.requests == [] and service.store.list_pending_sources()


@pytest.mark.asyncio
async def test_manual_edit_and_expired_eligibility_cannot_be_overwritten(tmp_path: Path) -> None:
    service, provider, scope = prepare(tmp_path)

    def edit() -> None:
        service.skill_path.parent.mkdir(parents=True)
        service.skill_path.write_text("用户新编辑", encoding="utf-8")

    provider.on_complete = edit
    result = await service.maintain_cycle(scope=scope)
    assert result.status == "conflict"
    assert service.skill_path.read_text(encoding="utf-8") == "用户新编辑"
    assert service.store.list_pending_sources()
    checks = 0

    async def expire(sources: tuple[EvolutionSource, ...]) -> bool:
        nonlocal checks
        checks += 1
        return checks == 1

    provider.on_complete = lambda: None
    with pytest.raises(asyncio.CancelledError):
        await service.maintain_cycle(
            scope=EvolutionMaintenanceScope(
                scope.allowed_sources, expire, frozenset(), eligible_session
            )
        )
    assert service.store.list_pending_sources()


@pytest.mark.asyncio
async def test_editable_strategy_cannot_remove_response_schema(tmp_path: Path) -> None:
    policy = tmp_path / "policy.md"
    policy.write_text("# policy\n原始经验必须注明适用范围。", encoding="utf-8")
    service, provider, scope = prepare(tmp_path, policy_skill="policy.md")
    (service.prompt_source.root / "project_skill_update.j2").write_text(
        "只输出 fake", encoding="utf-8"
    )
    await service.maintain_cycle(scope=scope)
    prompt = provider.requests[0].messages[0].text
    assert "原始经验必须注明适用范围" in prompt
    assert prompt.startswith("只输出 fake")
    schema = json.loads(prompt.rsplit("\n", 1)[1])
    assert set(schema["properties"]) == {"body", "reason", "issue"}


@pytest.mark.asyncio
async def test_budget_selects_complete_messages_and_leaves_remainder_pending(
    tmp_path: Path,
) -> None:
    service, provider, _ = prepare(tmp_path, input_budget_tokens=20)
    scope = append_source(service, "second")
    provider.estimate_input_tokens = lambda request: (
        len(json.loads(request.messages[1].text)["materials"]) * 20
    )
    first = await service.maintain_cycle(scope=scope)
    assert first.has_more and len(first.consumed_ranges) == 1
    assert len(service.store.list_pending_sources()) == 1
    second = await service.maintain_cycle(scope=scope)
    assert not second.has_more and len(second.consumed_ranges) == 1
    assert service.store.list_pending_sources() == ()
    assert all(
        len(json.loads(request.messages[1].text)["materials"]) == 1 for request in provider.requests
    )


@pytest.mark.asyncio
async def test_next_cycle_adopts_new_template_and_policy(tmp_path: Path) -> None:
    policy = tmp_path / "policy.md"
    policy.write_text("old policy", encoding="utf-8")
    service, provider, scope = prepare(tmp_path, policy_skill="policy.md")
    template = service.prompt_source.root / "project_skill_update.j2"
    template.write_text("old strategy", encoding="utf-8")

    def revise() -> None:
        template.write_text("new strategy", encoding="utf-8")
        policy.write_text("new policy", encoding="utf-8")

    provider.on_complete = revise
    await service.maintain_cycle(scope=scope)
    scope = append_source(service, "second")
    await service.maintain_cycle(scope=scope)
    assert "old strategy" in provider.requests[0].messages[0].text
    assert "old policy" in provider.requests[0].messages[0].text
    assert "new strategy" in provider.requests[1].messages[0].text
    assert "new policy" in provider.requests[1].messages[0].text


@pytest.mark.asyncio
async def test_cancelled_provider_late_response_cannot_publish(
    tmp_path: Path, observability: tuple[Observability, InMemorySpanExporter]
) -> None:
    from opentelemetry.trace import StatusCode

    observation, exporter = observability
    service, provider, scope = prepare(tmp_path, observability=observation)
    started = asyncio.Event()
    original = provider.complete

    async def suppress(request: LLMRequest) -> LLMResponse:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return await original(request)
        raise AssertionError("必须取消模型等待")

    provider.complete = suppress
    with observation.bind({"iris.maintenance.kind": "evolution"}), observation.scope("cycle"):
        task = asyncio.create_task(service.maintain_cycle(scope=scope))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not service.skill_path.exists() and service.store.list_pending_sources()
    model, cycle = exporter.get_finished_spans()
    assert model.attributes["iris.model.outcome"] == "completed"
    assert cycle.status.status_code is StatusCode.UNSET
    [event] = cycle.events
    assert event.attributes["iris.maintenance.status"] == "cancelled"
    assert event.attributes["iris.maintenance.stage"] == "experience"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["updated", "no_change"])
async def test_experience_cancelled_after_commit_keeps_committed_result_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    observability: tuple[Observability, InMemorySpanExporter],
) -> None:
    """A 提交后仍向外传播取消；事件保留已确认结果，不追加 cancelled。"""
    from opentelemetry.trace import StatusCode

    observation, exporter = observability
    service, provider, scope = prepare(tmp_path, observability=observation)
    if status == "no_change":
        provider.output = {"body": None, "reason": "没有新增项目经验"}
    committed: list[EvolutionResult] = []
    entered = threading.Event()
    release = threading.Event()
    original = service._commit

    def commit(
        baseline: str | None,
        content: str | None,
        result: EvolutionResult,
        publication: PublicationRecord,
    ) -> EvolutionResult:
        stored = original(baseline, content, result, publication)
        committed.append(stored)
        entered.set()
        release.wait()
        return stored

    monkeypatch.setattr(service, "_commit", commit)
    with observation.bind({"iris.maintenance.kind": "evolution"}), observation.scope("cycle"):
        task = asyncio.create_task(service.maintain_cycle(scope=scope))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await service.wait_pending_io()
    assert committed[0].status == status
    assert service.store.list_pending_sources() == ()
    assert service.skill_path.exists() is (status == "updated")
    model, cycle = exporter.get_finished_spans()
    assert model.attributes["iris.model.outcome"] == "completed"
    assert model.status.status_code is cycle.status.status_code is StatusCode.UNSET
    [event] = cycle.events
    assert event.attributes["iris.maintenance.stage"] == "experience"
    assert event.attributes["iris.maintenance.status"] == status


@pytest.mark.asyncio
async def test_new_skill_discovery_and_registered_tool_read_current_body(tmp_path: Path) -> None:
    service, provider, scope = prepare(tmp_path)
    options = SkillDiscoveryOptions(
        workspace_root=tmp_path, roots=((SkillScope.PROJECT, service.skill_path.parent.parent),)
    )
    old_registry = SkillRegistry(discover_skills(options))
    await service.maintain_cycle(scope=scope)
    assert old_registry.names() == ()
    registry = SkillRegistry(discover_skills(options))
    tool = LoadSkillTool(registry)
    context = ToolExecutionContext(workspace_root=tmp_path)
    first = await tool.arun(LoadSkillInput(name="project-experience"), context)
    assert not first.is_error and "使用 uv" in first.model_content
    scope = append_source(service, "second")
    provider.output = {"body": "# 新经验\n\n依赖变更后运行针对性测试。", "reason": "新约定"}
    await service.maintain_cycle(scope=scope)
    second = await tool.arun(LoadSkillInput(name="project-experience"), context)
    assert "针对性测试" in second.model_content and "使用 uv" not in second.model_content


@pytest.mark.asyncio
async def test_publication_failure_keeps_previous_skill_and_material(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observability: tuple[Observability, InMemorySpanExporter],
) -> None:
    from opentelemetry.trace import StatusCode

    observation, exporter = observability
    service, provider, scope = prepare(tmp_path, observability=observation)
    service.skill_path.parent.mkdir(parents=True)
    service.skill_path.write_text("原有经验", encoding="utf-8")
    error = OSError("文件写入失败")

    def fail(path: Path, content: str) -> None:
        raise error

    monkeypatch.setattr(evolution_service, "atomic_write_text", fail)
    with (
        pytest.raises(IrisEvolutionError) as captured,
        observation.bind({"iris.maintenance.kind": "evolution"}),
        observation.scope("cycle"),
    ):
        await service.maintain_cycle(scope=scope)
    assert captured.value.__cause__ is error
    assert service.skill_path.read_text(encoding="utf-8") == "原有经验"
    assert service.store.list_pending_sources()
    assert len(provider.requests) == 1
    model, cycle = exporter.get_finished_spans()
    assert model.attributes["iris.model.purpose"] == "evolution_experience"
    assert model.attributes["iris.model.outcome"] == "completed"
    assert model.status.status_code is StatusCode.UNSET
    assert cycle.status.status_code is StatusCode.ERROR
    [event] = cycle.events
    assert event.attributes["iris.maintenance.status"] == "failed"
    assert event.attributes["iris.maintenance.stage"] == "experience"
