"""验证无工具 flush/dream 的材料边界、消费与正式知识可见性。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from iris.exceptions import IrisMemoryError, IrisTemplateError
from iris.memory import (
    MemoryEpisode,
    MemoryGenerationConfig,
    MemoryObserveInput,
    MemoryRecord,
    MemorySearchQuery,
    MemoryService,
    MemorySourceType,
    MemoryWriteInput,
    SQLiteMemoryStore,
    _prompts,
)
from iris.memory.generation import _DreamResponse, _flush_input, _FlushResponse
from iris.memory.generation_models import EpisodeCursor, EpisodeProgress, EpisodeSlice
from iris.message import LLMRequest, LLMResponse, TextBlock
from iris.observability.service import Observability
from iris.prompts import PromptSource


class Provider:
    """按真实请求中的标识返回结构化内容，不替生成器执行持久写入。"""

    def __init__(self, respond: Callable[[dict[str, object]], dict[str, object]]) -> None:
        self.respond = respond
        self.requests: list[LLMRequest] = []
        self.finish_reason = "stop"

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        return len(request.messages[1].text)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            provider="controlled",
            content=[
                TextBlock(text=json.dumps(self.respond(json.loads(request.messages[1].text))))
            ],
            finish_reason=self.finish_reason,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )


def flush_one(source: dict[str, object]) -> dict[str, object]:
    """引用本批第一条真实证据。"""
    return {
        "observations": [
            {
                "text": "本项目使用 uv",
                "applicability": "本项目",
                "category": "reference",
                "kind": "fact",
                "reason": "用户明确约定",
                "evidence": [source["records"][0]["ref"]],
            }
        ]
    }


def service(
    tmp_path: Path,
    provider: Provider,
    *,
    observability: Observability | None = None,
    **budgets: int,
) -> MemoryService:
    """构造没有镜像也能显式生成的独立 SDK。"""
    return MemoryService(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        generation_provider=provider,
        generation_model="test-model",
        generation_config=MemoryGenerationConfig(**budgets),
        prompt_source=PromptSource.initialize(tmp_path),
        observability=observability,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("in_cycle", [False, True])
async def test_generation_purposes_and_results_belong_only_to_a_memory_cycle(
    tmp_path: Path,
    observability: tuple[Observability, InMemorySpanExporter],
    in_cycle: bool,
) -> None:
    observation, exporter = observability
    provider = Provider(lambda _: {"observations": []})
    memory = service(tmp_path, provider, observability=observation)
    memory.observe(MemoryObserveInput(text="记录原文"))
    with observation.bind({"iris.maintenance.kind": "memory"} if in_cycle else {}):
        with observation.scope("owner") as owner:
            assert (await memory.flush("project")).status == "completed"
            memory.remember(MemoryWriteInput(text="保留事实", reason="seed"))
            provider.respond = lambda _: {"operations": [], "resolutions": []}
            assert (await memory.dream("project")).status == "completed"
    flush_span, dream_span, owner_span = exporter.get_finished_spans()
    assert flush_span.attributes["iris.model.purpose"] == "memory_flush"
    assert dream_span.attributes["iris.model.purpose"] == "memory_dream"
    assert flush_span.parent == dream_span.parent == owner.get_span_context()
    assert owner_span.status.status_code is StatusCode.UNSET
    assert [
        (event.attributes["iris.maintenance.stage"], event.attributes["iris.maintenance.status"])
        for event in owner_span.events
    ] == ([("flush", "completed"), ("dream", "completed")] if in_cycle else [])
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_budget_blocked_records_domain_result_without_model_error(
    tmp_path: Path,
    observability: tuple[Observability, InMemorySpanExporter],
) -> None:
    observation, exporter = observability
    provider = Provider(lambda _: {"operations": [], "resolutions": []})
    memory = service(tmp_path, provider, observability=observation, dream_input_budget_tokens=5000)
    memory.remember(MemoryWriteInput(text="large fact " * 2000, reason="大段材料"))
    with observation.bind({"iris.maintenance.kind": "memory"}), observation.scope("cycle"):
        result = await memory.dream("project")
    assert result.status == "blocked"
    assert provider.requests == []
    [cycle] = exporter.get_finished_spans()
    assert cycle.status.status_code is StatusCode.UNSET
    [event] = cycle.events
    assert event.attributes["iris.maintenance.stage"] == "dream"
    assert event.attributes["iris.maintenance.status"] == "blocked"


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["flush", "dream", "block"])
async def test_generation_conflict_keeps_domain_status_and_actual_model_calls(
    tmp_path: Path,
    observability: tuple[Observability, InMemorySpanExporter],
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    observation, exporter = observability
    provider = Provider(
        lambda _: (
            {"observations": []} if boundary == "flush" else {"operations": [], "resolutions": []}
        )
    )
    memory = service(tmp_path, provider, observability=observation, dream_input_budget_tokens=5000)
    if boundary == "flush":
        memory.observe(MemoryObserveInput(text="原文"))
    else:
        memory.remember(
            MemoryWriteInput(
                text="large fact " * 2000 if boundary == "block" else "fact", reason="seed"
            )
        )
    monkeypatch.setattr(
        memory.store,
        "block_dream" if boundary == "block" else f"commit_{boundary}",
        lambda *args, **kwargs: False,
    )
    with observation.bind({"iris.maintenance.kind": "memory"}), observation.scope("cycle"):
        result = await (memory.flush("project") if boundary == "flush" else memory.dream("project"))
    assert result.status == "conflict"
    assert len(provider.requests) == int(boundary != "block")
    *models, cycle = exporter.get_finished_spans()
    assert len(models) == len(provider.requests)
    assert cycle.status.status_code is StatusCode.UNSET
    [event] = cycle.events
    assert event.attributes["iris.maintenance.status"] == "conflict"


def test_flush_input_keeps_semantic_context_without_repeated_source_ids() -> None:
    records = (
        MemoryRecord(
            id="source:run:12:0",
            role="user",
            text="本项目使用 uv",
            source_type=MemorySourceType.MESSAGE,
            metadata={"message_ordinal": 12, "block_index": 0},
        ),
        MemoryRecord(
            id="source:run:13:0",
            role="tool",
            text="下载超时",
            source_type=MemorySourceType.TOOL_EVENT,
            metadata={
                "message_ordinal": 13,
                "block_index": 0,
                "record_kind": "tool_result",
                "tool_name": "download",
                "call_id": "call-1",
                "is_error": True,
                "memory_item_ids": ["item-1"],
            },
        ),
    )
    episode = MemoryEpisode(
        id="episode-1",
        source_type=MemorySourceType.TASK,
        source_id="run",
        records=records,
        metadata={
            "lifecycle_source_id": "source",
            "session_id": "session",
            "start_message_count": 12,
            "end_message_count": 14,
            "outcome": None,
        },
    )
    payload = _flush_input(
        "project",
        [
            EpisodeSlice(episode.id, record.id, 0, len(record.text), record.text)
            for record in records
        ],
        (EpisodeProgress(episode, EpisodeCursor(), source_outcome="failed"),),
    )

    assert payload["episodes"] == [
        {
            "ref": "p0",
            "source_type": "task",
            "source_id": "run",
            "metadata": {"session_id": "session"},
            "run_outcome": "failed",
        }
    ]
    first, second = payload["records"]
    assert [first["ref"], second["ref"]] == ["e0", "e1"]
    assert [first["episode_ref"], second["episode_ref"]] == ["p0", "p0"]
    assert "episode_id" not in first and "record_id" not in first
    assert "metadata" not in first and "artifacts" not in first
    assert second["role"] == "tool" and second["text"] == "下载超时"
    assert second["metadata"] == {
        "record_kind": "tool_result",
        "tool_name": "download",
        "call_id": "call-1",
        "is_error": True,
        "memory_item_ids": ["item-1"],
    }


@pytest.mark.asyncio
async def test_flush_evidence_then_dream_publishes_formal_item(tmp_path: Path) -> None:
    provider = Provider(flush_one)
    memory = service(tmp_path, provider)
    episode = memory.observe(MemoryObserveInput(text="本项目以后用 uv 管理依赖"))
    result = await memory.flush("project")
    assert result.status == "completed" and result.counts["observations"] == 1
    assert result.usage["total_tokens"] == 15
    assert memory.search(MemorySearchQuery(query="uv"), ["project"]).items == ()
    observation = memory.store.list_observations("project")[0].observation
    assert observation.evidence[0].source_id == episode.id

    def dream(source: dict[str, object]) -> dict[str, object]:
        return {
            "operations": [
                {
                    "action": "add",
                    "new_key": "new-uv",
                    "text": "本项目使用 uv",
                    "category": "reference",
                    "kind": "fact",
                    "reason": "保留项目约定",
                    "evidence": source["observations"][0]["evidence"],
                }
            ],
            "resolutions": [
                {"observation_id": observation.id, "target_id": "new-uv", "reason": "形成项目约定"}
            ],
        }

    provider.respond = dream
    dreamed = await memory.dream("project")
    assert dreamed.status == "completed"
    assert dreamed.counts["add"] == 1
    (hit,) = memory.search(MemorySearchQuery(query="uv"), ["project"]).items
    item = memory.get_item(hit.item_id, ["project"])
    assert item.evidence == observation.evidence
    assert memory.generation_state("project").pending_observations == 0
    assert all(not request.tools for request in provider.requests)
    assert all(request.response_format == "json_object" for request in provider.requests)
    assert all(request.temperature == 0 for request in provider.requests)
    assert (await memory.dream("project")).status == "empty"
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_generation_prompts_preserve_json_schema_and_original_text(tmp_path: Path) -> None:
    provider = Provider(flush_one)
    memory = service(tmp_path, provider)
    original = '<topic> R&D "原文" {{ untouched }}'
    memory.observe(MemoryObserveInput(text=original))
    await memory.flush("project")
    observation = memory.store.list_observations("project")[0].observation
    provider.respond = lambda _: {
        "operations": [],
        "resolutions": [
            {"observation_id": observation.id, "target_id": None, "reason": "无新增知识"}
        ],
    }
    await memory.dream("project")

    flush_request, dream_request = provider.requests
    for request, schema, instruction in (
        (flush_request, _FlushResponse, "不调用工具"),
        (dream_request, _DreamResponse, "original_evidence"),
    ):
        prompt, schema_json = request.messages[0].text.rsplit("\n", 1)
        assert json.loads(schema_json) == schema.model_json_schema()
        assert "&quot;" not in request.messages[0].text
        assert instruction in prompt
    assert json.loads(flush_request.messages[1].text)["records"][0]["text"] == original
    original_evidence = json.loads(dream_request.messages[1].text)["original_evidence"]
    assert next(iter(original_evidence.values()))["text"] == original


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["flush", "dream"])
@pytest.mark.parametrize("template", ["{% invalid %}", "{{ missing }}"])
async def test_template_failure_does_not_consume_generation_input(
    tmp_path: Path, stage: str, template: str
) -> None:
    provider = Provider(flush_one)
    memory = service(tmp_path, provider)
    memory.observe(MemoryObserveInput(text="保留原文"))
    if stage == "dream":
        await memory.flush("project")
    before = memory.generation_state("project")
    before_requests = len(provider.requests)
    template_path = memory.prompt_source.root / f"memory_{stage}.j2"
    template_path.write_text(template, encoding="utf-8")

    with pytest.raises(IrisMemoryError) as captured:
        await (memory.flush("project") if stage == "flush" else memory.dream("project"))

    assert isinstance(captured.value.__cause__, IrisTemplateError)
    assert captured.value.context["path"] == str(template_path)
    assert len(provider.requests) == before_requests
    after = memory.generation_state("project")
    assert after.pending_episodes == before.pending_episodes
    assert after.pending_observations == before.pending_observations
    assert after.item_revision == before.item_revision
    failed = next(result for result in after.latest_results if result.stage == stage)
    assert failed.status == "failed"
    assert memory.list_items(["project"]) == []


@pytest.mark.asyncio
async def test_empty_generation_does_not_render_templates(
    tmp_path: Path, observability: tuple[Observability, InMemorySpanExporter]
) -> None:
    observation, exporter = observability
    provider = Provider(flush_one)
    memory = service(tmp_path, provider, observability=observation)
    for stage in ("flush", "dream"):
        (memory.prompt_source.root / f"memory_{stage}.j2").write_text(
            "{% invalid %}", encoding="utf-8"
        )
    with observation.bind({"iris.maintenance.kind": "memory"}), observation.scope("cycle"):
        assert (await memory.flush("project")).status == "empty"
        assert (await memory.dream("project")).status == "empty"
    assert provider.requests == []
    [cycle] = exporter.get_finished_spans()
    assert cycle.status.status_code is StatusCode.UNSET
    assert [event.attributes["iris.maintenance.status"] for event in cycle.events] == [
        "empty",
        "empty",
    ]


@pytest.mark.asyncio
async def test_dream_receives_original_attribution_and_tool_metadata(tmp_path: Path) -> None:
    def extract(source: dict[str, object]) -> dict[str, object]:
        response = flush_one(source)
        response["observations"][0]["evidence"] = [record["ref"] for record in source["records"]]
        return response

    provider = Provider(extract)
    memory = service(tmp_path, provider)
    records = (
        MemoryRecord(role="user", text="我认为失败与代理有关", occurred_at="2026-09-22T09:00:00Z"),
        MemoryRecord(
            role="tool",
            text="下载超时",
            occurred_at="2026-09-22T09:01:00Z",
            metadata={"is_error": True},
        ),
    )
    memory.observe(MemoryObserveInput(records=records))
    await memory.flush("project")
    observation = memory.store.list_observations("project")[0].observation
    provider.respond = lambda _: {
        "operations": [],
        "resolutions": [
            {"observation_id": observation.id, "target_id": None, "reason": "无新增知识"}
        ],
    }
    await memory.dream("project")
    original = json.loads(provider.requests[-1].messages[1].text)["original_evidence"]
    assert list(original.values()) == [
        {
            "role": record.role,
            "text": record.text,
            "occurred_at": record.occurred_at,
            "metadata": record.metadata,
        }
        for record in records
    ]


@pytest.mark.asyncio
async def test_empty_flush_consumes_and_truncated_flush_does_not(
    tmp_path: Path, observability: tuple[Observability, InMemorySpanExporter]
) -> None:
    observation, exporter = observability
    provider = Provider(lambda _: {"observations": []})
    memory = service(tmp_path, provider, observability=observation)
    memory.observe(MemoryObserveInput(text="无需要记录的闲聊"))
    provider.finish_reason = "length"
    with observation.bind({"iris.maintenance.kind": "memory"}), observation.scope("cycle"):
        with pytest.raises(IrisMemoryError):
            await memory.flush("project")
    model, cycle = exporter.get_finished_spans()
    assert model.status.status_code is StatusCode.UNSET
    assert cycle.status.status_code is StatusCode.ERROR
    [event] = cycle.events
    assert event.attributes["iris.maintenance.stage"] == "flush"
    assert event.attributes["iris.maintenance.status"] == "failed"
    assert memory.generation_state("project").pending_episodes == 1
    assert memory.generation_state("project").latest_results[0].usage["total_tokens"] == 15
    provider.finish_reason = "stop"
    result = await memory.flush("project")
    assert result.counts["observations"] == 0
    assert memory.generation_state("project").pending_episodes == 0


@pytest.mark.asyncio
async def test_partial_record_resumes_from_committed_character(tmp_path: Path) -> None:
    provider = Provider(lambda _: {"observations": []})
    memory = service(tmp_path, provider, flush_input_budget_tokens=1400)
    episode = memory.observe(MemoryObserveInput(text="a" * 4000))
    result = await memory.flush("project")
    first = json.loads(provider.requests[0].messages[1].text)["records"][0]
    assert first["start"] == 0 and 0 < first["end"] < 4000
    assert result.consumed_ranges[0].start == first["start"]
    assert result.consumed_ranges[0].end == first["end"]
    assert (
        memory.generation_state("project").latest_results[0].consumed_ranges
        == result.consumed_ranges
    )
    await memory.flush("project")
    second = json.loads(provider.requests[1].messages[1].text)["records"][0]
    assert second["start"] == first["end"]
    assert memory.store.get_episode(episode.id, "project").records[0].text == "a" * 4000


@pytest.mark.asyncio
async def test_invalid_evidence_does_not_consume_episode(tmp_path: Path) -> None:
    provider = Provider(
        lambda _: {
            "observations": [
                {
                    "text": "编造来源",
                    "reason": "无依据",
                    "evidence": ["unknown"],
                }
            ]
        }
    )
    memory = service(tmp_path, provider)
    memory.observe(MemoryObserveInput(text="真实材料"))
    with pytest.raises(IrisMemoryError, match="证据"):
        await memory.flush("project")
    assert memory.generation_state("project").pending_episodes == 1
    assert memory.store.list_observations("project") == []


@pytest.mark.asyncio
async def test_explicit_no_change_is_consumed_without_revision_churn(tmp_path: Path) -> None:
    provider = Provider(lambda _: {"operations": [], "resolutions": []})
    memory = service(tmp_path, provider)
    memory.remember(MemoryWriteInput(text="项目使用 uv", reason="明确写入"))
    revision = memory.generation_state("project").item_revision
    assert memory.generation_state("project").pending_changes == 1
    result = await memory.dream("project")
    assert result.status == "completed"
    assert result.counts["processed_changes"] == result.counts["unchanged"] == 1
    assert result.counts["processed_observations"] == 0
    state = memory.generation_state("project")
    assert state.pending_changes == 0 and state.item_revision == revision


@pytest.mark.asyncio
async def test_readback_only_records_are_consumed_without_model_call(tmp_path: Path) -> None:
    provider = Provider(flush_one)
    memory = service(tmp_path, provider)
    memory.observe(
        MemoryObserveInput(
            records=(
                MemoryRecord(
                    role="tool",
                    text="",
                    metadata={"evidence_allowed": False, "memory_item_ids": ["old"]},
                ),
            )
        )
    )
    result = await memory.flush("project")
    assert result.counts["observations"] == 0 and provider.requests == []
    assert memory.generation_state("project").pending_episodes == 0


@pytest.mark.asyncio
async def test_large_explicit_change_is_blocked_but_unrelated_small_change_advances(
    tmp_path: Path,
) -> None:
    provider = Provider(lambda _: {"operations": [], "resolutions": []})
    memory = service(tmp_path, provider, dream_input_budget_tokens=5000)
    memory.remember(MemoryWriteInput(text="large fact " * 2000, reason="大段有效材料"))
    memory.remember(MemoryWriteInput(text="独立的小事实", reason="独立材料"))
    with patch.object(
        _prompts, "render_memory_prompt", wraps=_prompts.render_memory_prompt
    ) as render:
        result = await memory.dream("project")
    assert render.call_count == 1
    assert result.status == "completed" and result.counts["blocked"] == 1
    state = memory.generation_state("project")
    assert state.blocked_changes == 1 and state.pending_changes == 0
    source = json.loads(provider.requests[0].messages[1].text)
    assert len(source["explicit_changes"]) == 1
    assert "独立的小事实" in provider.requests[0].messages[1].text
    assert "large fact" not in provider.requests[0].messages[1].text
    assert (await memory.dream("project")).status == "empty"
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_provider_suppressing_cancellation_cannot_commit_late_observations(
    tmp_path: Path,
    observability: tuple[Observability, InMemorySpanExporter],
) -> None:
    observation, exporter = observability
    started = asyncio.Event()

    class SlowProvider(Provider):
        """模拟底层 client 清理取消后仍返回已到达响应。"""

        async def complete(self, request: LLMRequest) -> LLMResponse:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return await super().complete(request)

    provider = SlowProvider(flush_one)
    memory = service(tmp_path, provider, observability=observation)
    memory.observe(MemoryObserveInput(text="本项目使用 uv"))
    with observation.bind({"iris.maintenance.kind": "memory"}), observation.scope("cycle"):
        task = asyncio.create_task(memory.flush("project"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    state = memory.generation_state("project")
    assert state.pending_episodes == 1 and state.pending_observations == 0
    assert state.latest_results[0].status == "cancelled"
    assert state.latest_results[0].usage["total_tokens"] == 15
    model, cycle = exporter.get_finished_spans()
    assert model.attributes["iris.model.outcome"] == "completed"
    assert model.status.status_code is cycle.status.status_code is StatusCode.UNSET
    [event] = cycle.events
    assert event.attributes["iris.maintenance.status"] == "cancelled"
