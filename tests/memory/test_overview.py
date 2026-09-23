"""验证两部分显式概览、版本发布和知识范围读回。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory import (
    FileMemoryMirror,
    MemoryCategory,
    MemoryEvent,
    MemoryEventType,
    MemoryItem,
    MemoryItemPatch,
    MemoryObserveInput,
    MemoryOverviewConfig,
    MemoryService,
    MemoryWriteInput,
    SQLiteMemoryStore,
)
from iris.message import LLMRequest, LLMResponse, TextBlock

from .test_search import _flush_observation


class _Provider:
    """返回可控的完整文本并保留实际请求。"""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []
        self.estimated_tokens = 20
        self.response = LLMResponse(
            provider="fake",
            content=[
                TextBlock(text='{"core_facts":"核心事实","knowledge_scope":"项目资料与用户偏好"}')
            ],
            finish_reason="stop",
            input_tokens=20,
            output_tokens=4,
            total_tokens=24,
        )

    def estimate_input_tokens(self, request: LLMRequest) -> int:
        return self.estimated_tokens

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self.response


def _service(tmp_path: Path, provider: _Provider | None = None) -> MemoryService:
    return MemoryService(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        mirror=FileMemoryMirror(tmp_path / "mirror"),
        overview_provider=provider,
        overview_model="fake-model" if provider is not None else None,
        overview_config=MemoryOverviewConfig(input_budget_tokens=100),
    )


@pytest.mark.asyncio
async def test_generation_uses_complete_active_namespace_and_two_part_overview(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    service = _service(tmp_path, provider)
    for index in range(105):
        service.store.add_item(
            item := MemoryItem(
                namespace="研究 / A", text=f"正文-{index}", category=MemoryCategory.REFERENCE
            ),
            event=MemoryEvent(
                namespace=item.namespace, event_type=MemoryEventType.ADD, item_id=item.id
            ),
        )
    service.remember(MemoryWriteInput(namespace="other", text="private-secret", reason="seed"))
    episode = service.observe(MemoryObserveInput(namespace="研究 / A", text="episode-secret"))
    _flush_observation(service.store, episode, text="observation-secret")
    assert provider.requests == []
    result = await service.refresh_overview("研究 / A")
    request = provider.requests[0]
    source = request.messages[1].text
    assert all(f"正文-{index}" in source for index in range(105))
    assert all(
        secret not in source
        for secret in ("private-secret", "episode-secret", "observation-secret")
    )
    assert request.tools == [] and request.max_tokens == 4096
    assert request.model == "fake-model"
    assert result.item_count == 105 and result.source_revision == 105
    assert result.published and result.usage["total_tokens"] == 24
    saved = next(
        result
        for result in service.generation_state("研究 / A").latest_results
        if result.stage == "overview"
    )
    assert saved.status == "completed"
    assert saved.usage == result.usage
    assert saved.item_revision == 105
    assert saved.counts == {"items": 105, "published": 1}
    (document,) = await service.aload_overviews(["研究 / A"])
    assert document.source_revision == 105
    assert "核心事实" in document.text
    assert "项目资料与用户偏好" in document.navigation
    assert "核心事实" not in document.navigation
    assert "未同步" in document.warning


@pytest.mark.asyncio
async def test_absent_overview_does_not_scan_database_or_generate_topics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path)
    service.remember(MemoryWriteInput(text="not a summary", reason="seed"))

    def unexpected_read(*args: object, **kwargs: object) -> None:
        pytest.fail("缺少概览时不得扫描条目或搜索主题")

    monkeypatch.setattr(service.store, "read_namespace_snapshot", unexpected_read)
    monkeypatch.setattr(service.store, "list_items", unexpected_read)
    monkeypatch.setattr(service.store, "search", unexpected_read)
    document, empty = await service.aload_overviews(["project", "empty"])
    assert document.source_revision is None
    assert document.text == document.navigation
    assert "not a summary" not in document.text
    assert document.navigation == "尚未生成概览，知识范围未知。"
    assert empty.navigation == document.navigation
    assert empty.warning is None
    assert not service.load_overviews([])
    with pytest.raises(IrisMemoryError, match="未配置"):
        await service.refresh_overview("project")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        '```json\n{"core_facts":"fact","knowledge_scope":"topic"}\n```',
        '{"knowledge_scope":"topic"}',
        '{"core_facts":"fact"}',
        '{"core_facts":"fact","knowledge_scope":"  "}',
        '{"core_facts":1,"knowledge_scope":"topic"}',
        '{"core_facts":"fact","knowledge_scope":"topic","extra":true}',
    ],
)
async def test_invalid_content_preserves_file_and_usage(tmp_path: Path, payload: str) -> None:
    provider = _Provider()
    service = _service(tmp_path, provider)
    service.remember(MemoryWriteInput(text="fact", reason="seed"))
    first = await service.refresh_overview("project")
    before = first.path.read_text(encoding="utf-8")
    provider.response = provider.response.model_copy(update={"content": [TextBlock(text=payload)]})
    with pytest.raises(IrisMemoryError) as captured:
        await service.refresh_overview("project")
    assert captured.value.context["usage"]["total_tokens"] == 24
    assert first.path.read_text(encoding="utf-8") == before
    assert len(provider.requests) == 2
    saved = service.generation_state("project").latest_results[0]
    assert saved.stage == "overview" and saved.status == "failed"
    assert saved.usage["total_tokens"] == 24
    assert saved.error


@pytest.mark.asyncio
async def test_core_facts_can_be_empty_and_sdk_without_mirror_stays_empty(tmp_path: Path) -> None:
    provider = _Provider()
    provider.response = provider.response.model_copy(
        update={
            "content": [TextBlock(text=json.dumps({"core_facts": "", "knowledge_scope": "主题"}))]
        }
    )
    service = _service(tmp_path, provider)
    service.remember(MemoryWriteInput(text="fact", reason="seed"))
    result = await service.refresh_overview("project")
    assert result.published
    (document,) = service.load_overviews(["project"])
    assert document.navigation == "## 可查询的知识\n\n主题\n"
    service_without_mirror = MemoryService(service.store)
    assert await service_without_mirror.aload_overviews(["project"]) == ()
    with pytest.raises(IrisMemoryError, match="未配置"):
        await service_without_mirror.refresh_overview("project")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "existing",
    [
        "<!-- iris-memory source_revision: 0 -->\n# 旧格式\n"
        "<!-- iris-memory-navigation -->\nUser/user.md\n",
        "没有合法版本的旧概览",
        "<!-- iris-memory source_revision: 0 -->\n# 缺少规定节头\n"
        "<!-- iris-memory-knowledge-scope -->\n## 可查询的知识\n\n主题\n",
    ],
)
async def test_invalid_published_format_requires_explicit_refresh(
    tmp_path: Path, existing: str
) -> None:
    provider = _Provider()
    service = _service(tmp_path, provider)
    service.remember(MemoryWriteInput(text="fact", reason="seed"))
    path = service.mirror.namespace_directory("project") / "Memory.md"
    path.write_text(existing, encoding="utf-8")
    with pytest.raises(IrisMemoryError):
        service.load_overviews(["project"])
    assert path.read_text(encoding="utf-8") == existing
    result = await service.refresh_overview("project")
    assert result.published
    assert "iris-memory-knowledge-scope" in path.read_text(encoding="utf-8")
    assert service.load_overviews(["project"])[0].source_revision == 1


def test_overview_publication_serializes_state_without_advancing_revisions(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.remember(MemoryWriteInput(text="fact", reason="seed"))
    before = service.store.read_namespace_state("project")
    token = object()

    def publish(state: object) -> object:
        assert state == before
        return token

    assert service.store.publish_overview("project", publish) is token
    assert service.store.read_namespace_state("project") == before
    error = IrisMemoryError("publication failed")

    def fail(state: object) -> object:
        raise error

    with pytest.raises(IrisMemoryError) as captured:
        service.store.publish_overview("project", fail)
    assert captured.value is error
    assert service.store.publish_overview("project", publish) is token


@pytest.mark.asyncio
async def test_budget_and_incomplete_response_preserve_last_complete_overview(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    service = _service(tmp_path, provider)
    service.remember(MemoryWriteInput(text="fact", reason="seed"))
    published = await service.refresh_overview("project")
    before = published.path.read_text(encoding="utf-8")
    provider.estimated_tokens = 101
    with pytest.raises(IrisMemoryError, match="容量"):
        await service.refresh_overview("project")
    assert len(provider.requests) == 1
    assert published.path.read_text(encoding="utf-8") == before
    provider.estimated_tokens = 100
    provider.response = provider.response.model_copy(update={"finish_reason": "length"})
    with pytest.raises(IrisMemoryError, match="完整") as captured:
        await service.refresh_overview("project")
    assert captured.value.context["usage"] == {
        "input_tokens": 20,
        "output_tokens": 4,
        "total_tokens": 24,
    }
    assert published.path.read_text(encoding="utf-8") == before
    provider.response = provider.response.model_copy(
        update={"finish_reason": "stop", "content": []}
    )
    with pytest.raises(IrisMemoryError, match="完整"):
        await service.refresh_overview("project")
    assert published.path.read_text(encoding="utf-8") == before


@pytest.mark.asyncio
async def test_older_generation_cannot_overwrite_newer_published_overview(tmp_path: Path) -> None:
    provider = _Provider()
    service = _service(tmp_path, provider)
    item = service.remember(MemoryWriteInput(text="v1", reason="seed"))
    started = asyncio.Event()
    release = asyncio.Event()

    async def complete(request: LLMRequest) -> LLMResponse:
        provider.requests.append(request)
        if len(provider.requests) == 1:
            started.set()
            await release.wait()
            return provider.response.model_copy(
                update={
                    "content": [
                        TextBlock(text='{"core_facts":"older","knowledge_scope":"项目版本"}')
                    ]
                }
            )
        return provider.response.model_copy(
            update={
                "content": [TextBlock(text='{"core_facts":"newer","knowledge_scope":"项目版本"}')]
            }
        )

    provider.complete = complete
    pending = asyncio.create_task(service.refresh_overview("project"))
    await started.wait()
    service.update(item.id, "project", MemoryItemPatch(text="v2"), reason="edit")
    newer = await service.refresh_overview("project")
    release.set()
    older = await pending
    assert newer.published and newer.source_revision == 2
    assert not older.published and older.source_revision == 1
    assert older.usage["total_tokens"] == 24
    assert "newer" in newer.path.read_text(encoding="utf-8")
    assert "\nolder\n" not in newer.path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_generation_behind_items_can_publish_with_stale_warning(tmp_path: Path) -> None:
    provider = _Provider()
    service = _service(tmp_path, provider)
    item = service.remember(MemoryWriteInput(text="v1", reason="seed"))

    async def complete(request: LLMRequest) -> LLMResponse:
        service.update(item.id, "project", MemoryItemPatch(text="v2"), reason="during model")
        return provider.response

    provider.complete = complete
    result = await service.refresh_overview("project")
    assert result.published
    assert (result.source_revision, result.current_revision) == (1, 2)
    (document,) = await service.aload_overviews(["project"])
    assert "陈旧" in document.warning


@pytest.mark.asyncio
async def test_empty_namespace_publishes_without_calling_provider(tmp_path: Path) -> None:
    provider = _Provider()
    service = _service(tmp_path, provider)

    def unexpected_estimate(request: LLMRequest) -> int:
        pytest.fail("空快照不需要构造或估算模型请求")

    provider.estimate_input_tokens = unexpected_estimate
    result = await service.refresh_overview("empty")
    assert provider.requests == []
    assert result.published and result.source_revision == 0 and result.item_count == 0
    assert result.usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    assert "当前无记忆" in result.path.read_text(encoding="utf-8")
    (document,) = await service.aload_overviews(["empty"])
    assert document.warning is None


@pytest.mark.asyncio
async def test_provider_failure_preserves_previous_overview_and_propagates_error(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    service = _service(tmp_path, provider)
    service.remember(MemoryWriteInput(text="fact", reason="seed"))
    result = await service.refresh_overview("project")
    before = result.path.read_text(encoding="utf-8")
    error = IrisMemoryError("fake provider failure")

    async def fail(request: LLMRequest) -> LLMResponse:
        raise error

    provider.complete = fail
    with pytest.raises(IrisMemoryError) as captured:
        await service.refresh_overview("project")
    assert captured.value is error
    assert result.path.read_text(encoding="utf-8") == before
    saved = service.generation_state("project").latest_results[0]
    assert saved.stage == "overview" and saved.status == "failed"
    assert saved.error == str(error)
    assert saved.usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


@pytest.mark.asyncio
async def test_overview_publication_failure_keeps_file_and_reports_generation_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _Provider()
    service = _service(tmp_path, provider)
    service.remember(MemoryWriteInput(text="fact", reason="seed"))
    result = await service.refresh_overview("project")
    before = result.path.read_text(encoding="utf-8")

    def fail(relative_path: str, content: str) -> None:
        raise IrisMemoryError("write failed")

    monkeypatch.setattr(service.mirror, "_atomic_replace", fail)
    with pytest.raises(IrisMemoryError, match="write failed") as captured:
        await service.refresh_overview("project")
    assert captured.value.context["usage"]["total_tokens"] == 24
    assert result.path.read_text(encoding="utf-8") == before
    saved = service.generation_state("project").latest_results[0]
    assert saved.stage == "overview" and saved.status == "failed"
    assert saved.usage["total_tokens"] == 24
