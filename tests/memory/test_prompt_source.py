"""Memory 生成使用显式项目来源、领域固定契约与操作内一致快照。"""

import json
from pathlib import Path

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory import (
    FileMemoryMirror,
    MemoryMaintenanceScope,
    MemoryObserveInput,
    MemorySearchQuery,
    MemoryService,
    MemoryWriteInput,
    SQLiteMemoryStore,
)
from iris.memory.generation import _DreamResponse, _FlushResponse
from iris.memory.generation_models import MemorySource
from iris.memory.models import MemoryOverviewContent
from iris.prompts import PromptSource

from .test_generation import Provider, flush_one


def _seed(source: PromptSource, version: str) -> None:
    for stage in ("flush", "dream", "overview"):
        (source.root / f"memory_{stage}.j2").write_text(
            f"{version}-{stage}: schema 仅允许伪造字段 fake", encoding="utf-8"
        )


async def _eligible(sources: tuple[MemorySource, ...]) -> bool:
    return True


@pytest.mark.asyncio
async def test_cycle_keeps_one_snapshot_and_next_cycle_adopts_changes(tmp_path: Path) -> None:
    source = PromptSource.initialize(tmp_path)
    _seed(source, "old")

    def respond(data: dict[str, object]) -> dict[str, object]:
        if "records" in data:
            _seed(source, "new")
            return flush_one(data)
        if "observations" in data:
            observation = data["observations"][0]
            return {
                "operations": [
                    {
                        "action": "add",
                        "new_key": "learned",
                        "text": "本项目使用 uv",
                        "category": "reference",
                        "kind": "fact",
                        "reason": "约定",
                        "evidence": observation["evidence"],
                    }
                ],
                "resolutions": [
                    {
                        "observation_id": observation["id"],
                        "target_id": "learned",
                        "reason": "保留约定",
                    }
                ],
            }
        return {"core_facts": "使用 uv", "knowledge_scope": "项目工具约定"}

    provider = Provider(respond)
    memory = MemoryService(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        mirror=FileMemoryMirror(tmp_path / "mirror"),
        prompt_source=source,
        generation_provider=provider,
        generation_model="test",
        overview_provider=provider,
        overview_model="test",
    )
    scope = MemoryMaintenanceScope(frozenset(), _eligible, frozenset())
    memory.observe(MemoryObserveInput(text="本项目使用 uv"))
    await memory.maintain_cycle("project", scope=scope, cycle_id="first-cycle")
    assert len(provider.requests) == 3
    for request, stage, response_model in zip(
        provider.requests,
        ("flush", "dream", "overview"),
        (_FlushResponse, _DreamResponse, MemoryOverviewContent),
        strict=True,
    ):
        prompt = request.messages[0].text
        assert prompt.startswith(f"old-{stage}")
        assert json.dumps(response_model.model_json_schema(), ensure_ascii=False) in prompt
    memory.observe(MemoryObserveInput(text="第二轮仍使用 uv"))
    await memory.maintain_cycle("project", scope=scope, cycle_id="next-cycle")
    assert all(request.messages[0].text.startswith("new-") for request in provider.requests[3:])


@pytest.mark.asyncio
async def test_each_sdk_operation_adopts_current_source(tmp_path: Path) -> None:
    source = PromptSource.initialize(tmp_path)
    _seed(source, "old")
    provider = Provider(lambda _: {"observations": []})
    memory = MemoryService(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        prompt_source=source,
        generation_provider=provider,
        generation_model="test",
    )
    memory.observe(MemoryObserveInput(text="第一批输入"))
    await memory.flush("project")
    _seed(source, "new")
    memory.observe(MemoryObserveInput(text="第二批输入"))
    await memory.flush("project")
    assert provider.requests[0].messages[0].text.startswith("old-flush")
    assert provider.requests[1].messages[0].text.startswith("new-flush")


@pytest.mark.asyncio
async def test_crud_needs_no_source_but_generation_requires_explicit_source(tmp_path: Path) -> None:
    provider = Provider(flush_one)
    memory = MemoryService(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        mirror=FileMemoryMirror(tmp_path / "mirror"),
        generation_provider=provider,
        generation_model="test",
        overview_provider=provider,
        overview_model="test",
    )
    item = memory.remember(MemoryWriteInput(text="项目使用 uv", reason="约定"))
    assert memory.get_item(item.id, ["project"]) == item
    assert memory.search(MemorySearchQuery(query="uv"), ["project"]).items
    for operation in (memory.flush, memory.dream, memory.refresh_overview):
        with pytest.raises(IrisMemoryError, match="prompt.*来源"):
            await operation("project")
    assert provider.requests == []
    assert not (tmp_path / ".iris/prompts").exists()
