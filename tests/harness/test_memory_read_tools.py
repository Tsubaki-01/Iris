"""真实数据库与Runner串联模型自主Search/Fetch及普通工具历史。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.exceptions import IrisMemoryError
from iris.harness import AgentRunner
from iris.lifecycle import AgentRunRequest, RunStopReason
from iris.memory import (
    FileMemoryMirror,
    MemoryCategory,
    MemoryItemPatch,
    MemorySearchQuery,
    MemorySearchResponse,
    MemoryService,
    MemoryWriteInput,
    SQLiteMemoryStore,
)
from iris.message import LLMRequest, LLMResponse, ToolUseBlock
from iris.store import SQLiteStore

from .fakes import StaticProvider, text_response, tool_response


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["none", "search", "fetch"])
async def test_model_controls_search_fetch_and_results_remain_normal_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    """无隐式查询；可直接用片段回答，或在修改后Fetch数据库当前完整记录。"""
    overview_provider = StaticProvider(
        text_response(json.dumps({"core_facts": "", "knowledge_scope": "项目部署版本与配置"}))
    )
    memory_store = SQLiteMemoryStore(tmp_path / "memory.db")
    mirror = FileMemoryMirror(tmp_path / "mirror")
    service = MemoryService(
        memory_store,
        mirror=mirror,
        overview_provider=overview_provider,
        overview_model="fake-model",
    )
    item = service.remember(
        MemoryWriteInput(
            text="deploytoken 当前部署版本 v1",
            category=MemoryCategory.REFERENCE,
            source_id="source-message",
            metadata={"version": 1},
            reason="用户明确保存部署资料",
        )
    )
    await service.refresh_overview("project")
    queries: list[tuple[MemorySearchQuery, tuple[str, ...]]] = []
    original_search = memory_store.search

    def search(query: MemorySearchQuery, namespaces: Sequence[str]) -> MemorySearchResponse:
        queries.append((query, tuple(namespaces)))
        return original_search(query, namespaces)

    monkeypatch.setattr(memory_store, "search", search)
    responses = []
    if strategy != "none":
        responses.append(
            tool_response(
                ToolUseBlock(
                    id="search-memory",
                    name="memory_search",
                    input={"query": "deploytoken", "categories": ["reference"], "kinds": ["note"]},
                )
            )
        )
    if strategy == "fetch":
        responses.append(
            tool_response(
                ToolUseBlock(id="fetch-memory", name="memory_fetch", input={"item_id": item.id})
            )
        )
    responses.extend([text_response("本轮完成"), text_response("下轮完成")])

    class ChoosingProvider(StaticProvider):
        """在Search完成后改变DB，模拟Fetch之前正常发生的记忆更新。"""

        async def complete(self, request: LLMRequest) -> LLMResponse:
            if strategy == "fetch" and len(self.requests) == 1:
                def fail_projection(*args: object, **kwargs: object) -> None:
                    raise IrisMemoryError("projection unavailable")

                monkeypatch.setattr(mirror, "rebuild_from_store", fail_projection)
                service.update(
                    item.id,
                    "project",
                    MemoryItemPatch(text="deploytoken 当前部署版本 v2", metadata={"version": 2}),
                    reason="部署已更新",
                )
            return await super().complete(request)

    provider = ChoosingProvider(*responses)
    lifecycle_store = SQLiteStore(tmp_path / "lifecycle.db")
    config = AgentConfig.model_validate(
        {
            "name": "reader",
            "model": "openai/fake-model",
            "system": "回答项目问题。",
            "permissions": {"workspace": str(tmp_path)},
            "tools": {"builtin": ["memory.search", "memory.fetch"]},
        }
    )
    runner = AgentRunner.from_config(
        config, provider=provider, memory_service=service, store=lifecycle_store
    )
    first = await runner.start(AgentRunRequest(input="确认项目部署版本", run_id="read-memory"))
    assert first.run.stop_reason is RunStopReason.COMPLETED, first.error
    expected_names = [] if strategy == "none" else ["memory_search"]
    if strategy == "fetch":
        expected_names.append("memory_fetch")
    session = lifecycle_store.load_session("default")
    results = [result for message in session.messages for result in message.tool_results]
    assert [result.name for result in results] == expected_names
    assert all(not result.is_error for result in results)
    assert len(queries) == int(strategy != "none")
    if queries:
        query, namespaces = queries[0]
        assert query.query == "deploytoken" and namespaces == ("project",)
        hit = json.loads(results[0].content)["items"][0]
        assert hit["item_id"] == item.id and hit["is_complete"]
        assert hit["snippet"] == item.text
        assert provider.requests[1].messages[-1].tool_results[0].content == results[0].content
    if strategy == "fetch":
        current = service.get_item(item.id, ["project"])
        fetched = json.loads(results[1].content)
        assert fetched == {"item": current.model_dump(mode="json")}
        assert fetched["item"]["metadata"] == {"version": 2}
        assert fetched["item"]["source_id"] == "source-message"
        assert "warning" not in fetched
    assert session.context_window is not None
    assert "项目部署版本与配置" in provider.requests[0].messages[0].text
    assert all(message.metadata.get("context_kind") != "memory" for message in session.messages)
    second = await runner.start(AgentRunRequest(input="继续讨论部署", run_id="next-round"))
    assert second.run.stop_reason is RunStopReason.COMPLETED, second.error
    assert len(queries) == int(strategy != "none")
    replayed = [
        result for message in provider.requests[-1].messages for result in message.tool_results
    ]
    assert replayed == results
    assert len(overview_provider.requests) == 1
