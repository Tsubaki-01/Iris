from __future__ import annotations

import pytest
from fakes import FakeProvider

from iris.agents import AgentConfig
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.memory import (
    MemoryCategory,
    MemoryContextBundle,
    MemoryContextFragment,
    MemoryItem,
    MemoryItemKind,
    MemoryLevel,
    MemoryQuery,
    MemoryScope,
    MemorySearchResult,
)
from iris.message import LLMResponse, Role, TextBlock
from iris.runtime import AgentRuntime
from iris.runtime.models import RuntimeOptions, RuntimeStatus


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="memory-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        system="你是本地助手。",
    )


def _context_input() -> ContextBuildInput:
    return ContextBuildInput(
        system=ContextSection(
            slots=[ContextSlot(name="instructions", content="遵守用户指令")]
        )
    )


def _assistant_response() -> LLMResponse:
    return LLMResponse(
        provider="fake",
        id="response-1",
        model="gpt-4o-mini",
        content=[TextBlock(text="收到。")],
        finish_reason="stop",
    )


def _memory_scope() -> MemoryScope:
    return MemoryScope(workspace_id="workspace", agent_id="agent")


def _memory_result(
    *, item_id: str = "memory-1", text: str = "用户喜欢简洁回答"
) -> MemorySearchResult:
    item = MemoryItem(
        id=item_id,
        scope=_memory_scope(),
        text=text,
        category=MemoryCategory.USER,
        kind=MemoryItemKind.PREFERENCE,
        level=MemoryLevel.SEMANTIC,
    )
    return MemorySearchResult(
        item=item,
        score=0.98,
        source="sqlite",
        matched_text=text,
    )


def _memory_query() -> MemoryQuery:
    return MemoryQuery(scope=_memory_scope(), text="偏好")


@pytest.mark.asyncio
async def test_run_turn_does_not_call_memory_service_by_default() -> None:
    memory_service = SpyMemoryService(
        MemoryContextBundle(fragments=[], total_chars=0, omitted_count=0, max_chars=100)
    )
    provider = FakeProvider([_assistant_response()])
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        memory_service=memory_service,
    )

    result = await runtime.run_turn("当前问题")

    assert result.status == RuntimeStatus.OK
    assert memory_service.calls == []
    assert [message.role for message in provider.requests[0].messages] == [
        Role.SYSTEM,
        Role.USER,
    ]
    assert provider.requests[0].messages[1].text == "当前问题"


@pytest.mark.asyncio
async def test_run_turn_injects_explicit_memory_results_as_context_slots() -> None:
    provider = FakeProvider([_assistant_response()])
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
    )

    result = await runtime.run_turn(
        "当前问题",
        options=RuntimeOptions(
            memory_results=[_memory_result()],
            memory_max_chars=100,
        ),
    )

    assert result.status == RuntimeStatus.OK
    assert [message.role for message in provider.requests[0].messages] == [
        Role.SYSTEM,
        Role.USER,
        Role.USER,
    ]
    memory_message = provider.requests[0].messages[1]
    assert memory_message.text.startswith("<memory_context>")
    assert "<memory " in memory_message.text
    assert 'item_id="memory-1"' in memory_message.text
    assert 'category="user"' in memory_message.text
    assert 'kind="preference"' in memory_message.text
    assert 'level="l2"' in memory_message.text
    assert 'truncated="false"' in memory_message.text
    assert "用户喜欢简洁回答" in memory_message.text
    assert "score" not in memory_message.text
    assert "sqlite" not in memory_message.text
    assert provider.requests[0].messages[2].text == "当前问题"


@pytest.mark.asyncio
async def test_run_turn_uses_memory_service_for_explicit_memory_query() -> None:
    bundle = MemoryContextBundle(
        fragments=[
            MemoryContextFragment(
                item_id="memory-2",
                text="用户要求中文回答",
                category=MemoryCategory.USER,
                kind=MemoryItemKind.PREFERENCE,
                level=MemoryLevel.SEMANTIC,
                warning="记忆内容可能过期。",
                truncated=True,
            )
        ],
        total_chars=8,
        omitted_count=0,
        max_chars=8,
    )
    memory_service = SpyMemoryService(bundle)
    provider = FakeProvider([_assistant_response()])
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        memory_service=memory_service,
    )
    query = _memory_query()

    result = await runtime.run_turn(
        "当前问题",
        options=RuntimeOptions(memory_query=query, memory_max_chars=8),
    )

    assert result.status == RuntimeStatus.OK
    assert memory_service.calls == [(query, 8)]
    memory_message = provider.requests[0].messages[1]
    assert 'item_id="memory-2"' in memory_message.text
    assert 'truncated="true"' in memory_message.text
    assert "用户要求中文回答" in memory_message.text


@pytest.mark.asyncio
async def test_run_turn_returns_memory_error_when_query_has_no_service() -> None:
    provider = FakeProvider([_assistant_response()])
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
    )

    result = await runtime.run_turn(
        "当前问题",
        options=RuntimeOptions(memory_query=_memory_query()),
    )

    assert result.status == RuntimeStatus.ERROR
    assert result.error is not None
    assert result.error.code == "MEMORY_ERROR"
    assert result.error.source == "memory"
    assert provider.requests == []


class SpyMemoryService:
    """记录 runtime 是否显式调用 memory service。"""

    def __init__(self, bundle: MemoryContextBundle) -> None:
        self.bundle = bundle
        self.calls: list[tuple[MemoryQuery, int]] = []

    def build_context(
        self, query: MemoryQuery, *, max_chars: int
    ) -> MemoryContextBundle:
        """模拟 `MemoryService.build_context()`。"""
        self.calls.append((query, max_chars))
        return self.bundle
