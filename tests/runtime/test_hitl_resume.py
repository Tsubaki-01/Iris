from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeProvider

from iris.agents import AgentConfig
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.hitl import (
    InMemoryInteractionStore,
    PermissionInteractionResponse,
    QuestionInteractionResponse,
)
from iris.hitl.tools import AskQuestionTool
from iris.message import LLMResponse, TextBlock, ToolUseBlock
from iris.runtime import AgentRuntime
from iris.runtime.models import RuntimeOptions, RuntimeStatus
from iris.session import InMemorySessionStore
from iris.tools import (
    DefaultPermissionPolicy,
    ToolCapability,
    ToolExecutor,
    ToolRegistry,
)


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="hitl-resume-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        system="你是本地助手。",
    )


def _context_input() -> ContextBuildInput:
    return ContextBuildInput(
        system=ContextSection(slots=[ContextSlot(name="instructions", content="遵守用户指令")])
    )


def _tool_response(call: ToolUseBlock) -> LLMResponse:
    return LLMResponse(
        provider="fake",
        id="response-1",
        model="gpt-4o-mini",
        content=[TextBlock(text="需要工具。"), call],
        finish_reason="tool_calls",
    )


@pytest.mark.asyncio
async def test_resume_question_returns_answer_without_another_provider_call() -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    session_store = InMemorySessionStore()
    interaction_store = InMemoryInteractionStore()
    provider = FakeProvider(
        [
            _tool_response(
                ToolUseBlock(
                    id="call_question",
                    name="ask_question",
                    input={"question": "选哪个？"},
                )
            )
        ]
    )
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        session_store=session_store,
        interaction_store=interaction_store,
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )

    waiting = await runtime.run_turn(
        "请询问我",
        options=RuntimeOptions(session_id="session-1", run_id="run-1"),
    )
    assert waiting.pending_interaction is not None

    resumed = await runtime.resume(
        waiting.pending_interaction.interaction_id,
        QuestionInteractionResponse(answer="选生产"),
    )

    assert resumed.status is RuntimeStatus.OK
    assert [result.model_content for result in resumed.tool_results] == ["选生产"]
    assert len(provider.requests) == 1
    assert session_store.load_messages("session-1")[-1]["content"][0]["content"] == "选生产"


@pytest.mark.asyncio
async def test_resume_approved_permission_executes_only_matching_call_once() -> None:
    calls: list[str] = []

    def write_note(value: str) -> str:
        calls.append(value)
        return "written"

    registry = ToolRegistry()
    registry.register_function(
        write_note,
        description="写入笔记",
        capabilities={ToolCapability.WRITE},
    )
    provider = FakeProvider(
        [_tool_response(ToolUseBlock(id="call_write", name="write_note", input={"value": "ok"}))]
    )
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=provider,
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(
            registry,
            permission_policy=DefaultPermissionPolicy(write_mode="confirm"),
        ),
        workspace_root=Path.cwd(),
    )

    waiting = await runtime.run_turn("写入", options=RuntimeOptions(run_id="run-2"))
    assert waiting.pending_interaction is not None

    resumed = await runtime.resume(
        waiting.pending_interaction.interaction_id,
        PermissionInteractionResponse(decision="approve"),
    )

    assert resumed.status is RuntimeStatus.OK
    assert calls == ["ok"]
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_resume_rejected_permission_does_not_execute_tool() -> None:
    calls: list[str] = []

    def write_note() -> str:
        calls.append("called")
        return "written"

    registry = ToolRegistry()
    registry.register_function(
        write_note, description="写入笔记", capabilities={ToolCapability.WRITE}
    )
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [_tool_response(ToolUseBlock(id="call_write", name="write_note", input={}))]
        ),
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(
            registry, permission_policy=DefaultPermissionPolicy(write_mode="confirm")
        ),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_turn("写入")
    assert waiting.pending_interaction is not None

    resumed = await runtime.resume(
        waiting.pending_interaction.interaction_id, PermissionInteractionResponse(decision="reject")
    )

    assert resumed.status is RuntimeStatus.OK
    assert resumed.tool_results[0].error is not None
    assert resumed.tool_results[0].error.code == "USER_REJECTED"
    assert calls == []


@pytest.mark.asyncio
async def test_resume_pauses_again_for_second_gate_in_same_batch() -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                LLMResponse(
                    provider="fake",
                    content=[
                        ToolUseBlock(id="first", name="ask_question", input={"question": "一？"}),
                        ToolUseBlock(id="second", name="ask_question", input={"question": "二？"}),
                    ],
                    finish_reason="tool_calls",
                )
            ]
        ),
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_turn("开始")
    assert waiting.pending_interaction is not None

    resumed = await runtime.resume(
        waiting.pending_interaction.interaction_id, QuestionInteractionResponse(answer="一")
    )

    assert resumed.status is RuntimeStatus.WAITING_HUMAN
    assert resumed.pending_interaction is not None
    assert resumed.pending_interaction.tool_call_id == "second"


@pytest.mark.asyncio
async def test_resume_claimed_interaction_without_result_fails_closed() -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                _tool_response(
                    ToolUseBlock(id="call", name="ask_question", input={"question": "继续？"})
                )
            ]
        ),
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_turn("开始")
    assert waiting.pending_interaction is not None
    interaction_id = waiting.pending_interaction.interaction_id
    runtime.interaction_service.resolve(interaction_id, QuestionInteractionResponse(answer="是"))
    runtime.interaction_service.claim(interaction_id, waiting.pending_interaction.checkpoint)

    resumed = await runtime.resume(interaction_id)

    assert resumed.status is RuntimeStatus.ERROR
    assert resumed.error is not None
    assert resumed.error.code == "HITL_EXECUTION_OUTCOME_UNKNOWN"


@pytest.mark.asyncio
async def test_resume_result_committed_is_idempotent() -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    store = InMemorySessionStore()
    runtime = AgentRuntime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                _tool_response(
                    ToolUseBlock(id="call", name="ask_question", input={"question": "继续？"})
                )
            ]
        ),
        session_store=store,
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )
    waiting = await runtime.run_turn("开始")
    assert waiting.pending_interaction is not None
    interaction_id = waiting.pending_interaction.interaction_id
    await runtime.resume(interaction_id, QuestionInteractionResponse(answer="是"))

    retried = await runtime.resume(interaction_id)

    assert retried.status is RuntimeStatus.OK
    assert len(store.load_tool_events("default")) == 1
