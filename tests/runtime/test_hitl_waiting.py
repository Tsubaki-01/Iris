from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeProvider, build_runtime

from iris.agents import AgentConfig
from iris.context import ContextBuildInput, ContextSection, ContextSlot
from iris.exceptions import HITLCheckpointInvalidError
from iris.hitl import InMemoryInteractionStore, QuestionInteractionResponse
from iris.hitl.models import InteractionKind, PermissionPrompt
from iris.message import LLMResponse, TextBlock, ToolUseBlock
from iris.runtime import AgentRuntime
from iris.runtime.models import RuntimeOptions, RuntimeStatus
from iris.session import InMemorySessionStore
from iris.tools import (
    AskQuestionTool,
    DefaultPermissionPolicy,
    ToolCapability,
    ToolExecutor,
    ToolRegistry,
)


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="hitl-runtime-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        system="你是本地助手。",
    )


def _context_input() -> ContextBuildInput:
    return ContextBuildInput(
        system=ContextSection(slots=[ContextSlot(name="instructions", content="遵守用户指令")])
    )


def _question_response() -> LLMResponse:
    return LLMResponse(
        provider="fake",
        id="response-1",
        model="gpt-4o-mini",
        content=[
            TextBlock(text="我需要询问用户。"),
            ToolUseBlock(
                id="call_question",
                name="ask_question",
                input={"question": "请选择部署环境", "options": ["测试", "生产"]},
            ),
        ],
        finish_reason="tool_calls",
    )


def _tool_response(*tool_calls: ToolUseBlock) -> LLMResponse:
    return LLMResponse(
        provider="fake",
        id="response-1",
        model="gpt-4o-mini",
        content=[TextBlock(text="我需要调用工具。"), *tool_calls],
        finish_reason="tool_calls",
    )


def _question_runtime(
    responses: list[LLMResponse],
    *,
    session_store: InMemorySessionStore | None = None,
    interaction_store: InMemoryInteractionStore | None = None,
) -> AgentRuntime:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    return build_runtime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(responses),
        session_store=session_store,
        interaction_store=interaction_store,
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )


@pytest.mark.asyncio
async def test_run_turn_waits_and_persists_question_interaction_before_execution() -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    interaction_store = InMemoryInteractionStore()
    session_store = InMemorySessionStore()
    runtime = build_runtime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider([_question_response()]),
        session_store=session_store,
        interaction_store=interaction_store,
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )

    result = await runtime.run_turn(
        "需要部署配置",
        options=RuntimeOptions(session_id="session-1", run_id="run-1"),
    )

    assert result.status is RuntimeStatus.WAITING_HUMAN
    assert result.pending_interaction is not None
    assert result.pending_interaction.request.prompt.question == "请选择部署环境"
    assert interaction_store.load_interaction(result.pending_interaction.interaction_id) == (
        result.pending_interaction
    )
    assert [message["role"] for message in session_store.load_messages("session-1")] == [
        "user",
        "assistant",
    ]
    latest_run = session_store.load_run_metadata("session-1")["latest_run"]
    assert latest_run["waiting_human"] is True
    assert latest_run["interaction_id"] == result.pending_interaction.interaction_id
    assert runtime.load_resumable_interaction("session-1") == result.pending_interaction
    checkpoint = result.pending_interaction.checkpoint
    assert checkpoint["run_mode"] == "turn"
    assert checkpoint["checkpoint_version"] == 2
    assert "call_fingerprint" not in checkpoint
    assert checkpoint["next_tool_index"] == 0
    assert checkpoint["tool_calls"][0]["id"] == "call_question"


@pytest.mark.asyncio
async def test_run_loop_waits_before_executing_earlier_read_only_tool() -> None:
    executed: list[str] = []

    def read_probe() -> str:
        executed.append("read")
        return "read"

    registry = ToolRegistry()
    registry.register_function(
        read_probe,
        description="记录只读调用",
        capabilities={ToolCapability.READ},
    )
    registry.register(AskQuestionTool())
    runtime = build_runtime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [
                _tool_response(
                    ToolUseBlock(id="call_read", name="read_probe", input={}),
                    ToolUseBlock(
                        id="call_question",
                        name="ask_question",
                        input={"question": "继续吗？"},
                    ),
                )
            ]
        ),
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )

    result = await runtime.run_loop("开始", options=RuntimeOptions(run_id="run-loop"))

    assert result.status is RuntimeStatus.WAITING_HUMAN
    assert result.pending_interaction is not None
    assert result.pending_interaction.request.tool_call.tool_call_id == "call_question"
    assert executed == []


@pytest.mark.asyncio
async def test_permission_gate_creates_permission_interaction() -> None:
    def write_probe() -> str:
        return "written"

    registry = ToolRegistry()
    registry.register_function(
        write_probe,
        description="写入探针",
        capabilities={ToolCapability.WRITE},
    )
    runtime = build_runtime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [_tool_response(ToolUseBlock(id="call_write", name="write_probe", input={}))]
        ),
        interaction_store=InMemoryInteractionStore(),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(
            registry,
            permission_policy=DefaultPermissionPolicy(write_mode="confirm"),
        ),
        workspace_root=Path.cwd(),
    )

    result = await runtime.run_turn("写入", options=RuntimeOptions(run_id="run-permission"))

    assert result.status is RuntimeStatus.WAITING_HUMAN
    assert result.pending_interaction is not None
    assert result.pending_interaction.request.prompt.kind is InteractionKind.PERMISSION
    request = result.pending_interaction.request
    assert isinstance(request.prompt, PermissionPrompt)
    assert request.tool_call.tool_call_id == "call_write"
    assert request.tool_call.tool_name == "write_probe"


@pytest.mark.asyncio
async def test_invalid_checkpoint_returns_structured_error_without_interaction() -> None:
    registry = ToolRegistry()
    registry.register(AskQuestionTool())
    interaction_store = InMemoryInteractionStore()
    runtime = build_runtime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider([_question_response()]),
        interaction_store=interaction_store,
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )

    result = await runtime.run_turn(
        "提问",
        options=RuntimeOptions(metadata={"invalid": object()}),
    )

    assert result.status is RuntimeStatus.ERROR
    assert result.error is not None
    assert result.error.code == "HITL_CHECKPOINT_INVALID"
    assert interaction_store.list_pending_interactions() == []


def test_load_resumable_interaction_returns_none_without_marker_or_pending() -> None:
    runtime = _question_runtime([])

    assert runtime.load_resumable_interaction("session-1") is None


@pytest.mark.asyncio
async def test_load_resumable_interaction_uses_orphan_pending_fallback() -> None:
    session_store = InMemorySessionStore()
    runtime = _question_runtime([_question_response()], session_store=session_store)
    waiting = await runtime.run_turn(
        "需要部署配置",
        options=RuntimeOptions(session_id="session-1", run_id="run-1"),
    )
    assert waiting.pending_interaction is not None
    session_store.save_run_metadata(
        "session-1",
        {"latest_run": {"status": "ok"}, "runs": []},
    )

    assert runtime.load_resumable_interaction("session-1") == waiting.pending_interaction


@pytest.mark.asyncio
async def test_load_resumable_interaction_uses_pending_when_marker_lacks_id() -> None:
    session_store = InMemorySessionStore()
    runtime = _question_runtime([_question_response()], session_store=session_store)
    waiting = await runtime.run_turn(
        "需要部署配置",
        options=RuntimeOptions(session_id="session-1", run_id="run-1"),
    )
    assert waiting.pending_interaction is not None
    session_store.save_run_metadata(
        "session-1",
        {"latest_run": {"waiting_human": True}, "runs": []},
    )

    assert runtime.load_resumable_interaction("session-1") == waiting.pending_interaction


def test_load_resumable_interaction_rejects_marker_without_id() -> None:
    session_store = InMemorySessionStore()
    session_store.save_run_metadata(
        "session-1",
        {"latest_run": {"waiting_human": True}, "runs": []},
    )
    runtime = _question_runtime([], session_store=session_store)

    with pytest.raises(HITLCheckpointInvalidError, match="interaction_id"):
        runtime.load_resumable_interaction("session-1")


def test_load_resumable_interaction_rejects_unknown_marker_target() -> None:
    session_store = InMemorySessionStore()
    session_store.save_run_metadata(
        "session-1",
        {
            "latest_run": {
                "waiting_human": True,
                "interaction_id": "int_00000000000000000000000000000000",
            },
            "runs": [],
        },
    )
    runtime = _question_runtime([], session_store=session_store)

    with pytest.raises(HITLCheckpointInvalidError, match="不存在"):
        runtime.load_resumable_interaction("session-1")


@pytest.mark.asyncio
async def test_load_resumable_interaction_rejects_session_mismatch() -> None:
    session_store = InMemorySessionStore()
    runtime = _question_runtime([_question_response()], session_store=session_store)
    waiting = await runtime.run_turn(
        "需要部署配置",
        options=RuntimeOptions(session_id="other-session", run_id="run-1"),
    )
    assert waiting.pending_interaction is not None
    session_store.save_run_metadata(
        "session-1",
        {
            "latest_run": {
                "waiting_human": True,
                "interaction_id": waiting.pending_interaction.interaction_id,
            },
            "runs": [],
        },
    )

    with pytest.raises(HITLCheckpointInvalidError, match="session"):
        runtime.load_resumable_interaction("session-1")


@pytest.mark.asyncio
async def test_load_resumable_interaction_rejects_conflicting_pending_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _question_runtime([_question_response()])
    waiting = await runtime.run_turn(
        "需要部署配置",
        options=RuntimeOptions(session_id="session-1", run_id="run-1"),
    )
    assert waiting.pending_interaction is not None
    runtime.environment.interaction_service.resolve(
        waiting.pending_interaction.interaction_id,
        QuestionInteractionResponse(answer="测试"),
    )
    other_pending = waiting.pending_interaction.model_copy(
        update={"interaction_id": "int_11111111111111111111111111111111"}
    )
    monkeypatch.setattr(
        runtime.environment.interaction_service,
        "list_pending",
        lambda session_id=None: [other_pending],
    )

    with pytest.raises(HITLCheckpointInvalidError, match="冲突"):
        runtime.load_resumable_interaction("session-1")


@pytest.mark.asyncio
async def test_run_turn_keeps_non_hitl_tool_execution_compatible() -> None:
    executed: list[str] = []

    def read_probe() -> str:
        executed.append("read")
        return "read"

    registry = ToolRegistry()
    registry.register_function(
        read_probe,
        description="记录只读调用",
        capabilities={ToolCapability.READ},
    )
    runtime = build_runtime(
        agent_config=_agent_config(),
        context_input=_context_input(),
        provider=FakeProvider(
            [_tool_response(ToolUseBlock(id="call_read", name="read_probe", input={}))]
        ),
        tool_registry=registry,
        tool_view=registry.view(),
        tool_executor=ToolExecutor(registry),
        workspace_root=Path.cwd(),
    )

    result = await runtime.run_turn("读取")

    assert result.status is RuntimeStatus.OK
    assert executed == ["read"]
