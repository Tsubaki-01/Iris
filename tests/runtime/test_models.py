from __future__ import annotations

import pytest
from pydantic import ValidationError

from iris.agents import AgentConfig
from iris.context import ContextBuilder, ContextBuildInput, ContextSection, ContextSlot
from iris.message import Msg
from iris.runtime.assembler import RuntimeMessageAssembler
from iris.runtime.models import (
    BoundedLoopOptions,
    Runstate,
    RuntimeErrorInfo,
    RuntimeOptions,
    RuntimeStatus,
    RuntimeTurnInput,
    RuntimeTurnResult,
    ToolBridgeResult,
    ToolErrorPolicy,
)


def _agent_config() -> AgentConfig:
    return AgentConfig(
        name="runtime-agent",
        model={"provider": "openai", "name": "gpt-4o-mini"},
        system="你是本地助手。",
    )


def _context_input() -> ContextBuildInput:
    return ContextBuildInput(
        system=ContextSection(slots=[ContextSlot(name="instructions", content="遵守用户指令")])
    )


def test_runtime_options_defaults_are_stable_and_isolated() -> None:
    first = RuntimeOptions()
    second = RuntimeOptions()

    assert first.session_id == "default"
    assert first.run_id
    assert second.run_id
    assert first.run_id != second.run_id
    assert first.include_tools is True
    assert first.request_options == {}
    assert first.metadata == {}
    assert first.memory_query is None
    assert first.memory_results is None
    assert first.memory_max_chars == 4000
    assert first.loop == BoundedLoopOptions()
    assert first.loop.max_steps == 20
    assert first.loop.tool_error_policy == ToolErrorPolicy.RETURN_TO_MODEL

    first.request_options["temperature"] = 0.2
    first.metadata["trace_id"] = "trace-1"

    assert second.request_options == {}
    assert second.metadata == {}


def test_runtime_options_reject_invalid_positive_fields() -> None:
    with pytest.raises(ValidationError):
        BoundedLoopOptions(max_steps=0)

    with pytest.raises(ValidationError):
        RuntimeOptions(memory_max_chars=0)


def test_runstate_rejects_negative_step_index() -> None:
    context_output = ContextBuilder().build(_context_input())
    current_input = Msg.user("当前问题")
    conversation = RuntimeMessageAssembler().build_conversation(
        context_output=context_output,
        history=[],
        current_input=current_input,
    )
    request = conversation.to_llm_request("gpt-4o-mini")

    with pytest.raises(ValidationError):
        Runstate(
            session_id="session-1",
            run_id="run-1",
            step_index=-1,
            context_output=context_output,
            history=[],
            current_input=current_input,
            conversation=conversation,
            request=request,
        )


def test_runstate_can_hold_current_assembler_output() -> None:
    history = [Msg.user("历史问题"), Msg.assistant("历史回答")]
    current_input = Msg.user("当前问题")
    context_output = ContextBuilder().build(_context_input())
    assembler = RuntimeMessageAssembler()

    conversation = assembler.build_conversation(
        context_output=context_output,
        history=history,
        current_input=current_input,
    )
    request = assembler.build_request(
        agent_config=_agent_config(),
        context_output=context_output,
        history=history,
        current_input=current_input,
    )

    runstate = Runstate(
        session_id="session-1",
        run_id="run-1",
        step_index=0,
        context_output=context_output,
        history=history,
        current_input=current_input,
        conversation=conversation,
        tools_schema=[{"name": "search"}],
        request=request,
        metadata={"trace_id": "trace-1"},
    )

    assert runstate.context_output == context_output
    assert runstate.conversation == conversation
    assert runstate.request == request
    assert runstate.tools_schema == [{"name": "search"}]
    assert runstate.metadata == {"trace_id": "trace-1"}


def test_turn_and_error_models_accept_minimal_runtime_data() -> None:
    turn_input = RuntimeTurnInput(user_input="你好", metadata={"source": "test"})
    error = RuntimeErrorInfo(
        code="PROVIDER_ERROR",
        message="provider failed",
        source="provider",
        details={"status_code": 500},
    )
    bridge_result = ToolBridgeResult(
        messages=[Msg.tool_result(tool_use_id="tool-1", content="done")],
        events=[{"tool_use_id": "tool-1", "status": "ok"}],
    )
    turn_result = RuntimeTurnResult(
        session_id="session-1",
        run_id="run-1",
        status=RuntimeStatus.ERROR,
        steps=1,
        error=error,
        tool_result_messages=bridge_result.messages,
    )

    assert turn_input.user_input == "你好"
    assert bridge_result.results == []
    assert bridge_result.messages[0].text == ""
    assert bridge_result.events == [{"tool_use_id": "tool-1", "status": "ok"}]
    assert turn_result.status == RuntimeStatus.ERROR
    assert turn_result.error == error


def test_turn_result_rejects_non_positive_steps() -> None:
    with pytest.raises(ValidationError):
        RuntimeTurnResult(
            session_id="session-1",
            run_id="run-1",
            status=RuntimeStatus.OK,
            steps=0,
        )
