from __future__ import annotations

from pathlib import Path

import pytest

from iris.hitl import (
    HumanInteractionRequest,
    HumanInteractionService,
    InMemoryInteractionStore,
    InteractionResumePhase,
    QuestionInteractionResponse,
    QuestionPrompt,
    ToolCallSnapshot,
)
from iris.message import TextBlock, ToolUseBlock
from iris.runtime.resume import (
    append_resumed_result,
    commit_ready_interaction,
    load_resumable_interaction,
    resolve_interaction_result,
)
from iris.session import InMemorySessionStore
from iris.tools import (
    PreparedToolCall,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
)


def _request() -> HumanInteractionRequest:
    return HumanInteractionRequest(
        tool_call=ToolCallSnapshot(
            tool_call_id="call-1",
            tool_name="ask_question",
            arguments={"question": "继续？"},
            workspace_root="/workspace",
            fingerprint="a" * 64,
        ),
        prompt=QuestionPrompt(question="继续？"),
    )


def test_load_resumable_interaction_uses_waiting_marker() -> None:
    session_store = InMemorySessionStore()
    service = HumanInteractionService(InMemoryInteractionStore())
    interaction = service.create(
        _request(),
        session_id="session-1",
        run_id="run-1",
        step_index=0,
        checkpoint={"checkpoint_version": 2},
    )
    session_store.save_run_metadata(
        "session-1",
        {
            "latest_run": {
                "waiting_human": True,
                "interaction_id": interaction.interaction_id,
            }
        },
    )

    loaded = load_resumable_interaction(
        session_store=session_store,
        interaction_service=service,
        session_id="session-1",
    )

    assert loaded == interaction


@pytest.mark.asyncio
async def test_resolve_question_interaction_projects_answer_to_tool_result() -> None:
    service = HumanInteractionService(InMemoryInteractionStore())
    pending = service.create(
        _request(),
        session_id="session-1",
        run_id="run-1",
        step_index=0,
        checkpoint={"checkpoint_version": 2},
    )
    interaction = service.resolve(
        pending.interaction_id,
        QuestionInteractionResponse(answer="继续"),
    )

    result = await resolve_interaction_result(
        interaction=interaction,
        prepared=PreparedToolCall(
            tool_use=ToolUseBlock(id="call-1", name="ask_question", input={"question": "继续？"})
        ),
        tool_executor=ToolExecutor(ToolRegistry()),
        tool_context=ToolExecutionContext(workspace_root=Path("/workspace")),
    )

    assert result.tool_use_id == "call-1"
    assert result.tool_name == "ask_question"
    assert result.content == [TextBlock(text="继续")]


def test_append_resumed_result_writes_message_and_event() -> None:
    session_store = InMemorySessionStore()
    result = ToolResult(
        tool_use_id="call-1",
        tool_name="ask_question",
        content=[TextBlock(text="继续")],
    )

    message = append_resumed_result(
        result=result,
        session_store=session_store,
        session_id="session-1",
        run_id="run-1",
        step_index=0,
        agent_id="agent-1",
    )

    assert message.tool_results[0].content == "继续"
    assert session_store.load_messages("session-1") == [message.model_dump(mode="json")]
    assert session_store.load_tool_events("session-1")[0]["event_id"] == (
        "tool_result:run-1:call-1"
    )


@pytest.mark.asyncio
async def test_commit_ready_interaction_advances_phase_and_persists_result() -> None:
    session_store = InMemorySessionStore()
    service = HumanInteractionService(InMemoryInteractionStore())
    result = ToolResult(
        tool_use_id="call-1",
        tool_name="ask_question",
        content=[TextBlock(text="继续")],
    )
    pending = service.create(
        _request(),
        session_id="session-1",
        run_id="run-1",
        step_index=0,
        checkpoint={"checkpoint_version": 2},
    )
    resolved = service.resolve(
        pending.interaction_id,
        QuestionInteractionResponse(answer="继续"),
    )
    claimed = service.claim(resolved.interaction_id, resolved.checkpoint)
    ready_checkpoint = {
        **claimed.checkpoint,
        "pending_result": result.model_dump(mode="json"),
        "all_tool_results": [result.model_dump(mode="json")],
    }
    ready = service.update_consumed(
        claimed.interaction_id,
        InteractionResumePhase.RESULT_READY,
        ready_checkpoint,
    )

    committed = await commit_ready_interaction(
        interaction=ready,
        session_store=session_store,
        interaction_service=service,
        agent_id="agent-1",
    )

    assert committed.tool_results == [result]
    assert service.get(ready.interaction_id).resume_phase is InteractionResumePhase.RESULT_COMMITTED
    assert len(session_store.load_messages("session-1")) == 1
    assert len(session_store.load_tool_events("session-1")) == 1
