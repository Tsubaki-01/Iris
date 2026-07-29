from __future__ import annotations

from pathlib import Path

import pytest

import iris.runtime.resume as resume_module
from iris.exceptions import HITLCheckpointInvalidError
from iris.hitl import (
    HumanInteraction,
    HumanInteractionRequest,
    InMemoryInteractionStore,
    InteractionResumePhase,
    QuestionInteractionResponse,
    QuestionPrompt,
    ToolCallSnapshot,
)
from iris.hitl._legacy_service import HumanInteractionService
from iris.message import Msg, TextBlock, ToolUseBlock
from iris.runtime.models import ToolResultCommit
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


def _request(*, tool_name: str = "ask_question") -> HumanInteractionRequest:
    return HumanInteractionRequest(
        tool_call=ToolCallSnapshot(
            tool_call_id="call-1",
            tool_name=tool_name,
            arguments={"question": "继续？"},
            workspace_root="/workspace",
            fingerprint="a" * 64,
        ),
        prompt=QuestionPrompt(question="继续？"),
    )


def _ready_interaction(
    *,
    pending_result: ToolResult,
    all_tool_results: list[ToolResult],
    request_tool_name: str = "ask_question",
) -> tuple[InMemorySessionStore, HumanInteractionService, HumanInteraction]:
    session_store = InMemorySessionStore()
    service = HumanInteractionService(InMemoryInteractionStore())
    pending = service.create(
        _request(tool_name=request_tool_name),
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
    checkpoint = {
        **claimed.checkpoint,
        "pending_result": pending_result.model_dump(mode="json"),
        "all_tool_results": [result.model_dump(mode="json") for result in all_tool_results],
    }
    ready = service.update_consumed(
        claimed.interaction_id,
        InteractionResumePhase.RESULT_READY,
        checkpoint,
        expected_phase=claimed.resume_phase,
        expected_version=claimed.version,
    )
    return session_store, service, ready


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


def test_append_resumed_result_delegates_to_committer(monkeypatch: pytest.MonkeyPatch) -> None:
    session_store = InMemorySessionStore()
    result = ToolResult(
        tool_use_id="call-1",
        tool_name="ask_question",
        content=[TextBlock(text="继续")],
    )
    observed: dict[str, object] = {}
    message = Msg.tool_result(tool_use_id="call-1", content="继续", name="ask_question")

    def commit(**kwargs: object) -> ToolResultCommit:
        observed.update(kwargs)
        return ToolResultCommit(results=[result], messages=[message])

    monkeypatch.setattr(resume_module, "commit_tool_results", commit)

    actual = append_resumed_result(
        result=result,
        session_store=session_store,
        session_id="session-1",
        run_id="run-1",
        step_index=0,
        agent_id="agent-1",
    )

    assert actual == message
    assert observed["results"] == [result]
    assert observed["deduplicate_messages"] is False


@pytest.mark.asyncio
async def test_commit_ready_interaction_delegates_to_idempotent_committer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = ToolResult(
        tool_use_id="call-1",
        tool_name="ask_question",
        content=[TextBlock(text="继续")],
    )
    session_store, service, ready = _ready_interaction(
        pending_result=result,
        all_tool_results=[result],
    )
    observed: dict[str, object] = {}

    def commit(**kwargs: object) -> ToolResultCommit:
        observed.update(kwargs)
        return ToolResultCommit(
            results=[result],
            messages=[Msg.tool_result(tool_use_id="call-1", content="继续")],
        )

    monkeypatch.setattr(resume_module, "commit_tool_results", commit)

    committed = await commit_ready_interaction(
        interaction=ready,
        session_store=session_store,
        interaction_service=service,
        agent_id="agent-1",
    )

    assert committed.tool_results == [result]
    assert observed["results"] == [result]
    assert observed["deduplicate_messages"] is True
    assert service.get(ready.interaction_id).resume_phase is InteractionResumePhase.RESULT_COMMITTED


@pytest.mark.asyncio
async def test_commit_ready_interaction_advances_phase_and_persists_result() -> None:
    result = ToolResult(
        tool_use_id="call-1",
        tool_name="ask_question",
        content=[TextBlock(text="继续")],
    )
    session_store, service, ready = _ready_interaction(
        pending_result=result,
        all_tool_results=[result],
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


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["wrong_id", "missing", "duplicate", "payload_mismatch"])
async def test_commit_ready_rejects_inconsistent_result_identity_before_writes(
    case: str,
) -> None:
    expected = ToolResult(
        tool_use_id="call-1",
        tool_name="ask_question",
        content=[TextBlock(text="继续")],
    )
    pending_result = expected
    all_results = [expected]
    if case == "wrong_id":
        pending_result = expected.model_copy(update={"tool_use_id": "call-2"})
        all_results = [pending_result]
    elif case == "missing":
        all_results = []
    elif case == "duplicate":
        all_results = [expected, expected]
    elif case == "payload_mismatch":
        all_results = [expected.model_copy(update={"content": [TextBlock(text="不同")]})]

    session_store, service, ready = _ready_interaction(
        pending_result=pending_result,
        all_tool_results=all_results,
    )

    with pytest.raises(HITLCheckpointInvalidError):
        await commit_ready_interaction(
            interaction=ready,
            session_store=session_store,
            interaction_service=service,
            agent_id="agent-1",
        )

    assert session_store.load_messages("session-1") == []
    assert session_store.load_tool_events("session-1") == []
    assert service.get(ready.interaction_id).resume_phase is InteractionResumePhase.RESULT_READY


@pytest.mark.asyncio
async def test_commit_ready_accepts_alias_request_with_canonical_result_name() -> None:
    result = ToolResult(
        tool_use_id="call-1",
        tool_name="canonical_question",
        content=[TextBlock(text="继续")],
    )
    session_store, service, ready = _ready_interaction(
        pending_result=result,
        all_tool_results=[result],
        request_tool_name="question_alias",
    )

    committed = await commit_ready_interaction(
        interaction=ready,
        session_store=session_store,
        interaction_service=service,
        agent_id="agent-1",
    )

    assert committed.tool_results == [result]
