from __future__ import annotations

import pytest

from iris.exceptions import HITLCheckpointInvalidError
from iris.hitl import HumanInteraction, HumanInteractionRequest, QuestionPrompt, ToolCallSnapshot
from iris.message import LLMResponse, Msg, TextBlock, ToolUseBlock
from iris.runtime.checkpoint import build_hitl_checkpoint, validate_hitl_checkpoint
from iris.runtime.models import RuntimeOptions


def _build_checkpoint() -> dict[str, object]:
    tool_call = ToolUseBlock(id="call-1", name="ask_question", input={"question": "继续？"})
    response = LLMResponse(
        provider="fake",
        id="response-1",
        model="gpt-4o-mini",
        content=[TextBlock(text="需要确认"), tool_call],
        finish_reason="tool_calls",
    )
    return build_hitl_checkpoint(
        run_mode="turn",
        agent_name="agent-1",
        runtime_options=RuntimeOptions(session_id="session-1", run_id="run-1"),
        assistant_message=Msg.assistant(response.content),
        response=response,
        step_index=0,
        next_tool_index=0,
        batch_results=[],
        all_tool_results=[],
        read_state={"/workspace/file.txt": {"digest": "abc"}},
    )


def _interaction(checkpoint: dict[str, object]) -> HumanInteraction:
    return HumanInteraction(
        session_id="session-1",
        run_id="run-1",
        step_index=0,
        request=HumanInteractionRequest(
            tool_call=ToolCallSnapshot(
                tool_call_id="call-1",
                tool_name="ask_question",
                arguments={"question": "继续？"},
                workspace_root="/workspace",
                fingerprint="a" * 64,
            ),
            prompt=QuestionPrompt(question="继续？"),
        ),
        checkpoint=checkpoint,
    )


def test_checkpoint_v2_builds_and_validates_as_runtime_snapshot() -> None:
    checkpoint = _build_checkpoint()

    validated, options = validate_hitl_checkpoint(
        _interaction(checkpoint),
        agent_name="agent-1",
    )

    assert validated == checkpoint
    assert validated["checkpoint_version"] == 2
    assert validated["tool_calls"] == [
        {
            "type": "tool_use",
            "id": "call-1",
            "name": "ask_question",
            "input": {"question": "继续？"},
        }
    ]
    assert validated["read_state"] == {"/workspace/file.txt": {"digest": "abc"}}
    assert validated["continuation_claim"] is None
    assert options.session_id == "session-1"
    assert options.run_id == "run-1"


def test_validate_hitl_checkpoint_rejects_v1() -> None:
    checkpoint = _build_checkpoint()
    checkpoint["checkpoint_version"] = 1

    with pytest.raises(HITLCheckpointInvalidError, match="checkpoint"):
        validate_hitl_checkpoint(_interaction(checkpoint), agent_name="agent-1")


def test_build_hitl_checkpoint_rejects_non_json_data() -> None:
    tool_call = ToolUseBlock(id="call-1", name="ask_question", input={"question": "继续？"})
    response = LLMResponse(
        provider="fake",
        model="gpt-4o-mini",
        content=[tool_call],
        finish_reason="tool_calls",
    )

    with pytest.raises(HITLCheckpointInvalidError, match="JSON-safe"):
        build_hitl_checkpoint(
            run_mode="turn",
            agent_name="agent-1",
            runtime_options=RuntimeOptions(session_id="session-1", run_id="run-1"),
            assistant_message=Msg.assistant(response.content),
            response=response,
            step_index=0,
            next_tool_index=0,
            batch_results=[],
            all_tool_results=[],
            read_state={"not_json": object()},
        )
