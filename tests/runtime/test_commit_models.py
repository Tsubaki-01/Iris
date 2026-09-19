"""进程内提交事实直接消费已经验证的 cursor。"""

from __future__ import annotations

import sys
from types import FrameType

import pytest
from pydantic import ValidationError

from iris.lifecycle import RuntimeExecutionOptions
from iris.message import Msg, ToolUseBlock
from iris.runtime import (
    ModelStepReservation,
    RuntimeActivationInput,
    RuntimeCursor,
    RuntimeModelStepCommit,
    RuntimeRunInputCommit,
    RuntimeToolCall,
    RuntimeToolResultCommit,
)
from iris.tools import ToolResult


@pytest.mark.parametrize("memory_results", [[], [{"item": {"id": "memory-1"}}]])
def test_runtime_options_reject_query_and_results_together(
    memory_results: list[dict[str, object]],
) -> None:
    """显式空结果也不能与显式查询同时提供。"""
    with pytest.raises(ValidationError, match="memory_query.*memory_results"):
        RuntimeExecutionOptions(memory_query={"text": "query"}, memory_results=memory_results)


def test_runtime_options_keep_explicit_empty_results() -> None:
    """空结果快照表示调用方已明确提供零条记忆。"""
    options = RuntimeExecutionOptions(memory_results=[])
    assert options.memory_results == []
    assert options.memory_query is None


@pytest.mark.parametrize("kind", ["input", "reservation", "model", "tool"])
def test_commit_facts_do_not_rescan_validated_tool_batch(kind: str) -> None:
    """提交边界不能随每个结果再次扫描整批调用及已提交前缀。"""
    calls = tuple(ToolUseBlock(id=f"call-{index}", name="echo") for index in range(8))
    assistant = Msg.assistant(list(calls))
    cursor = RuntimeCursor(
        position="tool_batch", step_index=0, tool_calls=calls, assistant_message=assistant
    )
    result = ToolResult(tool_use_id="call-0", tool_name="echo", content=[])
    after = cursor.model_copy(update={"next_tool_index": 1, "tool_results": (result,)})
    call = RuntimeToolCall(
        run_id="run",
        activation_id="activation",
        step_index=0,
        ordinal=1,
        tool_call_id="call-0",
        tool_name="echo",
        arguments={},
        fingerprint="0" * 64,
    )
    before_model = RuntimeCursor(position="before_model", step_index=0)
    before_input = RuntimeCursor(position="before_input", step_index=0)
    scans = 0
    validator_code = RuntimeCursor._validate_position.__code__

    def count_scan(frame: FrameType, event: str, arg: object) -> None:
        nonlocal scans
        if event == "call" and frame.f_code is validator_code:
            scans += 1

    previous_profile = sys.getprofile()
    sys.setprofile(count_scan)
    try:
        if kind == "input":
            RuntimeRunInputCommit(
                cursor_before=before_input,
                message_delta=(Msg.user("input"),),
                cursor_after=before_model,
            )
        elif kind == "reservation":
            ModelStepReservation(granted=True, step_index=0, cursor=before_model)
        elif kind == "model":
            RuntimeModelStepCommit(
                cursor_before=before_model, assistant_message=assistant, cursor_after=cursor
            )
        else:
            RuntimeToolResultCommit(
                tool_call=call, result=result, message_delta=(), cursor_after=after
            )
    finally:
        sys.setprofile(previous_profile)

    assert scans == 0


def test_cursor_raw_recovery_still_rejects_duplicate_call_identity() -> None:
    """减少内部扫描不改变 checkpoint raw parsing 的完整校验。"""
    with pytest.raises(ValidationError, match="tool call ID 不能重复"):
        RuntimeCursor.model_validate(
            {
                "position": "tool_batch",
                "step_index": 0,
                "assistant_message": {"role": "assistant", "content": "tools"},
                "tool_calls": [{"id": "same", "name": "echo"}, {"id": "same", "name": "echo"}],
            }
        )


@pytest.mark.parametrize("kind", ["start", "resume", "recover"])
def test_all_activation_kinds_carry_original_input_and_history_start(kind: str) -> None:
    """每次 activation 保留原 run 锚点，归档由 before_input 位置决定。"""
    activation = RuntimeActivationInput(
        run_id="run",
        activation_id="activation",
        session_id="session",
        kind=kind,
        run_input="原始请求",
        initial_session_message_count=4,
        cursor=RuntimeCursor(position="before_input", step_index=0),
        options=RuntimeExecutionOptions(),
    )
    assert activation.run_input == "原始请求"
    assert activation.initial_session_message_count == 4
