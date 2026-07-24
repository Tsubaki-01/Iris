from __future__ import annotations

from pathlib import Path

from iris.message import Role, TextBlock
from iris.runtime.tool_results import build_tool_result_event, build_tool_result_message
from iris.tools import ToolArtifact, ToolErrorInfo, ToolResult


def test_build_tool_result_message_preserves_success_metadata_and_artifact() -> None:
    result = ToolResult(
        tool_use_id="call_1",
        tool_name="write_note",
        content=[TextBlock(text="written")],
        artifact=ToolArtifact(
            path=Path("/tmp/result.txt"),
            mime_type="text/plain",
            size_bytes=7,
            preview="written",
        ),
        stats={"duration_ms": 3},
        metadata={"trace_id": "trace-1", "custom": "value"},
    )

    message = build_tool_result_message(result)

    assert message.role is Role.USER
    assert len(message.tool_results) == 1
    block = message.tool_results[0]
    assert block.tool_use_id == "call_1"
    assert block.name == "write_note"
    assert block.content == "written"
    assert block.is_error is False
    assert block.metadata == result.to_block_metadata()
    assert block.metadata["artifact"]["path"] == Path("/tmp/result.txt")


def test_build_tool_result_message_preserves_error_details() -> None:
    result = ToolResult(
        tool_use_id="call_1",
        tool_name="write_note",
        is_error=True,
        error=ToolErrorInfo(
            code="PERMISSION_ERROR",
            message="denied",
            details={"effect": "deny"},
        ),
    )

    message = build_tool_result_message(result)

    block = message.tool_results[0]
    assert block.content == "Error[PERMISSION_ERROR]: denied"
    assert block.is_error is True
    assert block.metadata["error"]["details"] == {"effect": "deny"}


def test_build_tool_result_event_returns_exact_json_safe_fields() -> None:
    result = ToolResult(
        tool_use_id="call_1",
        tool_name="write_note",
        content=[TextBlock(text="written")],
        artifact=ToolArtifact(
            path=Path("/tmp/result.txt"),
            mime_type="text/plain",
            size_bytes=7,
            preview="written",
        ),
    )

    event = build_tool_result_event(
        result,
        run_id="run_1",
        step_index=2,
        agent_id="agent_1",
        metadata={"trace_id": "trace-1"},
    )

    assert event == {
        "event_id": "tool_result:run_1:call_1",
        "type": "tool_result",
        "tool_call_id": "call_1",
        "tool_name": "write_note",
        "status": "ok",
        "error": None,
        "artifact": {
            "path": "/tmp/result.txt",
            "mime_type": "text/plain",
            "size_bytes": 7,
            "preview": "written",
        },
        "run_id": "run_1",
        "step_index": 2,
        "agent_id": "agent_1",
        "metadata": {"trace_id": "trace-1"},
    }
    assert "write_note" not in str(event["event_id"])


def test_build_tool_result_event_serializes_error() -> None:
    result = ToolResult(
        tool_use_id="call_1",
        tool_name="write_note",
        is_error=True,
        error=ToolErrorInfo(code="EXECUTION_ERROR", message="boom"),
    )

    event = build_tool_result_event(
        result,
        run_id="run_1",
        step_index=0,
        agent_id="agent_1",
        metadata=None,
    )

    assert event["status"] == "error"
    assert event["error"] == {
        "code": "EXECUTION_ERROR",
        "message": "boom",
        "retryable": False,
        "details": {},
    }
    assert event["artifact"] is None
    assert event["metadata"] == {}
