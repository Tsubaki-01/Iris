"""工具结果到消息的可信投影契约。"""

from pathlib import Path

import pytest

from iris.message import Msg, Role, TextBlock
from iris.tools import ToolArtifact, ToolErrorInfo, ToolResult


@pytest.mark.parametrize("failed", [False, True])
def test_tool_result_projection_preserves_output_and_metadata(failed: bool) -> None:
    result = ToolResult(
        tool_use_id="call-1",
        tool_name="read_file",
        content=[TextBlock(text="第一行"), TextBlock(text="第二行")],
        is_error=failed,
        error=ToolErrorInfo(code="READ_FAILED", message="读取失败") if failed else None,
        artifact=ToolArtifact(path=Path("result.txt"), size_bytes=10),
        stats={"duration_ms": 3},
        metadata={
            "trace_id": "trace-1",
            "permission": {"decision": "allow"},
            "custom": 7,
            "extra": {"nested": True},
            "tool_name": "不能覆盖真实名称",
        },
    )

    message = result.to_msg()

    assert message.role is Role.USER
    block = message.tool_results[0]
    assert block.tool_use_id == "call-1"
    assert block.name == "read_file"
    assert block.is_error is failed
    assert block.content == ("Error[READ_FAILED]: 读取失败" if failed else "第一行\n第二行")
    assert block.metadata == {
        "tool_name": "read_file",
        "artifact": result.artifact.model_dump(),
        "stats": {"duration_ms": 3},
        "trace_id": "trace-1",
        "permission": {"decision": "allow"},
        "extra": {"custom": 7, "nested": True},
        **({"error": result.error.model_dump()} if result.error is not None else {}),
    }
    restored = Msg.model_validate_json(message.model_dump_json())
    assert restored.tool_results[0].model_dump(mode="json") == block.model_dump(mode="json")


def test_raw_tool_result_message_still_normalizes_metadata() -> None:
    message = Msg.tool_result(
        tool_use_id="call-1", metadata={"trace_id": "t", "custom": 7, "extra": {"n": 1}}
    )
    assert message.tool_results[0].metadata == {"trace_id": "t", "extra": {"n": 1, "custom": 7}}
