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


def test_edit_patch_stays_in_complete_result_and_out_of_model_message() -> None:
    """宿主保留完整编辑 patch，模型消息只得到短摘要。"""
    file_change = {
        "file_path": "src/示例.py",
        "patch": "--- a/src/示例.py\n+++ b/src/示例.py\n@@ -1 +1 @@\n-旧值\n+新值\n",
    }
    result = ToolResult(
        tool_use_id="edit-1",
        tool_name="edit_file",
        content=[TextBlock(text="EDITED: src/示例.py")],
        data={"file_change": file_change},
    )

    restored = ToolResult.model_validate_json(result.model_dump_json())
    message = result.to_msg()

    assert restored.data == {"file_change": file_change}
    assert message.tool_results[0].content == "EDITED: src/示例.py"
    assert message.tool_results[0].metadata == {"tool_name": "edit_file"}
    assert "file_change" not in message.model_dump_json()
    assert "patch" not in message.model_dump_json()
