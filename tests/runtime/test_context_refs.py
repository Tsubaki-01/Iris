"""模型回读引用使用原始位置，不随摘要插入而移动。"""

from iris.lifecycle import SessionCompaction
from iris.message import Msg, TextBlock, ToolResultBlock
from iris.runtime._compaction_summary import serialize_history
from iris.runtime._context_refs import with_context_refs
from iris.runtime.compaction import project_history


def test_refs_survive_compaction_without_mutating_raw_messages() -> None:
    """摘要覆盖前缀后，结果仍用原 message/block 下标。"""
    result = ToolResultBlock(
        tool_use_id="same",
        name="read",
        content="preview",
        metadata={"artifact": {"path": "/result.txt", "text_path": "/result.txt"}},
    )
    messages = [
        Msg.user("旧问题"),
        Msg.assistant("旧答案"),
        Msg(role="user", content=[TextBlock(text="note"), result]),
    ]
    projected = project_history(
        with_context_refs(messages),
        SessionCompaction(summary="旧轮摘要", covered_message_count=2),
        (),
    )
    assert "result:2:1" in projected[1].tool_results[0].content
    assert messages[2].tool_results[0].content == "preview"
    records = serialize_history(messages[2:], 2)
    assert "ref=message:2" in records[0].header
    assert "ref=result:2:1" in records[1].header
    assert records[1].text.startswith("preview")
