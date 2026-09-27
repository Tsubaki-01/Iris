"""模型回读引用使用原始位置，不随摘要插入而移动。"""

from fakes import history_snapshot

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
    compaction = SessionCompaction(summary="旧轮摘要", covered_message_count=2)
    snapshot = history_snapshot(messages, initial_count=len(messages), compaction=compaction)
    projected = project_history(with_context_refs(snapshot), compaction)
    assert "result:2:1" in projected[1].tool_results[0].content
    assert messages[2].tool_results[0].content == "preview"
    records = serialize_history(messages[2:], 2)
    assert "ref=message:2" in records[0].header
    assert "ref=result:2:1" in records[1].header
    assert records[1].text.startswith("preview")


def test_refs_keep_absolute_indices_with_sparse_anchors_and_multiple_result_blocks() -> None:
    """前缀仅保留BCI/input，后缀结果仍按原始消息及block坐标生成引用。"""
    first = ToolResultBlock(
        tool_use_id="first", content="first-preview", metadata={"artifact": {"path": "/first"}}
    )
    second = ToolResultBlock(
        tool_use_id="second", content="second-preview", metadata={"artifact": {"path": "/second"}}
    )
    messages = [
        Msg.assistant("old"),
        Msg.user("BCI", sender="context", metadata={"context_kind": "before_current_input"}),
        Msg.user("current input"),
        Msg.assistant("work"),
        Msg(role="user", content=[TextBlock(text="note"), first, second]),
        Msg.user("latest steer"),
    ]
    compaction = SessionCompaction(summary="old work", covered_message_count=4)
    snapshot = history_snapshot(messages, initial_count=1, compaction=compaction)
    rendered = with_context_refs(snapshot)
    assert rendered.header is snapshot.header
    assert rendered.protected_prefix_messages == ((1, messages[1]), (2, messages[2]))
    assert rendered.raw_tail[0] is not snapshot.raw_tail[0]
    assert rendered.raw_tail[1] is snapshot.raw_tail[1]
    actual = project_history(rendered, compaction)
    assert actual[1:3] == [messages[1], messages[2]]
    assert "result:4:1" in actual[3].tool_results[0].content
    assert "result:4:2" in actual[3].tool_results[1].content
    assert snapshot.raw_tail[0].tool_results[0].content == "first-preview"
    assert snapshot.raw_tail[0].tool_results[1].content == "second-preview"


def test_empty_tail_ref_projection_keeps_prefix_only_history() -> None:
    """所有原文已覆盖时，回读投影无需任何未保留前缀。"""
    messages = [Msg.assistant("old"), Msg.user("input"), Msg.assistant("done")]
    compaction = SessionCompaction(summary="covered", covered_message_count=3)
    snapshot = history_snapshot(messages, initial_count=1, compaction=compaction)
    rendered = with_context_refs(snapshot)
    assert rendered.raw_tail == ()
    assert rendered.protected_prefix_messages == ((1, messages[1]),)
    assert project_history(rendered, compaction)[1:] == [messages[1]]
