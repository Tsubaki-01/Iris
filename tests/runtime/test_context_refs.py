"""模型回读引用使用原始位置，不随摘要插入而移动。"""

from pathlib import Path

import pytest
from fakes import history_snapshot

from iris.lifecycle import SessionCompaction
from iris.message import ImageBlock, ImageFileRef, Msg, TextBlock, ToolResultBlock, ToolUseBlock
from iris.runtime._compaction_summary import serialize_history
from iris.runtime._context_refs import with_context_refs
from iris.runtime.compaction import project_history


def test_refs_survive_compaction_without_mutating_raw_messages() -> None:
    """摘要覆盖前缀后，结果仍用原 message/block 下标。"""
    result = ToolResultBlock(
        tool_use_id="same",
        name="read",
        content=[TextBlock(text="preview")],
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
    assert "result:2:1" in projected[1].tool_results[0].text
    assert projected[1].tool_results[0].content[0] == result.content[0]
    assert len(projected[1].tool_results[0].content) == 2
    assert messages[2].tool_results[0].text == "preview"
    records = serialize_history(messages[2:], 2)
    assert "ref=message:2" in records[0].header
    assert "ref=result:2:1" in records[1].header
    assert records[1].text.startswith("preview")


def test_refs_keep_absolute_indices_with_sparse_anchors_and_multiple_result_blocks() -> None:
    """前缀仅保留BCI/input，后缀结果仍按原始消息及block坐标生成引用。"""
    first = ToolResultBlock(
        tool_use_id="first",
        content=[TextBlock(text="first-preview")],
        metadata={"artifact": {"path": "/first"}},
    )
    second = ToolResultBlock(
        tool_use_id="second",
        content=[TextBlock(text="second-preview")],
        metadata={"artifact": {"path": "/second"}},
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
    assert "result:4:1" in actual[3].tool_results[0].text
    assert "result:4:2" in actual[3].tool_results[1].text
    assert snapshot.raw_tail[0].tool_results[0].text == "first-preview"
    assert snapshot.raw_tail[0].tool_results[1].text == "second-preview"


def test_empty_tail_ref_projection_keeps_prefix_only_history() -> None:
    """所有原文已覆盖时，回读投影无需任何未保留前缀。"""
    messages = [Msg.assistant("old"), Msg.user("input"), Msg.assistant("done")]
    compaction = SessionCompaction(summary="covered", covered_message_count=3)
    snapshot = history_snapshot(messages, initial_count=1, compaction=compaction)
    rendered = with_context_refs(snapshot)
    assert rendered.raw_tail == ()
    assert rendered.protected_prefix_messages == ((1, messages[1]),)
    assert project_history(rendered, compaction)[1:] == [messages[1]]


@pytest.mark.parametrize("covered", [4, 6])
def test_image_refs_keep_absolute_blocks_with_repeated_sparse_compaction(
    tmp_path: Path, covered: int
) -> None:
    """图片不会改变原 block 编号，重复压缩后稀疏锚点与多结果引用仍准确。"""
    ref = ImageFileRef(path=tmp_path / "plot.png", mime_type="image/png", width=40, height=20)
    image = ImageBlock(original=ref, model=ref)

    def results(first: str, second: str) -> Msg:
        return Msg(
            role="user",
            content=[
                image,
                TextBlock(text="note"),
                *(
                    ToolResultBlock(
                        tool_use_id=call_id,
                        content=[image, TextBlock(text=call_id)],
                        metadata={"artifact": {"path": "/raw.json"}},
                    )
                    for call_id in (first, second)
                ),
            ],
        )

    messages = [
        Msg.assistant("old"),
        Msg.user("BCI", sender="context", metadata={"context_kind": "before_current_input"}),
        Msg.user([image]),
        Msg.assistant([ToolUseBlock(id="a", name="read"), ToolUseBlock(id="b", name="read")]),
        results("a", "b"),
        Msg.user([image]),
        Msg.assistant([ToolUseBlock(id="c", name="read"), ToolUseBlock(id="d", name="read")]),
        results("c", "d"),
    ]
    before = [message.model_dump_json() for message in messages]
    compaction = SessionCompaction(summary="summary", covered_message_count=covered)
    projected = project_history(
        with_context_refs(history_snapshot(messages, initial_count=1, compaction=compaction)),
        compaction,
    )
    last = projected[-1]
    assert last.blocks[0] is image
    assert last.tool_results[0].content[0] is image
    assert last.tool_results[1].content[0] is image
    assert "result:7:2" in last.tool_results[0].text and "result:7:3" in last.tool_results[1].text
    assert projected[2].blocks[0] is image
    records = serialize_history(messages[covered:], covered)
    final_records = [record for record in records if "message=7 " in record.header]
    assert "message:7" in final_records[0].text
    assert "result:7:2" in final_records[2].text and "result:7:3" in final_records[3].text
    assert [message.model_dump_json() for message in messages] == before
