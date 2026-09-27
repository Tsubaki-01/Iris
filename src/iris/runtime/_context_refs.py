"""将原始消息位置投影为模型可回读引用，不修改 durable history。"""

from dataclasses import replace

from ..lifecycle import SessionContextSnapshot
from ..message import Msg, ToolResultBlock


def with_context_refs(snapshot: SessionContextSnapshot) -> SessionContextSnapshot:
    """按绝对位置只投影有效后缀与保护前缀，不遍历已覆盖的其它原文。"""
    start = snapshot.compaction.covered_message_count if snapshot.compaction else 0
    return replace(
        snapshot,
        raw_tail=tuple(
            _with_message_refs(message, index)
            for index, message in enumerate(snapshot.raw_tail, start=start)
        ),
        protected_prefix_messages=tuple(
            (index, _with_message_refs(message, index))
            for index, message in snapshot.protected_prefix_messages
        ),
    )


def _with_message_refs(message: Msg, message_index: int) -> Msg:
    blocks = message.blocks
    changed = False
    for block_index, block in enumerate(blocks):
        if isinstance(block, ToolResultBlock) and "artifact" in block.metadata:
            ref = f"result:{message_index}:{block_index}"
            blocks[block_index] = block.model_copy(
                update={
                    "content": (
                        f"{block.content}\n[历史原文：{ref}；"
                        "使用 context_read 分页读取 text 或 raw。]"
                    )
                }
            )
            changed = True
    return message.model_copy(update={"content": blocks}) if changed else message
