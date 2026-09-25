"""将原始消息位置投影为模型可回读引用，不修改 durable history。"""

from ..message import Msg, ToolResultBlock


def with_context_refs(messages: list[Msg]) -> list[Msg]:
    """只给已外置工具结果追加其原始位置，返回写时复制的模型视图。"""
    projected: list[Msg] = []
    for message_index, message in enumerate(messages):
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
        projected.append(message.model_copy(update={"content": blocks}) if changed else message)
    return projected
