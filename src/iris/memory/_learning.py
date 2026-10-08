"""原文学习的有效证据判定，供捕获短事实和 flush 选材共用。"""

from .models import MemoryEpisode, MemoryRecord


def has_learning_text(record: MemoryRecord, start: int = 0) -> bool:
    """沿用 flush refs 的非空正文与 evidence_allowed 条件。"""
    return len(record.text) > start and bool(record.metadata.get("evidence_allowed", True))


def has_remaining_content(
    episode: MemoryEpisode, record_index: int = 0, text_offset: int = 0
) -> bool:
    """判断当前字符游标之后是否还存在能进入模型提炼的证据。"""
    return any(
        has_learning_text(record, text_offset if index == record_index else 0)
        for index, record in enumerate(episode.records[record_index:], record_index)
    )
