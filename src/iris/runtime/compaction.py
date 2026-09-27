"""自动摘要的历史投影与完整消息组切点。

本模块只消费已归档原文和本次请求的装配函数，不读取持久化存储，
也不提交压缩状态。原文锚点保留在模型视图，摘要仅替换其覆盖的历史前缀。
"""

from __future__ import annotations

from collections.abc import Callable

from ..agents import CompactionConfig
from ..lifecycle import SessionCompaction, SessionContextSnapshot
from ..lifecycle.history import is_ordinary_user
from ..message import Msg
from ._request_measurement import MeasuredRequest


def project_history(
    snapshot: SessionContextSnapshot,
    compaction: SessionCompaction | None,
) -> list[Msg]:
    """构造摘要、已覆盖锚点和未覆盖原文组成的模型历史视图。"""
    if compaction is None:
        return list(snapshot.raw_tail)
    return _project_with_summary(
        snapshot,
        covered_count=compaction.covered_message_count,
        summary=compaction.summary,
    )


def select_compaction_end(
    *,
    snapshot: SessionContextSnapshot,
    config: CompactionConfig,
    build_request: Callable[[list[Msg]], MeasuredRequest],
) -> int | None:
    """选择新增摘要前缀的完整组边界，以 K 为近期原文保留目标。

    所有容量估算均使用调用方的完整请求，并给摘要正文预留 S。锚点占请求
    容量，但不占近期目标；超大组放不下时保留已经选中的较新完整组。
    S 只是生成上限，连空 suffix 的规划都超额时仍返回最靠后的合法边界，
    由 runtime 使用真实摘要检查最终请求大小。None 只表示没有新增切点。
    """
    previous_end = snapshot.compaction.covered_message_count if snapshot.compaction else 0
    message_count = snapshot.header.message_count
    first_compressible = next(
        (
            index
            for index in range(previous_end, message_count)
            if index not in snapshot.protected_indices
        ),
        None,
    )
    if first_compressible is None:
        return None
    ends = [
        previous_end + end
        for end in _history_group_ends(list(snapshot.raw_tail))
        if previous_end + end > first_compressible
    ]
    if not ends:
        return None

    def planned_input_tokens(end: int) -> int:
        history = _project_with_summary(
            snapshot,
            covered_count=end,
            summary="",
        )
        return build_request(history).input_tokens

    # 空摘要仍保留完整包装；实际正文的最大额度在容量判定时单独预留。
    base_tokens = planned_input_tokens(message_count)
    selected_end = ends[-1]
    selected_tokens = (
        base_tokens if selected_end == message_count else planned_input_tokens(selected_end)
    )
    if selected_tokens + config.summary_tokens > config.trigger_tokens:
        return selected_end

    for end in reversed(ends[:-1]):
        if selected_tokens - base_tokens >= config.keep_recent_tokens:
            break
        tokens = planned_input_tokens(end)
        if tokens + config.summary_tokens > config.trigger_tokens:
            break
        selected_end = end
        selected_tokens = tokens
    return selected_end


def _project_with_summary(
    snapshot: SessionContextSnapshot,
    *,
    covered_count: int,
    summary: str,
) -> list[Msg]:
    start = snapshot.compaction.covered_message_count if snapshot.compaction else 0
    prefix = dict(snapshot.protected_prefix_messages)
    anchors = [
        prefix[index] if index < start else snapshot.raw_tail[index - start]
        for index in snapshot.protected_indices
        if index < covered_count
    ]
    return [
        Msg.user(f"<summary>\n{summary}\n</summary>", sender="context"),
        *anchors,
        *snapshot.raw_tail[covered_count - start :],
    ]


def _history_group_ends(messages: list[Msg]) -> list[int]:
    """返回工具整批闭合且没有拆开 BCI/input 的前缀结束位置。"""
    ends: list[int] = []
    pending_calls: set[str] = set()
    for index, message in enumerate(messages):
        pending_calls.update(call.id for call in message.tool_calls)
        pending_calls.difference_update(result.tool_use_id for result in message.tool_results)
        if pending_calls:
            continue
        if (
            message.metadata.get("context_kind") == "before_current_input"
            and index + 1 < len(messages)
            and is_ordinary_user(messages[index + 1])
        ):
            continue
        ends.append(index + 1)
    return ends


__all__ = ["project_history", "select_compaction_end"]
