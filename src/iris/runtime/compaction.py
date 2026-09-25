"""自动摘要的历史投影与完整消息组切点。

本模块只消费已归档原文和本次请求的装配函数，不读取持久化存储，
也不提交压缩状态。原文锚点保留在模型视图，摘要仅替换其覆盖的历史前缀。
"""

from __future__ import annotations

from collections.abc import Callable

from ..agents import CompactionConfig
from ..lifecycle import SessionCompaction
from ..message import LLMRequest, Msg, Role


def protected_message_indices(
    messages: list[Msg], initial_session_message_count: int
) -> tuple[int, ...]:
    """定位本 run 已归档的 BCI、原始 input 和最新普通用户 steer。

    输入阶段按 BCI（可选）、input 一次归档；run 起点之前的 context 不属于
    当前任务。工具结果虽然也使用 user role，但不能被当作 steer。
    """
    if initial_session_message_count == len(messages):
        return ()

    original_input = initial_session_message_count
    protected: set[int] = set()
    if messages[original_input].metadata.get("context_kind") == "before_current_input":
        protected.add(original_input)
        original_input += 1
    protected.add(original_input)

    latest_steer = next(
        (
            index
            for index in range(len(messages) - 1, original_input, -1)
            if _is_ordinary_user(messages[index])
        ),
        original_input,
    )
    protected.add(latest_steer)
    return tuple(sorted(protected))


def project_history(
    messages: list[Msg],
    compaction: SessionCompaction | None,
    protected_indices: tuple[int, ...],
) -> list[Msg]:
    """构造摘要、已覆盖锚点和未覆盖原文组成的模型历史视图。"""
    if compaction is None:
        return list(messages)
    return _project_with_summary(
        messages,
        covered_count=compaction.covered_message_count,
        protected_indices=protected_indices,
        summary=compaction.summary,
    )


def select_compaction_end(
    *,
    messages: list[Msg],
    previous_compaction: SessionCompaction | None,
    protected_indices: tuple[int, ...],
    config: CompactionConfig,
    build_request: Callable[[list[Msg]], LLMRequest],
    estimate_input_tokens: Callable[[LLMRequest], int],
) -> int | None:
    """选择新增摘要前缀的完整组边界，以 K 为近期原文保留目标。

    所有容量估算均使用调用方的完整请求，并给摘要正文预留 S。锚点占请求
    容量，但不占近期目标；超大组放不下时保留已经选中的较新完整组。
    S 只是生成上限，连空 suffix 的规划都超额时仍返回最靠后的合法边界，
    由 runtime 使用真实摘要检查最终请求大小。None 只表示没有新增切点。
    """
    previous_end = previous_compaction.covered_message_count if previous_compaction else 0
    first_compressible = next(
        (index for index in range(previous_end, len(messages)) if index not in protected_indices),
        None,
    )
    if first_compressible is None:
        return None
    ends = [end for end in _history_group_ends(messages) if end > first_compressible]
    if not ends:
        return None

    def planned_input_tokens(end: int) -> int:
        history = _project_with_summary(
            messages,
            covered_count=end,
            protected_indices=protected_indices,
            summary="",
        )
        return estimate_input_tokens(build_request(history))

    # 空摘要仍保留完整包装；实际正文的最大额度在容量判定时单独预留。
    base_tokens = planned_input_tokens(len(messages))
    selected_end = ends[-1]
    selected_tokens = (
        base_tokens if selected_end == len(messages) else planned_input_tokens(selected_end)
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
    messages: list[Msg],
    *,
    covered_count: int,
    protected_indices: tuple[int, ...],
    summary: str,
) -> list[Msg]:
    return [
        Msg.user(f"<summary>\n{summary}\n</summary>", sender="context"),
        *(messages[index] for index in protected_indices if index < covered_count),
        *messages[covered_count:],
    ]


def _is_ordinary_user(message: Msg) -> bool:
    return message.role == Role.USER and message.sender != "context" and not message.tool_results


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
            and _is_ordinary_user(messages[index + 1])
        ):
            continue
        ends.append(index + 1)
    return ends


__all__ = ["project_history", "protected_message_indices", "select_compaction_end"]
