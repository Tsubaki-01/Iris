"""完整请求压力下的确定性工具正文投影，不改变已提交历史。"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ..agents import ContextPolicyConfig
from ..context import ContextSnapshot
from ..context.source import render_context_snapshot
from ..message import LLMRequest, ToolResultBlock
from .compaction import _history_group_ends


@dataclass(frozen=True, slots=True)
class _Observation:
    """当前请求中的一个完整观察结果及其原文位置。"""

    position: tuple[int, int]
    ref: str
    block: ToolResultBlock
    key: tuple[str, str, str] | None


def project_context_request(
    request: LLMRequest,
    *,
    source_indices: Mapping[int, int],
    config: ContextPolicyConfig,
    trigger_tokens: int,
    estimate_input_tokens: Callable[[LLMRequest], int],
    snapshot: ContextSnapshot | None = None,
    select_optional: bool = False,
) -> tuple[LLMRequest, ContextSnapshot | None]:
    """装配快照后依次折叠重复、选择可选材料、短化旧观察。

    Args:
        request: 已包含实际 context、消息与工具 schema 的请求。
        source_indices: 本步骤原模型历史对象 identity 到原始消息下标的映射。
        config: 已校验的上下文保留策略。
        trigger_tokens: 既有 compaction 的压力线。
        estimate_input_tokens: 当前 provider 的完整请求计量器。
        snapshot: 本步骤已采集的完整快照或已冻结的选择。
        select_optional: 仅首次装配允许选择；后续候选沿用冻结材料。

    Returns:
        写时复制的完整请求与本步骤选定快照；不修改原历史。
    """
    if snapshot is not None:
        request = request.model_copy(
            update={"messages": [*request.messages, render_context_snapshot(snapshot)]}
        )
    tokens = estimate_input_tokens(request)
    if tokens < trigger_tokens:
        return request, snapshot
    groups = (
        _closed_observations(request, source_indices)
        if config.enabled and _can_read_context(request)
        else []
    )
    recent = config.preserve_recent_tool_groups
    older = [item for group in (groups[:-recent] if recent else groups) for item in group]
    latest = {item.key: item for group in groups for item in group if item.key is not None}
    replacements: dict[tuple[int, int], str] = {}
    representatives: set[tuple[int, int]] = set()
    for item in older:
        if item.key is None:
            continue
        representative = latest[item.key]
        if representative.position == item.position:
            continue
        text = (
            f"本次调用成功。重复正文见 {representative.ref}；"
            f"可用 context_read 读取本次原文 {item.ref}。"
        )
        if len(text) < len(item.block.content):
            replacements[item.position] = text
            representatives.add(representative.position)
    if replacements:
        candidate = _replace_contents(request, replacements)
        candidate_tokens = estimate_input_tokens(candidate)
        if candidate_tokens < tokens:
            request, tokens = candidate, candidate_tokens
        else:
            replacements.clear()
            representatives.clear()
    if select_optional and snapshot is not None and tokens >= trigger_tokens:
        optional = sorted(
            (
                (index, item)
                for index, item in enumerate(snapshot.contributions)
                if not item.required
            ),
            key=lambda entry: (entry[1].priority, -entry[0]),
        )
        for _, item in optional:
            snapshot = ContextSnapshot(
                tuple(
                    contribution
                    for contribution in snapshot.contributions
                    if contribution.key != item.key
                )
            )
            request = request.model_copy(
                update={
                    "messages": [
                        *request.messages[:-1],
                        render_context_snapshot(snapshot),
                    ]
                }
            )
            tokens = estimate_input_tokens(request)
            if tokens < trigger_tokens:
                break
    if tokens < trigger_tokens:
        return request, snapshot
    preview_chars = config.old_result_preview_chars
    for item in older:
        if item.position in replacements or item.position in representatives:
            continue
        content = item.block.content
        if len(content) <= preview_chars:
            continue
        head = preview_chars * 3 // 4
        tail = preview_chars - head
        preview = content[:head] + (content[-tail:] if tail else "")
        text = (
            "[本次工具调用成功；历史正文已移出当前窗口]\n"
            f"工具：{item.block.name}\n原文：{item.ref}（context_read 可分页读取）"
        )
        if preview_chars:
            text += f"\n预览：{preview}"
        if len(text) >= len(content):
            continue
        candidate = _replace_contents(request, {item.position: text})
        candidate_tokens = estimate_input_tokens(candidate)
        if candidate_tokens < tokens:
            request, tokens = candidate, candidate_tokens
            if tokens < trigger_tokens:
                break
    return request, snapshot


def _can_read_context(request: LLMRequest) -> bool:
    choice = request.tool_choice
    if choice == "none":
        return False
    if isinstance(choice, dict) and choice.get("function", {}).get("name") != "context_read":
        return False
    return any(tool["function"]["name"] == "context_read" for tool in request.tools)


def _closed_observations(
    request: LLMRequest, source_indices: Mapping[int, int]
) -> list[list[_Observation]]:
    groups: list[list[_Observation]] = []
    start = 0
    for end in _history_group_ends(request.messages):
        messages = request.messages[start:end]
        calls = {call.id: call for message in messages for call in message.tool_calls}
        if calls:
            observations: list[_Observation] = []
            for message_index in range(start, end):
                message = request.messages[message_index]
                for block_index, block in enumerate(message.blocks):
                    if not isinstance(block, ToolResultBlock) or block.is_error:
                        continue
                    metadata = block.metadata.get("extra", {})
                    if metadata.get("context_retention") != "observation":
                        continue
                    call = calls[block.tool_use_id]
                    key = (
                        None
                        if "artifact" in block.metadata
                        else (
                            metadata["context_tool_name"],
                            json.dumps(
                                call.input,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            block.content,
                        )
                    )
                    observations.append(
                        _Observation(
                            (message_index, block_index),
                            f"result:{source_indices[id(message)]}:{block_index}",
                            block,
                            key,
                        )
                    )
            groups.append(observations)
        start = end
    return groups


def _replace_contents(
    request: LLMRequest, replacements: Mapping[tuple[int, int], str]
) -> LLMRequest:
    messages = list(request.messages)
    for (message_index, block_index), content in replacements.items():
        message = messages[message_index]
        blocks = message.blocks
        blocks[block_index] = blocks[block_index].model_copy(update={"content": content})
        messages[message_index] = message.model_copy(update={"content": blocks})
    return request.model_copy(update={"messages": messages})
