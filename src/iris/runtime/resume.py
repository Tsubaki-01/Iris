"""Runtime HITL 恢复的独立发现、解析与提交辅助逻辑。"""

from __future__ import annotations

from ..exceptions import (
    HITLCheckpointInvalidError,
    HITLNotFoundError,
)
from ..hitl import (
    HumanInteraction,
    HumanInteractionService,
    InteractionResumePhase,
    InteractionStatus,
    PermissionInteractionResponse,
    QuestionInteractionResponse,
)
from ..message import Msg, TextBlock
from ..session import SessionStore
from ..tools import (
    PreparedToolCall,
    ToolErrorInfo,
    ToolExecutionContext,
    ToolExecutor,
    ToolResult,
)
from .models import RuntimeStatus, RuntimeTurnResult
from .tool_results import build_tool_result_event, build_tool_result_message


def load_resumable_interaction(
    *,
    session_store: SessionStore,
    interaction_service: HumanInteractionService,
    session_id: str,
) -> HumanInteraction | None:
    """读取当前 session 可安全恢复的 interaction，不修改持久化状态。"""
    metadata = session_store.load_run_metadata(session_id)
    pending = interaction_service.list_pending(session_id)
    if len(pending) > 1:
        raise HITLCheckpointInvalidError(
            "同一 session 存在多个 pending HITL interaction",
            session_id=session_id,
        )
    fallback = pending[0] if pending else None
    latest = metadata.get("latest_run")
    if not isinstance(latest, dict) or latest.get("waiting_human") is not True:
        return fallback

    interaction_id = latest.get("interaction_id")
    if not isinstance(interaction_id, str) or not interaction_id.strip():
        if fallback is not None:
            return fallback
        raise HITLCheckpointInvalidError(
            "HITL waiting marker 缺少 interaction_id",
            session_id=session_id,
        )
    try:
        target = interaction_service.get(interaction_id)
    except HITLNotFoundError as exc:
        raise HITLCheckpointInvalidError(
            "HITL waiting marker 指向的 interaction 不存在",
            session_id=session_id,
            interaction_id=interaction_id,
        ) from exc
    if target.session_id != session_id:
        raise HITLCheckpointInvalidError(
            "HITL waiting marker 与请求 session 不匹配",
            session_id=session_id,
            interaction_id=interaction_id,
        )
    if fallback is None or fallback.interaction_id == target.interaction_id:
        return target
    if (
        target.status is InteractionStatus.CONSUMED
        and target.resume_phase is InteractionResumePhase.RESULT_COMMITTED
    ):
        return fallback
    raise HITLCheckpointInvalidError(
        "HITL waiting marker 与 pending interaction 冲突",
        session_id=session_id,
        interaction_id=interaction_id,
        pending_interaction_id=fallback.interaction_id,
    )


async def resolve_interaction_result(
    *,
    interaction: HumanInteraction,
    prepared: PreparedToolCall,
    tool_executor: ToolExecutor,
    tool_context: ToolExecutionContext,
) -> ToolResult:
    """将 typed 人工响应收敛为一个工具结果。"""
    response = interaction.response
    tool_call_id = interaction.request.tool_call.tool_call_id
    if isinstance(response, PermissionInteractionResponse):
        if response.decision == "approve":
            return await tool_executor.execute_prepared(
                prepared,
                tool_context,
                approved_tool_call_id=tool_call_id,
            )
        return ToolResult(
            tool_use_id=tool_call_id,
            tool_name=prepared.tool_use.name,
            is_error=True,
            error=ToolErrorInfo(code="USER_REJECTED", message="用户拒绝了工具调用"),
        )
    if isinstance(response, QuestionInteractionResponse):
        return ToolResult(
            tool_use_id=tool_call_id,
            tool_name=prepared.tool_use.name,
            content=[TextBlock(text=response.answer)],
        )
    raise HITLCheckpointInvalidError("HITL interaction 缺少有效 response")


def append_resumed_result(
    *,
    result: ToolResult,
    session_store: SessionStore,
    session_id: str,
    run_id: str,
    step_index: int,
    agent_id: str,
) -> Msg:
    """投影已执行的普通续跑工具结果。

    此函数不更新 HITL checkpoint 或 interaction phase，也不对 message 去重；调用方必须先
    持久化 continuation claim。若任一步写入失败，claim 会保留，使后续恢复 fail closed，
    而不是重放可能已经产生副作用的工具。event 使用稳定 ID，避免同一结果重复写入事件流。

    Args:
        result: 已完成的普通续跑工具结果。
        session_store: 保存 session message 与工具事件的存储。
        session_id: 目标 session 标识。
        run_id: 当前 runtime run 标识。
        step_index: 当前 loop 步骤索引。
        agent_id: 产生工具调用的 agent 标识。

    Returns:
        Msg: 写入 session history 的 tool-result message。
    """
    message = build_tool_result_message(result)
    history = [Msg.from_dict(item) for item in session_store.load_messages(session_id)]
    history.append(message)
    session_store.save_messages(
        session_id,
        [item.model_dump(mode="json") for item in history],
    )
    event = build_tool_result_event(
        result,
        run_id=run_id,
        step_index=step_index,
        agent_id=agent_id,
        metadata=None,
    )
    session_store.append_tool_event(session_id, event)
    return message


def _validate_ready_results(
    interaction: HumanInteraction,
) -> tuple[ToolResult, list[ToolResult]]:
    """验证待提交结果与 interaction checkpoint 的身份和内容一致。

    针对checkpoint损坏的 fail-closed 函数，正常流程下不会出现问题。
    """
    checkpoint = interaction.checkpoint
    try:
        result = ToolResult.model_validate(checkpoint.get("pending_result"))
        all_tool_results = [
            ToolResult.model_validate(item) for item in checkpoint.get("all_tool_results", [])
        ]
    except (TypeError, ValueError) as exc:
        raise HITLCheckpointInvalidError("HITL 待提交工具结果结构无效") from exc

    expected_id = interaction.request.tool_call.tool_call_id
    if result.tool_use_id != expected_id:
        raise HITLCheckpointInvalidError("HITL 待提交工具结果与 interaction 不匹配")
    matches = [item for item in all_tool_results if item.tool_use_id == expected_id]
    if len(matches) != 1:
        raise HITLCheckpointInvalidError("HITL 工具结果列表必须包含唯一的 interaction 结果")
    if matches[0] != result:
        raise HITLCheckpointInvalidError("HITL 待提交工具结果 payload 不一致")
    return result, all_tool_results


async def commit_ready_interaction(
    *,
    interaction: HumanInteraction,
    session_store: SessionStore,
    interaction_service: HumanInteractionService,
    agent_id: str,
) -> RuntimeTurnResult:
    """幂等提交 checkpoint 中已持久化的当前 gate 结果。

    ``RESULT_READY`` 阶段的 ``pending_result`` 已是可恢复的 durable result；因此本函数
    可在 message/event 写入中断后安全重试。它先校验 result 与 interaction 身份及完整结果
    列表一致，再按 tool call ID 去重 message、以稳定 event ID 追加事件，最后通过 CAS 推进
    interaction 至 ``RESULT_COMMITTED``。本函数不执行工具。

    Args:
        interaction: 处于 result-ready 或 result-committed 的已消费 interaction。
        session_store: 保存 session message 与工具事件的存储。
        interaction_service: 推进 interaction 恢复阶段的服务。
        agent_id: 产生工具调用的 agent 标识。

    Returns:
        RuntimeTurnResult: 包含 checkpoint 全部已提交工具结果的恢复结果。

    Raises:
        HITLCheckpointInvalidError: checkpoint 结果缺失、结构无效或与 interaction 不一致时。
    """
    checkpoint = interaction.checkpoint
    result, all_tool_results = _validate_ready_results(interaction)
    message = build_tool_result_message(result)
    messages = [Msg.from_dict(item) for item in session_store.load_messages(interaction.session_id)]
    if not any(
        block.tool_use_id == result.tool_use_id
        for existing in messages
        for block in existing.tool_results
    ):
        messages.append(message)
        session_store.save_messages(
            interaction.session_id,
            [item.model_dump(mode="json") for item in messages],
        )
    event = build_tool_result_event(
        result,
        run_id=interaction.run_id,
        step_index=interaction.step_index,
        agent_id=agent_id,
        metadata=None,
    )
    session_store.append_tool_event(interaction.session_id, event)
    if interaction.resume_phase is not InteractionResumePhase.RESULT_COMMITTED:
        interaction = interaction_service.update_consumed(
            interaction.interaction_id,
            InteractionResumePhase.RESULT_COMMITTED,
            checkpoint,
            expected_phase=interaction.resume_phase,
            expected_version=interaction.version,
        )
    return RuntimeTurnResult(
        session_id=interaction.session_id,
        run_id=interaction.run_id,
        status=RuntimeStatus.OK,
        tool_result_messages=[build_tool_result_message(item) for item in all_tool_results],
        tool_results=all_tool_results,
        steps=interaction.step_index + 1,
    )


__all__ = [
    "append_resumed_result",
    "commit_ready_interaction",
    "load_resumable_interaction",
    "resolve_interaction_result",
]
