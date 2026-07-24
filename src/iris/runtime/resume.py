"""Runtime HITL 恢复的独立发现、解析与提交辅助逻辑。"""

from __future__ import annotations

from ..exceptions import (
    HITLCheckpointInvalidError,
    HITLExecutionOutcomeUnknownError,
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
    """保存续跑工具结果，并使用稳定 event ID 避免重复追加。"""
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


async def commit_ready_interaction(
    *,
    interaction: HumanInteraction,
    session_store: SessionStore,
    interaction_service: HumanInteractionService,
    agent_id: str,
) -> RuntimeTurnResult:
    """幂等写入已准备工具结果的消息与事件。"""
    checkpoint = interaction.checkpoint
    raw_result = checkpoint.get("pending_result")
    if not isinstance(raw_result, dict):
        raise HITLExecutionOutcomeUnknownError("HITL 已消费 interaction 缺少待提交结果")
    result = ToolResult.model_validate(raw_result)
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
        )
    all_tool_results = [
        ToolResult.model_validate(item) for item in checkpoint.get("all_tool_results", [])
    ]
    if not any(item.tool_use_id == result.tool_use_id for item in all_tool_results):
        all_tool_results.append(result)
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
