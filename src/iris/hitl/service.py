"""无状态 Human-in-the-loop 领域服务。"""

from __future__ import annotations

from datetime import UTC, datetime

from ..exceptions import (
    HITLConflictError,
    HITLResponseMismatchError,
    IrisRunRecoveryError,
    IrisRunStateError,
)
from ..lifecycle import RunPhase, RunSnapshot
from ..message import TextBlock
from ..tools import ToolErrorInfo, ToolResult
from .models import (
    ApprovedToolCall,
    HumanInteraction,
    HumanInteractionRequest,
    HumanInteractionResponse,
    InteractionStatus,
    QuestionInteractionResponse,
)


class HumanInteractionService:
    """只构造、校验和投影 HITL 值，不拥有 persistence 或 clock。"""

    def create_pending(
        self,
        request: HumanInteractionRequest,
        *,
        run: RunSnapshot,
        step_index: int,
        expires_at: datetime | None,
    ) -> HumanInteraction:
        """从 active run snapshot 构造一条尚未持久化的 pending interaction。"""
        if run.phase is not RunPhase.ACTIVE or run.current_activation_id is None:
            raise IrisRunStateError("只有 active run 可以创建 interaction", run_id=run.run_id)
        if run.pending_interaction_id is not None:
            raise IrisRunStateError("active run 已包含 pending interaction", run_id=run.run_id)
        normalized_expiry = _optional_aware_utc(expires_at, field_name="expires_at")
        if normalized_expiry is not None and normalized_expiry <= run.updated_at:
            raise IrisRunStateError("interaction expiry 必须晚于创建时间", run_id=run.run_id)
        return HumanInteraction(
            session_id=run.session_id,
            run_id=run.run_id,
            step_index=step_index,
            tool_call_id=request.tool_call.tool_call_id,
            request=request,
            created_at=run.updated_at,
            expires_at=normalized_expiry,
        )

    def validate_response(
        self,
        interaction: HumanInteraction,
        *,
        run: RunSnapshot,
        response: HumanInteractionResponse,
        now: datetime,
        environment_fingerprint: str,
    ) -> None:
        """校验一次 response 是否能安全绑定当前 durable waiting facts。"""
        normalized_now = _aware_utc(now, field_name="now")
        if run.phase is not RunPhase.WAITING:
            raise IrisRunStateError("只有 waiting run 可以接收 interaction response")
        if (
            interaction.run_id != run.run_id
            or interaction.session_id != run.session_id
            or run.pending_interaction_id != interaction.interaction_id
        ):
            raise HITLResponseMismatchError("interaction 与 waiting run identity 不匹配")
        if interaction.status not in {InteractionStatus.PENDING, InteractionStatus.RESOLVED}:
            raise IrisRunStateError("interaction 已关闭或不可响应")
        if response.kind != interaction.request.prompt.kind:
            raise HITLResponseMismatchError("response kind 与 interaction request 不匹配")
        if run.environment_fingerprint != environment_fingerprint:
            raise IrisRunRecoveryError("恢复环境 fingerprint 已变化", run_id=run.run_id)
        if (
            interaction.status is InteractionStatus.PENDING
            and interaction.expires_at is not None
            and normalized_now >= interaction.expires_at
        ):
            raise IrisRunStateError("interaction 已过期", run_id=run.run_id)
        if interaction.status is InteractionStatus.RESOLVED and interaction.response != response:
            raise HITLConflictError("interaction 已由不同 response 解决")

    def project_response(
        self,
        interaction: HumanInteraction,
        response: HumanInteractionResponse,
    ) -> ToolResult | ApprovedToolCall:
        """把 resolved response 投影为无副作用结果或精确批准 DTO。"""
        if interaction.status is not InteractionStatus.RESOLVED or interaction.response is None:
            raise IrisRunStateError("只有 resolved interaction 可以投影 response")
        if interaction.response != response or response.kind != interaction.request.prompt.kind:
            raise HITLResponseMismatchError("response 与 resolved interaction 不匹配")
        subject = interaction.request.tool_call
        if isinstance(response, QuestionInteractionResponse):
            return ToolResult(
                tool_use_id=subject.tool_call_id,
                tool_name=subject.tool_name,
                content=[TextBlock(text=response.answer)],
                data={"answer": response.answer},
            )
        if response.decision == "reject":
            return ToolResult(
                tool_use_id=subject.tool_call_id,
                tool_name=subject.tool_name,
                is_error=True,
                error=ToolErrorInfo(
                    code="USER_REJECTED",
                    message="用户拒绝了工具调用",
                ),
            )
        return ApprovedToolCall(
            interaction_id=interaction.interaction_id,
            tool_call_id=subject.tool_call_id,
            tool_name=subject.tool_name,
            fingerprint=subject.fingerprint,
        )


def _aware_utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise IrisRunStateError(f"{field_name} 必须包含时区")
    return value.astimezone(UTC)


def _optional_aware_utc(
    value: datetime | None,
    *,
    field_name: str,
) -> datetime | None:
    if value is None:
        return None
    return _aware_utc(value, field_name=field_name)


__all__ = ["HumanInteractionService"]
