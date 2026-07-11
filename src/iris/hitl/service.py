"""HITL interaction 生命周期服务。"""

from __future__ import annotations

import json
from typing import Any

from ..exceptions import (
    HITLAlreadyConsumedError,
    HITLCheckpointInvalidError,
    HITLConflictError,
    HITLNotFoundError,
    HITLResponseMismatchError,
    HITLResponseRequiredError,
)
from .models import (
    HumanInteraction,
    HumanInteractionResponse,
    InteractionKind,
    InteractionResumePhase,
    InteractionStatus,
    PermissionInteractionRequest,
    QuestionInteractionRequest,
    make_call_fingerprint,
)
from .store import InteractionStore


class HumanInteractionService:
    """通过 ``InteractionStore`` 执行最小 HITL 状态转换。"""

    def __init__(self, store: InteractionStore) -> None:
        self.store = store

    def create_permission(
        self,
        *,
        session_id: str,
        run_id: str,
        step_index: int,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        reason: str,
        workspace_root: str,
        checkpoint: dict[str, Any],
    ) -> HumanInteraction:
        """创建并保存一条 pending 权限确认。"""
        request = PermissionInteractionRequest(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            reason=reason,
            workspace_root=workspace_root,
            call_fingerprint=make_call_fingerprint(
                session_id=session_id,
                run_id=run_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=arguments,
                workspace_root=workspace_root,
            ),
        )
        interaction = HumanInteraction(
            session_id=session_id,
            run_id=run_id,
            step_index=step_index,
            tool_call_id=tool_call_id,
            kind=InteractionKind.PERMISSION,
            request=request,
            checkpoint=self._validate_checkpoint(checkpoint),
        )
        self.store.create_interaction(interaction)
        return interaction

    def create_question(
        self,
        *,
        session_id: str,
        run_id: str,
        step_index: int,
        tool_call_id: str,
        question: str,
        options: list[str] | None = None,
        checkpoint: dict[str, Any],
    ) -> HumanInteraction:
        """创建并保存一条 pending 人工问答。"""
        request = QuestionInteractionRequest(
            tool_call_id=tool_call_id,
            question=question,
            options=options or [],
        )
        interaction = HumanInteraction(
            session_id=session_id,
            run_id=run_id,
            step_index=step_index,
            tool_call_id=tool_call_id,
            kind=InteractionKind.QUESTION,
            request=request,
            checkpoint=self._validate_checkpoint(checkpoint),
        )
        self.store.create_interaction(interaction)
        return interaction

    def get(self, interaction_id: str) -> HumanInteraction:
        """读取 interaction；不存在时返回稳定 HITL 错误。"""
        interaction = self.store.load_interaction(interaction_id)
        if interaction is None:
            raise HITLNotFoundError("未找到 HITL interaction", interaction_id=interaction_id)
        return interaction

    def list_pending(self, session_id: str | None = None) -> list[HumanInteraction]:
        """列出 pending interaction。"""
        return self.store.list_pending_interactions(session_id)

    def resolve(
        self,
        interaction_id: str,
        response: HumanInteractionResponse,
    ) -> HumanInteraction:
        """保存人工响应，并保证重复相同响应幂等。"""
        interaction = self.get(interaction_id)
        if response.kind != interaction.kind:
            raise HITLResponseMismatchError(
                "HITL response kind 与 interaction 不匹配",
                interaction_id=interaction_id,
            )
        if interaction.status is InteractionStatus.RESOLVED:
            if interaction.response == response:
                return interaction
            raise HITLConflictError(
                "HITL interaction 已有不同 response", interaction_id=interaction_id
            )
        if interaction.status is InteractionStatus.CONSUMED:
            raise HITLAlreadyConsumedError(
                "HITL interaction 已被消费", interaction_id=interaction_id
            )
        return self.store.resolve_interaction(
            interaction_id,
            response,
            expected_version=interaction.version,
        )

    def claim(self, interaction_id: str, checkpoint: dict[str, Any]) -> HumanInteraction:
        """在 runtime 产生副作用前领取已响应 interaction。"""
        interaction = self.get(interaction_id)
        checkpoint = self._validate_checkpoint(checkpoint)
        if interaction.status is InteractionStatus.PENDING:
            raise HITLResponseRequiredError(
                "HITL interaction 尚未收到 response", interaction_id=interaction_id
            )
        if interaction.status is InteractionStatus.CONSUMED:
            raise HITLAlreadyConsumedError(
                "HITL interaction 已被消费", interaction_id=interaction_id
            )
        return self.store.claim_interaction(
            interaction_id,
            checkpoint,
            expected_version=interaction.version,
        )

    def update_consumed(
        self,
        interaction_id: str,
        phase: InteractionResumePhase,
        checkpoint: dict[str, Any],
    ) -> HumanInteraction:
        """推进已消费 interaction 的恢复阶段。"""
        interaction = self.get(interaction_id)
        checkpoint = self._validate_checkpoint(checkpoint)
        if interaction.status is not InteractionStatus.CONSUMED:
            raise HITLConflictError(
                "只能更新已消费的 HITL interaction", interaction_id=interaction_id
            )
        if phase is InteractionResumePhase.WAITING:
            raise HITLConflictError(
                "已消费的 HITL interaction 不能回到 waiting", interaction_id=interaction_id
            )
        return self.store.update_consumed_interaction(
            interaction_id,
            resume_phase=phase,
            checkpoint=checkpoint,
            expected_version=interaction.version,
        )

    @staticmethod
    def _validate_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(checkpoint, dict):
            raise HITLCheckpointInvalidError("HITL checkpoint 必须是 object")
        try:
            json.dumps(checkpoint, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise HITLCheckpointInvalidError("HITL checkpoint 必须是 JSON-safe 数据") from exc
        return checkpoint


__all__ = ["HumanInteractionService"]
