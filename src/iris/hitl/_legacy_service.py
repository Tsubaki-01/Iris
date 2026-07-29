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
    HumanInteractionRequest,
    HumanInteractionResponse,
    InteractionResumePhase,
    InteractionStatus,
)
from .store import InteractionStore


class HumanInteractionService:
    """通过 ``InteractionStore`` 执行最小 HITL 状态转换。"""

    def __init__(self, store: InteractionStore) -> None:
        self.store = store

    def create(
        self,
        request: HumanInteractionRequest,
        *,
        session_id: str,
        run_id: str,
        step_index: int,
        checkpoint: dict[str, Any],
    ) -> HumanInteraction:
        """创建并保存一条 pending 人工 gate。"""
        interaction = HumanInteraction(
            session_id=session_id,
            run_id=run_id,
            step_index=step_index,
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
        if response.kind != interaction.request.prompt.kind:
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
        *,
        expected_phase: InteractionResumePhase,
        expected_version: int,
    ) -> HumanInteraction:
        """按调用方读取的阶段与版本推进 consumed interaction。"""
        interaction = self.get(interaction_id)
        checkpoint = self._validate_checkpoint(checkpoint)
        if interaction.status is not InteractionStatus.CONSUMED:
            raise HITLConflictError(
                "只能更新已消费的 HITL interaction", interaction_id=interaction_id
            )
        if interaction.resume_phase is not expected_phase:
            raise HITLConflictError(
                "HITL interaction 恢复阶段已变更", interaction_id=interaction_id
            )
        if phase is InteractionResumePhase.WAITING:
            raise HITLConflictError(
                "已消费的 HITL interaction 不能回到 waiting", interaction_id=interaction_id
            )
        return self.store.update_consumed_interaction(
            interaction_id,
            resume_phase=phase,
            checkpoint=checkpoint,
            expected_phase=expected_phase,
            expected_version=expected_version,
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
