"""进程内 HITL interaction 存储实现。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..exceptions import HITLConflictError, HITLResponseMismatchError
from .models import (
    HumanInteraction,
    HumanInteractionResponse,
    InteractionResumePhase,
    InteractionStatus,
)


class InMemoryInteractionStore:
    """使用进程内字典实现 ``InteractionStore`` 的 CAS 语义。"""

    def __init__(self) -> None:
        self._interactions: dict[str, HumanInteraction] = {}

    def create_interaction(self, interaction: HumanInteraction) -> None:
        """保存 interaction，并保证同一 session 仅有一个 active interaction。"""
        if interaction.interaction_id in self._interactions or any(
            item.session_id == interaction.session_id and _is_active(item)
            for item in self._interactions.values()
        ):
            raise HITLConflictError(
                "同一 session 不能存在多个 active HITL interaction",
                session_id=interaction.session_id,
            )
        self._interactions[interaction.interaction_id] = interaction.model_copy(deep=True)

    def load_interaction(self, interaction_id: str) -> HumanInteraction | None:
        """按 ID 读取 interaction。"""
        interaction = self._interactions.get(interaction_id)
        return interaction.model_copy(deep=True) if interaction is not None else None

    def list_pending_interactions(self, session_id: str | None = None) -> list[HumanInteraction]:
        """列出 pending interaction。"""
        pending = (
            interaction
            for interaction in self._interactions.values()
            if interaction.status is InteractionStatus.PENDING
            and (session_id is None or interaction.session_id == session_id)
        )
        return [
            interaction.model_copy(deep=True)
            for interaction in sorted(
                pending,
                key=lambda item: (item.created_at, item.interaction_id),
            )
        ]

    def resolve_interaction(
        self,
        interaction_id: str,
        response: HumanInteractionResponse,
        *,
        expected_version: int,
    ) -> HumanInteraction:
        """以 CAS 写入人工响应。"""
        interaction = self._require(interaction_id)
        if response.kind != interaction.request.prompt.kind:
            raise HITLResponseMismatchError(
                "HITL response kind 与 interaction 不匹配",
                interaction_id=interaction_id,
            )
        if (
            interaction.status is not InteractionStatus.PENDING
            or interaction.version != expected_version
        ):
            raise HITLConflictError(
                "HITL resolve compare-and-set 失败", interaction_id=interaction_id
            )
        updated = _replace_interaction(
            interaction,
            status=InteractionStatus.RESOLVED,
            response=response,
            resolved_at=datetime.now(),
            version=interaction.version + 1,
        )
        self._interactions[interaction_id] = updated
        return updated.model_copy(deep=True)

    def claim_interaction(
        self,
        interaction_id: str,
        checkpoint: dict[str, Any],
        *,
        expected_version: int,
    ) -> HumanInteraction:
        """以 CAS 将 resolved interaction 标记为 consumed/claimed。"""
        interaction = self._require(interaction_id)
        if (
            interaction.status is not InteractionStatus.RESOLVED
            or interaction.resume_phase is not InteractionResumePhase.WAITING
            or interaction.version != expected_version
        ):
            raise HITLConflictError(
                "HITL claim compare-and-set 失败", interaction_id=interaction_id
            )
        updated = _replace_interaction(
            interaction,
            status=InteractionStatus.CONSUMED,
            resume_phase=InteractionResumePhase.CLAIMED,
            checkpoint=checkpoint,
            consumed_at=datetime.now(),
            version=interaction.version + 1,
        )
        self._interactions[interaction_id] = updated
        return updated.model_copy(deep=True)

    def update_consumed_interaction(
        self,
        interaction_id: str,
        *,
        resume_phase: InteractionResumePhase,
        checkpoint: dict[str, Any],
        expected_version: int,
    ) -> HumanInteraction:
        """以 CAS 更新 consumed interaction 的恢复进度。"""
        interaction = self._require(interaction_id)
        if (
            interaction.status is not InteractionStatus.CONSUMED
            or resume_phase is InteractionResumePhase.WAITING
            or interaction.version != expected_version
        ):
            raise HITLConflictError(
                "HITL update compare-and-set 失败", interaction_id=interaction_id
            )
        updated = _replace_interaction(
            interaction,
            resume_phase=resume_phase,
            checkpoint=checkpoint,
            version=interaction.version + 1,
        )
        self._interactions[interaction_id] = updated
        return updated.model_copy(deep=True)

    def _require(self, interaction_id: str) -> HumanInteraction:
        interaction = self._interactions.get(interaction_id)
        if interaction is None:
            raise HITLConflictError(
                "HITL interaction 不存在或已变更", interaction_id=interaction_id
            )
        return interaction


def _is_active(interaction: HumanInteraction) -> bool:
    return interaction.status in {InteractionStatus.PENDING, InteractionStatus.RESOLVED} or (
        interaction.status is InteractionStatus.CONSUMED
        and interaction.resume_phase is not InteractionResumePhase.RESULT_COMMITTED
    )


def _replace_interaction(
    interaction: HumanInteraction,
    **changes: Any,
) -> HumanInteraction:
    return HumanInteraction.model_validate(interaction.model_dump() | changes)


__all__ = ["InMemoryInteractionStore"]
