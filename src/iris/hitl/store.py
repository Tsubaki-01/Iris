"""HITL interaction 的持久化协议。"""

from __future__ import annotations

from typing import Any, Protocol

from .models import (
    HumanInteraction,
    HumanInteractionResponse,
    InteractionResumePhase,
)


class InteractionStore(Protocol):
    """人工交互记录的 compare-and-set 存储边界。"""

    def create_interaction(self, interaction: HumanInteraction) -> None:
        """创建一条 pending interaction。"""

    def load_interaction(self, interaction_id: str) -> HumanInteraction | None:
        """按 ID 读取 interaction。"""

    def list_pending_interactions(self, session_id: str | None = None) -> list[HumanInteraction]:
        """列出指定 session 或全部 session 的 pending interaction。"""

    def resolve_interaction(
        self,
        interaction_id: str,
        response: HumanInteractionResponse,
        *,
        expected_version: int,
    ) -> HumanInteraction:
        """以 CAS 写入人工响应。"""

    def claim_interaction(
        self,
        interaction_id: str,
        checkpoint: dict[str, Any],
        *,
        expected_version: int,
    ) -> HumanInteraction:
        """以 CAS 将已响应 interaction 标记为已消费。"""

    def update_consumed_interaction(
        self,
        interaction_id: str,
        *,
        resume_phase: InteractionResumePhase,
        checkpoint: dict[str, Any],
        expected_phase: InteractionResumePhase,
        expected_version: int,
    ) -> HumanInteraction:
        """按恢复阶段和版本 CAS 更新 consumed interaction。"""


__all__ = ["InteractionStore"]
