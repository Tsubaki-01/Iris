"""可选记忆编排器。

本模块只实现候选记忆链路：`observe()` 产生可审计候选，
`process_candidates()` 在显式调用时才把候选晋升为 L2 item。

Example:
    orchestrator = MemoryOrchestrator(service, extractor=RuleMemoryExtractor())
"""

# region imports
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, TypeVar

from .models import (
    MemoryActor,
    MemoryCandidate,
    MemoryCandidateStatus,
    MemoryCategory,
    MemoryContextBundle,
    MemoryEpisode,
    MemoryItem,
    MemoryItemKind,
    MemoryLevel,
    MemoryObserveInput,
    MemoryQuery,
    MemoryScope,
    MemoryWriteInput,
)
from .service import MemoryService

# endregion

EnumT = TypeVar("EnumT", bound=StrEnum)


class MemoryExtractor(Protocol):
    """从 episode 中抽取候选记忆的协议。"""

    def extract(self, episode: MemoryEpisode) -> list[MemoryCandidate]:
        """返回 episode 对应的候选记忆列表。"""


class NoOpMemoryExtractor:
    """默认关闭的抽取器，不产生候选记忆。"""

    def extract(self, episode: MemoryEpisode) -> list[MemoryCandidate]:
        """忽略 episode 并返回空候选列表。"""
        return []


class RuleMemoryExtractor:
    """基于 episode 字段和 metadata hint 的轻量规则抽取器。"""

    def extract(self, episode: MemoryEpisode) -> list[MemoryCandidate]:
        """把非空 episode 文本转换为一个候选记忆。"""
        if not episode.text.strip():
            return []
        return [
            MemoryCandidate(
                scope=episode.scope,
                episode_ids=[episode.id],
                category=_enum_hint(
                    episode.metadata,
                    keys=("memory_category", "category"),
                    enum_type=MemoryCategory,
                    default=episode.category,
                ),
                suggested_level=_enum_hint(
                    episode.metadata,
                    keys=("memory_level", "suggested_level"),
                    enum_type=MemoryLevel,
                    default=MemoryLevel.SEMANTIC,
                ),
                text=episode.text,
                confidence=_score_hint(episode.metadata, "memory_confidence", default=0.8),
                importance=_score_hint(episode.metadata, "memory_importance", default=0.6),
                reason=str(episode.metadata.get("memory_reason") or "rule extractor candidate"),
                metadata=dict(episode.metadata),
            )
        ]


class MemoryClassifier(Protocol):
    """候选记忆分类器协议。"""

    def classify(self, candidate: MemoryCandidate) -> MemoryCandidate:
        """返回分类后的候选记忆。"""


class RuleMemoryClassifier:
    """Stage 5 默认规则分类器。"""

    def classify(self, candidate: MemoryCandidate) -> MemoryCandidate:
        """当前阶段只保留 extractor 已给出的显式分类。"""
        return candidate


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """记忆策略判断结果。"""

    allowed: bool
    reason: str


class MemoryPolicy:
    """候选记忆接受与晋升策略。"""

    def __init__(
        self,
        *,
        min_confidence: float = 0.6,
        min_importance: float = 0.4,
    ) -> None:
        """初始化候选策略阈值。"""
        self.min_confidence = min_confidence
        self.min_importance = min_importance

    def should_accept_candidate(self, candidate: MemoryCandidate) -> PolicyDecision:
        """判断候选是否允许进入 pending 队列。"""
        return self._score_decision(candidate)

    def should_promote(self, candidate: MemoryCandidate) -> PolicyDecision:
        """判断候选是否允许在显式处理时晋升为 L2。"""
        if candidate.status != MemoryCandidateStatus.PENDING:
            return PolicyDecision(False, "candidate is not pending")
        if candidate.suggested_level != MemoryLevel.SEMANTIC:
            return PolicyDecision(False, "candidate is not suggested for L2")
        return self._score_decision(candidate)

    def should_merge(
        self,
        candidate: MemoryCandidate,
        existing: MemoryItem,
    ) -> PolicyDecision:
        """不自动合并候选。"""
        return PolicyDecision(False, "merge is not implemented in this policy")

    def _score_decision(self, candidate: MemoryCandidate) -> PolicyDecision:
        """按置信度和重要性阈值判断候选。"""
        if candidate.confidence is not None and candidate.confidence < self.min_confidence:
            return PolicyDecision(False, "candidate confidence is below policy threshold")
        if candidate.importance is not None and candidate.importance < self.min_importance:
            return PolicyDecision(False, "candidate importance is below policy threshold")
        return PolicyDecision(True, "candidate satisfies policy thresholds")


class MemoryOrchestrator:
    """可选记忆编排器。"""

    def __init__(
        self,
        service: MemoryService,
        *,
        extractor: MemoryExtractor | None = None,
        classifier: MemoryClassifier | None = None,
        policy: MemoryPolicy | None = None,
    ) -> None:
        """创建编排器实例。"""
        self.service = service
        self.extractor = extractor or NoOpMemoryExtractor()
        self.classifier = classifier or RuleMemoryClassifier()
        self.policy = policy or MemoryPolicy()

    def observe(self, input: MemoryObserveInput) -> list[MemoryCandidate]:
        """记录 episode，并按显式配置的 extractor 生成候选记忆。"""
        episode = self.service.observe(input)
        stored_candidates: list[MemoryCandidate] = []
        for raw_candidate in self.extractor.extract(episode):
            candidate = self.classifier.classify(raw_candidate)
            decision = self.policy.should_accept_candidate(candidate)
            status = (
                MemoryCandidateStatus.PENDING
                if decision.allowed
                else MemoryCandidateStatus.REJECTED
            )
            candidate = candidate.model_copy(update={"status": status})
            stored_candidates.append(
                self.service.add_candidate(
                    candidate,
                    actor=input.actor,
                    reason=decision.reason,
                )
            )
        return stored_candidates

    def process_candidates(
        self,
        scope: MemoryScope,
        *,
        limit: int = 50,
    ) -> list[MemoryItem]:
        """显式处理 pending candidates，并在策略允许时晋升为 L2 item。"""
        promoted_items: list[MemoryItem] = []
        candidates = self.service.list_candidates(
            scope,
            status=MemoryCandidateStatus.PENDING,
            limit=limit,
        )
        for candidate in candidates:
            decision = self.policy.should_promote(candidate)
            if not decision.allowed:
                self.service.reject_candidate(
                    candidate.id,
                    scope,
                    actor=MemoryActor.SDK,
                    reason=decision.reason,
                )
                continue
            item = self.service.remember(
                MemoryWriteInput(
                    scope=candidate.scope,
                    text=candidate.text,
                    reason=candidate.reason,
                    category=candidate.category,
                    kind=_candidate_kind(candidate),
                    episode_id=candidate.episode_ids[0],
                    source_id=candidate.id,
                    confidence=candidate.confidence,
                    importance=candidate.importance,
                    metadata={**candidate.metadata, "candidate_id": candidate.id},
                )
            )
            self.service.accept_candidate(
                candidate.id,
                scope,
                actor=MemoryActor.SDK,
                reason=decision.reason,
            )
            promoted_items.append(item)
        return promoted_items

    def build_context(self, query: MemoryQuery, *, max_chars: int) -> MemoryContextBundle:
        """复用 MemoryService 构建记忆上下文。"""
        return self.service.build_context(query, max_chars=max_chars)


def _candidate_kind(candidate: MemoryCandidate) -> MemoryItemKind:
    """从候选 metadata hint 中解析长期记忆 kind。"""
    return _enum_hint(
        candidate.metadata,
        keys=("memory_kind", "kind"),
        enum_type=MemoryItemKind,
        default=MemoryItemKind.NOTE,
    )


def _enum_hint(  # noqa: UP047
    metadata: dict[str, Any],
    *,
    keys: tuple[str, ...],
    enum_type: type[EnumT],
    default: EnumT,
) -> EnumT:
    """从 metadata 中读取枚举 hint，无效 hint 保持默认值。"""
    for key in keys:
        raw = metadata.get(key)
        if isinstance(raw, enum_type):
            return raw
        if isinstance(raw, str):
            for member in enum_type.__members__.values():
                if getattr(member, "value", None) == raw:
                    return member
    return default


def _score_hint(metadata: dict[str, Any], key: str, *, default: float) -> float:
    """从 metadata 中读取 0..1 分数 hint。"""
    raw = metadata.get(key)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and 0.0 <= raw <= 1.0:
        return float(raw)
    return default
