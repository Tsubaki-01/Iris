"""记忆 SDK 服务层。"""

# region imports
from __future__ import annotations

from ..exceptions import IrisMemoryError
from .context import MemoryContextBuilder
from .mirror import FileMemoryMirror
from .models import (
    MemoryActor,
    MemoryCandidate,
    MemoryCandidateStatus,
    MemoryContextBundle,
    MemoryEpisode,
    MemoryEvent,
    MemoryEventType,
    MemoryItem,
    MemoryObserveInput,
    MemoryQuery,
    MemoryScope,
    MemorySearchResult,
    MemoryWriteInput,
)
from .store import MemoryStore

# endregion


class MemoryService:
    """长期记忆内核的 Python SDK 门面。

    Args:
        store: 权威记忆存储实现。
        context_builder: 可选上下文构建器；默认使用 `MemoryContextBuilder`。
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        mirror: FileMemoryMirror | None = None,
        context_builder: MemoryContextBuilder | None = None,
    ) -> None:
        """初始化记忆服务。"""
        self.store = store
        self.mirror = mirror
        self.context_builder = context_builder or MemoryContextBuilder()

    def observe(self, input: MemoryObserveInput) -> MemoryEpisode:
        """记录 L1 观察片段。"""
        episode = MemoryEpisode(
            scope=input.scope,
            source_type=input.source_type,
            source_id=input.source_id,
            text=input.text,
            category=input.category,
            artifacts=input.artifacts,
            metadata=input.metadata,
        )
        event = MemoryEvent(
            scope=input.scope,
            event_type=MemoryEventType.OBSERVE,
            actor=input.actor,
            episode_id=episode.id,
            reason=input.reason,
        )
        stored = self.store.add_episode(episode, event=event)
        if self.mirror is not None:
            self.mirror.mirror_event(event)
        return stored

    def remember(self, input: MemoryWriteInput) -> MemoryItem:
        """写入 L2 长期记忆条目。"""
        item = MemoryItem(
            scope=input.scope,
            text=input.text,
            category=input.category,
            kind=input.kind,
            episode_id=input.episode_id,
            source_type=input.source_type,
            source_id=input.source_id,
            reason=input.reason,
            confidence=input.confidence,
            importance=input.importance,
            artifacts=input.artifacts,
            metadata=input.metadata,
        )
        event = MemoryEvent(
            scope=input.scope,
            event_type=MemoryEventType.ADD,
            actor=input.actor,
            item_id=item.id,
            reason=input.reason,
        )
        stored = self.store.add_item(item, event=event)
        if self.mirror is not None:
            self.mirror.mirror_item(stored)
        return stored

    def recall(self, query: MemoryQuery) -> list[MemorySearchResult]:
        """召回长期记忆。"""
        return self.store.search(query)

    def forget(
        self,
        item_id: str,
        scope: MemoryScope,
        *,
        actor: MemoryActor = MemoryActor.SDK,
        reason: str,
    ) -> None:
        """删除指定 scope 下的长期记忆条目。"""
        if not reason.strip():
            raise IrisMemoryError("删除记忆必须提供原因", item_id=item_id)
        event = MemoryEvent(
            scope=scope,
            event_type=MemoryEventType.DELETE,
            actor=actor,
            item_id=item_id,
            reason=reason,
        )
        self.store.delete_item(item_id, scope, event=event)
        if self.mirror is not None:
            self.mirror.rebuild_from_store(self.store, scope)

    def get_item(self, item_id: str, scope: MemoryScope) -> MemoryItem | None:
        """读取指定 scope 下的活跃长期记忆条目。"""
        return self.store.get_item(item_id, scope)

    def list_items(self, scope: MemoryScope, *, limit: int = 50) -> list[MemoryItem]:
        """列出指定 scope 下的活跃长期记忆条目。"""
        return self.store.list_items(scope, limit=limit)

    def list_events(
        self,
        scope: MemoryScope,
        *,
        item_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        """列出指定 scope 下的审计事件。"""
        return self.store.list_events(scope, item_id=item_id, limit=limit)

    def add_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        actor: MemoryActor = MemoryActor.SDK,
        reason: str = "",
    ) -> MemoryCandidate:
        """保存候选记忆并记录审计事件。"""
        event = MemoryEvent(
            scope=candidate.scope,
            event_type=MemoryEventType.CANDIDATE_ADD,
            actor=actor,
            episode_id=candidate.episode_ids[0],
            reason=reason or candidate.reason,
            metadata={
                "candidate_id": candidate.id,
                "candidate_status": candidate.status.value,
                "episode_ids": candidate.episode_ids,
            },
        )
        stored = self.store.add_candidate(candidate, event=event)
        if self.mirror is not None:
            self.mirror.mirror_event(event)
        return stored

    def list_candidates(
        self,
        scope: MemoryScope,
        *,
        status: MemoryCandidateStatus | None = None,
        limit: int = 50,
    ) -> list[MemoryCandidate]:
        """列出指定 scope 下的候选记忆。"""
        return self.store.list_candidates(scope, status=status, limit=limit)

    def accept_candidate(
        self,
        candidate_id: str,
        scope: MemoryScope,
        *,
        actor: MemoryActor = MemoryActor.SDK,
        reason: str,
    ) -> MemoryCandidate:
        """将候选记忆标记为已接受。"""
        return self._update_candidate_status(
            candidate_id,
            scope,
            MemoryCandidateStatus.ACCEPTED,
            event_type=MemoryEventType.CANDIDATE_ACCEPT,
            actor=actor,
            reason=reason,
        )

    def reject_candidate(
        self,
        candidate_id: str,
        scope: MemoryScope,
        *,
        actor: MemoryActor = MemoryActor.SDK,
        reason: str,
    ) -> MemoryCandidate:
        """将候选记忆标记为已拒绝。"""
        return self._update_candidate_status(
            candidate_id,
            scope,
            MemoryCandidateStatus.REJECTED,
            event_type=MemoryEventType.CANDIDATE_REJECT,
            actor=actor,
            reason=reason,
        )

    def build_context(self, query: MemoryQuery, *, max_chars: int) -> MemoryContextBundle:
        """召回并构建结构化记忆上下文。"""
        return self.context_builder.build(self.recall(query), max_chars=max_chars)

    def _update_candidate_status(
        self,
        candidate_id: str,
        scope: MemoryScope,
        status: MemoryCandidateStatus,
        *,
        event_type: MemoryEventType,
        actor: MemoryActor,
        reason: str,
    ) -> MemoryCandidate:
        """更新候选状态并同步审计事件。"""
        event = MemoryEvent(
            scope=scope,
            event_type=event_type,
            actor=actor,
            reason=reason,
            metadata={"candidate_id": candidate_id, "candidate_status": status.value},
        )
        stored = self.store.update_candidate_status(
            candidate_id,
            scope,
            status,
            event=event,
        )
        if self.mirror is not None:
            self.mirror.mirror_event(event)
        return stored
