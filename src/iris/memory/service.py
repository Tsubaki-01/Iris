"""记忆 SDK 服务层。"""

# region imports
from __future__ import annotations

from ..exceptions import IrisMemoryError
from .context import MemoryContextBuilder
from .models import (
    MemoryActor,
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
        context_builder: MemoryContextBuilder | None = None,
    ) -> None:
        """初始化记忆服务。"""
        self.store = store
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
        return self.store.add_episode(episode, event=event)

    def remember(self, input: MemoryWriteInput) -> MemoryItem:
        """写入 L2 长期记忆条目。"""
        item = MemoryItem(
            scope=input.scope,
            text=input.text,
            category=input.category,
            kind=input.kind,
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
        return self.store.add_item(item, event=event)

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

    def build_context(self, query: MemoryQuery, *, max_chars: int) -> MemoryContextBundle:
        """召回并构建结构化记忆上下文。"""
        return self.context_builder.build(self.recall(query), max_chars=max_chars)
