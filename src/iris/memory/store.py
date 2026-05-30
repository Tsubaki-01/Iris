"""记忆存储协议。"""

from __future__ import annotations

from typing import Protocol

from .models import (
    MemoryEpisode,
    MemoryEvent,
    MemoryItem,
    MemoryItemPatch,
    MemoryQuery,
    MemoryScope,
    MemorySearchResult,
)


class MemoryStore(Protocol):
    """长期记忆权威存储契约。"""

    def add_episode(self, episode: MemoryEpisode, *, event: MemoryEvent) -> MemoryEpisode:
        """保存 L1 片段记忆和对应审计事件。"""

    def add_item(self, item: MemoryItem, *, event: MemoryEvent) -> MemoryItem:
        """保存 L2 长期记忆条目和对应审计事件。"""

    def update_item(
        self,
        item_id: str,
        scope: MemoryScope,
        patch: MemoryItemPatch,
        *,
        event: MemoryEvent,
    ) -> MemoryItem:
        """更新长期记忆条目并记录审计事件。"""

    def delete_item(self, item_id: str, scope: MemoryScope, *, event: MemoryEvent) -> None:
        """将长期记忆条目标记为删除并记录审计事件。"""

    def get_item(self, item_id: str, scope: MemoryScope) -> MemoryItem | None:
        """读取指定 scope 下的活跃长期记忆条目。"""

    def search(self, query: MemoryQuery) -> list[MemorySearchResult]:
        """按查询条件召回长期记忆。"""

    def list_items(
        self,
        scope: MemoryScope,
        *,
        limit: int = 50,
        include_deleted: bool = False,
    ) -> list[MemoryItem]:
        """列出指定 scope 下的长期记忆条目。"""

    def list_events(
        self,
        scope: MemoryScope,
        *,
        item_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        """列出指定 scope 下的审计事件。"""


__all__ = ["MemoryStore"]
