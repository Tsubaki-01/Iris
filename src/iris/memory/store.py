"""记忆存储协议。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .models import (
    MemoryCandidate,
    MemoryCandidateStatus,
    MemoryCategory,
    MemoryEpisode,
    MemoryEvent,
    MemoryItem,
    MemoryItemKind,
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
        limit: int | None = 50,
        include_deleted: bool = False,
        categories: Sequence[MemoryCategory] | None = None,
        kinds: Sequence[MemoryItemKind] | None = None,
    ) -> list[MemoryItem]:
        """列出指定 scope 下的长期记忆条目。

        `categories` 与 `kinds` 必须由 store 在读取层过滤，再应用 `limit`。
        `limit=None` 表示读取完整投影，主要供 mirror 重建使用。
        """

    def list_events(
        self,
        scope: MemoryScope,
        *,
        item_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        """列出指定 scope 下的审计事件。"""

    def add_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        event: MemoryEvent,
    ) -> MemoryCandidate:
        """保存候选记忆和对应审计事件。"""

    def list_candidates(
        self,
        scope: MemoryScope,
        *,
        status: MemoryCandidateStatus | None = None,
        limit: int = 50,
    ) -> list[MemoryCandidate]:
        """列出指定 scope 下的候选记忆。"""

    def update_candidate_status(
        self,
        candidate_id: str,
        scope: MemoryScope,
        status: MemoryCandidateStatus,
        *,
        event: MemoryEvent,
    ) -> MemoryCandidate:
        """更新候选记忆状态并记录审计事件。"""


__all__ = ["MemoryStore"]
