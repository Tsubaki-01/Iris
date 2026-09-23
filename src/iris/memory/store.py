"""记忆存储协议。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar

from .generation_models import (
    DreamPlan,
    DreamSnapshot,
    EpisodeProgress,
    FlushCommit,
    GenerationResult,
    GenerationState,
    MemoryCaptureSource,
    ObservationState,
)
from .models import (
    MemoryCategory,
    MemoryEpisode,
    MemoryEvent,
    MemoryItem,
    MemoryItemKind,
    MemoryItemPatch,
    MemoryNamespaceSnapshot,
    MemoryNamespaceState,
    MemorySearchQuery,
    MemorySearchResponse,
)

PublicationT = TypeVar("PublicationT")


class MemoryStore(Protocol):
    """长期记忆权威存储契约。"""

    def add_episode(self, episode: MemoryEpisode, *, event: MemoryEvent) -> MemoryEpisode:
        """保存 原始经历材料和对应审计事件。"""

    def add_item(self, item: MemoryItem, *, event: MemoryEvent) -> MemoryItem:
        """保存 正式长期记忆条目和对应审计事件。"""

    def update_item(
        self,
        item_id: str,
        namespace: str,
        patch: MemoryItemPatch,
        *,
        event: MemoryEvent,
    ) -> MemoryItem:
        """更新长期记忆条目并记录审计事件。"""

    def delete_item(self, item_id: str, namespace: str, *, event: MemoryEvent) -> bool:
        """将长期记忆条目标记为删除并记录审计事件，返回是否实际删除。"""

    def get_item(self, item_id: str, namespaces: Sequence[str]) -> MemoryItem | None:
        """读取指定 namespace 下的活跃长期记忆条目。"""

    def read_namespace_state(self, namespace: str) -> MemoryNamespaceState:
        """读取 namespace 的条目版本与完整投影版本。"""

    def read_namespace_snapshot(self, namespace: str) -> MemoryNamespaceSnapshot:
        """在同一读事务中读取完整 active 正式 条目和版本。"""

    def publish_projection(
        self, namespace: str, publish: Callable[[MemoryNamespaceSnapshot], None]
    ) -> MemoryNamespaceState:
        """在短写事务内现读快照、发布正文并推进完整投影版本。"""

    def publish_overview(
        self, namespace: str, publish: Callable[[MemoryNamespaceState], PublicationT]
    ) -> PublicationT:
        """在同一短发布事务内读取当前状态并发布已生成概览。"""

    def search(self, query: MemorySearchQuery, namespaces: Sequence[str]) -> MemorySearchResponse:
        """在允许范围内搜索活跃条目，返回原文片段和剩余候选标记。"""

    def list_items(
        self,
        namespaces: Sequence[str],
        *,
        limit: int | None = 50,
        include_deleted: bool = False,
        categories: Sequence[MemoryCategory] | None = None,
        kinds: Sequence[MemoryItemKind] | None = None,
    ) -> list[MemoryItem]:
        """列出指定 namespace 下的长期记忆条目。

        `categories` 与 `kinds` 必须由 store 在读取层过滤，再应用 `limit`。
        `limit` 必须在 1 到 100 之间；`None` 表示读取完整投影，主要供 mirror 重建使用。
        """

    def list_events(
        self,
        namespace: str,
        *,
        item_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        """列出指定 namespace 下的审计事件，`limit` 必须在 1 到 100 之间。"""

    def get_episode(self, episode_id: str, namespace: str) -> MemoryEpisode | None:
        """读取不可变来源内容。"""

    def list_pending_episodes(self, namespace: str, *, limit: int = 100) -> list[EpisodeProgress]:
        """读取待提炼材料和精确处理位置。"""

    def register_source(self, source: MemoryCaptureSource) -> MemoryCaptureSource:
        """登记或读取同一个 run 的捕获水位。"""

    def list_capture_sources(
        self, lifecycle_source_id: str, namespace: str
    ) -> list[MemoryCaptureSource]:
        """读取当前 lifecycle source 尚未封口的来源。"""

    def commit_capture(
        self,
        source: MemoryCaptureSource,
        *,
        expected_captured_until: int,
        episode: MemoryEpisode | None,
    ) -> bool:
        """按水位 CAS 提交未捕获后缀。"""

    def commit_flush(self, commit: FlushCommit) -> bool:
        """原子提交观察和原文位置。"""

    def list_observations(
        self, namespace: str, *, status: str | None = None, limit: int = 100
    ) -> list[ObservationState]:
        """读取观察与其独立处理结果。"""

    def read_dream_snapshot(
        self,
        namespace: str,
        *,
        observation_ids: Sequence[str] | None = None,
        change_ids: Sequence[str] | None = None,
        limit: int = 16,
        related_limit: int = 8,
    ) -> DreamSnapshot:
        """同一读事务获取固定输入、目标、关联及纠正。"""

    def commit_dream(
        self, snapshot: DreamSnapshot, plan: DreamPlan, *, result: GenerationResult
    ) -> bool:
        """CAS 原子应用知识计划并消费本批输入。"""

    def block_dream(
        self,
        snapshot: DreamSnapshot,
        *,
        reason: str,
        budget: int,
        dependency_item_ids: Sequence[str],
    ) -> bool:
        """原子记录容量阻塞，不消费输入。"""

    def retry_blocked(self, namespace: str, *, budget: int | None = None) -> int:
        """预算变化或显式重试时恢复相关输入。"""

    def record_generation_result(self, result: GenerationResult) -> None:
        """保存独立阶段状态及已发生用量。"""

    def generation_state(self, namespace: str) -> GenerationState:
        """读取生成状态与最近阶段结果。"""


__all__ = ["MemoryStore"]
