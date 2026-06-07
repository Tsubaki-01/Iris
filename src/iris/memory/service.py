"""记忆 SDK 服务层。

屏蔽了底层的存储细节与文件镜像等逻辑，统一对外提供操作长期记忆的聚合方法。
核心暴露 MemoryService 类实例作为访问入口点。

Example:
    service = MemoryService(store=sqlite_store)
    episode = service.observe(input_data)
    candidates = service.list_candidates(scope)
"""

# region imports
from __future__ import annotations

from ..exceptions import IrisMemoryError
from .context import MemoryContextBuilder
from .mirror import FileMemoryMirror
from .models import (
    MemoryActor,
    MemoryCandidate,
    MemoryCandidateStatus,
    MemoryCategory,
    MemoryContextBundle,
    MemoryEpisode,
    MemoryEvent,
    MemoryEventType,
    MemoryItem,
    MemoryItemKind,
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

    提供对底层存储引擎的高级抽象，确保状态变更与审计事件的原子同步写入，
    并自动处理向文件系统镜像备份的逻辑。

    Attributes:
        store (MemoryStore): 权威记忆存储实现。
        mirror (FileMemoryMirror | None): 用于将变更双写到文件系统的镜像实例。
        context_builder (MemoryContextBuilder): 用于组合上下文的辅助类实例。

    Example:
        service = MemoryService(store=sqlite_store)
        episode = service.observe(input_data)
    """

    # ==========================================
    #               Initialization
    # ==========================================
    # region
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

    # endregion

    # ==========================================
    #           L1 / L2 Memory Core
    # ==========================================
    # region
    def observe(self, input: MemoryObserveInput) -> MemoryEpisode:
        """记录 L1 观察片段，保存临时感知的原始信息。

        直接将用户观察或系统事件转为不可变的 Episode 记录。
        同时生成审计追踪事件以记录操作来源与原因。

        Args:
            input (MemoryObserveInput): 包含作用域、文本、来源等字段的聚合入参。

        Returns:
            MemoryEpisode: 持久化后被分配唯一标识的观察片段。
        """
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

        # 将产生的变更同步投递至文件系统以实现灾备与外挂修改监测。
        if self.mirror is not None:
            self.mirror.mirror_event(event)

        return stored

    def remember(self, input: MemoryWriteInput) -> MemoryItem:
        """写入 L2 长期记忆条目，固化关键知识或意图总结。

        用于跨会话的高价值信息持久化，通常在处理完 L1 观察片段后被触发。

        Args:
            input (MemoryWriteInput): 包含作用域、分类及置信度等元数据的写请求。

        Returns:
            MemoryItem: 构造完整并被持久化后的权威长期记忆记录。
        """
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

        # 将产生的变更同步投递至文件系统以实现灾备与外挂修改监测。
        if self.mirror is not None:
            self.mirror.mirror_item(stored)
            self.mirror.mirror_event(event)

        return stored

    # endregion

    # ==========================================
    #         Query & Management Methods
    # ==========================================
    # region
    def recall(self, query: MemoryQuery) -> list[MemorySearchResult]:
        """召回长期记忆。

        将搜索请求直接路由至存储引擎执行。支持基于文本或元数据的高级过滤。

        Args:
            query (MemoryQuery): 限定搜索范围及条件的聚合参数。

        Returns:
            list[MemorySearchResult]: 按相关度或时间排序的命中结果。
        """
        return self.store.search(query)

    def forget(
        self,
        item_id: str,
        scope: MemoryScope,
        *,
        actor: MemoryActor = MemoryActor.SDK,
        reason: str,
    ) -> bool:
        """删除指定 scope 下的长期记忆条目。

        软删除或硬删除取决于底层引擎的具体实现，但业务上要求提供显式原因
        用于审计归档。

        Args:
            item_id (str): 目标记忆记录的全局唯一标识。
            scope (MemoryScope): 定位资源所在的作用域。
            actor (MemoryActor): 执行该操作的参与者。
            reason (str): 请求删除的原因以供审计。

        Raises:
            IrisMemoryError: 当没有提供具体的删除原因时。

        Returns:
            bool: 找到 active item 并实际删除时返回 True；未命中时返回 False。
        """
        if not reason.strip():
            raise IrisMemoryError("删除记忆必须提供原因", item_id=item_id)

        event = MemoryEvent(
            scope=scope,
            event_type=MemoryEventType.DELETE,
            actor=actor,
            item_id=item_id,
            reason=reason,
        )
        deleted = self.store.delete_item(item_id, scope, event=event)

        # 强制由真实数据源重构该作用域镜像，以此确保本地文件与数据库强一致。
        if deleted and self.mirror is not None:
            self.mirror.rebuild_from_store(self.store, scope)
        return deleted

    def get_item(self, item_id: str, scope: MemoryScope) -> MemoryItem | None:
        """读取指定 scope 下的活跃长期记忆条目。

        Args:
            item_id (str): 需要检索的条目标识。
            scope (MemoryScope): 隔离的作用域范围。

        Returns:
            MemoryItem | None: 定位到对应的对象则返回，否则返回 None。
        """
        return self.store.get_item(item_id, scope)

    def list_items(
        self,
        scope: MemoryScope,
        *,
        limit: int | None = 50,
        categories: list[MemoryCategory] | None = None,
        kinds: list[MemoryItemKind] | None = None,
    ) -> list[MemoryItem]:
        """列出指定 scope 下的近期活跃长期记忆条目。

        Args:
            scope (MemoryScope): 获取资源所在的作用域。
            limit (int | None): 最大返回数量约束；None 表示读取完整 active 投影。
            categories (list[MemoryCategory] | None): 可选的类别过滤条件。
            kinds (list[MemoryItemKind] | None): 可选的条目类型过滤条件。

        Returns:
            list[MemoryItem]: 符合限制大小的数据片段集。
        """
        return self.store.list_items(
            scope,
            limit=limit,
            categories=categories,
            kinds=kinds,
        )

    def list_events(
        self,
        scope: MemoryScope,
        *,
        item_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        """列出指定 scope 下的审计事件日志。

        Args:
            scope (MemoryScope): 日志相关的作用域空间。
            item_id (str | None): 可选的限定日志针对某一条记忆发生。
            limit (int): 限制结果数。

        Returns:
            list[MemoryEvent]: 用于还原操作历史记录的事件集。
        """
        return self.store.list_events(scope, item_id=item_id, limit=limit)

    # endregion

    # ==========================================
    #           Candidate Operations
    # ==========================================
    # region
    def add_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        actor: MemoryActor = MemoryActor.SDK,
        reason: str = "",
    ) -> MemoryCandidate:
        """保存候选记忆并记录审计事件。

        候选态主要用于人类确认或延后批处理固化，防止低置信度信息污染权威记忆。

        Args:
            candidate (MemoryCandidate): 预生成的候选条目对象。
            actor (MemoryActor): 触发操作的参与实体。
            reason (str): 候选写入的补充说明。

        Returns:
            MemoryCandidate: 包含唯一 ID 的候选对象。
        """
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

        # 将候选态事件录入镜像事件流中，保持完整审计闭环。
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
        """列出指定 scope 下的候选记忆。

        Args:
            scope (MemoryScope): 获取候选态记录的所属范围。
            status (MemoryCandidateStatus | None): 对记录审核阶段进行过滤。
            limit (int): 分页最大返回长度。

        Returns:
            list[MemoryCandidate]: 对应的候选态结果集合。
        """
        return self.store.list_candidates(scope, status=status, limit=limit)

    def promote_candidate(
        self,
        candidate_id: str,
        scope: MemoryScope,
        *,
        kind: MemoryItemKind,
        actor: MemoryActor = MemoryActor.SDK,
        reason: str,
    ) -> MemoryItem:
        """原子晋升 pending candidate 为 L2 item。

        Args:
            candidate_id (str): 目标候选记忆 ID。
            scope (MemoryScope): 候选资源所在隔离范围。
            kind (MemoryItemKind): 晋升后的长期记忆类型。
            actor (MemoryActor): 发起晋升操作的参与实体。
            reason (str): 通过晋升策略的原因。

        Returns:
            MemoryItem: 晋升后可召回的 L2 item。
        """
        item = self.store.promote_candidate(
            candidate_id,
            scope,
            kind=kind,
            actor=actor,
            reason=reason,
        )
        if self.mirror is not None:
            self.mirror.rebuild_from_store(self.store, scope)
        return item

    def accept_candidate(
        self,
        candidate_id: str,
        scope: MemoryScope,
        *,
        actor: MemoryActor = MemoryActor.SDK,
        reason: str,
    ) -> MemoryCandidate:
        """将候选记忆标记为已接受。

        标志候选信息通过评估阶段，允许被视作已验证的知识内容。

        Args:
            candidate_id (str): 唯一的候选实体标识。
            scope (MemoryScope): 候选资源所在隔离范围。
            actor (MemoryActor): 发起评估操作对象。
            reason (str): 批准其通过的特定原因。

        Returns:
            MemoryCandidate: 状态更新后的全新实体对象。
        """
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
        """将候选记忆标记为已拒绝。

        丢弃低价值信息或已失效的识别结论。

        Args:
            candidate_id (str): 唯一的候选实体标识。
            scope (MemoryScope): 候选资源所在隔离范围。
            actor (MemoryActor): 发起拒绝的执行方。
            reason (str): 说明信息为何不满足长期记忆要求。

        Returns:
            MemoryCandidate: 退回并标记丢弃后的实体表示。
        """
        return self._update_candidate_status(
            candidate_id,
            scope,
            MemoryCandidateStatus.REJECTED,
            event_type=MemoryEventType.CANDIDATE_REJECT,
            actor=actor,
            reason=reason,
        )

    # endregion

    # ==========================================
    #           Context & Helpers
    # ==========================================
    # region
    def build_context(self, query: MemoryQuery, *, max_chars: int) -> MemoryContextBundle:
        """召回并构建结构化记忆上下文。

        包装 context_builder 以代理执行查询并返回限定字数的大文本。

        Args:
            query (MemoryQuery): 记忆库检索的检索维度集合。
            max_chars (int): 限制向模型吐出的拼接字符串最大长度。

        Returns:
            MemoryContextBundle: 装载核心 prompt 内所需的组合数据块。
        """
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
        """更新候选状态并同步审计事件。

        收敛候选接受与拒绝时的底层数据变更以及事件触发复用流程。

        Args:
            candidate_id (str): 目标候选对象的标识。
            scope (MemoryScope): 所属空间范围。
            status (MemoryCandidateStatus): 要刷新为的目标态值。
            event_type (MemoryEventType): 同步写入的对应的事件类型。
            actor (MemoryActor): 触发操作发起方。
            reason (str): 操作详细理由。

        Returns:
            MemoryCandidate: 成功变更状态位后的候选实例对象。
        """
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

        # 当候选状态变化时，将其同步分发以保留文件层镜像状态一致。
        if self.mirror is not None:
            self.mirror.mirror_event(event)

        return stored

    # endregion
