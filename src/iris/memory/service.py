"""记忆 SDK 服务层。

屏蔽了底层的存储细节与文件镜像等逻辑，统一对外提供操作长期记忆的聚合方法。
核心暴露 MemoryService 类实例作为访问入口点。

Example:
    service = MemoryService(store=sqlite_store)
    episode = service.observe(input_data)
    candidates = service.list_candidates(namespace)
"""

# region imports
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable, Sequence
from enum import StrEnum
from time import perf_counter
from typing import TypeVar

from ..exceptions import IrisMemoryError
from ..providers.protocols import CompletionProvider
from .files import MemoryFileAccess, freshness_warning
from .mirror import FileMemoryMirror
from .models import (
    MemoryActor,
    MemoryCandidate,
    MemoryCandidateStatus,
    MemoryCategory,
    MemoryEpisode,
    MemoryEvent,
    MemoryEventType,
    MemoryItem,
    MemoryItemKind,
    MemoryItemPatch,
    MemoryObserveInput,
    MemoryOverviewConfig,
    MemoryOverviewContent,
    MemoryOverviewDocument,
    MemoryOverviewGenerationResult,
    MemorySearchQuery,
    MemorySearchResponse,
    MemoryWriteInput,
)
from .overview import build_overview_request, complete_overview_content
from .store import MemoryStore

# endregion

ResultT = TypeVar("ResultT")
logger = logging.getLogger(__name__)


class MemoryIOExecutionMode(StrEnum):
    """控制完整同步 Memory 操作在 async 调用方中的执行位置。"""

    INLINE = "inline"
    THREAD = "thread"


class MemoryService:
    """长期记忆内核的 Python SDK 门面。

    提供对底层存储引擎的高级抽象，确保状态变更与审计事件的原子同步写入，
    并自动处理向文件系统镜像备份的逻辑。

    Attributes:
        store (MemoryStore): 权威记忆存储实现。
        mirror (FileMemoryMirror | None): 权威数据库的可选文件投影。

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
        overview_provider: CompletionProvider | None = None,
        overview_model: str | None = None,
        overview_config: MemoryOverviewConfig | None = None,
        io_execution_mode: MemoryIOExecutionMode = MemoryIOExecutionMode.INLINE,
    ) -> None:
        """初始化记忆服务。"""
        self.store = store
        self.mirror = mirror
        self.overview_provider = overview_provider
        self.overview_model = overview_model
        self.overview_config = overview_config or MemoryOverviewConfig()
        self._io_execution_mode = io_execution_mode

    @property
    def io_execution_mode(self) -> MemoryIOExecutionMode:
        """返回 async IO 适配器使用的固定执行模式。"""
        return self._io_execution_mode

    async def run_async_io(self, operation: Callable[[], ResultT]) -> ResultT:
        """按配置执行完整同步 IO 操作，并保留其返回值和异常。"""
        if self._io_execution_mode is MemoryIOExecutionMode.THREAD:
            return await asyncio.to_thread(operation)
        return operation()

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
            namespace=input.namespace,
            source_type=input.source_type,
            source_id=input.source_id,
            text=input.text,
            category=input.category,
            artifacts=input.artifacts,
            metadata=input.metadata,
        )
        event = MemoryEvent(
            namespace=input.namespace,
            event_type=MemoryEventType.OBSERVE,
            actor=input.actor,
            episode_id=episode.id,
            reason=input.reason,
        )
        stored = self.store.add_episode(episode, event=event)

        return stored

    def remember(self, input: MemoryWriteInput) -> MemoryItem:
        """写入 L2 长期记忆条目，固化关键知识或意图总结。

        用于跨会话的高价值信息持久化，通常在处理完 L1 观察片段后被触发。

        Args:
            input (MemoryWriteInput): 包含作用域、分类及置信度等元数据的写请求。

        Returns:
            MemoryItem: 构造完整并被持久化后的权威长期记忆记录。
        """
        item = MemoryItem.model_construct(
            namespace=input.namespace,
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
            namespace=input.namespace,
            event_type=MemoryEventType.ADD,
            actor=input.actor,
            item_id=item.id,
            reason=input.reason,
        )
        stored = self.store.add_item(item, event=event)

        self._rebuild_committed(stored.namespace)

        return stored

    async def aremember(self, input: MemoryWriteInput) -> MemoryItem:
        """在一次 async IO 操作中完成条目写入及派生镜像刷新。"""
        return await self.run_async_io(lambda: self.remember(input))

    def update(
        self,
        item_id: str,
        namespace: str,
        patch: MemoryItemPatch,
        *,
        actor: MemoryActor = MemoryActor.SDK,
        reason: str,
    ) -> MemoryItem:
        """更新同一记忆条目，并重建其 namespace 的派生镜像。

        Args:
            item_id: 目标条目的稳定 ID。
            namespace: 条目所属 namespace。
            patch: 已校验的字段更新。
            actor: 发起更新的参与者。
            reason: 记录在更新事件中的原因。

        Returns:
            MemoryItem: 数据库提交后的最新条目。
        """
        event = MemoryEvent(
            namespace=namespace,
            event_type=MemoryEventType.UPDATE,
            actor=actor,
            item_id=item_id,
            reason=reason,
        )
        stored = self.store.update_item(item_id, namespace, patch, event=event)
        self._rebuild_committed(namespace)
        return stored

    async def aupdate(
        self,
        item_id: str,
        namespace: str,
        patch: MemoryItemPatch,
        *,
        actor: MemoryActor = MemoryActor.SDK,
        reason: str,
    ) -> MemoryItem:
        """在一次 async IO 操作中完成更新及派生镜像刷新。"""
        return await self.run_async_io(
            lambda: self.update(item_id, namespace, patch, actor=actor, reason=reason)
        )

    # endregion

    # ==========================================
    #         Query & Management Methods
    # ==========================================
    # region
    def search(
        self, query: MemorySearchQuery, namespaces: Sequence[str]
    ) -> MemorySearchResponse:
        """在调用方绑定的读取范围内搜索当前活跃记忆。"""
        return self.store.search(query, namespaces)

    async def asearch(
        self, query: MemorySearchQuery, namespaces: Sequence[str]
    ) -> MemorySearchResponse:
        """用一次完整同步操作适配搜索，连接与结果组装留在同一 worker。"""
        return await self.run_async_io(lambda: self.search(query, namespaces))

    def forget(
        self,
        item_id: str,
        namespace: str,
        *,
        actor: MemoryActor = MemoryActor.SDK,
        reason: str,
    ) -> bool:
        """删除指定 namespace 下的长期记忆条目。

        软删除后默认读取不再返回条目；已经写入会话的快照不受影响。

        Args:
            item_id (str): 目标记忆记录的全局唯一标识。
            namespace (str): 定位资源所在的作用域。
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
            namespace=namespace,
            event_type=MemoryEventType.DELETE,
            actor=actor,
            item_id=item_id,
            reason=reason,
        )
        deleted = self.store.delete_item(item_id, namespace, event=event)

        if deleted:
            self._rebuild_committed(namespace)
        return deleted

    async def aforget(
        self,
        item_id: str,
        namespace: str,
        *,
        actor: MemoryActor = MemoryActor.SDK,
        reason: str,
    ) -> bool:
        """在一次 async IO 操作中完成软删除及派生镜像刷新。"""
        return await self.run_async_io(
            lambda: self.forget(item_id, namespace, actor=actor, reason=reason)
        )

    def get_item(self, item_id: str, namespaces: Sequence[str]) -> MemoryItem | None:
        """在允许的 namespace 集合中读取活跃长期记忆条目。

        Args:
            item_id (str): 需要检索的条目标识。
            namespaces: 允许读取的 namespace 集合。

        Returns:
            MemoryItem | None: 定位到对应的对象则返回，否则返回 None。
        """
        return self.store.get_item(item_id, namespaces)

    async def aget_item(self, item_id: str, namespaces: Sequence[str]) -> MemoryItem | None:
        """异步适配单条长期记忆读取。"""
        return await self.run_async_io(lambda: self.get_item(item_id, namespaces))

    def list_items(
        self,
        namespaces: Sequence[str],
        *,
        limit: int | None = 50,
        categories: list[MemoryCategory] | None = None,
        kinds: list[MemoryItemKind] | None = None,
    ) -> list[MemoryItem]:
        """联合列出允许 namespace 中的近期活跃长期记忆条目。

        Args:
            namespaces: 允许读取的 namespace 集合。
            limit (int | None): 最大返回数量约束；None 表示读取完整 active 投影。
            categories (list[MemoryCategory] | None): 可选的类别过滤条件。
            kinds (list[MemoryItemKind] | None): 可选的条目类型过滤条件。

        Returns:
            list[MemoryItem]: 符合限制大小的数据片段集。
        """
        return self.store.list_items(
            namespaces,
            limit=limit,
            categories=categories,
            kinds=kinds,
        )

    async def alist_items(
        self,
        namespaces: Sequence[str],
        *,
        limit: int | None = 50,
        categories: list[MemoryCategory] | None = None,
        kinds: list[MemoryItemKind] | None = None,
    ) -> list[MemoryItem]:
        """异步适配长期记忆列表读取。"""
        return await self.run_async_io(
            lambda: self.list_items(
                namespaces,
                limit=limit,
                categories=categories,
                kinds=kinds,
            )
        )

    def list_events(
        self,
        namespace: str,
        *,
        item_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        """列出指定 namespace 下的审计事件日志。

        Args:
            namespace (str): 日志相关的作用域空间。
            item_id (str | None): 可选的限定日志针对某一条记忆发生。
            limit (int): 限制结果数。

        Returns:
            list[MemoryEvent]: 用于还原操作历史记录的事件集。
        """
        return self.store.list_events(namespace, item_id=item_id, limit=limit)

    async def alist_events(
        self,
        namespace: str,
        *,
        item_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        """异步适配记忆审计事件列表读取。"""
        return await self.run_async_io(
            lambda: self.list_events(namespace, item_id=item_id, limit=limit)
        )

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
            namespace=candidate.namespace,
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

        return stored

    def list_candidates(
        self,
        namespace: str,
        *,
        status: MemoryCandidateStatus | None = None,
        limit: int = 50,
    ) -> list[MemoryCandidate]:
        """列出指定 namespace 下的候选记忆。

        Args:
            namespace (str): 获取候选态记录的所属范围。
            status (MemoryCandidateStatus | None): 对记录审核阶段进行过滤。
            limit (int): 分页最大返回长度。

        Returns:
            list[MemoryCandidate]: 对应的候选态结果集合。
        """
        return self.store.list_candidates(namespace, status=status, limit=limit)

    def promote_candidate(
        self,
        candidate_id: str,
        namespace: str,
        *,
        kind: MemoryItemKind,
        actor: MemoryActor = MemoryActor.SDK,
        reason: str,
    ) -> MemoryItem:
        """原子晋升 pending candidate 为 L2 item。

        Args:
            candidate_id (str): 目标候选记忆 ID。
            namespace (str): 候选资源所在隔离范围。
            kind (MemoryItemKind): 晋升后的长期记忆类型。
            actor (MemoryActor): 发起晋升操作的参与实体。
            reason (str): 通过晋升策略的原因。

        Returns:
            MemoryItem: 晋升后可召回的 L2 item。
        """
        return self.promote_candidates(
            namespace,
            [(candidate_id, kind, reason)],
            actor=actor,
        )[0]

    def promote_candidates(
        self,
        namespace: str,
        promotions: Iterable[tuple[str, MemoryItemKind, str]],
        *,
        actor: MemoryActor = MemoryActor.SDK,
    ) -> list[MemoryItem]:
        """逐项原子晋升，并统一刷新本批已提交条目的镜像。

        Args:
            namespace: 本批候选所属的隔离范围。
            promotions: 按处理顺序提供候选 ID、长期记忆类型和晋升原因。
            actor: 发起晋升操作的参与实体。

        Returns:
            list[MemoryItem]: 按输入顺序返回已晋升的条目。

        Raises:
            IrisMemoryError: store 操作失败；此前成功项仍已提交。自动镜像失败只告警。
        """
        items: list[MemoryItem] = []
        try:
            for candidate_id, kind, reason in promotions:
                items.append(
                    self.store.promote_candidate(
                        candidate_id, namespace, kind=kind, actor=actor, reason=reason
                    )
                )
        finally:
            if items:
                self._rebuild_committed(namespace)
        return items

    def accept_candidate(
        self,
        candidate_id: str,
        namespace: str,
        *,
        actor: MemoryActor = MemoryActor.SDK,
        reason: str,
    ) -> MemoryCandidate:
        """将候选记忆标记为已接受。

        标志候选信息通过评估阶段，允许被视作已验证的知识内容。

        Args:
            candidate_id (str): 唯一的候选实体标识。
            namespace (str): 候选资源所在隔离范围。
            actor (MemoryActor): 发起评估操作对象。
            reason (str): 批准其通过的特定原因。

        Returns:
            MemoryCandidate: 状态更新后的全新实体对象。
        """
        return self._update_candidate_status(
            candidate_id,
            namespace,
            MemoryCandidateStatus.ACCEPTED,
            event_type=MemoryEventType.CANDIDATE_ACCEPT,
            actor=actor,
            reason=reason,
        )

    def reject_candidate(
        self,
        candidate_id: str,
        namespace: str,
        *,
        actor: MemoryActor = MemoryActor.SDK,
        reason: str,
    ) -> MemoryCandidate:
        """将候选记忆标记为已拒绝。

        丢弃低价值信息或已失效的识别结论。

        Args:
            candidate_id (str): 唯一的候选实体标识。
            namespace (str): 候选资源所在隔离范围。
            actor (MemoryActor): 发起拒绝的执行方。
            reason (str): 说明信息为何不满足长期记忆要求。

        Returns:
            MemoryCandidate: 退回并标记丢弃后的实体表示。
        """
        return self._update_candidate_status(
            candidate_id,
            namespace,
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
    def file_access(self, read_namespaces: Sequence[str]) -> MemoryFileAccess | None:
        """提供通用文件工具所需的当前只读路径和版本接口。"""
        if self.mirror is None:
            return None
        return MemoryFileAccess(self.mirror.root, tuple(read_namespaces), self.store)

    def projection_warning(self, namespace: str) -> str | None:
        """返回已保存数据库条目尚未完整投影时的可见说明。"""
        if self.mirror is None:
            return None
        state = self.store.read_namespace_state(namespace)
        return freshness_warning(state, state.item_revision)

    def load_overviews(self, namespaces: Sequence[str]) -> tuple[MemoryOverviewDocument, ...]:
        """一次读取已发布概览；缺文件只返回确定的缺产物说明。"""
        if self.mirror is None:
            return ()
        documents: list[MemoryOverviewDocument] = []
        for namespace in namespaces:
            published = self.mirror.read_overview(namespace)
            if published is None:
                revision = None
                text = navigation = "尚未生成概览，知识范围未知。"
                warning = None
            else:
                revision, text, navigation = published
                state = self.store.read_namespace_state(namespace)
                warning = freshness_warning(state, revision, overview=True)
            documents.append(
                MemoryOverviewDocument(
                    namespace=namespace,
                    path=self.mirror.document_path(namespace),
                    source_revision=revision,
                    text=text,
                    navigation=navigation,
                    warning=warning,
                )
            )
        return tuple(documents)

    async def aload_overviews(
        self, namespaces: Sequence[str]
    ) -> tuple[MemoryOverviewDocument, ...]:
        """把全部 namespace 的概览读取放在同一个 async IO job 内。"""
        return await self.run_async_io(lambda: self.load_overviews(namespaces))

    async def refresh_overview(self, namespace: str) -> MemoryOverviewGenerationResult:
        """显式生成和发布概览，保留旧完整产物及实际模型用量。"""
        if self.mirror is None or self.overview_provider is None or self.overview_model is None:
            raise IrisMemoryError("memory 概览生成依赖未配置", namespace=namespace)
        started = perf_counter()
        snapshot = await self.run_async_io(lambda: self.store.read_namespace_snapshot(namespace))
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        response = None
        if snapshot.items:
            request = build_overview_request(snapshot, self.overview_model, self.overview_config)
            estimated_tokens = self.overview_provider.estimate_input_tokens(request)
            if estimated_tokens > self.overview_config.input_budget_tokens:
                raise IrisMemoryError(
                    "memory 概览生成输入容量不足，保留最后完整产物",
                    namespace=namespace,
                    estimated_tokens=estimated_tokens,
                    input_budget_tokens=self.overview_config.input_budget_tokens,
                )
            response = await self.overview_provider.complete(request)
            usage = {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.total_tokens,
            }
        try:
            content = (
                complete_overview_content(response)
                if response is not None
                else MemoryOverviewContent(core_facts="", knowledge_scope="当前无记忆。")
            )
            published, state = await self.run_async_io(
                lambda: self.mirror.publish_overview(self.store, snapshot, content)
            )
        except IrisMemoryError as exc:
            exc.context.update(
                namespace=namespace,
                source_revision=snapshot.state.item_revision,
                usage=usage,
                elapsed_seconds=perf_counter() - started,
            )
            raise
        return MemoryOverviewGenerationResult(
            namespace=namespace,
            path=self.mirror.namespace_directory(namespace) / "Memory.md",
            source_revision=snapshot.state.item_revision,
            current_revision=state.item_revision,
            projection_revision=state.projection_revision,
            item_count=len(snapshot.items),
            published=published,
            publication_reason=None if published else "已生成但未发布，已有更新版本",
            usage=usage,
            elapsed_seconds=perf_counter() - started,
        )

    def _update_candidate_status(
        self,
        candidate_id: str,
        namespace: str,
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
            namespace (str): 所属空间范围。
            status (MemoryCandidateStatus): 要刷新为的目标态值。
            event_type (MemoryEventType): 同步写入的对应的事件类型。
            actor (MemoryActor): 触发操作发起方。
            reason (str): 操作详细理由。

        Returns:
            MemoryCandidate: 成功变更状态位后的候选实例对象。
        """
        event = MemoryEvent(
            namespace=namespace,
            event_type=event_type,
            actor=actor,
            reason=reason,
            metadata={"candidate_id": candidate_id, "candidate_status": status.value},
        )
        stored = self.store.update_candidate_status(
            candidate_id,
            namespace,
            status,
            event=event,
        )

        return stored

    def _rebuild_committed(self, namespace: str) -> None:
        """数据库成功后同步完整正文，失败由版本状态对读取方明示。"""
        if self.mirror is not None:
            try:
                self.mirror.rebuild_from_store(self.store, namespace)
            except Exception:
                logger.warning(
                    "memory mirror 自动重建失败；数据库写入已提交 namespace=%s",
                    namespace,
                    exc_info=True,
                )

    # endregion
