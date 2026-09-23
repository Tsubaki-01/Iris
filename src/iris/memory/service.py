"""记忆 SDK 服务层。

屏蔽了底层的存储细节与文件镜像等逻辑，统一对外提供操作长期记忆的聚合方法。
核心暴露 MemoryService 类实例作为访问入口点。

Example:
    service = MemoryService(store=sqlite_store)
    episode = service.observe(input_data)
    observations = service.store.list_observations(namespace)
"""

# region imports
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from enum import StrEnum
from time import perf_counter
from typing import TypeVar

from ..exceptions import IrisMemoryError
from ..providers.protocols import CompletionProvider
from .files import MemoryFileAccess, freshness_warning
from .generation import before_generation_commit, dream, flush, raise_if_generation_cancelled
from .generation_models import GenerationResult, GenerationState, MemoryGenerationConfig
from .mirror import FileMemoryMirror
from .models import (
    MemoryActor,
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
    MemoryRecord,
    MemorySearchQuery,
    MemorySearchResponse,
    MemorySourceType,
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
        generation_provider: CompletionProvider | None = None,
        generation_model: str | None = None,
        generation_config: MemoryGenerationConfig | None = None,
        io_execution_mode: MemoryIOExecutionMode = MemoryIOExecutionMode.INLINE,
    ) -> None:
        """初始化记忆服务。"""
        self.store = store
        self.mirror = mirror
        self.overview_provider = overview_provider
        self.overview_model = overview_model
        self.overview_config = overview_config or MemoryOverviewConfig()
        self.generation_provider = generation_provider
        self.generation_model = generation_model
        self.generation_config = generation_config or MemoryGenerationConfig()
        self._io_execution_mode = io_execution_mode
        self._io_tasks: set[asyncio.Task[object]] = set()
        self._change_listeners: list[Callable[[str], None]] = []

    @property
    def io_execution_mode(self) -> MemoryIOExecutionMode:
        """返回 async IO 适配器使用的固定执行模式。"""
        return self._io_execution_mode

    async def run_async_io(
        self, operation: Callable[[], ResultT], *, complete_on_cancel: bool = False
    ) -> ResultT:
        """执行完整同步 IO；短提交可要求取消后收取真实回执。

        complete_on_cancel 保留当前 task 的取消状态，由阶段在结果落账后继续传播。
        """
        if self._io_execution_mode is MemoryIOExecutionMode.THREAD:
            task = asyncio.create_task(asyncio.to_thread(operation))
            self._io_tasks.add(task)
            task.add_done_callback(self._finish_io)
            while True:
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    if not complete_on_cancel or task.cancelled():
                        raise
        return operation()

    def _finish_io(self, task: asyncio.Task[object]) -> None:
        """保留取消等待后真实 IO 的生命周期，并回收已完成的结果。"""
        self._io_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def wait_pending_io(self) -> None:
        """等待已派发数据库工作真正结束，供维护关闭时调用。"""
        while self._io_tasks:
            await asyncio.gather(*tuple(self._io_tasks), return_exceptions=True)

    def add_change_listener(self, callback: Callable[[str], None]) -> None:
        """订阅新材料和正式写入；回调在实际写入所在线程执行。"""
        self._change_listeners.append(callback)

    def remove_change_listener(self, callback: Callable[[str], None]) -> None:
        """释放当前宿主注册的变更回调。"""
        self._change_listeners.remove(callback)

    def _notify_change(self, namespace: str) -> None:
        """持久写入后唤醒维护；通知失败不改变已提交写入的结果。"""
        for callback in tuple(self._change_listeners):
            try:
                callback(namespace)
            except Exception:
                logger.warning("memory 维护通知失败 namespace=%s", namespace, exc_info=True)

    def generation_state(self, namespace: str) -> GenerationState:
        """读取生成积压、受阻输入及最近阶段结果。"""
        state = self.store.generation_state(namespace)
        if self.mirror is None:
            return state
        overview = self.mirror.read_overview(namespace)
        return replace(state, overview_revision=overview[0] if overview is not None else None)

    async def ageneration_state(self, namespace: str) -> GenerationState:
        """在一次 IO 内读取生成状态。"""
        return await self.run_async_io(lambda: self.generation_state(namespace))

    async def flush(self, namespace: str) -> GenerationResult:
        """从已捕获经历提炼一批观察；正式知识在 dreaming 后才可读取。"""
        return await flush(self, namespace)

    async def dream(self, namespace: str, *, retry_blocked: bool = False) -> GenerationResult:
        """将一批观察和显式修改原子整理到正式知识，不隐式执行 flush。"""
        return await dream(self, namespace, retry_blocked=retry_blocked)

    # endregion

    # ==========================================
    #           Episode / Item Core
    # ==========================================
    # region
    def observe(self, input: MemoryObserveInput) -> MemoryEpisode:
        """记录有明确边界的原始经历，留给 flush 提炼。

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
            records=input.records
            or (
                MemoryRecord(
                    text=input.text,
                    source_type=input.source_type,
                    source_id=input.source_id,
                    artifacts=tuple(input.artifacts),
                ),
            ),
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
        self._notify_change(stored.namespace)
        return stored

    def remember(self, input: MemoryWriteInput) -> MemoryItem:
        """显式写入正式知识，并保留本次声明的真实来源。

        用于跨会话的高价值信息持久化，不需要先构造观察或调用模型。

        Args:
            input (MemoryWriteInput): 包含作用域、分类及来源等元数据的写请求。

        Returns:
            MemoryItem: 构造完整并被持久化后的权威长期记忆记录。
        """
        item = MemoryItem.model_construct(
            namespace=input.namespace,
            text=input.text,
            category=input.category,
            kind=input.kind,
            source_type=input.source_type,
            source_id=input.source_id,
            reason=input.reason,
            evidence=input.evidence,
            artifacts=input.artifacts,
            metadata=input.metadata,
        )
        event = MemoryEvent(
            namespace=input.namespace,
            event_type=MemoryEventType.ADD,
            actor=input.actor,
            item_id=item.id,
            reason=input.reason,
            source_type=input.source_type,
            source_id=input.source_id,
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
        source_type: MemorySourceType = MemorySourceType.SDK,
        source_id: str = "",
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
            source_type=source_type,
            source_id=source_id,
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
        source_type: MemorySourceType = MemorySourceType.SDK,
        source_id: str = "",
    ) -> MemoryItem:
        """在一次 async IO 操作中完成更新及派生镜像刷新。"""
        return await self.run_async_io(
            lambda: self.update(
                item_id,
                namespace,
                patch,
                actor=actor,
                reason=reason,
                source_type=source_type,
                source_id=source_id,
            )
        )

    # endregion

    # ==========================================
    #         Query & Management Methods
    # ==========================================
    # region
    def search(self, query: MemorySearchQuery, namespaces: Sequence[str]) -> MemorySearchResponse:
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
        source_type: MemorySourceType = MemorySourceType.SDK,
        source_id: str = "",
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
            source_type=source_type,
            source_id=source_id,
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
        source_type: MemorySourceType = MemorySourceType.SDK,
        source_id: str = "",
    ) -> bool:
        """在一次 async IO 操作中完成软删除及派生镜像刷新。"""
        return await self.run_async_io(
            lambda: self.forget(
                item_id,
                namespace,
                actor=actor,
                reason=reason,
                source_type=source_type,
                source_id=source_id,
            )
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
        try:
            response = None
            if snapshot.items:
                request = build_overview_request(
                    snapshot, self.overview_model, self.overview_config
                )
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
            content = (
                complete_overview_content(response)
                if response is not None
                else MemoryOverviewContent(core_facts="", knowledge_scope="当前无记忆。")
            )
            await before_generation_commit()
            published, state = await self.run_async_io(
                lambda: self.mirror.publish_overview(self.store, snapshot, content),
                complete_on_cancel=True,
            )
        except (Exception, asyncio.CancelledError) as exc:
            if isinstance(exc, IrisMemoryError):
                exc.context.update(
                    namespace=namespace,
                    source_revision=snapshot.state.item_revision,
                    usage=usage,
                    elapsed_seconds=perf_counter() - started,
                )
            failure = GenerationResult(
                namespace=namespace,
                stage="overview",
                status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                usage=usage,
                elapsed_seconds=perf_counter() - started,
                error=str(exc) or type(exc).__name__,
                item_revision=snapshot.state.item_revision,
            )
            await self.run_async_io(
                lambda: self.store.record_generation_result(failure), complete_on_cancel=True
            )
            raise
        result = GenerationResult(
            namespace=namespace,
            stage="overview",
            status="completed",
            usage=usage,
            elapsed_seconds=perf_counter() - started,
            item_revision=snapshot.state.item_revision,
            counts={"items": len(snapshot.items), "published": int(published)},
        )
        await self.run_async_io(
            lambda: self.store.record_generation_result(result), complete_on_cancel=True
        )
        raise_if_generation_cancelled()
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
        self._notify_change(namespace)

    # endregion
