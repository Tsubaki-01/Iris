"""Runner 的原文捕获队列；只保存事实，不调度学习或持有维护锁。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from functools import partial

from ..evolution.models import EvolutionCaptureBlock, EvolutionRecord, EvolutionSource
from ..evolution.service import EvolutionService
from ..lifecycle import LifecycleStore, RunRecord
from ..memory import MemoryService
from ..memory.generation_models import GenerationResult
from ._capture_records import capture_records
from ._memory_capture import capture_episode, capture_source

logger = logging.getLogger(__name__)
_CAPTURE_BATCH_SIZE = 128


class RunCapture:
    """按原文水位分批捕获一个 lifecycle 来源。"""

    def __init__(
        self,
        *,
        memory_service: MemoryService | None = None,
        namespace: str = "project",
        evolution_service: EvolutionService | None = None,
        lifecycle_store: LifecycleStore,
        on_memory_capture: Callable[[], None] = lambda: None,
        on_evolution_capture: Callable[[], None] = lambda: None,
    ) -> None:
        """绑定捕获依赖；不拥有共享服务和 lifecycle reader。"""
        self.service = memory_service
        self.evolution_service = evolution_service
        self.namespace = namespace
        self.lifecycle_store = lifecycle_store
        self._on_memory_capture = on_memory_capture
        self._on_evolution_capture = on_evolution_capture
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._capture_requests: dict[str, int | None] = {}
        self._capture_task: asyncio.Task[None] | None = None
        self._known_runs: dict[str, RunRecord] = {}

    async def prepare(self) -> None:
        """首次进入事件循环时补采已登记来源。"""
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
            await self.capture_pending()

    async def aclose(self) -> None:
        """排空原文捕获，不生成内容或关闭共享依赖。"""
        if self._closed:
            return
        self._closed = True
        if self._capture_task is not None:
            await self._capture_task
        await self.capture_pending()

    async def register_run(self, run: RunRecord) -> None:
        """第一条原文提交前登记来源；失败不影响主 run，后续边界再次尝试。"""
        self._known_runs[run.run_id] = run
        failed = False
        if self.service is not None:
            try:
                await self.service.run_async_io(
                    lambda: self.service.store.register_source(
                        capture_source(
                            run, source_id=self.lifecycle_store.source_id, namespace=self.namespace
                        )
                    )
                )
            except Exception as exc:
                failed = True
                await self._capture_failed(exc, run.run_id)
        if self.evolution_service is not None:
            try:
                await self.evolution_service.run_async_io(
                    lambda: self.evolution_service.store.register_source(
                        EvolutionSource.model_construct(
                            lifecycle_source_id=self.lifecycle_store.source_id,
                            run_id=run.run_id,
                            session_id=run.session_id,
                        ),
                        run.initial_session_message_count,
                    )
                )
            except Exception as exc:
                failed = True
                await self._capture_failed(exc, run.run_id, evolution=True)
        if not failed:
            self._known_runs.pop(run.run_id, None)

    def request_capture(self, run_id: str, through_count: int) -> None:
        """合并 runtime 提示，捕获可在前台执行中进行，模型维护仍等待空闲。"""
        current = self._capture_requests.get(run_id, 0)
        self._capture_requests[run_id] = None if current is None else max(current, through_count)
        if self._loop is not None and not self._closed and self._capture_task is None:
            self._capture_task = asyncio.create_task(self._drain_capture_requests())

    async def capture_pending(self) -> None:
        """恢复同源已登记后缀，并补采当前 runner 已知但登记曾失败的 run。"""
        for run in tuple(self._known_runs.values()):
            await self.register_run(run)
        if self.service is not None:
            try:
                sources = await self.service.run_async_io(
                    lambda: self.service.store.list_capture_sources(
                        self.lifecycle_store.source_id, self.namespace
                    )
                )
                for source in sources:
                    await self._capture_memory_source(source.run_id, None)
            except Exception as exc:
                await self._capture_failed(exc)
        if self.evolution_service is not None:
            try:
                project_sources = await self.evolution_service.run_async_io(
                    lambda: self.evolution_service.store.list_capture_sources(
                        self.lifecycle_store.source_id
                    )
                )
                for source in project_sources:
                    await self._capture_evolution_source(source.source.run_id, None)
            except Exception as exc:
                await self._capture_failed(exc, evolution=True)

    async def _capture_source(self, run_id: str, through_count: int | None) -> None:
        """将同一事实提示分发给两类消费者，失败互不阻塞。"""
        if self.service is not None:
            try:
                await self._capture_memory_source(run_id, through_count)
            except Exception as exc:
                await self._capture_failed(exc, run_id)
        if self.evolution_service is not None:
            try:
                await self._capture_evolution_source(run_id, through_count)
            except Exception as exc:
                await self._capture_failed(exc, run_id, evolution=True)

    async def _capture_memory_source(self, run_id: str, through_count: int | None) -> None:
        """每页独立读取和提交，在页间让出事件循环，不占用后台生成 worker。"""
        while True:
            more, captured = await self.service.run_async_io(
                partial(self._capture_source_sync, run_id, through_count)
            )
            if captured:
                self._on_memory_capture()
            if not more:
                return
            await asyncio.sleep(0)

    def _capture_source_sync(self, run_id: str, through_count: int | None) -> tuple[bool, bool]:
        """读取一页精确后缀；返回是否继续，水位 CAS 防止重复经历。"""
        run = self.lifecycle_store.load_run(run_id)
        if run is None:
            return False, False
        source = self.service.store.register_source(
            capture_source(run, source_id=self.lifecycle_store.source_id, namespace=self.namespace)
        )
        if source.terminal_message_count is not None:
            return False, False
        messages = self.lifecycle_store.load_run_message_slice(
            run_id, source.captured_until, limit=_CAPTURE_BATCH_SIZE
        )
        updated, episode = capture_episode(source, messages, through_count=through_count)
        if updated == source:
            return False, False
        committed = self.service.store.commit_capture(
            updated, expected_captured_until=source.captured_until, episode=episode
        )
        more = not committed or (
            updated.terminal_message_count is None
            and len(messages.messages) == _CAPTURE_BATCH_SIZE
            and (through_count is None or updated.captured_until < through_count)
        )
        return more, committed

    async def _capture_evolution_source(self, run_id: str, through_count: int | None) -> None:
        """项目材料独立发布分页，不借用 Memory 数据库或生成 worker。"""
        while True:
            more, captured = await self.evolution_service.run_async_io(
                partial(self._capture_evolution_source_sync, run_id, through_count)
            )
            if captured:
                self._on_evolution_capture()
            if not more:
                return
            await asyncio.sleep(0)

    def _capture_evolution_source_sync(
        self, run_id: str, through_count: int | None
    ) -> tuple[bool, bool]:
        """发布完整且独立命名的块，连续水位由材料 store 根据已发布区间投影。"""
        run = self.lifecycle_store.load_run(run_id)
        if run is None:
            return False, False
        store = self.evolution_service.store
        source = EvolutionSource.model_construct(
            lifecycle_source_id=self.lifecycle_store.source_id,
            run_id=run.run_id,
            session_id=run.session_id,
        )
        state = store.register_source(source, run.initial_session_message_count)
        if state.terminal_message_count is not None:
            return False, False
        messages = self.lifecycle_store.load_run_message_slice(
            run_id, state.captured_until, limit=_CAPTURE_BATCH_SIZE
        )
        end = messages.end_message_count
        if through_count is not None:
            end = max(messages.start_message_count, min(end, through_count))
        terminal = (
            messages.terminal_message_count if end == messages.terminal_message_count else None
        )
        if end == state.captured_until and terminal is None:
            return False, False
        records = tuple(
            EvolutionRecord.model_construct(
                ref=record.ref,
                role=record.role,
                text=record.text,
                occurred_at=record.occurred_at,
                message_ordinal=record.message_ordinal,
                block_index=record.block_index,
                metadata=record.metadata,
            )
            for record in capture_records(messages, end=end)
        )
        updated = store.commit_capture(
            EvolutionCaptureBlock.model_construct(
                source=source,
                initial_message_count=state.initial_message_count,
                start_message_count=messages.start_message_count,
                end_message_count=end,
                terminal_message_count=terminal,
                outcome=messages.outcome.value
                if terminal is not None and messages.outcome
                else None,
                records=records,
            )
        )
        more = (
            updated.terminal_message_count is None
            and len(messages.messages) == _CAPTURE_BATCH_SIZE
            and (through_count is None or updated.captured_until < through_count)
        )
        return more, True

    async def _drain_capture_requests(self) -> None:
        """处理压缩/同步取消提示；请求到达时只扩展内存中的待取范围。"""
        try:
            while self._capture_requests:
                run_id, through_count = self._capture_requests.popitem()
                await self._capture_source(run_id, through_count)
        finally:
            self._capture_task = None

    async def _capture_failed(
        self, error: Exception, run_id: str | None = None, *, evolution: bool = False
    ) -> None:
        """将 capture 故障留在记忆状态中，主任务仍使用自己的完成结果。"""
        if evolution:
            logger.warning("项目经验原文捕获失败 run=%s", run_id, exc_info=error)
            return
        logger.warning("记忆原文捕获失败", exc_info=error)
        result = GenerationResult(
            namespace=self.namespace,
            stage="capture",
            status="failed",
            error=str(error),
            input_ids=() if run_id is None else (run_id,),
        )
        try:
            await self.service.run_async_io(
                lambda: self.service.store.record_generation_result(result)
            )
        except Exception:
            logger.exception("无法记录记忆捕获失败状态")
