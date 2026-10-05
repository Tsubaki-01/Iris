"""Runner 的原文捕获队列；只保存事实，不调度学习或持有维护锁。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from functools import partial

from ..lifecycle import LifecycleStore, RunRecord
from ..memory import MemoryService
from ..memory.generation_models import GenerationResult
from ._memory_capture import capture_episode, capture_source

logger = logging.getLogger(__name__)
_CAPTURE_BATCH_SIZE = 128


class MemoryCapture:
    """按原文水位分批捕获一个 lifecycle 来源。"""

    def __init__(
        self,
        *,
        service: MemoryService,
        namespace: str,
        lifecycle_store: LifecycleStore,
        on_capture: Callable[[], None] = lambda: None,
    ) -> None:
        """绑定捕获依赖；不拥有共享服务和 lifecycle reader。"""
        self.service = service
        self.namespace = namespace
        self.lifecycle_store = lifecycle_store
        self._on_capture = on_capture
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
        try:
            await self.service.run_async_io(
                lambda: self.service.store.register_source(
                    capture_source(
                        run, source_id=self.lifecycle_store.source_id, namespace=self.namespace
                    )
                )
            )
            self._known_runs.pop(run.run_id, None)
        except Exception as exc:
            await self._capture_failed(exc, run.run_id)

    def request_capture(self, run_id: str, through_count: int) -> None:
        """合并 runtime 提示，捕获可在前台执行中进行，模型维护仍等待空闲。"""
        current = self._capture_requests.get(run_id, 0)
        self._capture_requests[run_id] = None if current is None else max(current, through_count)
        if self._loop is not None and not self._closed and self._capture_task is None:
            self._capture_task = asyncio.create_task(self._drain_capture_requests())

    async def capture_pending(self) -> None:
        """恢复同源已登记后缀，并补采当前 runner 已知但登记曾失败的 run。"""
        try:
            for run in tuple(self._known_runs.values()):
                await self.register_run(run)
            sources = await self.service.run_async_io(
                lambda: self.service.store.list_capture_sources(
                    self.lifecycle_store.source_id, self.namespace
                )
            )
            for source in sources:
                await self._capture_source(source.run_id, None)
        except Exception as exc:
            await self._capture_failed(exc)

    async def _capture_source(self, run_id: str, through_count: int | None) -> None:
        """每页独立读取和提交，在页间让出事件循环，不占用后台生成 worker。"""
        while True:
            more, captured = await self.service.run_async_io(
                partial(self._capture_source_sync, run_id, through_count)
            )
            if captured:
                self._on_capture()
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

    async def _drain_capture_requests(self) -> None:
        """处理压缩/同步取消提示；请求到达时只扩展内存中的待取范围。"""
        try:
            while self._capture_requests:
                run_id, through_count = self._capture_requests.popitem()
                try:
                    await self._capture_source(run_id, through_count)
                except Exception as exc:
                    await self._capture_failed(exc, run_id)
        finally:
            self._capture_task = None

    async def _capture_failed(self, error: Exception, run_id: str | None = None) -> None:
        """将 capture 故障留在记忆状态中，主任务仍使用自己的完成结果。"""
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
