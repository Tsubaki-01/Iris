"""Runner 独占的空闲记忆维护；前台优先，已派发 IO 在关闭时完整收口。"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import cast

from ..lifecycle import LifecycleStore, RunRecord
from ..memory import MemoryService
from ..memory.generation_models import GenerationResult
from ..memory.mirror import FileMemoryMirror
from ._memory_capture import capture_episode, capture_source

logger = logging.getLogger(__name__)


class MemoryMaintenance:
    """串行驱动一个 namespace 的 flush、dream 与概览发布。"""

    def __init__(
        self, *, service: MemoryService, namespace: str, lifecycle_store: LifecycleStore
    ) -> None:
        """绑定服务；构造时不创建 task 或模型调用。"""
        self.service = service
        self.namespace = namespace
        self.lifecycle_store = lifecycle_store
        self._loop: asyncio.AbstractEventLoop | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._task: asyncio.Task[None] | None = None
        self._foreground = 0
        self._dirty = False
        self._wake_revision = 0
        self._cycle_revision = 0
        self._closed = False
        self._capture_requests: dict[str, int | None] = {}
        self._capture_task: asyncio.Task[None] | None = None
        self._known_runs: dict[str, RunRecord] = {}

    @property
    def foreground_active(self) -> bool:
        """是否仍有 admission 或 activation 调用尚未完整退出。"""
        return self._foreground > 0

    async def prepare(self) -> None:
        """首次准备订阅服务变化并安排一次积压检查。"""
        if self._loop is not None:
            return
        self._loop = asyncio.get_running_loop()
        self.service.add_change_listener(self._on_change)
        try:
            await self.service.run_async_io(
                lambda: self.service.store.retry_blocked(
                    self.namespace, budget=self.service.generation_config.dream_input_budget_tokens
                )
            )
        except Exception as exc:
            await self._capture_failed(exc)
        await self.capture_pending()
        self._dirty = True
        self._schedule()

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
            await self.service.run_async_io(self._capture_pending_sync)
        except Exception as exc:
            await self._capture_failed(exc)
        self._dirty = True

    def _capture_pending_sync(self) -> None:
        """一份 worker job 完成来源读取和持久 capture，连接各归原 store 管理。"""
        for run in tuple(self._known_runs.values()):
            self.service.store.register_source(
                capture_source(
                    run, source_id=self.lifecycle_store.source_id, namespace=self.namespace
                )
            )
            self._known_runs.pop(run.run_id, None)
        sources = self.service.store.list_capture_sources(
            self.lifecycle_store.source_id, self.namespace
        )
        for source in sources:
            self._capture_source_sync(source.run_id, None)

    def _capture_source_sync(self, run_id: str, through_count: int | None) -> None:
        """读取精确后缀，水位 CAS 防止重复提示创建重复经历。"""
        run = self.lifecycle_store.load_run(run_id)
        if run is None:
            return
        source = self.service.store.register_source(
            capture_source(run, source_id=self.lifecycle_store.source_id, namespace=self.namespace)
        )
        if source.terminal_message_count is not None:
            return
        messages = self.lifecycle_store.load_run_message_slice(run_id, source.captured_until)
        updated, episode = capture_episode(source, messages, through_count=through_count)
        if updated == source:
            return
        self.service.store.commit_capture(
            updated, expected_captured_until=source.captured_until, episode=episode
        )

    async def _drain_capture_requests(self) -> None:
        """处理压缩/同步取消提示；请求到达时只扩展内存中的待取范围。"""
        try:
            while self._capture_requests:
                run_id, through_count = self._capture_requests.popitem()
                try:
                    await self.service.run_async_io(
                        partial(self._capture_source_sync, run_id, through_count)
                    )
                except Exception as exc:
                    await self._capture_failed(exc, run_id)
        finally:
            self._capture_task = None
            self._wake()

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

    def foreground_enter(self) -> None:
        """立即停止空闲计时和未提交的模型工作，不阻塞前台入场。"""
        self._foreground += 1
        self._dirty = True
        self._wake_revision += 1
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._task is not None:
            self._task.cancel()

    def foreground_exit(self) -> None:
        """完整前台调用退出后重新计算安静时间。"""
        self._foreground -= 1
        self._schedule()

    async def aclose(self) -> None:
        """取消维护并等待真实 IO 结束，不关闭宿主拥有的服务。"""
        if self._closed:
            return
        self._closed = True
        if self._loop is not None:
            self.service.remove_change_listener(self._on_change)
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        task = self._task
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self._capture_task is not None:
            await self._capture_task
        await self.capture_pending()
        await self.service.wait_pending_io()

    def _on_change(self, namespace: str) -> None:
        """服务写入可在线程中完成，通知回到本 runner 的 event loop。"""
        if namespace == self.namespace and self._loop is not None and not self._closed:
            self._loop.call_soon_threadsafe(self._wake)

    def _wake(self) -> None:
        """合并变化；运行中的流水线结束后再决定是否继续。"""
        self._dirty = True
        self._wake_revision += 1
        self._schedule()

    def _schedule(self) -> None:
        """仅有待检查工作且没有前台/维护任务时安排一次计时。"""
        if (
            self._closed
            or self._loop is None
            or self._foreground
            or self._task is not None
            or self._timer is not None
            or not self._dirty
        ):
            return
        self._timer = self._loop.call_later(
            self.service.generation_config.idle_seconds, self._start
        )

    def _start(self) -> None:
        """计时完成后启动唯一维护 task。"""
        self._timer = None
        self._dirty = False
        self._cycle_revision = self._wake_revision
        self._task = asyncio.create_task(self._run_cycle())
        self._task.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Task[None]) -> None:
        """真实模型 task 已退出才允许下一轮；失败等待新的外部触发。"""
        self._task = None
        if not task.cancelled() and task.exception() is not None:
            logger.error("记忆维护失败", exc_info=task.exception())
            self._dirty = self._wake_revision != self._cycle_revision
        self._schedule()

    async def _run_cycle(self) -> None:
        """先让一批知识整理并发布，再安排下一批待提炼资料。"""
        state = await self.service.ageneration_state(self.namespace)
        if state.pending_observations or state.pending_changes:
            result = await self.service.dream(self.namespace)
            if result.status in {"failed", "cancelled", "conflict"}:
                self._dirty = self._wake_revision != self._cycle_revision
                return
        elif state.pending_episodes:
            result = await self.service.flush(self.namespace)
            if result.status in {"failed", "cancelled", "conflict"}:
                self._dirty = self._wake_revision != self._cycle_revision
                return
            state = await self.service.ageneration_state(self.namespace)
            if state.pending_observations or state.pending_changes:
                result = await self.service.dream(self.namespace)
                if result.status in {"failed", "cancelled", "conflict"}:
                    self._dirty = self._wake_revision != self._cycle_revision
                    return
        state = await self.service.ageneration_state(self.namespace)
        if state.projection_revision != state.item_revision:
            mirror = cast(FileMemoryMirror, self.service.mirror)
            await self.service.run_async_io(
                lambda: mirror.rebuild_from_store(self.service.store, self.namespace)
            )
        if state.item_revision and state.overview_revision != state.item_revision:
            await self.service.refresh_overview(self.namespace)
        state = await self.service.ageneration_state(self.namespace)
        self._dirty = bool(
            state.pending_episodes
            or state.pending_observations
            or state.pending_changes
            or (state.item_revision and state.overview_revision != state.item_revision)
            or self._wake_revision != self._cycle_revision
        )
