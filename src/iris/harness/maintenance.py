"""宿主级共享维护调度；Memory 负责领域执行，OS 锁保护跨进程完整周期。"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock, Timeout

from ..exceptions import IrisConfigError, IrisRunStateError
from ..lifecycle import LifecycleStore, RunPhase
from ..memory import MemoryService
from ..memory._generation_worker import GenerationWorker, generation_worker
from ..memory.generation_models import MemoryMaintenanceScope, MemorySource

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryMaintenanceBinding:
    """宿主选定的实际 Memory 资源和完整领域服务，不探测 store 私有属性。"""

    service: MemoryService
    database_path: Path
    namespace: str


@dataclass(slots=True)
class _MemoryResource:
    """一个 DB/namespace 的进程内调度状态。"""

    binding: MemoryMaintenanceBinding
    lock_path: Path
    dirty: bool = True
    revision: int = 0
    ready_at: float = 0
    listener: Callable[[str], None] | None = None
    attachments: int = 0


class MaintenanceCoordinator:
    """一个宿主共享的空闲计时、Memory 任务和生命周期资格 owner。"""

    def __init__(self, *, idle_seconds: float = 300) -> None:
        """创建协调器；实际任务和监听在 runner 准备时启动。"""
        if not 0 <= idle_seconds < math.inf:
            raise IrisConfigError("maintenance.idle_seconds 必须为有限非负数")
        self.idle_seconds = idle_seconds
        self._resources: dict[tuple[str, str], _MemoryResource] = {}
        self._readers: dict[str, LifecycleStore] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._task: asyncio.Task[None] | None = None
        self._active_resource: _MemoryResource | None = None
        self._worker = GenerationWorker(on_idle=self._schedule)
        self._foreground = 0
        self._quiet_until = 0.0
        self._next_resource = 0
        self._closed = False
        self._cancelling = False

    def _attach(
        self, binding: MemoryMaintenanceBinding | None, reader: LifecycleStore
    ) -> _MemoryResource | None:
        """注册实际资源和 reader；runner detach 不撤销宿主持有的资格读取能力。"""
        if self._closed:
            raise IrisRunStateError("维护协调器已关闭")
        if binding is None:
            self._readers[reader.source_id] = reader
            for resource in self._resources.values():
                self._wake(resource)
            return None
        path = Path(os.path.normcase(str(binding.database_path.resolve())))
        key = (str(path), binding.namespace)
        resource = self._resources.get(key)
        if resource is not None:
            if resource.binding.service is not binding.service:
                raise IrisConfigError("同一 Memory 资源必须绑定同一宿主持有的 service")
        else:
            namespace = binding.namespace.encode("utf-8").hex()
            resource = _MemoryResource(
                binding=binding,
                lock_path=path.with_name(f".{path.name}.iris-memory-{namespace}.lock"),
            )
            self._resources[key] = resource
            if self._loop is not None:
                self._subscribe(resource)
        self._readers[reader.source_id] = reader
        resource.attachments += 1
        self._wake(resource)
        return resource

    async def unbind_memory(self, binding: MemoryMaintenanceBinding) -> None:
        """关闭借用该资源的 runner 后撤销绑定，排空该资源作业但不关闭 service。"""
        key = (os.path.normcase(str(binding.database_path.resolve())), binding.namespace)
        resource = self._resources[key]
        if resource.attachments:
            raise IrisRunStateError("Memory 资源仍有绑定 runner，请先关闭这些 runner")
        del self._resources[key]
        if resource.listener is not None:
            resource.binding.service.remove_change_listener(resource.listener)
        if self._active_resource is resource:
            self._cancel_task()
            await asyncio.gather(self._task, return_exceptions=True)

    async def prepare(self) -> None:
        """首次使用时订阅所有绑定；资源按统一 idle 进入有界维护。"""
        if self._closed:
            raise IrisRunStateError("维护协调器已关闭")
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
            self._quiet_until = self._loop.time() + self.idle_seconds
            for resource in self._resources.values():
                self._subscribe(resource)
        self._schedule()

    def _subscribe(self, resource: _MemoryResource) -> None:
        """服务通知可以来自 IO 线程，统一送回宿主事件循环。"""

        def changed(namespace: str) -> None:
            if (
                namespace == resource.binding.namespace
                and not self._closed
                and generation_worker.get() is not self._worker
            ):
                self._loop.call_soon_threadsafe(self._wake, resource)

        resource.listener = changed
        resource.binding.service.add_change_listener(changed)

    def _wake(self, resource: _MemoryResource) -> None:
        """合并原文、服务变化和新 reader 到一个待检查位置。"""
        resource.dirty = True
        resource.revision += 1
        self._schedule()

    def _foreground_enter(self) -> None:
        """立即停止维护 admission 并撤销未提交生成，不等待模型或锁。"""
        self._foreground += 1
        for resource in self._resources.values():
            resource.dirty = True
            resource.revision += 1
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._cancel_task()

    def _foreground_exit(self) -> None:
        """完整前台退出后重新计算宿主的安静时间。"""
        self._foreground -= 1
        if self._loop is not None:
            self._quiet_until = self._loop.time() + self.idle_seconds
        self._schedule()

    def _cancel_task(self) -> None:
        """只取消当前 coroutine 一次，后续前台不能再次打断真实 IO 收尾。"""
        self._worker.cancel()
        if self._task is not None and not self._cancelling:
            self._cancelling = True
            self._task.cancel()

    def _schedule(self) -> None:
        """所有资源共享一个 timer；锁忙的资源延迟后轮流再试。"""
        if (
            self._closed
            or self._loop is None
            or self._foreground
            or self._task is not None
            or self._timer is not None
        ):
            return
        ready = [resource.ready_at for resource in self._resources.values() if resource.dirty]
        if ready:
            self._timer = self._loop.call_at(max(self._quiet_until, min(ready)), self._start)

    def _start(self) -> None:
        """轮转选择一个到期资源，保持同宿主最多一项 Memory 作业。"""
        self._timer = None
        resources = tuple(self._resources.values())
        for offset in range(len(resources)):
            index = (self._next_resource + offset) % len(resources)
            resource = resources[index]
            if resource.dirty and resource.ready_at <= self._loop.time():
                self._next_resource = (index + 1) % len(resources)
                self._active_resource = resource
                resource.dirty = False
                self._cancelling = False
                self._task = asyncio.create_task(self._run_cycle(resource))
                self._task.add_done_callback(self._finished)
                return
        self._schedule()

    async def _eligible(self, sources: tuple[MemorySource, ...]) -> bool:
        """每次发布前重读权威 Run/lane；WAITING 只排除该来源会话。"""
        if self._closed or self._foreground:
            return False
        eligible = all(self._source_eligible(source) for source in sources)
        if not eligible:
            # 外部进程改变资格不会发本地通知；重选一次以便其他会话继续。
            self._active_resource.revision += 1
        return eligible

    def _source_eligible(self, source: MemorySource) -> bool:
        reader = self._readers.get(source.lifecycle_source_id)
        if reader is None:
            return False
        run = reader.load_run(source.run_id)
        if run is None or run.phase is not RunPhase.TERMINAL:
            return False
        lane = reader.load_session_lane(source.session_id)
        current = reader.load_run(lane) if lane is not None else None
        return current is None or current.phase is not RunPhase.WAITING

    async def _run_cycle(self, resource: _MemoryResource) -> None:
        """有界周期持有 OS 锁，取消后直到真实 worker 排空才释放。"""
        lock = FileLock(resource.lock_path, timeout=0)
        try:
            lock.acquire()
        except Timeout:
            resource.dirty = True
            resource.ready_at = self._loop.time() + max(self.idle_seconds, 1)
            return
        revision = resource.revision
        try:
            with self._worker.bind():
                service = resource.binding.service
                sources = await service.alist_pending_sources(resource.binding.namespace)
                scope = MemoryMaintenanceScope(
                    allowed_sources=frozenset(
                        (source.lifecycle_source_id, source.run_id)
                        for source in sources
                        if self._source_eligible(source)
                    ),
                    check=self._eligible,
                )
                more = await service.maintain_cycle(resource.binding.namespace, scope=scope)
                resource.dirty = more or resource.revision != revision
                resource.ready_at = self._loop.time() + self.idle_seconds
        except asyncio.CancelledError:
            resource.dirty = resource.revision != revision
            raise
        except Exception:
            resource.dirty = resource.revision != revision
            raise
        finally:
            await self._worker.wait_idle()
            lock.release()

    def _finished(self, task: asyncio.Task[None]) -> None:
        """真实作业收尾完成后才释放本地任务位置。"""
        self._task = None
        self._active_resource = None
        if not task.cancelled() and task.exception() is not None:
            logger.error("共享 Memory 维护失败", exc_info=task.exception())
        self._schedule()

    async def aclose(self) -> None:
        """宿主停止派发、排空实际 IO；服务和 lifecycle reader 仍由宿主关闭。"""
        if self._closed:
            return
        self._closed = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        for resource in self._resources.values():
            if resource.listener is not None:
                resource.binding.service.remove_change_listener(resource.listener)
        self._cancel_task()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
        await self._worker.aclose()


class MaintenanceAttachment:
    """runner 的借用关系与本地前台计数，不拥有资源或后台任务。"""

    def __init__(
        self, coordinator: MaintenanceCoordinator, resource: _MemoryResource | None
    ) -> None:
        """建立前台借用关系；Memory 资源可选。"""
        self.coordinator = coordinator
        self.resource = resource
        self._foreground = 0
        self._detached = False

    @property
    def foreground_active(self) -> bool:
        """只描述当前 runner，不使同宿主其他 runner 无法关闭。"""
        return self._foreground > 0

    def foreground_enter(self) -> None:
        """登记当前 runner 的前台 admission 或 handoff。"""
        self._foreground += 1
        self.coordinator._foreground_enter()

    def foreground_exit(self) -> None:
        """释放当前 runner 的前台计数。"""
        self._foreground -= 1
        self.coordinator._foreground_exit()

    def detach(self) -> None:
        """runner 关闭解除借用；宿主的 reader 与资源注册保持有效。"""
        if not self._detached and self.resource is not None:
            self.resource.attachments -= 1
            self._detached = True
