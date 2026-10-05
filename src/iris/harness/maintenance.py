"""宿主级共享维护调度；Memory 负责领域执行，OS 锁保护跨进程完整周期。"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import cast

from filelock import FileLock, Timeout

from ..evolution.models import (
    EvolutionMaintenanceScope,
    EvolutionResult,
    EvolutionSession,
    EvolutionSource,
    RevisionRequest,
)
from ..evolution.service import EvolutionService
from ..exceptions import IrisConfigError, IrisRunStateError
from ..lifecycle import LifecycleStore, RunPhase
from ..memory import MemoryService
from ..memory.generation_models import MemoryMaintenanceScope, MemorySource
from ..utils.generation_worker import GenerationWorker, generation_worker

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryMaintenanceBinding:
    """宿主选定的实际 Memory 资源和完整领域服务，不探测 store 私有属性。"""

    service: MemoryService
    database_path: Path
    namespace: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ProjectEvolutionBinding:
    """宿主选定的项目与经验服务，服务固定生成策略及唯一发布目标。"""

    workspace_root: Path
    service: EvolutionService


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


@dataclass(frozen=True, slots=True)
class _RevisionWaiter:
    """等待具体持久请求的宿主调用，不存储第二份运行资格。"""

    future: asyncio.Future[EvolutionResult]
    session: EvolutionSession | None


@dataclass(slots=True)
class _EvolutionResource:
    """一个项目的经验维护状态，与 Memory 独立占用任务和锁。"""

    binding: ProjectEvolutionBinding
    lock_path: Path
    dirty: bool = True
    revision: int = 0
    ready_at: float = 0
    attachments: int = 0
    request: asyncio.Future[EvolutionResult] | None = None
    revision_requests: dict[str, _RevisionWaiter] = field(default_factory=dict)


class MaintenanceCoordinator:
    """一个宿主共享空闲计时与资格，两类维护分别持有任务、worker 和锁。"""

    def __init__(self, *, idle_seconds: float = 300) -> None:
        """创建协调器；实际任务和监听在 runner 准备时启动。"""
        if not 0 <= idle_seconds < math.inf:
            raise IrisConfigError("maintenance.idle_seconds 必须为有限非负数")
        self.idle_seconds = idle_seconds
        self._resources: dict[tuple[str, str], _MemoryResource] = {}
        self._projects: dict[str, _EvolutionResource] = {}
        self._readers: dict[str, LifecycleStore] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._task: asyncio.Task[None] | None = None
        self._active_resource: _MemoryResource | None = None
        self._worker = GenerationWorker(on_idle=self._schedule)
        self._evolution_task: asyncio.Task[EvolutionResult | None] | None = None
        self._active_project: _EvolutionResource | None = None
        self._evolution_worker = GenerationWorker(on_idle=self._schedule)
        self._next_project = 0
        self._foreground = 0
        self._quiet_until = 0.0
        self._next_resource = 0
        self._closed = False

    def _attach(
        self,
        binding: MemoryMaintenanceBinding | None,
        reader: LifecycleStore,
        *,
        evolution: ProjectEvolutionBinding | None = None,
    ) -> tuple[_MemoryResource | None, _EvolutionResource | None]:
        """注册实际资源和 reader；runner detach 不撤销宿主持有的资格读取能力。"""
        if self._closed:
            raise IrisRunStateError("维护协调器已关闭")
        resource = None
        project = None
        if binding is not None:
            path = Path(os.path.normcase(str(binding.database_path.resolve())))
            key = (str(path), binding.namespace)
            resource = self._resources.get(key)
            if resource is not None and resource.binding.service is not binding.service:
                raise IrisConfigError("同一 Memory 资源必须绑定同一宿主持有的 service")
        if evolution is not None:
            project_key = os.path.normcase(str(evolution.workspace_root.resolve()))
            project = self._projects.get(project_key)
            if project is not None and project.binding.service is not evolution.service:
                raise IrisConfigError("同一项目经验资源必须绑定同一宿主持有的 service")
        if binding is not None:
            if resource is None:
                namespace = binding.namespace.encode("utf-8").hex()
                resource = _MemoryResource(
                    binding=binding,
                    lock_path=path.with_name(f".{path.name}.iris-memory-{namespace}.lock"),
                )
                self._resources[key] = resource
                if self._loop is not None:
                    self._subscribe(resource)
            resource.attachments += 1
        if evolution is not None:
            if project is None:
                project = _EvolutionResource(
                    binding=evolution,
                    lock_path=Path(project_key) / ".iris" / "evolution.lock",
                )
                self._projects[project_key] = project
            project.attachments += 1
        self._readers[reader.source_id] = reader
        for existing in self._resources.values():
            self._wake(existing)
        for existing_project in self._projects.values():
            self._wake(existing_project)
        return resource, project

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

    async def unbind_evolution(self, binding: ProjectEvolutionBinding) -> None:
        """撤销已无 runner 借用的项目，排空该类作业但不关闭宿主服务。"""
        key = os.path.normcase(str(binding.workspace_root.resolve()))
        project = self._projects[key]
        if project.attachments:
            raise IrisRunStateError("项目经验资源仍有绑定 runner，请先关闭这些 runner")
        del self._projects[key]
        if self._active_project is project:
            self._cancel_evolution_task()
            await asyncio.gather(self._evolution_task, return_exceptions=True)
        await project.binding.service.wait_pending_io()
        self._fail_request(project, IrisRunStateError("项目经验维护绑定已撤销"))
        self._fail_revisions(project, IrisRunStateError("项目经验维护绑定已撤销"))

    async def request_project_experience(self, binding: ProjectEvolutionBinding) -> EvolutionResult:
        """合并同项目主动请求，跳过普通 idle；前台、资格和项目锁仍然有效。"""
        await self.prepare()
        project = self._projects[os.path.normcase(str(binding.workspace_root.resolve()))]
        if project.binding.service is not binding.service:
            raise IrisConfigError("主动整理必须使用已绑定的项目经验服务")
        if project.request is None:
            project.request = asyncio.get_running_loop().create_future()
            project.request.add_done_callback(_consume_request_error)
            project.ready_at = 0
        request = project.request
        if self._active_project is not project:
            self._wake(project)
        return await asyncio.shield(request)

    async def request_revision(
        self,
        binding: ProjectEvolutionBinding,
        request: RevisionRequest,
    ) -> EvolutionResult:
        """保存显式 B 请求并等待其自身结算；不把其他 A/B 结果当成本请求完成。"""
        await self.prepare()
        key = os.path.normcase(str(binding.workspace_root.resolve()))
        project = self._projects[key]
        if project.binding.service is not binding.service:
            raise IrisConfigError("主动修订必须使用已绑定的项目经验服务")
        item = await binding.service.enqueue_revision(request)
        if self._closed or self._projects.get(key) is not project:
            raise IrisRunStateError("项目维护已关闭或绑定已撤销，请求已保存")
        future: asyncio.Future[EvolutionResult] = asyncio.get_running_loop().create_future()
        future.add_done_callback(_consume_request_error)
        project.revision_requests[item.id] = _RevisionWaiter(future, request.session)
        project.ready_at = 0
        self._wake(project)
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise asyncio.CancelledError
        return await asyncio.shield(future)

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

    def _wake(self, resource: _MemoryResource | _EvolutionResource) -> None:
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
        for project in self._projects.values():
            project.dirty = True
            project.revision += 1
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._cancel_task()
        self._cancel_evolution_task()

    def _foreground_exit(self) -> None:
        """完整前台退出后重新计算宿主的安静时间。"""
        self._foreground -= 1
        if self._loop is not None:
            self._quiet_until = self._loop.time() + self.idle_seconds
        self._schedule()

    def _cancel_task(self) -> None:
        """只取消当前 coroutine 一次，后续前台不能再次打断真实 IO 收尾。"""
        self._worker.cancel()
        if self._task is not None and not self._task.cancelling():
            self._task.cancel()

    def _cancel_evolution_task(self) -> None:
        """仅撤销项目经验 slot，不打断 Memory 或已经开始的本类 IO 收尾。"""
        self._evolution_worker.cancel()
        if self._evolution_task is not None and not self._evolution_task.cancelling():
            self._evolution_task.cancel()

    def _schedule(self) -> None:
        """所有资源共享一个 timer；锁忙的资源延迟后轮流再试。"""
        if self._closed or self._loop is None or self._foreground:
            return
        ready = (
            [
                max(self._quiet_until, resource.ready_at)
                for resource in self._resources.values()
                if resource.dirty
            ]
            if self._task is None
            else []
        )
        if self._evolution_task is None:
            ready.extend(
                max(
                    project.ready_at,
                    0
                    if project.request is not None or project.revision_requests
                    else self._quiet_until,
                )
                for project in self._projects.values()
                if project.dirty
            )
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if ready:
            self._timer = self._loop.call_at(min(ready), self._start)

    def _start(self) -> None:
        """两类分别轮转一个到期资源，同宿主最多各一项作业。"""
        self._timer = None
        resources = tuple(self._resources.values()) if self._task is None else ()
        for offset in range(len(resources)):
            index = (self._next_resource + offset) % len(resources)
            resource = resources[index]
            if resource.dirty and max(resource.ready_at, self._quiet_until) <= self._loop.time():
                self._next_resource = (index + 1) % len(resources)
                self._active_resource = resource
                resource.dirty = False
                self._task = asyncio.create_task(self._run_cycle(resource))
                self._task.add_done_callback(self._finished)
                break
        projects = tuple(self._projects.values()) if self._evolution_task is None else ()
        for offset in range(len(projects)):
            index = (self._next_project + offset) % len(projects)
            project = projects[index]
            ready_at = max(
                project.ready_at,
                0
                if project.request is not None or project.revision_requests
                else self._quiet_until,
            )
            if project.dirty and ready_at <= self._loop.time():
                self._next_project = (index + 1) % len(projects)
                self._active_project = project
                project.dirty = False
                self._evolution_task = asyncio.create_task(self._run_project_cycle(project))
                self._evolution_task.add_done_callback(self._evolution_finished)
                break
        self._schedule()

    async def _eligible(
        self,
        resource: _MemoryResource | _EvolutionResource,
        sources: tuple[MemorySource | EvolutionSource, ...],
    ) -> bool:
        """每次发布前重读权威 Run/lane；WAITING 只排除该来源会话。"""
        if self._closed or self._foreground:
            return False
        eligible = all(
            self._source_eligible(source.lifecycle_source_id, source.run_id, source.session_id)
            for source in sources
        )
        if not eligible:
            # 外部进程改变资格不会发本地通知；重选一次以便其他会话继续。
            resource.revision += 1
        return eligible

    def _source_eligible(self, lifecycle_source_id: str, run_id: str, session_id: str) -> bool:
        reader = self._readers.get(lifecycle_source_id)
        if reader is None:
            return False
        run = reader.load_run(run_id)
        if run is None or run.phase is not RunPhase.TERMINAL:
            return False
        lane = reader.load_session_lane(session_id)
        current = reader.load_run(lane) if lane is not None else None
        return current is None or current.phase is not RunPhase.WAITING

    def _session_eligible(self, session: EvolutionSession | None) -> bool:
        """宿主明确请求没有虚构 Run；携带会话时必须能读取其真实 lane。"""
        if session is None:
            return True
        reader = self._readers.get(session.lifecycle_source_id)
        if reader is None or reader.load_session(session.session_id) is None:
            return False
        lane = reader.load_session_lane(session.session_id)
        current = reader.load_run(lane) if lane is not None else None
        return current is None or current.phase is not RunPhase.WAITING

    async def _eligible_session(
        self, project: _EvolutionResource, session: EvolutionSession | None
    ) -> bool:
        """B 发布前重查宿主请求会话与前台状态。"""
        if self._closed or self._foreground:
            return False
        eligible = self._session_eligible(session)
        if not eligible:
            project.revision += 1
        return eligible

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
                        if self._source_eligible(
                            source.lifecycle_source_id, source.run_id, source.session_id
                        )
                    ),
                    check=partial(self._eligible, resource),
                )
                more = await service.maintain_cycle(resource.binding.namespace, scope=scope)
                resource.dirty = more or resource.revision != revision
                resource.ready_at = self._loop.time() + self.idle_seconds
        except (Exception, asyncio.CancelledError):
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

    async def _run_project_cycle(self, project: _EvolutionResource) -> EvolutionResult | None:
        """一个有界 A 或 B 持项目锁；真实短 IO 排空前不释放该类位置。"""
        lock = FileLock(project.lock_path, timeout=0)
        try:
            lock.acquire()
        except Timeout:
            project.dirty = True
            project.ready_at = self._loop.time() + max(self.idle_seconds, 1)
            return None
        revision = project.revision
        try:
            with self._evolution_worker.bind():
                service = project.binding.service
                for item_id in tuple(project.revision_requests):
                    settled = await service.run_async_io(
                        partial(service.store.revision_result, item_id)
                    )
                    if settled is not None:
                        project.revision_requests.pop(item_id).future.set_result(settled)
                sources = await service.alist_pending_sources()
                sessions = await service.alist_pending_sessions()
                scope = EvolutionMaintenanceScope(
                    allowed_sources=frozenset(
                        (source.lifecycle_source_id, source.run_id)
                        for source in sources
                        if self._source_eligible(
                            source.lifecycle_source_id, source.run_id, source.session_id
                        )
                    ),
                    check=partial(self._eligible, project),
                    allowed_sessions=frozenset(
                        (session.lifecycle_source_id, session.session_id)
                        for session in sessions
                        if self._session_eligible(session)
                    ),
                    check_session=partial(self._eligible_session, project),
                    experience_only=project.request is not None,
                    requested_revision_id=next(
                        (
                            item_id
                            for item_id, waiter in project.revision_requests.items()
                            if self._session_eligible(waiter.session)
                        ),
                        None,
                    ),
                )
                result = await service.maintain_cycle(scope=scope)
                project.dirty = result.has_more or project.revision != revision
                project.ready_at = (
                    0
                    if project.request is not None or project.revision_requests
                    else self._loop.time() + self.idle_seconds
                )
                return result
        except BaseException:
            project.dirty = project.revision != revision
            raise
        finally:
            await self._evolution_worker.wait_idle()
            lock.release()

    def _evolution_finished(self, task: asyncio.Task[EvolutionResult | None]) -> None:
        """项目真实收尾完成后才释放该类任务位置，Memory 独立继续。"""
        project = cast(_EvolutionResource, self._active_project)
        self._evolution_task = None
        self._active_project = None
        if task.cancelled():
            self._fail_request(project, asyncio.CancelledError())
        elif (error := task.exception()) is not None:
            self._fail_request(project, error)
            self._fail_revisions(project, error)
            logger.error("项目经验维护失败", exc_info=error)
        elif (result := task.result()) is not None:
            completed_request = False
            if result.stage == "experience" and project.request is not None:
                project.request.set_result(result)
                project.request = None
                completed_request = True
            if result.revision_id in project.revision_requests:
                project.revision_requests.pop(result.revision_id).future.set_result(result)
                completed_request = True
            if project.request is not None or (
                completed_request
                and any(
                    self._session_eligible(waiter.session)
                    for waiter in project.revision_requests.values()
                )
            ):
                project.dirty = True
                project.ready_at = 0
        self._schedule()

    @staticmethod
    def _fail_request(project: _EvolutionResource, error: BaseException) -> None:
        """结束主动等待者；调用方取消等待不会取消这份共享 future。"""
        if project.request is not None:
            project.request.set_exception(error)
            project.request = None

    @staticmethod
    def _fail_revisions(project: _EvolutionResource, error: BaseException) -> None:
        """宿主关闭或撤销绑定时结束全部显式等待，持久 pending 仍保留。"""
        for waiter in project.revision_requests.values():
            waiter.future.set_exception(error)
        project.revision_requests.clear()

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
        self._cancel_evolution_task()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
        if self._evolution_task is not None:
            await asyncio.gather(self._evolution_task, return_exceptions=True)
        for project in self._projects.values():
            await project.binding.service.wait_pending_io()
            self._fail_request(project, IrisRunStateError("维护协调器已关闭"))
            self._fail_revisions(project, IrisRunStateError("维护协调器已关闭"))
        await self._worker.aclose()
        await self._evolution_worker.aclose()


class MaintenanceAttachment:
    """runner 的借用关系与本地前台计数，不拥有资源或后台任务。"""

    def __init__(
        self,
        coordinator: MaintenanceCoordinator,
        resource: _MemoryResource | None,
        project: _EvolutionResource | None,
    ) -> None:
        """建立前台借用关系；Memory 资源可选。"""
        self.coordinator = coordinator
        self.resource = resource
        self.project = project
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
        if not self._detached:
            if self.resource is not None:
                self.resource.attachments -= 1
            if self.project is not None:
                self.project.attachments -= 1
            self._detached = True


def _consume_request_error(request: asyncio.Future[EvolutionResult]) -> None:
    """外部等待者全部取消时，也回收最终共享请求的异常。"""
    if not request.cancelled():
        request.exception()
