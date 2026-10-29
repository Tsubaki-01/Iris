"""Root 环境恢复点的停写协调与成套发布。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager, nullcontext
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from ..command.docker import DockerCommandService
from ..exceptions import IrisCommandCleanupError, IrisConfigError, IrisRunStateError
from ..lifecycle import RunPhase, RunSnapshot, ToolCallPhase
from ..lifecycle.history import RunCursor, SessionCursor
from ..runtime.environment import RuntimeExecutionScope
from ..store import SQLiteStore
from ..utils.background_io import BackgroundIO
from ._recovery_bindings import backup_state_snapshots, collect_state_snapshots
from ._recovery_files import SnapshotStorage, archive_workspace
from .recovery_models import EnvironmentSnapshotManifest, SnapshotFiles, StateSnapshot

if TYPE_CHECKING:
    from .runner import AgentRunner


class RootRecoveryCoordinator:
    """仅启用时创建的 root 独占区间；普通执行之间保持并发。"""

    def __init__(self, root: AgentRunner) -> None:
        self.root = root
        self.io = BackgroundIO()
        self._snapshot_lock = asyncio.Lock()
        self._execution_ready = asyncio.Event()
        self._execution_ready.set()
        self._operations = 0
        self._blocked: IrisCommandCleanupError | None = None
        self._saves: set[asyncio.Task[EnvironmentSnapshotManifest]] = set()
        self._storage: SnapshotStorage | None = None

    @contextmanager
    def operation(self) -> Iterator[None]:
        """同步准入普通写入，独占时在任何副作用之前报忙碌。"""
        if self._blocked is not None:
            raise self._blocked
        if not self._execution_ready.is_set():
            raise IrisRunStateError("环境快照正在保存")
        self._operations += 1
        try:
            yield
        finally:
            self._operations -= 1

    @asynccontextmanager
    async def internal_operation(self) -> AsyncIterator[None]:
        """内部调度等独占区间结束，随后原子登记。"""
        while not self._execution_ready.is_set():
            await self._execution_ready.wait()
        with self.operation():
            yield

    def track_task[Result](self, task: asyncio.Task[Result]) -> None:
        """在派发时占用计数，真实作业结束后释放。"""
        self._operations += 1

        def finished(_: asyncio.Task[Result]) -> None:
            self._operations -= 1

        task.add_done_callback(finished)

    def begin_snapshot(self) -> None:
        """只在所有已准入执行与资源收尾结束后进入独占。"""
        if self.root._closed:
            raise IrisRunStateError("runner 已关闭")
        if self._blocked is not None:
            raise self._blocked
        hooks = self.root._hook_lifecycle
        if hooks.admission_error is not None:
            raise hooks.admission_error
        if (
            self._operations
            or self.root._command_lifecycle.pending
            or any(not entry.completion.is_set() for entry in hooks._finished.values())
        ):
            raise IrisRunStateError("root 仍有执行或资源收尾，不能保存环境")
        self._execution_ready.clear()

    def release_execution(self) -> None:
        """资源状态已恢复，普通执行可以继续。"""
        self._execution_ready.set()

    def block_execution(self, error: IrisCommandCleanupError) -> None:
        """解除暂停失败后保留资源错误，并唤醒内部等待者。"""
        self._blocked = error
        self._execution_ready.set()

    async def wait_for_saves(self) -> None:
        """关闭前收取本 root 所有保存的最终收尾。"""
        while self._saves:
            await asyncio.shield(asyncio.gather(*tuple(self._saves), return_exceptions=True))

    async def save(self, directory: Path) -> EnvironmentSnapshotManifest:
        """确认本次支持范围后，派发拥有完整收尾的保存作业。"""
        root = self.root
        environment = root.runtime.environment
        if environment.execution_scope is not RuntimeExecutionScope.ROOT:
            raise IrisConfigError("环境快照只能由 root 保存")
        if not isinstance(root.store, SQLiteStore):
            raise IrisConfigError("环境快照要求专用 SQLiteStore")
        binding = environment.command_binding
        if binding is None or not isinstance(binding.service, DockerCommandService):
            raise IrisConfigError("环境快照要求 DockerCommandService")
        directory = directory.resolve()
        if directory.is_relative_to(environment.workspace_root):
            raise IrisConfigError("快照目录必须位于 workspace 外")
        root._check_maintenance_binding()
        states = collect_state_snapshots(root)
        task = asyncio.create_task(
            self._save(SnapshotStorage(directory), root.store, binding.service, states)
        )
        self._saves.add(task)
        task.add_done_callback(self._saves.discard)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            if not task.cancelled():
                task.result()
            raise

    async def _save(
        self,
        storage: SnapshotStorage,
        store: SQLiteStore,
        service: DockerCommandService,
        states: tuple[StateSnapshot, ...],
    ) -> EnvironmentSnapshotManifest:
        async with self._snapshot_lock:
            self.begin_snapshot()
            published = False
            candidate: SnapshotFiles | None = None
            previous: EnvironmentSnapshotManifest | None = None
            preparing: asyncio.Task[None] | None = None
            pruning: asyncio.Task[tuple[str, ...]] | None = None
            try:
                previous = await self.io.run(storage.load_latest)
                if previous is not None and previous.source_id != store.source_id:
                    raise IrisConfigError("快照目录已属于另一个 lifecycle source")
                self._storage = storage
                keep = None if previous is None else previous.docker.image_ref
                preparing = asyncio.create_task(service.prepare())
                await asyncio.shield(preparing)
                pruning = asyncio.create_task(
                    service.prune_snapshot_images(source_id=store.source_id, keep_image_ref=keep)
                )
                await asyncio.shield(pruning)
                await self.io.run(partial(_prune_files, storage, previous))
                snapshot_id = uuid4()
                candidate = SnapshotFiles(storage.directory, snapshot_id)
                await self.io.run(partial(storage.prepare_candidate, snapshot_id))
                attachment = self.root._maintenance
                async with (
                    attachment.coordinator.suspend_for_snapshot()
                    if attachment is not None
                    else nullcontext()
                ):
                    try:
                        if self.root._capture is not None:
                            await self.root._capture.drain_for_snapshot()
                        memory = self.root.runtime.environment.memory_service
                        if memory is not None:
                            await memory.wait_pending_io()
                        if attachment is not None and attachment.project is not None:
                            await attachment.project.binding.service.wait_pending_io()
                        await self.io.run(partial(_require_stable_runs, store))
                        async with service.snapshot(
                            source_id=store.source_id, snapshot_id=snapshot_id
                        ) as docker:
                            try:
                                await self.io.run(partial(store.backup, candidate.lifecycle_backup))
                                await self.io.run(
                                    partial(backup_state_snapshots, self.root, states, candidate)
                                )
                                workspace = self.root.runtime.environment.workspace_root
                                excluded = frozenset(
                                    [
                                        store.path.resolve(),
                                        *(workspace / state.database_path for state in states),
                                    ]
                                )
                                await self.io.run(
                                    partial(
                                        archive_workspace,
                                        workspace,
                                        candidate.workspace_archive,
                                        excluded_files=excluded,
                                    )
                                )
                                manifest = EnvironmentSnapshotManifest.model_construct(
                                    snapshot_id=snapshot_id,
                                    created_at=self.root._now(),
                                    source_id=store.source_id,
                                    workspace_root=workspace,
                                    lifecycle_path=store.path.resolve(),
                                    docker=docker,
                                    states=states,
                                )
                                await self.io.run(
                                    partial(storage.publish, manifest), complete_on_cancel=True
                                )
                                published = True
                                service.adopt_snapshot_retention(
                                    source_id=store.source_id, keep_image_ref=docker.image_ref
                                )
                                if cast(asyncio.Task[object], asyncio.current_task()).cancelling():
                                    raise asyncio.CancelledError
                            finally:
                                await self.io.wait_pending()
                    finally:
                        await self.io.wait_pending()
                self.release_execution()
                pruning = asyncio.create_task(
                    service.prune_snapshot_images(
                        source_id=store.source_id, keep_image_ref=manifest.docker.image_ref
                    )
                )
                await asyncio.shield(pruning)
                await self.io.run(partial(_prune_files, storage, manifest))
                return manifest
            except IrisCommandCleanupError as error:
                self.block_execution(error)
                raise
            finally:
                if preparing is not None:
                    await asyncio.gather(preparing, return_exceptions=True)
                if pruning is not None:
                    await asyncio.gather(pruning, return_exceptions=True)
                await self.io.wait_pending()
                if self._blocked is None:
                    self.release_execution()
                if candidate is not None and not published:
                    await self.io.run(
                        partial(storage.remove_candidate, candidate.snapshot_id),
                        complete_on_cancel=True,
                    )
                    await service.prune_snapshot_images(
                        source_id=store.source_id,
                        keep_image_ref=None if previous is None else previous.docker.image_ref,
                    )

    async def cleanup_files(self) -> None:
        """正常关闭时收回专用目录中的非 latest 产物。"""
        if self._storage is not None:
            latest = await self.io.run(self._storage.load_latest)
            await self.io.run(partial(_prune_files, self._storage, latest))


def _prune_files(storage: SnapshotStorage, latest: EnvironmentSnapshotManifest | None) -> None:
    for candidate in storage.list_candidates():
        if latest is None or candidate != latest.snapshot_id:
            storage.remove_candidate(candidate)


def _require_stable_runs(store: SQLiteStore) -> None:
    """分页覆盖所有根会话和后代，拒绝遗留执行或未结算 claim。"""

    def visit(run: RunSnapshot) -> None:
        if run.phase not in (RunPhase.WAITING, RunPhase.TERMINAL) or any(
            tool.phase is ToolCallPhase.CLAIMED for tool in store.list_tool_calls(run.run_id)
        ):
            raise IrisRunStateError("保存前需要先恢复未结算 Run", run_id=run.run_id)
        child_cursor: RunCursor | None = None
        while True:
            children = store.list_child_runs(run.run_id, after=child_cursor)
            for child in children.items:
                visit(child.run)
            child_cursor = children.next_cursor
            if child_cursor is None:
                break

    session_cursor: SessionCursor | None = None
    while True:
        sessions = store.list_sessions(after=session_cursor)
        for session in sessions.items:
            run_cursor: RunCursor | None = None
            while True:
                runs = store.list_runs(session.session_id, after=run_cursor)
                for run in runs.items:
                    visit(run)
                run_cursor = runs.next_cursor
                if run_cursor is None:
                    break
        session_cursor = sessions.next_cursor
        if session_cursor is None:
            return
