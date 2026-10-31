"""Root 环境恢复点的停写协调与成套发布。"""

from __future__ import annotations

import asyncio
import platform
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager, nullcontext
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from ..command.config import CommandConfig
from ..command.docker import DockerCommandService
from ..command.models import CommandEnvironment, CommandMode
from ..command.service import CommandBinding
from ..exceptions import (
    IrisCommandCleanupError,
    IrisCommandError,
    IrisConfigError,
    IrisRunRecoveryError,
    IrisRunStateError,
)
from ..lifecycle import RunPhase, RunSnapshot, ToolCallPhase
from ..lifecycle.history import RunCursor, SessionCursor
from ..store import SQLiteStore
from ..utils.background_io import BackgroundIO
from ._recovery_bindings import (
    backup_state_snapshots,
    collect_state_snapshots,
    validate_memory_restore,
)
from ._recovery_files import SnapshotStorage, archive_workspace, restore_database, restore_workspace
from .recovery_models import (
    EnvironmentSnapshotManifest,
    MemoryStateSnapshot,
    RestoredEnvironment,
    SnapshotFiles,
    StateSnapshot,
)

if TYPE_CHECKING:
    from ..agents import AgentConfig
    from ..memory import MemoryService
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

        def finished(completed: asyncio.Task[Result]) -> None:
            self._operations -= 1
            if not completed.cancelled():
                completed.exception()

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
            if self._storage is not None and self._storage.directory != storage.directory:
                raise IrisConfigError("同一 root 必须使用同一个专用快照目录")
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


async def restore_environment(
    config: AgentConfig,
    directory: Path,
    *,
    memory_service: MemoryService | None,
) -> RestoredEnvironment:
    """固定一次 latest，先确认绑定和目标，再还原同套工作副本。"""
    if not config.environment_recovery.enabled:
        raise IrisConfigError("environment_recovery 未启用")
    io = BackgroundIO()
    service: DockerCommandService | None = None
    try:
        storage = SnapshotStorage(directory.resolve())
        manifest = await io.run(storage.load_latest)
        if manifest is None:
            raise IrisRunRecoveryError("没有已发布的环境恢复点", path=str(directory))
        expected = next(
            (state for state in manifest.states if isinstance(state, MemoryStateSnapshot)), None
        )
        validate_memory_restore(
            config.memory, memory_service, workspace=manifest.workspace_root, expected=expected
        )
        files = SnapshotFiles(storage.directory, manifest.snapshot_id)
        await io.run(partial(_require_restore_targets, manifest, files))
        service = DockerCommandService.from_snapshot(manifest.workspace_root, manifest.docker)
        try:
            await service.prepare()
        except IrisCommandError as error:
            raise IrisRunRecoveryError(
                "恢复镜像或本地 Docker 引擎不可用", error=str(error)
            ) from error
        store = await io.run(partial(_restore_files, manifest, files))
        command_config = CommandConfig.model_construct(
            mode=CommandMode.DOCKER,
            docker=manifest.docker.config,
            timeout_seconds=config.command.timeout_seconds,
        )
        return RestoredEnvironment(
            manifest,
            store,
            CommandBinding(
                config=command_config,
                service=service,
                environment=CommandEnvironment(
                    host_os=platform.system(),
                    mode=CommandMode.DOCKER,
                    command_os="Linux",
                    command_shell="/bin/sh",
                ),
            ),
        )
    except BaseException:

        async def cleanup() -> None:
            await io.wait_pending()
            if service is not None:
                await service.aclose()

        task = asyncio.create_task(cleanup())
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                if task.done():
                    task.result()
                    break
        raise


def _require_restore_targets(manifest: EnvironmentSnapshotManifest, files: SnapshotFiles) -> None:
    """离线恢复边界只检查本次使用的固定产物和原路径。"""
    workspace = manifest.workspace_root
    try:
        if files.directory.is_relative_to(workspace):
            raise IrisRunRecoveryError("快照目录必须位于 workspace 外")
        if workspace.exists() and (not workspace.is_dir() or any(workspace.iterdir())):
            raise IrisRunRecoveryError("恢复 workspace 必须为空或不存在", path=str(workspace))
        if manifest.lifecycle_path.exists():
            raise IrisRunRecoveryError(
                "恢复工作数据库路径必须已腾空", path=str(manifest.lifecycle_path)
            )
        required = [files.workspace_archive, files.lifecycle_backup]
        required.extend(
            files.memory_backup if state.role == "memory" else files.evolution_backup
            for state in manifest.states
        )
        for path in required:
            if not path.is_file():
                raise IrisRunRecoveryError("恢复点缺少必要产物", path=str(path))
    except OSError as error:
        raise IrisRunRecoveryError("恢复目标不可用", path=str(workspace)) from error


def _restore_files(manifest: EnvironmentSnapshotManifest, files: SnapshotFiles) -> SQLiteStore:
    """文件工作副本还原完成后才打开 lifecycle 的解析边界。"""
    restore_workspace(files.workspace_archive, manifest.workspace_root)
    restore_database(files.lifecycle_backup, manifest.lifecycle_path)
    for state in manifest.states:
        backup = files.memory_backup if state.role == "memory" else files.evolution_backup
        restore_database(backup, manifest.workspace_root / state.database_path)
    store = SQLiteStore(manifest.lifecycle_path)
    if store.source_id != manifest.source_id:
        raise IrisRunRecoveryError("恢复点数据库与 manifest 的 source_id 不一致")
    return store
