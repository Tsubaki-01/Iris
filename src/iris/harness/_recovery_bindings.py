"""恢复点的实际持久资源描述与备份分派。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from ..evolution.materials import EvolutionMaterialStore
from ..evolution.service import EvolutionService
from ..exceptions import IrisConfigError, IrisError, IrisRunPersistenceError
from ..memory import MemoryService
from ..memory.mirror import FileMemoryMirror
from ..memory.sqlite import SQLiteMemoryStore
from .recovery_models import (
    EvolutionStateSnapshot,
    MemoryStateSnapshot,
    SnapshotFiles,
    StateSnapshot,
)

if TYPE_CHECKING:
    from .runner import AgentRunner


def _relative(path: Path, workspace: Path) -> str:
    """在环境 API 边界确认实际持久路径属于本 workspace。"""
    try:
        return path.resolve().relative_to(workspace).as_posix()
    except ValueError as exc:
        raise IrisConfigError("环境恢复的状态路径必须位于 workspace", path=str(path)) from exc


def describe_memory_state(
    service: MemoryService, *, workspace: Path, namespace: str
) -> MemoryStateSnapshot:
    """投影本次输入的实际内置 Memory 资源。"""
    if not isinstance(service.store, SQLiteMemoryStore):
        raise IrisConfigError("环境恢复要求 SQLiteMemoryStore")
    if service.mirror is not None and not isinstance(service.mirror, FileMemoryMirror):
        raise IrisConfigError("环境恢复要求 FileMemoryMirror")
    return MemoryStateSnapshot.model_construct(
        role="memory",
        database_path=_relative(service.store.path, workspace),
        projection_root=None
        if service.mirror is None
        else _relative(service.mirror.root, workspace),
        write_namespace=namespace,
    )


def describe_evolution_state(
    service: EvolutionService, *, workspace: Path
) -> EvolutionStateSnapshot:
    """投影实际项目材料、Skill 及当前开放的发布目标。"""
    if not isinstance(service.store, EvolutionMaterialStore):
        raise IrisConfigError("环境恢复要求 EvolutionMaterialStore")
    return EvolutionStateSnapshot.model_construct(
        role="evolution",
        database_path=_relative(service.store.path, workspace),
        skill_path=_relative(service.skill_path, workspace),
        prompt_paths={
            target.name: _relative(service.prompt_source.root / f"{target.name}.j2", workspace)
            for target in service.prompt_targets
        },
        config_path=None
        if service.config_target is None
        else _relative(service.config_target.path, workspace),
    )


def collect_state_snapshots(runner: AgentRunner) -> tuple[StateSnapshot, ...]:
    """保存入口收集当前 root 的确切绑定，不从配置猜测注入服务。"""
    environment = runner.runtime.environment
    states: list[StateSnapshot] = []
    attachment = runner._maintenance
    if environment.memory_service is not None:
        if (
            attachment is not None
            and attachment.resource is not None
            and attachment.resource.attachments > 1
        ):
            raise IrisConfigError("环境恢复要求本 root 独占 Memory 服务")
        states.append(
            describe_memory_state(
                environment.memory_service,
                workspace=environment.workspace_root,
                namespace=environment.agent_config.memory.write_namespace,
            )
        )
    if attachment is not None and attachment.project is not None:
        if attachment.project.attachments > 1:
            raise IrisConfigError("环境恢复要求本 root 独占 Evolution 服务")
        states.append(
            describe_evolution_state(
                attachment.project.binding.service, workspace=environment.workspace_root
            )
        )
    return tuple(states)


def backup_state_snapshots(
    runner: AgentRunner, states: tuple[StateSnapshot, ...], files: SnapshotFiles
) -> None:
    """消费入口已确认的内置资源，在停写区间导出各自数据库。"""
    try:
        for state in states:
            if state.role == "memory":
                store = cast(SQLiteMemoryStore, runner.runtime.environment.memory_service.store)
                store.backup(files.memory_backup)
            else:
                runner._maintenance.project.binding.service.store.backup(files.evolution_backup)
    except IrisError as exc:
        raise IrisRunPersistenceError("环境恢复点状态库备份失败", role=state.role) from exc
