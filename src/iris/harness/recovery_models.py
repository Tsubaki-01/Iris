"""环境恢复包的磁盘契约与内部路径投影。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from ..sandbox.snapshot import DockerSnapshot

if TYPE_CHECKING:
    from ..command.service import CommandBinding
    from ..store import SQLiteStore


class MemoryStateSnapshot(BaseModel):
    """实际 Memory 数据库、投影及写入 namespace。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["memory"]
    database_path: str
    projection_root: str | None
    write_namespace: str


class EvolutionStateSnapshot(BaseModel):
    """实际项目经验数据库与已开放发布目标。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["evolution"]
    database_path: str
    skill_path: str
    prompt_paths: dict[str, str]
    config_path: str | None


StateSnapshot = Annotated[MemoryStateSnapshot | EvolutionStateSnapshot, Field(discriminator="role")]


class EnvironmentSnapshotManifest(BaseModel):
    """同一停写区间发布的整套 root 环境。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: Literal[1] = 1
    snapshot_id: UUID
    created_at: AwareDatetime
    source_id: str
    workspace_root: Path
    lifecycle_path: Path
    docker: DockerSnapshot
    states: tuple[StateSnapshot, ...] = ()

    @field_validator("created_at")
    @classmethod
    def _utc_time(cls, value: datetime) -> datetime:
        """将边界时间统一为 UTC。"""
        return value.astimezone(UTC)

    @field_validator("workspace_root", "lifecycle_path")
    @classmethod
    def _absolute_path(cls, value: Path) -> Path:
        """恢复描述保留原绝对路径。"""
        if not value.is_absolute():
            raise ValueError("恢复目标必须是绝对路径")
        return value

    @field_validator("states")
    @classmethod
    def _unique_roles(cls, value: tuple[StateSnapshot, ...]) -> tuple[StateSnapshot, ...]:
        """每个角色在恢复点中只有一个 owner。"""
        if len({state.role for state in value}) != len(value):
            raise ValueError("恢复状态角色不能重复")
        return value


class LatestSnapshot(BaseModel):
    """唯一已发布恢复点的指针。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_id: UUID


@dataclass(frozen=True, slots=True)
class SnapshotFiles:
    """专用目录中固定产物的纯路径投影。"""

    directory: Path
    snapshot_id: UUID

    @property
    def candidate_directory(self) -> Path:
        """返回当前候选目录。"""
        return self.directory / str(self.snapshot_id)

    @property
    def manifest_path(self) -> Path:
        """返回 manifest 路径。"""
        return self.candidate_directory / "manifest.json"

    @property
    def lifecycle_backup(self) -> Path:
        """返回 lifecycle 备份路径。"""
        return self.candidate_directory / "lifecycle.sqlite"

    @property
    def workspace_archive(self) -> Path:
        """返回 workspace 归档路径。"""
        return self.candidate_directory / "workspace.tar"

    @property
    def memory_backup(self) -> Path:
        """返回 Memory 备份路径。"""
        return self.candidate_directory / "state" / "memory.sqlite"

    @property
    def evolution_backup(self) -> Path:
        """返回 Evolution 备份路径。"""
        return self.candidate_directory / "state" / "evolution.sqlite"


@dataclass(frozen=True, slots=True)
class RestoredEnvironment:
    """已经恢复的资源；仅在 harness 内部交接。"""

    manifest: EnvironmentSnapshotManifest
    store: SQLiteStore
    command_binding: CommandBinding


__all__ = [
    "DockerSnapshot",
    "EnvironmentSnapshotManifest",
    "EvolutionStateSnapshot",
    "LatestSnapshot",
    "MemoryStateSnapshot",
    "RestoredEnvironment",
    "SnapshotFiles",
    "StateSnapshot",
]
