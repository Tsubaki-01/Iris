"""恢复包文件操作；停写和真实后台作业由 harness 编排。"""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path, PurePosixPath
from uuid import UUID

from ..exceptions import IrisRunPersistenceError, IrisRunRecoveryError
from ..utils.files import atomic_write_text
from .recovery_models import (
    EnvironmentSnapshotManifest,
    EvolutionStateSnapshot,
    LatestSnapshot,
    SnapshotFiles,
)


class SnapshotStorage:
    """专用快照目录的文件 owner，每次读取磁盘上的 latest。"""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def prepare_candidate(self, snapshot_id: UUID) -> SnapshotFiles:
        """创建一个全新的候选目录。"""
        files = SnapshotFiles(self.directory, snapshot_id)
        try:
            files.candidate_directory.mkdir(parents=True, exist_ok=False)
            (files.candidate_directory / "state").mkdir()
        except OSError as exc:
            raise IrisRunPersistenceError(
                "无法创建恢复点候选目录", path=str(files.candidate_directory)
            ) from exc
        return files

    def load_latest(self) -> EnvironmentSnapshotManifest | None:
        """解析唯一指针和对应 manifest，不寻找替代候选。"""
        latest_path = self.directory / "latest.json"
        try:
            try:
                content = latest_path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return None
            latest = LatestSnapshot.model_validate_json(content)
            files = SnapshotFiles(self.directory, latest.snapshot_id)
            manifest = EnvironmentSnapshotManifest.model_validate_json(
                files.manifest_path.read_text(encoding="utf-8")
            )
            if manifest.snapshot_id != latest.snapshot_id:
                raise ValueError("manifest 身份与所在目录不一致")
            for state in manifest.states:
                paths = [state.database_path]
                if isinstance(state, EvolutionStateSnapshot):
                    paths.extend((state.skill_path, *state.prompt_paths.values()))
                    if state.config_path is not None:
                        paths.append(state.config_path)
                elif state.projection_root is not None:
                    paths.append(state.projection_root)
                for path in paths:
                    relative = PurePosixPath(path)
                    if (
                        not path
                        or "\\" in path
                        or ":" in path
                        or relative.is_absolute()
                        or ".." in relative.parts
                        or relative.as_posix() != path
                    ):
                        raise ValueError(f"状态路径必须是 workspace 相对 POSIX 路径：{path}")
            return manifest
        except (OSError, UnicodeError, ValueError) as exc:
            raise IrisRunRecoveryError(
                "无法读取最新环境恢复点", path=str(latest_path), error=str(exc)
            ) from exc

    def publish(self, manifest: EnvironmentSnapshotManifest) -> None:
        """完整写入 manifest 后原子替换唯一发布指针。"""
        files = SnapshotFiles(self.directory, manifest.snapshot_id)
        try:
            atomic_write_text(files.manifest_path, manifest.model_dump_json(indent=2))
            atomic_write_text(
                self.directory / "latest.json",
                LatestSnapshot.model_construct(snapshot_id=manifest.snapshot_id).model_dump_json(),
            )
        except OSError as exc:
            raise IrisRunPersistenceError(
                "环境恢复点发布失败", path=str(files.manifest_path), operation="publish"
            ) from exc

    def remove_candidate(self, snapshot_id: UUID) -> None:
        """删除上层已确认不再保留的专用候选。"""
        path = self.directory / str(snapshot_id)
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise IrisRunPersistenceError("恢复点清理失败", path=str(path)) from exc

    def list_candidates(self) -> tuple[UUID, ...]:
        """枚举本目录直接包含的 UUID 候选，不扫描其他目录。"""
        try:
            if not self.directory.exists():
                return ()
            candidates = []
            for entry in self.directory.iterdir():
                if entry.is_dir():
                    try:
                        candidate = UUID(entry.name)
                    except ValueError:
                        continue
                    if entry.name == str(candidate):
                        candidates.append(candidate)
            return tuple(sorted(candidates))
        except OSError as exc:
            raise IrisRunPersistenceError("恢复点目录读取失败", path=str(self.directory)) from exc


def archive_workspace(
    workspace: Path, destination: Path, *, excluded_files: frozenset[Path]
) -> None:
    """保存全部 workspace 文件，精确排除单独备份的 SQLite 文件及辅助文件。"""
    excluded = {
        Path(f"{path}{suffix}").resolve()
        for path in excluded_files
        for suffix in ("", "-wal", "-shm", "-journal")
    }

    def include(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
        return None if (workspace / member.name).resolve() in excluded else member

    try:
        with tarfile.open(destination, "w") as archive:
            for entry in sorted(workspace.iterdir()):
                archive.add(entry, arcname=entry.name, filter=include)
    except (OSError, tarfile.TarError) as exc:
        raise IrisRunPersistenceError(
            "workspace 归档失败", path=str(destination), operation="archive"
        ) from exc


def restore_workspace(archive: Path, destination: Path) -> None:
    """把归档还原到外层已确认的空目标，使用标准解包边界。"""
    try:
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r") as source:
            source.extractall(destination, filter="data")
    except (OSError, tarfile.TarError) as exc:
        raise IrisRunRecoveryError(
            "workspace 还原失败", path=str(archive), destination=str(destination)
        ) from exc


def restore_database(backup: Path, destination: Path) -> None:
    """从已关闭的备份复制独立工作库，不把恢复包作为工作数据库。"""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, destination)
    except OSError as exc:
        raise IrisRunRecoveryError(
            "数据库还原失败", path=str(backup), destination=str(destination)
        ) from exc
