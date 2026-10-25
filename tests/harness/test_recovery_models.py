"""恢复包的唯一解析契约。"""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from iris.harness.recovery_models import (
    EnvironmentSnapshotManifest,
    EvolutionStateSnapshot,
    LatestSnapshot,
    MemoryStateSnapshot,
    SnapshotFiles,
)
from iris.sandbox.config import DockerConfig
from iris.sandbox.snapshot import DockerSnapshot


def manifest(workspace: Path) -> EnvironmentSnapshotManifest:
    """构造含两种持久状态的恢复点。"""
    return EnvironmentSnapshotManifest(
        snapshot_id=uuid4(),
        created_at=datetime.now(UTC),
        source_id="test-source",
        workspace_root=workspace,
        lifecycle_path=workspace / ".iris/lifecycle.sqlite",
        docker=DockerSnapshot(
            image_ref="iris-recovery/test-source:test",
            config=DockerConfig(image="sha256:test", environment={"PATH": "/custom/bin"}),
            user="1000:1000",
            working_dir="/workspace",
            entrypoint=("/bin/sh",),
            command=("-c", "sleep infinity"),
            workspace_writable=True,
        ),
        states=(
            MemoryStateSnapshot(
                role="memory",
                database_path=".iris/memory.sqlite",
                projection_root=".iris/memory",
                write_namespace="default",
            ),
            EvolutionStateSnapshot(
                role="evolution",
                database_path=".iris/evolution.sqlite",
                skill_path=".agents/skills/project/SKILL.md",
                prompt_paths={"system": ".iris/prompts/system.md"},
                config_path=None,
            ),
        ),
    )


def test_manifest_round_trip(tmp_path: Path) -> None:
    original = manifest(tmp_path)
    parsed = EnvironmentSnapshotManifest.model_validate_json(original.model_dump_json())
    assert parsed == original
    assert parsed.docker.config.environment == {"PATH": "/custom/bin"}
    assert isinstance(parsed.states[0], MemoryStateSnapshot)
    assert isinstance(parsed.states[1], EvolutionStateSnapshot)
    assert (
        LatestSnapshot.model_validate_json(
            LatestSnapshot(snapshot_id=original.snapshot_id).model_dump_json()
        ).snapshot_id
        == original.snapshot_id
    )


@pytest.mark.parametrize("change", ["duplicate", "version", "naive", "relative"])
def test_invalid_manifest_fails_at_parse(tmp_path: Path, change: str) -> None:
    raw = manifest(tmp_path).model_dump(mode="json")
    if change == "duplicate":
        raw["states"].append(raw["states"][0])
    elif change == "version":
        raw["format_version"] = 2
    elif change == "naive":
        raw["created_at"] = "2026-11-01T12:00:00"
    else:
        raw["workspace_root"] = "relative"
    with pytest.raises(ValidationError):
        EnvironmentSnapshotManifest.model_validate(raw)


def test_snapshot_files_are_pure_paths(tmp_path: Path) -> None:
    snapshot_id = uuid4()
    files = SnapshotFiles(tmp_path, snapshot_id)
    assert files.candidate_directory == tmp_path / str(snapshot_id)
    assert files.manifest_path == files.candidate_directory / "manifest.json"
    assert files.memory_backup == files.candidate_directory / "state/memory.sqlite"
    assert files.evolution_backup == files.candidate_directory / "state/evolution.sqlite"
    assert not list(tmp_path.iterdir())
