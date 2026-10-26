"""文件恢复、精确排除和 latest 原子发布。"""

from pathlib import Path
from uuid import uuid4

import pytest

from iris.exceptions import IrisRunPersistenceError, IrisRunRecoveryError
from iris.harness import _recovery_files as files_module
from iris.harness._recovery_files import (
    SnapshotStorage,
    archive_workspace,
    restore_database,
    restore_workspace,
)
from iris.harness.recovery_models import LatestSnapshot

from .test_recovery_models import manifest


def test_archive_includes_untracked_and_only_excludes_owned_databases(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    contents = {
        "project.txt": "first",
        ".iris/todo/session.md": "- [ ] work",
        ".iris/artifacts/output.txt": "generated",
        ".iris/image-cache/image.bin": "image",
        "other.db": "application",
        ".gitignore": "ignored.txt",
        "ignored.txt": "included",
        ".iris/state.db": "database",
        ".iris/state.db-wal": "wal",
        ".iris/state.db-shm": "shm",
        ".iris/state.db-journal": "journal",
    }
    for name, content in contents.items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    archive = tmp_path / "workspace.tar"
    archive_workspace(workspace, archive, excluded_files=frozenset({workspace / ".iris/state.db"}))
    restored = tmp_path / "restored"
    restore_workspace(archive, restored)
    for name, content in contents.items():
        if name.startswith(".iris/state.db"):
            assert not (restored / name).exists()
        else:
            assert (restored / name).read_text() == content


def test_publish_failure_keeps_previous_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = SnapshotStorage(tmp_path / "snapshots")
    first = manifest(tmp_path / "workspace")
    storage.prepare_candidate(first.snapshot_id)
    storage.publish(first)
    second = first.model_copy(update={"snapshot_id": uuid4()})
    storage.prepare_candidate(second.snapshot_id)
    atomic = files_module.atomic_write_text

    def fail_latest(path: Path, content: str) -> None:
        if path.name == "latest.json":
            raise OSError("publication failed")
        atomic(path, content)

    monkeypatch.setattr(files_module, "atomic_write_text", fail_latest)
    with pytest.raises(IrisRunPersistenceError):
        storage.publish(second)
    assert storage.load_latest() == first
    monkeypatch.setattr(files_module, "atomic_write_text", atomic)
    storage.publish(second)
    assert storage.load_latest() == second
    storage.remove_candidate(first.snapshot_id)
    assert storage.list_candidates() == (second.snapshot_id,)


@pytest.mark.parametrize("corruption", ["missing", "id", "path"])
def test_latest_does_not_fall_back(tmp_path: Path, corruption: str) -> None:
    storage = SnapshotStorage(tmp_path)
    original = manifest(tmp_path / "workspace")
    files = storage.prepare_candidate(original.snapshot_id)
    storage.publish(original)
    if corruption == "missing":
        files.manifest_path.unlink()
    elif corruption == "id":
        (tmp_path / "latest.json").write_text(LatestSnapshot(snapshot_id=uuid4()).model_dump_json())
    else:
        raw = original.model_dump(mode="json")
        raw["states"][0]["database_path"] = "../outside.db"
        import json

        files.manifest_path.write_text(json.dumps(raw))
    with pytest.raises(IrisRunRecoveryError):
        storage.load_latest()


def test_database_restore_creates_working_copy(tmp_path: Path) -> None:
    backup = tmp_path / "backup.sqlite"
    backup.write_bytes(b"saved state")
    working = tmp_path / "working/database.sqlite"
    restore_database(backup, working)
    working.write_bytes(b"new state")
    assert backup.read_bytes() == b"saved state"
