"""Memory 备份保留捕获水位。"""

from pathlib import Path

import pytest

from iris.exceptions import IrisMemoryError
from iris.memory.generation_models import MemoryCaptureSource
from iris.memory.sqlite import SQLiteMemoryStore


def test_backup_keeps_capture_progress(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    source = store.register_source(
        MemoryCaptureSource(
            lifecycle_source_id="source",
            run_id="run",
            session_id="session",
            namespace="project",
            initial_message_count=2,
            captured_until=4,
        )
    )
    backup = tmp_path / "snapshot/memory.sqlite"
    store.backup(backup)
    saved = SQLiteMemoryStore(backup)
    assert saved.list_capture_sources("source", "project") == [source]
    store.commit_capture(
        source.model_copy(update={"captured_until": 5}), expected_captured_until=4, episode=None
    )
    assert saved.list_capture_sources("source", "project")[0].captured_until == 4
    with pytest.raises(IrisMemoryError):
        store.backup(store.path)
