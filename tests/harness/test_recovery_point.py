"""完整恢复点保存与失败原子性。"""

import asyncio
import threading
from pathlib import Path

import pytest

from iris.agents import AgentConfig
from iris.exceptions import (
    IrisCommandCleanupError,
    IrisConfigError,
    IrisRunPersistenceError,
    IrisRunStateError,
)
from iris.harness import AgentRunner, recovery_point
from iris.harness._recovery_files import SnapshotStorage
from iris.lifecycle import AgentRunRequest
from iris.store import SQLiteStore

from ..command.test_docker import FakeClient, driver
from .fakes import StaticProvider, text_response

__all__ = ["driver"]


def snapshot_runner(tmp_path: Path, *, enabled: bool = True) -> AgentRunner:
    """独立 workspace、工作库与 fake Docker root。"""
    config = AgentConfig.model_validate(
        {
            "name": "snapshot",
            "model": "openai/test",
            "system": "test",
            "permissions": {"workspace": str(tmp_path / "workspace")},
            "session": {"backend": "sqlite", "path": str(tmp_path / "live.sqlite")},
            "command": {"mode": "docker"},
            "environment_recovery": {"enabled": enabled},
        }
    )
    return AgentRunner.from_config(config, provider=StaticProvider(text_response("done")))


@pytest.mark.asyncio
async def test_save_replaces_latest_without_mutating_working_state(
    tmp_path: Path,
    driver: FakeClient,
) -> None:
    runner = snapshot_runner(tmp_path)
    path = runner.runtime.environment.workspace_root / "value.txt"
    path.write_text("first")
    result = await runner.start(AgentRunRequest(input="hello", session_id="a"))
    directory = tmp_path / "snapshots"
    first = await runner.save_environment_snapshot(directory)
    assert (
        SQLiteStore(directory / str(first.snapshot_id) / "lifecycle.sqlite").load_result(
            result.run.run_id
        )
        == result
    )
    path.write_text("second")
    second = await runner.save_environment_snapshot(directory)
    assert SnapshotStorage(directory).load_latest() == second
    assert SnapshotStorage(directory).list_candidates() == (second.snapshot_id,)
    assert first.docker.image_ref not in driver.image_refs
    assert second.docker.image_ref in driver.image_refs
    assert not driver.containers.created
    await runner.aclose()
    assert second.docker.image_ref in driver.image_refs


@pytest.mark.asyncio
async def test_disabled_fails_before_directory_io(tmp_path: Path, driver: FakeClient) -> None:
    runner = snapshot_runner(tmp_path, enabled=False)
    with pytest.raises(IrisConfigError):
        await runner.save_environment_snapshot(tmp_path / "snapshots")
    assert not (tmp_path / "snapshots").exists()
    assert not driver.info_calls
    await runner.aclose()


@pytest.mark.asyncio
async def test_archive_failure_preserves_previous_point(
    tmp_path: Path,
    driver: FakeClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = snapshot_runner(tmp_path)
    directory = tmp_path / "snapshots"
    first = await runner.save_environment_snapshot(directory)

    def fail(*args: object, **kwargs: object) -> None:
        raise IrisRunPersistenceError("archive failed")

    monkeypatch.setattr(recovery_point, "archive_workspace", fail)
    with pytest.raises(IrisRunPersistenceError):
        await runner.save_environment_snapshot(directory)
    assert SnapshotStorage(directory).load_latest() == first
    assert SnapshotStorage(directory).list_candidates() == (first.snapshot_id,)
    with runner._recovery.operation():
        pass
    await runner.aclose()


@pytest.mark.asyncio
async def test_cancelled_archive_waits_for_thread_and_close(
    tmp_path: Path,
    driver: FakeClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = snapshot_runner(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    archive = recovery_point.archive_workspace

    def block(*args: object, **kwargs: object) -> None:
        entered.set()
        assert release.wait(5)
        archive(*args, **kwargs)

    monkeypatch.setattr(recovery_point, "archive_workspace", block)
    directory = tmp_path / "snapshots"
    task = asyncio.create_task(runner.save_environment_snapshot(directory))
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    with pytest.raises(IrisRunStateError):
        with runner._recovery.operation():
            pass
    closing = asyncio.create_task(runner.aclose())
    await asyncio.sleep(0)
    assert not closing.done() and not driver.closed
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await closing
    assert SnapshotStorage(directory).load_latest() is None
    assert not SnapshotStorage(directory).list_candidates()


@pytest.mark.asyncio
async def test_cancel_at_latest_replacement_preserves_publication(
    tmp_path: Path,
    driver: FakeClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = snapshot_runner(tmp_path)
    directory = tmp_path / "snapshots"
    entered, release = threading.Event(), threading.Event()
    publish = SnapshotStorage.publish

    def delayed(storage: SnapshotStorage, manifest: object) -> None:
        publish(storage, manifest)
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(SnapshotStorage, "publish", delayed)
    task = asyncio.create_task(runner.save_environment_snapshot(directory))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    latest = SnapshotStorage(directory).load_latest()
    assert latest is not None
    assert (directory / str(latest.snapshot_id) / "workspace.tar").exists()
    await runner.aclose()
    assert latest.docker.image_ref in driver.image_refs


@pytest.mark.asyncio
async def test_unpause_failure_retains_latest_and_blocks_execution(
    tmp_path: Path,
    driver: FakeClient,
) -> None:
    from iris.command.models import CommandScope

    from ..command.test_docker import request

    runner = snapshot_runner(tmp_path)
    service = runner.runtime.environment.command_binding.service
    await service.execute(CommandScope("s", "r"), request(tmp_path / "workspace", "first"))
    driver.containers.container.unpause_error = True
    directory = tmp_path / "snapshots"
    with pytest.raises(IrisCommandCleanupError):
        await runner.save_environment_snapshot(directory)
    latest = SnapshotStorage(directory).load_latest()
    assert latest is not None
    with pytest.raises(IrisCommandCleanupError):
        await runner.start(AgentRunRequest(input="blocked"))
    await runner.aclose()
    assert latest.docker.image_ref in driver.image_refs


@pytest.mark.asyncio
async def test_capture_failure_keeps_previous_latest(
    tmp_path: Path,
    driver: FakeClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iris.harness._capture import RunCapture
    from iris.memory import MemoryService, SQLiteMemoryStore

    runner = snapshot_runner(tmp_path)
    directory = tmp_path / "snapshots"
    previous = await runner.save_environment_snapshot(directory)
    memory = MemoryService(SQLiteMemoryStore(tmp_path / "workspace/memory.db"))
    capture = RunCapture(memory_service=memory, namespace="project", lifecycle_store=runner.store)
    runner._capture = capture

    def fail(source: str, namespace: str) -> list[object]:
        raise OSError("last page failed")

    monkeypatch.setattr(memory.store, "list_capture_sources", fail)
    with pytest.raises(IrisRunPersistenceError, match="捕获"):
        await runner.save_environment_snapshot(directory)
    assert SnapshotStorage(directory).load_latest() == previous
    assert not capture._closed
    runner._capture = None
    await runner.aclose()
