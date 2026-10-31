"""快照 scope、取消收尾及定向镜像保留。"""

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

from iris.command.docker import DockerCommandService
from iris.command.models import CommandScope
from iris.exceptions import IrisCommandCleanupError, IrisCommandError
from iris.sandbox import DockerConfig

from .test_docker import FakeClient, driver, request

__all__ = ["driver"]


@pytest.mark.asyncio
async def test_snapshot_holds_launch_and_restores_exact_configuration(
    tmp_path: Path,
    driver: FakeClient,
) -> None:
    service = DockerCommandService(
        tmp_path, DockerConfig(environment={"PATH": "/custom"}), workspace_writable=True
    )
    await service.execute(CommandScope("session", "run"), request(tmp_path, "first"))
    async with service.snapshot(source_id="source", snapshot_id=uuid4()) as snapshot:
        assert driver.containers.container.paused
        queued = asyncio.create_task(
            service.execute(CommandScope("session", "next"), request(tmp_path, "next"))
        )
        await asyncio.sleep(0)
        assert not queued.done()
    await queued
    service.adopt_snapshot_retention(source_id="source", keep_image_ref=snapshot.image_ref)
    await service.aclose()
    restored = DockerCommandService.from_snapshot(tmp_path, snapshot)
    assert restored._service_id != service._service_id
    driver.closed = False
    await restored.prepare()
    config = restored._sandbox._container_config()
    assert config["Image"] == snapshot.config.image
    assert config["User"] == snapshot.user
    assert dict(value.split("=", 1) for value in config["Env"]) == snapshot.config.environment
    await restored.aclose()


@pytest.mark.asyncio
async def test_cancel_waits_for_commit_before_unpause(tmp_path: Path, driver: FakeClient) -> None:
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    await service.execute(CommandScope("session", "run"), request(tmp_path, "first"))
    container = driver.containers.container
    container.commit_gate = asyncio.Event()

    async def save() -> None:
        async with service.snapshot(source_id="source", snapshot_id=uuid4()):
            pytest.fail("cancelled save must not publish")

    task = asyncio.create_task(save())
    await container.commit_entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    assert container.paused and not task.done()
    container.commit_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not container.paused and container.unpauses == 1
    await service.aclose()
    assert driver.image_refs == {"iris-command:local": "sha256:base"}


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["commit", "unpause"])
async def test_snapshot_failures_are_reported(
    tmp_path: Path, driver: FakeClient, error: str
) -> None:
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    await service.execute(CommandScope("s", "r"), request(tmp_path, "first"))
    container = driver.containers.container
    container.commit_error = error == "commit"
    container.unpause_error = error == "unpause"
    expected = IrisCommandError if error == "commit" else IrisCommandCleanupError
    with pytest.raises(expected):
        async with service.snapshot(source_id="source", snapshot_id=uuid4()):
            pass
    container.unpause_error = False
    await service.aclose()


@pytest.mark.asyncio
async def test_prune_defers_active_image_and_close_releases_it(
    tmp_path: Path, driver: FakeClient
) -> None:
    service = DockerCommandService(
        tmp_path, DockerConfig(image="sha256:old"), workspace_writable=True
    )
    driver.image_refs.update(
        {
            "iris-recovery/source:old": "sha256:old",
            "iris-recovery/source:new": "sha256:new",
            "iris-recovery/other:keep": "sha256:other",
        }
    )
    await service.execute(CommandScope("s", "r"), request(tmp_path, "first"))
    service.adopt_snapshot_retention(source_id="source", keep_image_ref="iris-recovery/source:new")
    assert await service.prune_snapshot_images(
        source_id="source", keep_image_ref="iris-recovery/source:new"
    ) == ("iris-recovery/source:old",)
    await service.aclose()
    assert "iris-recovery/source:old" not in driver.image_refs
    assert "iris-recovery/source:new" in driver.image_refs
    assert "iris-recovery/other:keep" in driver.image_refs
    assert "iris-command:local" in driver.image_refs


@pytest.mark.asyncio
async def test_cancel_during_failed_pause_still_reports_failed_cleanup(
    tmp_path: Path,
    driver: FakeClient,
) -> None:
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    await service.execute(CommandScope("s", "r"), request(tmp_path, "first"))
    container = driver.containers.container
    container.pause_gate = asyncio.Event()
    container.pause_error = container.unpause_error = True

    async def save() -> None:
        async with service.snapshot(source_id="source", snapshot_id=uuid4()):
            pytest.fail("pause failed")

    task = asyncio.create_task(save())
    await container.pause_entered.wait()
    task.cancel()
    container.pause_gate.set()
    try:
        with pytest.raises(IrisCommandCleanupError):
            await task
        assert container.paused
    finally:
        container.unpause_error = False
        await service.aclose()
