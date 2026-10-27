"""三态物理快照和有效容器描述。"""

from pathlib import Path
from uuid import uuid4

import pytest

from iris.exceptions import IrisSandboxError
from iris.sandbox import DockerConfig
from iris.sandbox.docker import DockerSandbox

from ..command.test_docker import FakeClient, driver

__all__ = ["driver"]


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["absent", "running", "stopped"])
async def test_physical_snapshot_states(tmp_path: Path, driver: FakeClient, state: str) -> None:
    sandbox = DockerSandbox(
        tmp_path,
        DockerConfig(environment={"PATH": "/custom"}),
        workspace_writable=False,
        owner_id="owner",
    )
    await sandbox.prepare()
    if state != "absent":
        await sandbox.create()
    if state == "running":
        await sandbox.start()
    paused = await sandbox.freeze_for_snapshot()
    snapshot = await sandbox.commit_snapshot(source_id="source", snapshot_id=uuid4())
    container = driver.containers.container
    assert paused is (state == "running")
    assert container.paused is paused
    assert snapshot.config.environment["PATH"] == "/custom"
    assert snapshot.config.endpoint == driver.kwargs["url"]
    assert not snapshot.workspace_writable
    assert snapshot.config.image.startswith("sha256:")
    if state == "absent":
        assert not driver.containers.created and not container.commits
    else:
        assert container.commits[0]["pause"] is False
        assert container.commits[0]["timeout"].total is None
    await sandbox.release_snapshot(paused_here=paused)
    assert not container.paused
    assert container.running is (state == "running")
    await sandbox.stop()
    await sandbox.aclose()


@pytest.mark.asyncio
async def test_pause_lost_response_is_reconciled(tmp_path: Path, driver: FakeClient) -> None:
    sandbox = DockerSandbox(tmp_path, DockerConfig(), workspace_writable=True, owner_id="owner")
    await sandbox.prepare()
    await sandbox.create()
    await sandbox.start()
    driver.containers.container.pause_error = True
    with pytest.raises(IrisSandboxError):
        await sandbox.freeze_for_snapshot()
    assert not driver.containers.container.paused
    assert driver.containers.container.unpauses == 1
    await sandbox.stop()
    await sandbox.aclose()
