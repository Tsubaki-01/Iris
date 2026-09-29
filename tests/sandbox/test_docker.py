"""Docker 资源层负责物理容器，不协调命令任务。"""

import sys
from pathlib import Path

import pytest

from iris.exceptions import IrisSandboxError
from iris.sandbox import DockerConfig
from iris.sandbox.docker import DockerSandbox

from ..command.test_docker import FakeClient, FakeContainer, driver

__all__ = ["driver"]


@pytest.mark.asyncio
async def test_prepare_checks_explicit_endpoint_and_image_without_creating_container(
    tmp_path: Path, driver: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote:2375")
    config = DockerConfig(endpoint="unix:///local/docker.sock", image="custom:local")
    sandbox = DockerSandbox(tmp_path, config, workspace_writable=True, owner_id="owner")

    await sandbox.prepare()

    assert driver.kwargs["url"] == config.endpoint
    assert driver.info_calls == 1
    assert driver.image_calls == ["custom:local"]
    assert not driver.containers.created
    await sandbox.aclose()
    assert driver.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["missing-extra", "windows-engine", "missing-image"])
async def test_prepare_failure_uses_sandbox_error_without_creating_container(
    tmp_path: Path, driver: FakeClient, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    if problem == "missing-extra":
        monkeypatch.setitem(sys.modules, "aiodocker", None)
    elif problem == "windows-engine":
        driver.os_type = "windows"
    else:
        driver.image_missing = True
    sandbox = DockerSandbox(tmp_path, DockerConfig(), workspace_writable=True, owner_id="owner")

    with pytest.raises(IrisSandboxError):
        await sandbox.prepare()

    assert not driver.containers.created
    await sandbox.aclose()


@pytest.mark.asyncio
async def test_create_passes_resource_configuration_and_stop_allows_reuse(
    tmp_path: Path, driver: FakeClient
) -> None:
    config = DockerConfig(
        image="custom:local",
        network="bridge",
        cpus=0.5,
        memory_mb=256,
        pids_limit=32,
        environment={"CUSTOM": "value"},
    )
    sandbox = DockerSandbox(tmp_path, config, workspace_writable=False, owner_id="owner")
    await sandbox.prepare()
    await sandbox.create()
    container = driver.containers.container
    assert sandbox.container is container
    assert not container.running
    created, name = driver.containers.created[0]
    assert "owner" in name
    assert created["Image"] == "custom:local"
    host = created["HostConfig"]
    assert host["Mounts"] == [
        {"Type": "bind", "Source": str(tmp_path), "Target": "/workspace", "ReadOnly": True}
    ]
    assert host["NetworkMode"] == "bridge"
    assert host["NanoCpus"] == 500_000_000
    assert host["Memory"] == 256 * 1024 * 1024
    assert host["PidsLimit"] == 32
    environment = dict(value.split("=", 1) for value in created["Env"])
    assert environment["CUSTOM"] == "value"
    assert environment["IMAGE_ONLY"] == "yes"

    await sandbox.start()
    await sandbox.stop()
    assert not container.running
    assert container.stops == 1
    assert not container.deleted
    await sandbox.create()
    await sandbox.start()
    assert len(driver.containers.created) == 1
    assert container.starts == 2
    await sandbox.stop()
    await sandbox.aclose()
    assert container.deleted
    assert driver.closed


@pytest.mark.asyncio
async def test_close_removes_only_stopped_owned_container_and_keeps_workspace(
    tmp_path: Path, driver: FakeClient
) -> None:
    report = tmp_path / "report.txt"
    report.write_text("keep", encoding="utf-8")
    external = FakeContainer()
    external.running = True
    sandbox = DockerSandbox(tmp_path, DockerConfig(), workspace_writable=True, owner_id="owner")
    await sandbox.prepare()
    await sandbox.create()
    owned = driver.containers.container
    await sandbox.start()
    await sandbox.stop()
    driver.containers.container = external

    await sandbox.aclose()

    assert owned.deleted
    assert owned.stops == 1
    assert external.running
    assert not external.deleted
    assert not driver.containers.gets
    assert driver.closed
    assert report.read_text(encoding="utf-8") == "keep"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "start", "stop", "aclose"])
async def test_driver_failures_are_reported_as_sandbox_errors(
    tmp_path: Path, driver: FakeClient, operation: str
) -> None:
    sandbox = DockerSandbox(tmp_path, DockerConfig(), workspace_writable=True, owner_id="owner")
    await sandbox.prepare()
    container = driver.containers.container
    if operation == "create":
        driver.containers.create_error = True
        action = sandbox.create
    else:
        await sandbox.create()
        if operation == "start":
            container.start_response_error = True
            action = sandbox.start
        else:
            await sandbox.start()
            if operation == "stop":
                container.stop_failures.append(False)
                action = sandbox.stop
            else:
                await sandbox.stop()
                container.delete_failures = 1
                action = sandbox.aclose

    with pytest.raises(IrisSandboxError):
        await action()
