"""Python 源码在容器准备期上传，复用共享命令环境。"""

import asyncio
from pathlib import Path

import pytest

from iris.command import CommandRequest, CommandScope, CommandStatus, PythonCode
from iris.command.docker import DockerCommandService
from iris.exceptions import IrisCommandError
from iris.sandbox import DockerConfig

from .test_docker import FakeClient, driver

__all__ = ["driver"]


@pytest.mark.asyncio
async def test_python_upload_works_with_readonly_root_and_child_cwd(
    tmp_path: Path, driver: FakeClient
) -> None:
    child = tmp_path / "child 中文"
    child.mkdir()
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=False)
    code = "print('中文')\n" + "# large\n" * 20_000
    try:
        result = await service.execute(
            CommandScope("r", "s"), CommandRequest("py", PythonCode(code), child, 5)
        )
        container = driver.containers.container
        execution = next(item for item in container.execs if item.command is not None)
        assert execution.cmd[4] == "python"
        assert code not in execution.cmd
        assert execution.cmd[5].startswith("/tmp/iris-python-")
        assert container.sources[execution.cmd[5]] == code.encode()
        assert execution.kwargs["workdir"] == "/workspace/child 中文"
        assert not execution.kwargs["stdin"]
        assert driver.containers.created[0][0]["HostConfig"]["Mounts"][0]["ReadOnly"]
        assert result.status is CommandStatus.EXITED
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_failed_upload_is_known_not_dispatched(tmp_path: Path, driver: FakeClient) -> None:
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    container = driver.containers.container
    container.archive_error = True
    try:
        with pytest.raises(IrisCommandError) as caught:
            await service.execute(
                CommandScope("r", "s"), CommandRequest("py", PythonCode("print(1)"), tmp_path, 5)
            )
        assert caught.value.context["started"] is False
        assert not container.execs
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_cancel_during_upload_does_not_start_user_program(
    tmp_path: Path, driver: FakeClient
) -> None:
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    container = driver.containers.container
    container.archive_gate = asyncio.Event()
    task = asyncio.create_task(
        service.execute(
            CommandScope("r", "s"), CommandRequest("py", PythonCode("print(1)"), tmp_path, 5)
        )
    )
    try:
        await container.archive_entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        container.archive_gate.set()
        result = await asyncio.wait_for(task, 2)
        assert result.status is CommandStatus.CANCELLED
        assert not container.execs
        assert result.stop_receipt is not None
    finally:
        container.archive_gate.set()
        await service.aclose()
