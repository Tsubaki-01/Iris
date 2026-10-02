"""Python 源码在容器准备期上传，复用共享命令环境。"""

import asyncio
from pathlib import Path

import pytest

from iris.command import CommandRequest, CommandScope, CommandStatus, PythonCode, ShellCommand
from iris.command.docker import DockerCommandService
from iris.exceptions import IrisCommandError
from iris.sandbox import DockerConfig

from .test_docker import FakeClient, driver

__all__ = ["driver"]


@pytest.mark.asyncio
@pytest.mark.parametrize("python", [False, True])
@pytest.mark.parametrize(
    "stdin", [None, b"", "中文无换行".encode() + b"\x00\xff"], ids=["none", "empty", "binary"]
)
async def test_stdin_uses_per_call_binary_upload_and_one_helper_protocol(
    tmp_path: Path, driver: FakeClient, python: bool, stdin: bytes | None
) -> None:
    """源码和输入可共用一次上传；stdin 字节不进入 helper argv。"""
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=False)
    payload = PythonCode("print('ready')") if python else ShellCommand("echo ready")
    try:
        results = await asyncio.gather(
            *(
                service.execute(
                    CommandScope("r", "s"),
                    CommandRequest(str(index), payload, tmp_path, 5, stdin=stdin),
                )
                for index in range(2)
            )
        )
        container = driver.containers.container
        executions = [item for item in container.execs if item.command is not None]
        assert len(executions) == 2
        assert all(result.status is CommandStatus.EXITED for result in results)
        for execution in executions:
            assert len(execution.cmd) == 9
            assert execution.kwargs["stdin"] is False
            input_path = execution.cmd[7]
            if stdin is None:
                assert input_path == ""
            else:
                assert input_path.startswith("/tmp/iris-stdin-")
                assert container.sources[input_path] == stdin
        if stdin is not None:
            assert executions[0].cmd[7] != executions[1].cmd[7]
        if python or stdin is not None:
            assert len(container.archives) == 2
            assert all(
                len(items) == int(python) + int(stdin is not None) for items in container.archives
            )
        else:
            assert container.archives == []
    finally:
        await service.aclose()


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
@pytest.mark.parametrize("python", [False, True])
async def test_cancel_during_upload_does_not_start_user_program(
    tmp_path: Path, driver: FakeClient, python: bool
) -> None:
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    container = driver.containers.container
    container.archive_gate = asyncio.Event()
    task = asyncio.create_task(
        service.execute(
            CommandScope("r", "s"),
            CommandRequest(
                "py",
                PythonCode("print(1)") if python else ShellCommand("echo ready"),
                tmp_path,
                5,
                stdin=b"one-shot input",
            ),
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
        await service.wait_drained(result.stop_receipt)
        assert container.starts == 1
        assert not container.running
        assert any(path.startswith("/tmp/iris-stdin-") for path in container.sources)
    finally:
        container.archive_gate.set()
        await service.aclose()
    assert container.deleted
