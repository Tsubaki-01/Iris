"""Docker driver 的受控契约测试；不代表真实 Docker 集成证据。"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import tarfile
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from iris.command.docker import DockerCommandService
from iris.command.models import (
    CommandRequest,
    CommandScope,
    CommandStatus,
    ShellCommand,
)
from iris.exceptions import IrisCommandError, IrisToolOutcomeUnknownError
from iris.sandbox import DockerConfig


class FakeDockerError(Exception):
    """提供真实 driver 的状态码字段。"""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class FakeStream:
    """分别控制 start 响应、前台退出和输出 EOF。"""

    def __init__(self, execution: FakeExec) -> None:
        self.execution = execution
        self.queue: asyncio.Queue[SimpleNamespace | None | Exception] = asyncio.Queue()
        self.closed = False

    async def __aenter__(self) -> FakeStream:
        execution = self.execution
        execution.started.set()
        if execution.container.exec_start_gate is not None and execution.command is not None:
            await execution.container.exec_start_gate.wait()
        execution.running = True
        if execution.command is None:
            await execution.control()
        elif execution.command not in execution.container.blocked:
            execution.finish()
        return self

    async def read_out(self) -> SimpleNamespace | None:
        value = await self.queue.get()
        if isinstance(value, Exception):
            raise value
        return value

    async def close(self) -> None:
        self.closed = True
        self.queue.put_nowait(None)


class FakeExec:
    """模拟 helper exec 和小型结果控制 exec。"""

    def __init__(self, container: FakeContainer, cmd: list[str], kwargs: dict[str, Any]) -> None:
        self.container = container
        self.cmd = cmd
        self.kwargs = kwargs
        self.command = cmd[5] if len(cmd) == 8 else None
        if len(cmd) == 8 and cmd[4] == "python":
            self.command = container.sources[cmd[5]].decode("utf-8")
        self.started = asyncio.Event()
        self.running = False
        self.exit_code = 0
        self.stream = FakeStream(self)

    def start(self, *, detach: bool, timeout: Any = None) -> FakeStream:
        assert not detach
        return self.stream

    async def inspect(self) -> dict[str, Any]:
        return {"Running": self.running, "ExitCode": self.exit_code}

    def finish(self, *, reason: str = "exited", returncode: int = 0) -> None:
        command = self.command
        assert command is not None
        if command.startswith("exit "):
            returncode = int(command.split()[1])
        if command == "timeout":
            reason, returncode = "timed_out", 0
        if command != "missing-result":
            self.container.results[self.cmd[-1]] = json.dumps(
                {"reason": reason, "returncode": returncode}
            )
        data = b"x" * (2 * 1024 * 1024) if command == "large" else command.encode()
        self.stream.queue.put_nowait(SimpleNamespace(stream=1, data=data))
        self.stream.queue.put_nowait(SimpleNamespace(stream=2, data=b"stderr"))
        self.running = False
        if command != "inherited-pipe":
            self.stream.queue.put_nowait(None)

    async def control(self) -> None:
        path = self.cmd[-1]
        if "unlink" in self.cmd[2]:
            if self.container.delete_result_error:
                raise FakeDockerError(500, "delete result failed")
            self.container.results.pop(path, None)
        else:
            if path not in self.container.results:
                self.exit_code = 1
            else:
                payload = self.container.results[path].encode()
                self.stream.queue.put_nowait(SimpleNamespace(stream=1, data=payload))
        self.running = False
        self.stream.queue.put_nowait(None)


class FakeContainer:
    """同一 root 容器，在停止时中断所有仍运行的 exec。"""

    def __init__(self) -> None:
        self.id = "container-1"
        self.running = False
        self.starts = 0
        self.stops = 0
        self.shows = 0
        self.deleted = False
        self.execs: list[FakeExec] = []
        self.results: dict[str, str] = {}
        self.blocked: set[str] = set()
        self.start_gate: asyncio.Event | None = None
        self.exec_start_gate: asyncio.Event | None = None
        self.stop_gate: asyncio.Event | None = None
        self.start_entered = asyncio.Event()
        self.stop_entered = asyncio.Event()
        self.stop_failures: deque[bool] = deque()
        self.delete_result_error = False
        self.delete_failures = 0
        self.start_response_error = False
        self.sources: dict[str, bytes] = {}
        self.archive_gate: asyncio.Event | None = None
        self.archive_entered = asyncio.Event()
        self.archive_error = False

    async def put_archive(self, path: str, data: bytes) -> None:
        assert path == "/tmp"
        self.archive_entered.set()
        if self.archive_gate is not None:
            await self.archive_gate.wait()
        if self.archive_error:
            raise FakeDockerError(500, "upload response lost")
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            member = archive.getmembers()[0]
            assert member.mode & 0o444
            assert member.uid == member.gid == 1000
            source = archive.extractfile(member)
            assert source is not None
            self.sources[f"/tmp/{member.name}"] = source.read()

    async def start(self) -> None:
        self.starts += 1
        self.start_entered.set()
        if self.start_gate is not None:
            await self.start_gate.wait()
        self.running = True
        if self.start_response_error:
            raise FakeDockerError(500, "start response lost")

    async def exec(self, cmd: list[str], **kwargs: Any) -> FakeExec:
        assert self.running
        execution = FakeExec(self, cmd, kwargs)
        self.execs.append(execution)
        return execution

    async def stop(self, *, t: int, timeout: Any) -> None:
        assert t == 0
        self.stops += 1
        self.stop_entered.set()
        if self.stop_gate is not None:
            await self.stop_gate.wait()
        if self.stop_failures:
            stopped = self.stop_failures.popleft()
            if stopped:
                self.running = False
            raise FakeDockerError(500, "stop response lost")
        self.running = False
        for execution in self.execs:
            if execution.running:
                execution.running = False
                execution.exit_code = 137
                execution.stream.queue.put_nowait(None)

    async def show(self) -> dict[str, Any]:
        self.shows += 1
        return {"State": {"Running": self.running}}

    async def delete(self, **kwargs: Any) -> None:
        assert not self.running
        if self.delete_failures:
            self.delete_failures -= 1
            raise FakeDockerError(500, "delete response failed")
        self.deleted = True


class FakeContainers:
    """只允许 create/get；故意不提供会自动 pull 的 run。"""

    def __init__(self) -> None:
        self.container = FakeContainer()
        self.created: list[tuple[dict[str, Any], str]] = []
        self.create_gate: asyncio.Event | None = None
        self.create_entered = asyncio.Event()
        self.create_error = False
        self.gets: list[str] = []
        self.get_error: FakeDockerError | None = None

    async def create(self, config: dict[str, Any], *, name: str) -> FakeContainer:
        self.created.append((config, name))
        self.create_entered.set()
        if self.create_gate is not None:
            await self.create_gate.wait()
        if self.create_error:
            raise FakeDockerError(500, "create response lost")
        return self.container

    async def get(self, name: str) -> FakeContainer:
        self.gets.append(name)
        if self.get_error is not None:
            raise self.get_error
        return self.container


class FakeClient:
    """记录明确 endpoint、一次准备与资源释放。"""

    def __init__(self) -> None:
        self.containers = FakeContainers()
        self.images = SimpleNamespace(inspect=self.inspect_image)
        self.system = SimpleNamespace(info=self.info)
        self.info_calls = 0
        self.image_calls: list[str] = []
        self.closed = False
        self.os_type = "linux"
        self.image_missing = False
        self.kwargs: dict[str, Any] = {}

    async def info(self) -> dict[str, Any]:
        self.info_calls += 1
        return {"OSType": self.os_type, "OperatingSystem": "Docker Desktop"}

    async def inspect_image(self, image: str) -> dict[str, Any]:
        self.image_calls.append(image)
        if self.image_missing:
            raise FakeDockerError(404, "missing image")
        return {"Config": {"Env": ["PATH=/image/bin:/usr/bin", "IMAGE_ONLY=yes"]}}

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def driver(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    """替换唯一可选 driver 导入，不连接真实 daemon。"""
    client = FakeClient()

    def create_client(**kwargs: Any) -> FakeClient:
        client.kwargs = kwargs
        return client

    monkeypatch.setitem(sys.modules, "aiodocker", SimpleNamespace(Docker=create_client))
    monkeypatch.setitem(
        sys.modules, "aiodocker.exceptions", SimpleNamespace(DockerError=FakeDockerError)
    )
    return client


def request(root: Path, command: str, *, call_id: str = "call") -> CommandRequest:
    """构造已经由工具边界解析的请求。"""
    return CommandRequest(call_id, ShellCommand(command), root, 0.5)


async def started(container: FakeContainer, count: int) -> None:
    """等待具体命令实际进入 driver start。"""
    async with asyncio.timeout(2):
        while len([item for item in container.execs if item.command is not None]) < count:
            await asyncio.sleep(0)
        commands = [item for item in container.execs if item.command is not None]
        await commands[count - 1].started.wait()


@pytest.mark.asyncio
async def test_prepare_once_explicit_local_endpoint_without_container(
    tmp_path: Path, driver: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote:2375")
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    await asyncio.gather(service.prepare(), service.prepare())
    assert driver.kwargs["url"].startswith(("unix:///", "npipe:////"))
    assert driver.info_calls == 1
    assert driver.image_calls == ["iris-command:local"]
    assert not driver.containers.created
    await service.aclose()
    assert driver.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["missing-extra", "windows-engine", "missing-image"])
async def test_prepare_failure_never_falls_back(
    tmp_path: Path, driver: FakeClient, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    if problem == "missing-extra":
        monkeypatch.setitem(sys.modules, "aiodocker", None)
    elif problem == "windows-engine":
        driver.os_type = "windows"
    else:
        driver.image_missing = True
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    with pytest.raises(IrisCommandError) as caught:
        await service.prepare()
    if problem == "missing-image":
        assert "显式构建" in str(caught.value)
        assert "iris-command:local" in str(caught.value)
    assert not driver.containers.created
    await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [0, 7, 124, 137])
async def test_real_result_protocol_preserves_exit_codes(
    tmp_path: Path, driver: FakeClient, exit_code: int
) -> None:
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    outcome = await service.execute(CommandScope("r", "s"), request(tmp_path, f"exit {exit_code}"))
    assert outcome.status is CommandStatus.EXITED
    assert outcome.exit_code == exit_code
    assert outcome.stderr == "stderr"
    assert not driver.containers.container.results
    assert driver.containers.container.running
    assert driver.containers.container.stops == 0
    await service.aclose()


@pytest.mark.asyncio
async def test_shared_container_configuration_and_child_cwd(
    tmp_path: Path, driver: FakeClient
) -> None:
    config = DockerConfig(
        image="my-command:local", network="bridge", environment={"CUSTOM": "value"}
    )
    service = DockerCommandService(tmp_path, config, workspace_writable=False)
    child = tmp_path / "中文 space"
    child.mkdir()
    outcomes = await asyncio.gather(
        service.execute(CommandScope("r1", "s1"), request(child, "one")),
        service.execute(CommandScope("r2", "s2"), request(tmp_path, "two")),
    )
    assert len(driver.containers.created) == 1
    container_config, _name = driver.containers.created[0]
    assert driver.image_calls == ["my-command:local"]
    assert container_config["Image"] == "my-command:local"
    host = container_config["HostConfig"]
    assert host["Mounts"] == [
        {"Type": "bind", "Source": str(tmp_path), "Target": "/workspace", "ReadOnly": True}
    ]
    assert host["CapDrop"] == ["ALL"]
    assert host["SecurityOpt"] == ["no-new-privileges:true"]
    assert host["NetworkMode"] == "bridge"
    assert host["NanoCpus"] == 2_000_000_000
    assert host["Memory"] == 1024 * 1024 * 1024
    assert host["PidsLimit"] == 128
    assert host["Init"] is True
    assert host["AutoRemove"] is False
    assert container_config["User"] == "1000:1000"
    assert container_config["Entrypoint"] == ["sleep"]
    assert container_config["Cmd"] == ["infinity"]
    environment = dict(value.split("=", 1) for value in container_config["Env"])
    assert environment["HOME"] == "/tmp"
    assert environment["PYTHONUSERBASE"] == "/tmp/.local"
    assert environment["PATH"] == "/tmp/.local/bin:/image/bin:/usr/bin"
    assert environment["IMAGE_ONLY"] == "yes"
    assert environment["CUSTOM"] == "value"
    commands = [item for item in driver.containers.container.execs if item.command is not None]
    assert commands[0].kwargs["workdir"] == "/workspace/中文 space"
    assert commands[0].cmd[-1].startswith("/tmp/iris-command-")
    assert commands[0].cmd[-1] != commands[1].cmd[-1]
    assert outcomes[0].cwd == "中文 space"
    await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["large", "inherited-pipe", "timeout"])
async def test_output_and_business_timeout_are_bounded(
    tmp_path: Path, driver: FakeClient, command: str
) -> None:
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    outcome = await asyncio.wait_for(
        service.execute(CommandScope("r", "s"), request(tmp_path, command)), 3
    )
    if command == "timeout":
        assert outcome.status is CommandStatus.TIMED_OUT
        assert outcome.exit_code is None
    else:
        assert outcome.status is CommandStatus.EXITED
        assert outcome.output_truncated
        if command == "large":
            assert outcome.output_stats.stdout_retained_bytes == 512 * 1024
            assert outcome.output_stats.stdout_bytes == 2 * 1024 * 1024
        else:
            assert outcome.output_stats.truncation_reasons == frozenset({"drain_timeout"})
    assert driver.containers.container.stops == 0
    await service.aclose()


@pytest.mark.asyncio
async def test_missing_result_is_unknown_with_confirmed_stop_receipt(
    tmp_path: Path, driver: FakeClient
) -> None:
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    with pytest.raises(IrisToolOutcomeUnknownError) as caught:
        await service.execute(CommandScope("r", "s"), request(tmp_path, "missing-result"))
    receipt = caught.value.stop_receipt
    assert receipt is not None
    await service.wait_drained(receipt)
    assert driver.containers.container.stops == 1
    await service.aclose()


@pytest.mark.asyncio
async def test_result_delete_failure_preserves_known_outcome(
    tmp_path: Path, driver: FakeClient
) -> None:
    driver.containers.container.delete_result_error = True
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    outcome = await service.execute(CommandScope("r", "s"), request(tmp_path, "exit 7"))
    assert outcome.exit_code == 7
    assert not outcome.output_truncated
    assert driver.containers.container.stops == 0
    await service.aclose()
