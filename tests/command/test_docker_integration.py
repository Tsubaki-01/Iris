"""显式启用的本地 Docker 实测，仅操作当前服务创建的容器。"""

from __future__ import annotations

import asyncio
import json
import shlex
import sys
import zipfile
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

from iris.command import (
    CommandOutcome,
    CommandRequest,
    CommandScope,
    CommandStatus,
    DockerConfig,
)
from iris.command.docker import DockerCommandService


@pytest.fixture(autouse=True)
def require_real_docker(request: pytest.FixtureRequest) -> None:
    """未显式请求时跳过，不探测引擎或自动拉取镜像。"""
    if not request.config.getoption("--run-docker"):
        pytest.skip("真实 Docker 需显式 --run-docker")


@pytest_asyncio.fixture
async def docker_service(tmp_path: Path) -> AsyncIterator[DockerCommandService]:
    """每例独立 root 服务；prepare 失败必须作为未满足前提报告。"""
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    try:
        await service.prepare()
        yield service
    finally:
        await service.aclose()


async def _python(
    service: DockerCommandService,
    cwd: Path,
    code: str,
    *,
    session: str = "a",
    timeout: float = 5,
) -> CommandOutcome:
    return await service.execute(
        CommandScope(f"run-{session}", session),
        CommandRequest(uuid4().hex, shlex.join(["python", "-c", code]), cwd, timeout),
    )


@pytest.mark.asyncio
async def test_real_shared_container_cwd_and_host_writeback(
    docker_service: DockerCommandService, tmp_path: Path
) -> None:
    child = tmp_path / "中文 space"
    child.mkdir()
    first, second = await asyncio.gather(
        _python(docker_service, tmp_path, "import socket; print(socket.gethostname())"),
        _python(
            docker_service,
            child,
            "import socket, pathlib; pathlib.Path('written.txt').write_text('shared'); "
            "print(socket.gethostname()); print(pathlib.Path.cwd())",
            session="child",
        ),
    )
    assert first.exit_code == second.exit_code == 0
    assert first.stdout.strip() == second.stdout.splitlines()[0]
    assert second.stdout.splitlines()[1] == "/workspace/中文 space"
    assert second.cwd == "中文 space"
    assert (child / "written.txt").read_text() == "shared"


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [124, 137])
async def test_real_program_exit_is_not_timeout(
    docker_service: DockerCommandService, tmp_path: Path, exit_code: int
) -> None:
    result = await _python(docker_service, tmp_path, f"raise SystemExit({exit_code})")
    assert result.status is CommandStatus.EXITED
    assert result.exit_code == exit_code


@pytest.mark.asyncio
async def test_real_timeout_is_local_and_stop_then_restart_retains_files(
    docker_service: DockerCommandService, tmp_path: Path
) -> None:
    saved = await _python(
        docker_service,
        tmp_path,
        "from pathlib import Path; Path('/tmp/iris-retained.txt').write_text('retained')",
    )
    assert saved.exit_code == 0
    timed, other = await asyncio.gather(
        _python(docker_service, tmp_path, "import time; time.sleep(30)", timeout=0.3),
        _python(
            docker_service,
            tmp_path,
            "import time; time.sleep(1); print('other completed')",
            session="b",
        ),
    )
    assert timed.status is CommandStatus.TIMED_OUT
    assert other.status is CommandStatus.EXITED
    assert other.stdout.strip() == "other completed"

    receipt = await docker_service.stop(CommandScope("run-a", "a")).wait_drained()
    restarted = await _python(
        docker_service,
        tmp_path,
        "from pathlib import Path; print(Path('/tmp/iris-retained.txt').read_text())",
        session="c",
    )
    assert restarted.stdout.strip() == "retained"
    await docker_service.wait_drained(receipt)
    again = await _python(docker_service, tmp_path, "print('still running')", session="c")
    assert again.stdout.strip() == "still running"


@pytest.mark.asyncio
async def test_real_shared_stop_interrupts_other_command(
    docker_service: DockerCommandService, tmp_path: Path
) -> None:
    command = asyncio.create_task(
        _python(
            docker_service,
            tmp_path,
            "from pathlib import Path; import time; Path('ready').touch(); time.sleep(30)",
            session="b",
        )
    )
    try:
        async with asyncio.timeout(10):
            while not (tmp_path / "ready").exists():
                await asyncio.sleep(0.02)
        operation = docker_service.stop(CommandScope("run-a", "a"))
        result = await asyncio.wait_for(command, 15)
        assert result.status is CommandStatus.ENVIRONMENT_INTERRUPTED
        assert result.exit_code is None
        assert result.stop_receipt == await operation.wait_drained()
    finally:
        if not command.done():
            command.cancel()
        await asyncio.gather(command, return_exceptions=True)


@pytest.mark.asyncio
async def test_real_root_readonly_bind_prevents_command_write(tmp_path: Path) -> None:
    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=False)
    try:
        await service.prepare()
        result = await _python(service, tmp_path, "from pathlib import Path; Path('deny').touch()")
        assert result.status is CommandStatus.EXITED
        assert result.exit_code != 0
        assert "Read-only file system" in result.stderr
        assert not (tmp_path / "deny").exists()
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_real_none_network_and_container_resource_configuration(
    docker_service: DockerCommandService, tmp_path: Path
) -> None:
    result = await _python(
        docker_service,
        tmp_path,
        "from pathlib import Path; import json; print(json.dumps({"
        "'cpu': Path('/sys/fs/cgroup/cpu.max').read_text().strip(),"
        "'memory': Path('/sys/fs/cgroup/memory.max').read_text().strip(),"
        "'pids': Path('/sys/fs/cgroup/pids.max').read_text().strip(),"
        "'interfaces': sorted(p.name for p in Path('/sys/class/net').iterdir())}))",
    )
    assert result.exit_code == 0, result.stderr
    facts = json.loads(result.stdout)
    quota, period = map(int, facts["cpu"].split())
    assert quota / period == 2
    assert int(facts["memory"]) == 1024 * 1024 * 1024
    assert int(facts["pids"]) == 128
    assert facts["interfaces"] == ["lo"]


def _offline_wheel(workspace: Path) -> str:
    """创建无构建依赖的微型离线 wheel，用于验证容器用户安装目录。"""
    name = "iris_sandbox_probe-0.0.1-py3-none-any.whl"
    entries = {
        "iris_sandbox_probe.py": "VALUE = 'persisted dependency'\n",
        "iris_sandbox_probe-0.0.1.dist-info/METADATA": (
            "Metadata-Version: 2.1\nName: iris-sandbox-probe\nVersion: 0.0.1\n"
        ),
        "iris_sandbox_probe-0.0.1.dist-info/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: iris-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    record = "iris_sandbox_probe-0.0.1.dist-info/RECORD"
    entries[record] = "".join(f"{path},,\n" for path in [*entries, record])
    with zipfile.ZipFile(workspace / name, "w") as wheel:
        for path, content in entries.items():
            wheel.writestr(path, content)
    return name


@pytest.mark.asyncio
async def test_real_offline_dependency_and_background_service_are_shared(
    docker_service: DockerCommandService, tmp_path: Path
) -> None:
    wheel = _offline_wheel(tmp_path)
    installed = await docker_service.execute(
        CommandScope("install", "a"),
        CommandRequest(
            uuid4().hex,
            f"python -m pip install --user --no-index --no-deps --no-compile /workspace/{wheel}",
            tmp_path,
            15,
        ),
    )
    assert installed.exit_code == 0, installed.stderr
    server = tmp_path / "server.py"
    server.write_text(
        "import socket\nfrom pathlib import Path\n"
        "server = socket.socket()\nserver.bind(('127.0.0.1', 0))\nserver.listen()\n"
        "Path('/tmp/probe-port').write_text(str(server.getsockname()[1]))\n"
        "while True:\n"
        "    connection, _ = server.accept()\n"
        "    connection.sendall(b'background alive')\n"
        "    connection.close()\n",
        encoding="utf-8",
    )
    started = await docker_service.execute(
        CommandScope("server", "a"),
        CommandRequest(
            uuid4().hex, "python /workspace/server.py >/tmp/server.log 2>&1 &", tmp_path, 5
        ),
    )
    assert started.exit_code == 0
    used = await _python(
        docker_service,
        tmp_path,
        "import socket, time; from pathlib import Path; import iris_sandbox_probe\n"
        "while not Path('/tmp/probe-port').exists(): time.sleep(0.01)\n"
        "s = socket.create_connection(('127.0.0.1', int(Path('/tmp/probe-port').read_text())))\n"
        "print(s.recv(100).decode()); s.close(); print(iris_sandbox_probe.VALUE)",
        session="b",
    )
    assert used.stdout.splitlines() == ["background alive", "persisted dependency"]
    await docker_service.stop(CommandScope("stop", "a")).wait_drained()
    restarted = await _python(
        docker_service,
        tmp_path,
        "import socket; from pathlib import Path; import iris_sandbox_probe\n"
        "print(iris_sandbox_probe.VALUE)\n"
        "s = socket.socket(); s.settimeout(1)\n"
        "print(s.connect_ex(('127.0.0.1', int(Path('/tmp/probe-port').read_text()))))\n"
        "s.close()",
        session="c",
    )
    assert restarted.exit_code == 0, restarted.stderr
    assert restarted.stdout.splitlines()[0] == "persisted dependency"
    assert int(restarted.stdout.splitlines()[1]) != 0


@pytest.mark.asyncio
async def test_real_large_output_and_background_pipe_return_bounded(
    docker_service: DockerCommandService, tmp_path: Path
) -> None:
    result = await _python(
        docker_service,
        tmp_path,
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "sys.stdout.buffer.write(b'x' * (2 * 1024 * 1024))\n",
    )
    assert result.status is CommandStatus.EXITED
    assert result.exit_code == 0
    assert result.output_truncated
    assert len(result.stdout) == 1024 * 1024
    assert result.duration_seconds < 10


@pytest.mark.asyncio
async def test_real_close_removes_only_owned_container_and_keeps_host_files(tmp_path: Path) -> None:
    from aiodocker import Docker
    from aiodocker.exceptions import DockerError

    service = DockerCommandService(tmp_path, DockerConfig(), workspace_writable=True)
    try:
        await service.prepare()
        result = await _python(
            service,
            tmp_path,
            "import socket; from pathlib import Path; "
            "Path('retained.txt').write_text('host'); print(socket.gethostname())",
        )
        assert result.exit_code == 0
        container_id = result.stdout.strip()
    finally:
        await service.aclose()
    endpoint = (
        "npipe:////./pipe/docker_engine"
        if sys.platform == "win32"
        else "unix:///var/run/docker.sock"
    )
    async with Docker(url=endpoint) as observer:
        with pytest.raises(DockerError) as missing:
            await observer.containers.get(container_id)
        assert missing.value.status == 404
    assert (tmp_path / "retained.txt").read_text() == "host"
