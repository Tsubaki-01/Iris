"""用本地目标和受控负载验证真实 Docker 网络及 cgroup 限额。"""

from __future__ import annotations

import errno
import json
import shlex
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

from iris.command import (
    CommandOutcome,
    CommandRequest,
    CommandScope,
    CommandStatus,
    ShellCommand,
)
from iris.command.docker import DockerCommandService
from iris.sandbox import DockerConfig


@pytest.fixture(autouse=True)
def require_real_docker(request: pytest.FixtureRequest) -> None:
    """普通测试不连接引擎，显式启用后前提不满足应报错。"""
    if not request.config.getoption("--run-docker"):
        pytest.skip("真实 Docker 需显式 --run-docker")


@pytest_asyncio.fixture
async def engine_evidence() -> None:
    """只读取本地引擎版本和 cgroup 条件，供实测证据记录。"""
    from aiodocker import Docker

    endpoint = (
        "npipe:////./pipe/docker_engine"
        if sys.platform == "win32"
        else "unix:///var/run/docker.sock"
    )
    async with Docker(url=endpoint) as observer:
        info = await observer.system.info()
        _evidence(
            "engine",
            {
                key: info.get(key)
                for key in (
                    "ServerVersion",
                    "OperatingSystem",
                    "KernelVersion",
                    "OSType",
                    "Architecture",
                    "CgroupDriver",
                    "CgroupVersion",
                    "NCPU",
                    "MemTotal",
                )
            },
        )


@asynccontextmanager
async def _service(workspace: Path, config: DockerConfig) -> AsyncIterator[DockerCommandService]:
    """只创建、关闭本次服务自己的容器，不拉取镜像。"""
    service = DockerCommandService(workspace, config, workspace_writable=True)
    try:
        await service.prepare()
        yield service
    finally:
        await service.aclose()


async def _python(service: DockerCommandService, cwd: Path, code: str) -> CommandOutcome:
    """由真实命令入口执行有限 Python 负载，保留助手的真实结果语义。"""
    result = await service.execute(
        CommandScope("limit-probe", "probe"),
        CommandRequest(uuid4().hex, ShellCommand(shlex.join(["python", "-c", code])), cwd, 15),
    )
    assert result.status is CommandStatus.EXITED, result
    assert result.exit_code == 0, result.stderr
    return result


def _evidence(case: str, facts: dict[str, object]) -> None:
    """将实际观测值交给 pytest 输出，不写入仓库或用户环境。"""
    print(json.dumps({"case": case, **facts}, sort_keys=True))


@pytest.mark.asyncio
async def test_real_none_blocks_local_target_while_bridge_connects(
    tmp_path: Path, engine_evidence: None
) -> None:
    token = uuid4().hex
    port_file = f"/tmp/iris-limit-port-{uuid4().hex}"
    server_code = (
        "import socket\nfrom pathlib import Path\n"
        "server = socket.socket()\nserver.bind(('0.0.0.0', 0))\nserver.listen()\n"
        f"Path({port_file!r}).write_text(str(server.getsockname()[1]))\n"
        "while True:\n"
        "    client, _ = server.accept()\n"
        f"    client.sendall({token.encode()!r})\n"
        "    client.close()\n"
    )
    async with (
        _service(tmp_path, DockerConfig(network="bridge")) as target,
        _service(tmp_path, DockerConfig(network="bridge")) as bridge,
        _service(tmp_path, DockerConfig(network="none")) as none,
    ):
        started = await _python(
            target,
            tmp_path,
            "import json, socket, subprocess, sys, time\nfrom pathlib import Path\n"
            f"subprocess.Popen([sys.executable, '-c', {server_code!r}], "
            "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, "
            "start_new_session=True)\n"
            f"port = Path({port_file!r})\n"
            "deadline = time.monotonic() + 5\n"
            "while not port.exists() or not port.read_text():\n"
            "    assert time.monotonic() < deadline, 'local target did not start'\n"
            "    time.sleep(0.01)\n"
            "print(json.dumps({'host': socket.gethostbyname(socket.gethostname()), "
            "'port': int(port.read_text())}))",
        )
        address = json.loads(started.stdout)
        probe = (
            "import errno, json, socket\n"
            "try:\n"
            f"    connection = socket.create_connection(({address['host']!r}, "
            f"{address['port']}), timeout=1)\n"
            "    with connection:\n"
            "        reply = connection.recv(100).decode()\n"
            "        print(json.dumps({'connected': True, 'reply': reply}))\n"
            "except OSError as error:\n"
            "    print(json.dumps({'connected': False, 'errno': error.errno, "
            "'error_name': errno.errorcode[error.errno]}))\n"
        )
        bridge_result = json.loads((await _python(bridge, tmp_path, probe)).stdout)
        none_result = json.loads((await _python(none, tmp_path, probe)).stdout)
        assert bridge_result == {"connected": True, "reply": token}
        assert not none_result["connected"]
        assert none_result["error_name"] == "ENETUNREACH"
        _evidence(
            "network",
            {
                "bridge_connected": True,
                "none_errno": none_result["errno"],
                "none_error_name": none_result["error_name"],
            },
        )


@pytest.mark.asyncio
async def test_real_half_cpu_quota_throttles_cpu_bound_work(tmp_path: Path) -> None:
    async with _service(tmp_path, DockerConfig(cpus=0.5)) as service:
        result = await _python(
            service,
            tmp_path,
            "import json, time\nfrom pathlib import Path\n"
            "def stats():\n"
            "    return {k: int(v) for k, v in (line.split() for line in "
            "Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())}\n"
            "before = stats()\nstart = time.monotonic()\nvalue = 0\n"
            "while time.monotonic() - start < 2:\n"
            "    value += 1\n"
            "elapsed = time.monotonic() - start\nafter = stats()\n"
            "print(json.dumps({'quota': Path('/sys/fs/cgroup/cpu.max').read_text().strip(), "
            "'wall_seconds': elapsed, 'iterations': value, 'nr_throttled_delta': "
            "after['nr_throttled'] - before['nr_throttled'], 'throttled_usec_delta': "
            "after['throttled_usec'] - before['throttled_usec'], 'usage_usec_delta': "
            "after['usage_usec'] - before['usage_usec']}))",
        )
        facts = json.loads(result.stdout)
        quota, period = map(int, facts["quota"].split())
        assert quota / period == 0.5
        assert facts["iterations"] > 0
        assert facts["nr_throttled_delta"] > 0
        assert facts["throttled_usec_delta"] > 0
        _evidence("cpu", facts)


@pytest.mark.asyncio
async def test_real_memory_limit_kills_only_controlled_allocator_with_known_result(
    tmp_path: Path,
) -> None:
    allocator = (
        "blocks = []\n"
        "for _ in range(64):\n"
        "    block = bytearray(8 * 1024 * 1024)\n"
        "    for offset in range(0, len(block), 4096): block[offset] = 1\n"
        "    blocks.append(block)\n"
    )
    async with _service(tmp_path, DockerConfig(memory_mb=128)) as service:
        result = await _python(
            service,
            tmp_path,
            "import json, subprocess, sys\nfrom pathlib import Path\n"
            "def events():\n"
            "    return {k: int(v) for k, v in (line.split() for line in "
            "Path('/sys/fs/cgroup/memory.events').read_text().splitlines())}\n"
            "before = events()\n"
            f"child = subprocess.run([sys.executable, '-c', {allocator!r}], timeout=10)\n"
            "after = events()\n"
            "print(json.dumps({'memory_max': int(Path('/sys/fs/cgroup/memory.max').read_text()), "
            "'child_returncode': child.returncode, 'oom_delta': after['oom'] - before['oom'], "
            "'oom_kill_delta': after['oom_kill'] - before['oom_kill']}))",
        )
        facts = json.loads(result.stdout)
        assert facts["memory_max"] == 128 * 1024 * 1024
        assert facts["child_returncode"] == -9
        assert facts["oom_delta"] > 0
        assert facts["oom_kill_delta"] > 0
        # 负载子进程退出已知，helper/父命令仍正常完成，后续调用继续使用同一环境。
        assert (
            await _python(service, tmp_path, "print('still alive')")
        ).stdout.strip() == "still alive"
        _evidence("memory", {**facts, "command_status": result.status.value, "exit_code": 0})


@pytest.mark.asyncio
async def test_real_pid_limit_rejects_spawn_and_reaps_owned_children(tmp_path: Path) -> None:
    async with _service(tmp_path, DockerConfig(pids_limit=16)) as service:
        result = await _python(
            service,
            tmp_path,
            "import json, subprocess\nfrom pathlib import Path\n"
            "def events():\n"
            "    return {k: int(v) for k, v in (line.split() for line in "
            "Path('/sys/fs/cgroup/pids.events').read_text().splitlines())}\n"
            "before = events()\nchildren = []\nfailed_errno = None\n"
            "try:\n"
            "    for _ in range(32):\n"
            "        try:\n"
            "            children.append(subprocess.Popen(['/bin/sleep', '30']))\n"
            "        except OSError as error:\n"
            "            failed_errno = error.errno\n"
            "            break\n"
            "    peak = int(Path('/sys/fs/cgroup/pids.current').read_text())\n"
            "finally:\n"
            "    for child in children: child.terminate()\n"
            "    for child in children: child.wait(timeout=5)\n"
            "after = events()\n"
            "print(json.dumps({'pids_max': int(Path('/sys/fs/cgroup/pids.max').read_text()), "
            "'spawned': len(children), 'spawn_errno': failed_errno, 'peak': peak, "
            "'after_cleanup': int(Path('/sys/fs/cgroup/pids.current').read_text()), "
            "'max_events_delta': after['max'] - before['max'], "
            "'all_reaped': all(child.returncode is not None for child in children)}))",
        )
        facts = json.loads(result.stdout)
        assert facts["pids_max"] == 16
        assert facts["spawned"] > 0
        assert facts["spawn_errno"] == errno.EAGAIN
        assert facts["peak"] == 16
        assert facts["max_events_delta"] > 0
        assert facts["all_reaped"]
        assert facts["after_cleanup"] < facts["peak"]
        _evidence("pids", facts)
