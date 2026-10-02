"""Native 服务使用真实宿主进程的退出、管道与取消契约。"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from iris.command.models import (
    CommandMode,
    CommandRequest,
    CommandScope,
    CommandStatus,
    ShellCommand,
)
from iris.command.native import NativeCommandService
from iris.exceptions import IrisCommandCleanupError, IrisToolOutcomeUnknownError


def _request(cwd: Path, code: str, *, call_id: str = "call", timeout: float = 5) -> CommandRequest:
    """用脚本文件避开各宿主 shell 对内联 Python 的不同转义规则。"""
    script = cwd / f"{call_id}.py"
    script.write_text(code, encoding="utf-8")
    arguments = [sys.executable, "-u", str(script)]
    command = subprocess.list2cmdline(arguments) if os.name == "nt" else shlex.join(arguments)
    return CommandRequest(call_id, ShellCommand(command), cwd, timeout)


async def _wait_file(path: Path) -> str:
    """等待测试子进程发布已启动标记。"""
    async with asyncio.timeout(5):
        while True:
            if path.exists():
                content = path.read_text(encoding="utf-8")
                if content:
                    return content
            await asyncio.sleep(0.01)


def _alive(pid: int) -> bool:
    """只检查测试程序发布的精确 PID，不扫描宿主进程。"""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
def background_pid(tmp_path: Path) -> Iterator[Path]:
    """回收本测试命令显式创建的后台进程，即使断言失败也执行。"""
    marker = tmp_path / "background.pid"
    yield marker
    if marker.exists():
        pid = int(marker.read_text(encoding="utf-8"))
        if _alive(pid):
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    timeout=5,
                    check=False,
                )
            else:
                os.kill(pid, 9)


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [0, 7, 124, 137])
async def test_native_preserves_exit_code_output_and_unicode_cwd(
    tmp_path: Path, exit_code: int
) -> None:
    """普通 124/137 不被当作 timeout，含空格中文 cwd 原样生效。"""
    cwd = tmp_path / "中文 space"
    cwd.mkdir()
    request = _request(
        cwd,
        "import os, sys\n"
        "sys.stdout.buffer.write(('输出:' + os.getcwd()).encode('utf-8'))\n"
        "sys.stderr.buffer.write('诊断'.encode('utf-8'))\n"
        f"sys.exit({exit_code})\n",
    )
    service = NativeCommandService(tmp_path)
    try:
        await service.prepare()
        result = await service.execute(CommandScope("run", "session"), request)
    finally:
        await service.aclose()

    assert result.mode is CommandMode.NATIVE
    assert result.status is CommandStatus.EXITED
    assert result.exit_code == exit_code
    assert result.stdout == f"输出:{cwd}"
    assert result.stderr == "诊断"
    assert result.cwd == "中文 space"
    assert not result.output_truncated
    assert result.stop_receipt is None


@pytest.mark.asyncio
async def test_native_drains_output_after_limit(tmp_path: Path) -> None:
    """保留上限以外仍读取管道，避免写入较多输出的进程卡住。"""
    request = _request(
        tmp_path,
        "import sys\n"
        "sys.stdout.buffer.write(b'x' * (2 * 1024 * 1024))\n"
        "sys.stderr.buffer.write(b'y' * (2 * 1024 * 1024))\n",
    )
    service = NativeCommandService(tmp_path)
    try:
        result = await service.execute(CommandScope("run", "session"), request)
    finally:
        await service.aclose()
    assert result.status is CommandStatus.EXITED
    assert result.exit_code == 0
    assert result.output_stats.stdout_retained_bytes == 512 * 1024
    assert result.output_stats.stderr_retained_bytes == 512 * 1024
    assert result.output_stats.stdout_bytes == result.output_stats.stderr_bytes == 2 * 1024 * 1024
    assert result.output_truncated


@pytest.mark.asyncio
async def test_native_local_timeout_leaves_other_session_running(tmp_path: Path) -> None:
    """当前期限终止真实命令树，独立调用继续得到自己的结果。"""
    marker = tmp_path / "timed.pid"
    timed = _request(
        tmp_path,
        "import os, pathlib, time\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        "print('started', flush=True)\ntime.sleep(30)\n",
        call_id="timed",
        timeout=0.6,
    )
    other = _request(tmp_path, "import time\ntime.sleep(1)\nprint('other done')\n", call_id="other")
    service = NativeCommandService(tmp_path)
    try:
        first, second = await asyncio.gather(
            service.execute(CommandScope("run-a", "a"), timed),
            service.execute(CommandScope("run-b", "b"), other),
        )
    finally:
        await service.aclose()
    assert first.status is CommandStatus.TIMED_OUT
    assert first.exit_code is None
    assert first.stop_receipt is None
    assert not _alive(int(marker.read_text()))
    assert second.status is CommandStatus.EXITED
    assert second.stdout.strip() == "other done"


@pytest.mark.asyncio
async def test_native_repeated_cancel_waits_for_real_process_exit(tmp_path: Path) -> None:
    """重复取消不会打断停止和管道收尾；调用返回前确认测试前台已退出。"""
    marker = tmp_path / "cancel.pid"
    request = _request(
        tmp_path,
        "import os, pathlib, time\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        "print('started', flush=True)\ntime.sleep(30)\n",
    )
    service = NativeCommandService(tmp_path)
    task = asyncio.create_task(service.execute(CommandScope("run", "session"), request))
    try:
        pid = int(await _wait_file(marker))
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        result = await asyncio.wait_for(task, 5)
        assert result.status is CommandStatus.CANCELLED
        assert result.exit_code is None
        assert result.stop_receipt is not None
        await service.wait_drained(result.stop_receipt)
        assert not _alive(pid)
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_native_cancel_during_inflight_launch_recovers_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """模拟 spawn 已成功但句柄尚未交回，仍在取消后收回并停止真实进程。"""
    loop = asyncio.get_running_loop()
    method = "_make_subprocess_transport" if os.name == "nt" else "subprocess_exec"
    original = getattr(loop, method)
    spawned = asyncio.Event()
    release = asyncio.Event()
    marker = tmp_path / "launch.pid"

    async def delayed_launch(*args: Any, **kwargs: Any) -> Any:
        result = await original(*args, **kwargs)
        spawned.set()
        await release.wait()
        return result

    monkeypatch.setattr(loop, method, delayed_launch)
    service = NativeCommandService(tmp_path)
    task = asyncio.create_task(
        service.execute(
            CommandScope("run", "session"),
            _request(
                tmp_path,
                "import os, pathlib, time\n"
                f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
                "time.sleep(30)\n",
            ),
        )
    )
    try:
        await asyncio.wait_for(spawned.wait(), 5)
        pid = int(await _wait_file(marker))
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release.set()
        result = await asyncio.wait_for(task, 5)
        assert result.status is CommandStatus.CANCELLED
        assert not _alive(pid)
    finally:
        release.set()
        await service.aclose()


@pytest.mark.asyncio
async def test_native_foreground_exit_does_not_wait_for_background_pipe(
    tmp_path: Path, background_pid: Path
) -> None:
    """后台继承输出管道时前台仍返回真实 exit，fixture 清理自己的后台程序。"""
    request = _request(
        tmp_path,
        "import pathlib, subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"pathlib.Path({str(background_pid)!r}).write_text(str(child.pid))\n"
        "print('foreground done', flush=True)\n",
    )
    service = NativeCommandService(tmp_path)
    try:
        result = await asyncio.wait_for(service.execute(CommandScope("run", "session"), request), 4)
    finally:
        await service.aclose()
    assert result.status is CommandStatus.EXITED
    assert result.exit_code == 0
    assert result.stdout.strip() == "foreground done"
    assert result.output_truncated
    assert _alive(int(background_pid.read_text()))


@pytest.mark.asyncio
async def test_native_works_without_optional_docker_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native 的准备和执行不导入可选 Docker 驱动。"""
    monkeypatch.setitem(sys.modules, "aiodocker", None)
    service = NativeCommandService(tmp_path)
    try:
        await service.prepare()
        result = await service.execute(
            CommandScope("run", "session"), _request(tmp_path, "print('native')\n")
        )
    finally:
        await service.aclose()
    assert result.stdout.strip() == "native"


@pytest.mark.asyncio
async def test_native_unknown_retains_frontend_for_stop_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """首次停止失败不丢失真实前台句柄，随后显式 stop 能继续回收。"""
    marker = tmp_path / "retry.pid"
    service = NativeCommandService(tmp_path)
    original = service._terminate_process

    async def failed_stop(protocol: Any) -> None:
        raise IrisCommandCleanupError("模拟首次控制请求失败")

    monkeypatch.setattr(service, "_terminate_process", failed_stop)
    scope = CommandScope("run", "session")
    request = _request(
        tmp_path,
        "import os, pathlib, time\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n",
        timeout=0.5,
    )
    try:
        with pytest.raises(IrisToolOutcomeUnknownError):
            await service.execute(scope, request)
        pid = int(marker.read_text())
        assert _alive(pid)
        monkeypatch.setattr(service, "_terminate_process", original)
        await service.stop(scope).wait_drained()
        assert not _alive(pid)
    finally:
        monkeypatch.setattr(service, "_terminate_process", original)
        await service.aclose()


@pytest.mark.asyncio
async def test_native_drain_failure_keeps_physical_stop_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """物理停止后输出收尾失败，unknown 仍携带既有停止证明。"""
    marker = tmp_path / "receipt.pid"
    service = NativeCommandService(tmp_path)
    original = service._release_call
    first = True

    async def fail_first_drain(call: Any) -> None:
        nonlocal first
        if first:
            first = False
            raise IrisCommandCleanupError("模拟输出收尾失败")
        await original(call)

    monkeypatch.setattr(service, "_release_call", fail_first_drain)
    scope = CommandScope("run", "session")
    task = asyncio.create_task(
        service.execute(
            scope,
            _request(
                tmp_path,
                "import os, pathlib, time\n"
                f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
                "time.sleep(30)\n",
            ),
        )
    )
    try:
        await _wait_file(marker)
        operation = service.stop(scope)
        receipt = await operation.wait_stopped()
        with pytest.raises(IrisToolOutcomeUnknownError) as caught:
            await task
        assert caught.value.stop_receipt == receipt
        await service.wait_drained(receipt)
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_native_cancel_reports_failed_stop_before_business_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """外层取消后停止失败立即反馈 unknown，长业务期限不能阻挡 close 重试。"""
    marker = tmp_path / "cancel-failed-stop.pid"
    service = NativeCommandService(tmp_path)
    original = service._terminate_process
    stop_attempts = 0

    async def failed_stop(protocol: Any) -> None:
        nonlocal stop_attempts
        stop_attempts += 1
        raise IrisCommandCleanupError("模拟外层取消的停止控制失败")

    monkeypatch.setattr(service, "_terminate_process", failed_stop)
    task = asyncio.create_task(
        service.execute(
            CommandScope("run", "session"),
            _request(
                tmp_path,
                "import os, pathlib, time\n"
                f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
                "time.sleep(30)\n",
                timeout=120,
            ),
        )
    )
    try:
        pid = int(await _wait_file(marker))
        task.cancel()
        completed, _pending = await asyncio.wait({task}, timeout=1)
        assert task in completed, "停止已经失败，不应继续等待 120 秒业务期限"
        with pytest.raises(IrisToolOutcomeUnknownError) as caught:
            await task
        assert caught.value.stop_receipt is None
        assert stop_attempts == 1
        assert _alive(pid)
        monkeypatch.setattr(service, "_terminate_process", original)
        await service.aclose()
        assert not _alive(pid)
    finally:
        monkeypatch.setattr(service, "_terminate_process", original)
        await service.aclose()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_native_cancel_during_admission_does_not_launch_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """等待旧停止操作的调用被取消时，明确返回未启动的已知结果。"""
    marker = tmp_path / "first.pid"
    forbidden = tmp_path / "must-not-start"
    service = NativeCommandService(tmp_path)
    original = service._terminate_process
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_stop(protocol: Any) -> None:
        entered.set()
        await release.wait()
        await original(protocol)

    monkeypatch.setattr(service, "_terminate_process", blocked_stop)
    scope = CommandScope("run", "session")
    first = asyncio.create_task(
        service.execute(
            scope,
            _request(
                tmp_path,
                "import os, pathlib, time\n"
                f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
                "time.sleep(30)\n",
            ),
        )
    )
    try:
        await _wait_file(marker)
        operation = service.stop(scope)
        await entered.wait()
        waiting = asyncio.create_task(
            service.execute(
                scope,
                _request(
                    tmp_path,
                    f"import pathlib\npathlib.Path({str(forbidden)!r}).touch()\n",
                    call_id="waiting",
                ),
            )
        )
        await asyncio.sleep(0)
        waiting.cancel()
        result = await waiting
        assert result.status is CommandStatus.CANCELLED
        assert "尚未启动" in result.stderr
        assert not forbidden.exists()
        release.set()
        await first
        await operation.wait_drained()
    finally:
        release.set()
        await service.aclose()


@pytest.mark.skipif(os.name == "nt", reason="需要真实 POSIX 进程组")
@pytest.mark.asyncio
async def test_native_timeout_kills_group_after_foreground_term_exit(
    tmp_path: Path, background_pid: Path
) -> None:
    """前台响应 TERM 后，仍对同组忽略 TERM 的普通子进程执行 KILL。"""
    child_code = (
        "import os, pathlib, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(background_pid)!r}).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    request = _request(
        tmp_path,
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        "time.sleep(30)\n",
        timeout=0.6,
    )
    service = NativeCommandService(tmp_path)
    try:
        result = await asyncio.wait_for(service.execute(CommandScope("run", "session"), request), 4)
    finally:
        await service.aclose()
    assert background_pid.exists()
    assert result.status is CommandStatus.TIMED_OUT
    assert not result.output_truncated
