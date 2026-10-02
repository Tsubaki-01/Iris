"""Native 一次性文件 stdin 的真实字节、EOF 与启动收尾契约。"""

import asyncio
import os
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryFile
from typing import Any, BinaryIO

import pytest

from iris.command import CommandRequest, CommandScope, CommandStatus, PythonCode, native
from iris.command.native import NativeCommandService
from iris.exceptions import IrisCommandError

from .test_native import _alive, _request, _wait_file


def _stdin_request(
    tmp_path: Path, code: str, kind: str, data: bytes | None, *, timeout: float = 5
) -> CommandRequest:
    """同一真实读取程序分别经 Shell 与 Python 直接启动。"""
    if kind == "python":
        return CommandRequest("stdin", PythonCode(code), tmp_path, timeout, stdin=data)
    return replace(_request(tmp_path, code, timeout=timeout), stdin=data)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["shell", "python"])
@pytest.mark.parametrize("data", [None, b"", "中文，没有末尾换行".encode(), b"\x00\xff\r\n"])
async def test_native_stdin_delivers_exact_bytes_and_eof(
    tmp_path: Path, kind: str, data: bytes | None
) -> None:
    code = (
        "import sys\ndata = sys.stdin.buffer.read()\nprint(data.hex())\n"
        "assert sys.stdin.buffer.read() == b''\n"
    )
    service = NativeCommandService(tmp_path)
    try:
        result = await service.execute(
            CommandScope("run", "session"), _stdin_request(tmp_path, code, kind, data)
        )
    finally:
        await service.aclose()
    assert result.status is CommandStatus.EXITED
    assert result.exit_code == 0, result.stderr
    assert bytes.fromhex(result.stdout.strip()) == (data or b"")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["shell", "python"])
@pytest.mark.parametrize("spawn_fails", [False, True])
async def test_native_stdin_parent_file_closes_after_spawn_or_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, spawn_fails: bool
) -> None:
    files: list[BinaryIO] = []

    def temporary_file() -> BinaryIO:
        file = TemporaryFile()
        files.append(file)
        return file

    async def fail_spawn(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["stdin"] is files[0]
        assert not files[0].closed
        raise OSError("spawn failed")

    monkeypatch.setattr(native, "TemporaryFile", temporary_file)
    if spawn_fails:
        method = (
            "_make_subprocess_transport"
            if kind == "shell" and os.name == "nt"
            else "subprocess_exec"
        )
        monkeypatch.setattr(asyncio.get_running_loop(), method, fail_spawn)
    service = NativeCommandService(tmp_path)
    request = _stdin_request(
        tmp_path, "import sys\nassert sys.stdin.buffer.read() == b'input'", kind, b"input"
    )
    try:
        if spawn_fails:
            with pytest.raises(IrisCommandError) as caught:
                await service.execute(CommandScope("run", "session"), request)
            assert caught.value.context["started"] is False
        else:
            outcome = await service.execute(CommandScope("run", "session"), request)
            assert outcome.exit_code == 0, outcome.stderr
        assert len(files) == 1 and files[0].closed
        assert not service._calls
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_native_shell_keeps_quoted_command_and_stdin(tmp_path: Path) -> None:
    workspace = tmp_path / "quoted workspace"
    workspace.mkdir()
    request = _stdin_request(
        workspace,
        "import sys\nassert sys.stdin.buffer.read() == b'input'\nprint('quoted \\\" value')\n",
        "shell",
        b"input",
    )
    service = NativeCommandService(workspace)
    try:
        outcome = await service.execute(CommandScope("run", "session"), request)
    finally:
        await service.aclose()
    assert outcome.exit_code == 0, outcome.stderr
    assert outcome.stdout.strip() == 'quoted " value'


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["shell", "python"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_native_stdin_preserves_timeout_and_cancel_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, cancel: bool
) -> None:
    files: list[BinaryIO] = []

    def temporary_file() -> BinaryIO:
        file = TemporaryFile()
        files.append(file)
        return file

    monkeypatch.setattr(native, "TemporaryFile", temporary_file)
    marker = tmp_path / "stdin.pid"
    received = tmp_path / "received.bin"
    data = "取消前已读完输入".encode()
    code = (
        "import os, pathlib, sys, time\n"
        f"pathlib.Path({str(received)!r}).write_bytes(sys.stdin.buffer.read())\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    service = NativeCommandService(tmp_path)
    execution = asyncio.create_task(
        service.execute(
            CommandScope("run", "session"),
            _stdin_request(tmp_path, code, kind, data, timeout=5 if cancel else 0.8),
        )
    )
    try:
        pid = int(await _wait_file(marker))
        assert received.read_bytes() == data
        assert len(files) == 1 and files[0].closed
        if cancel:
            execution.cancel()
        result = await asyncio.wait_for(execution, 5)
        assert result.status is (CommandStatus.CANCELLED if cancel else CommandStatus.TIMED_OUT)
        if result.stop_receipt is not None:
            await service.wait_drained(result.stop_receipt)
        assert not _alive(pid)
        assert not service._calls
    finally:
        await service.aclose()
