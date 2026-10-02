"""Native 命令前台、有限输出与 session 范围停止。"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryFile
from typing import Any, BinaryIO, cast
from uuid import uuid4

from ..exceptions import IrisCommandCleanupError, IrisCommandError, IrisToolOutcomeUnknownError
from ._output import OutputBuffer
from ._python import PYTHON_LOADER_SOURCE, write_python_source
from .models import (
    CommandMode,
    CommandOutcome,
    CommandRequest,
    CommandScope,
    CommandStatus,
    CommandStopReceipt,
    PythonCode,
    ShellCommand,
)

_DRAIN_SECONDS = 1.0
_TERM_GRACE_SECONDS = 0.5
_CONTROL_SECONDS = 10.0
logger = logging.getLogger(__name__)


class _ProcessProtocol(asyncio.SubprocessProtocol):
    """分别接收前台退出和管道 EOF，避免后台继承管道阻塞返回。"""

    def __init__(self) -> None:
        loop = asyncio.get_running_loop()
        self.output = OutputBuffer()
        self.exited: asyncio.Future[int] = loop.create_future()
        self.closed: asyncio.Future[None] = loop.create_future()
        self.pipes_closed: asyncio.Future[None] = loop.create_future()
        self.transport: asyncio.SubprocessTransport
        self._pipes = {1, 2}

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        """保留本次进程 transport。"""
        self.transport = cast(asyncio.SubprocessTransport, transport)

    def pipe_data_received(self, fd: int, data: bytes) -> None:
        """持续排空两条输出管道，只保留有限字节。"""
        self.output.append(fd, data)

    def pipe_connection_lost(self, fd: int, exc: Exception | None) -> None:
        """记录单条管道关闭；读取失败也不能无限等待 EOF。"""
        self._pipes.discard(fd)
        if exc is not None:
            self.output.mark_truncated("stream_error")
        if not self._pipes and not self.pipes_closed.done():
            self.pipes_closed.set_result(None)

    def process_exited(self) -> None:
        """只依据进程句柄记录真实前台退出码。"""
        self.exited.set_result(cast(int, self.transport.get_returncode()))

    def connection_lost(self, exc: Exception | None) -> None:
        """确认 transport 及其输出资源已经关闭。"""
        if exc is not None:
            self.output.mark_truncated("stream_error")
        self.closed.set_result(None)


@dataclass(slots=True)
class _Call:
    """当前调用持有的启动、前台与收尾状态，不记录历史后台进程。"""

    scope: CommandScope
    request: CommandRequest
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    stop_requested: asyncio.Event = field(default_factory=asyncio.Event)
    protocol: _ProcessProtocol | None = None
    stop_operation: _NativeStopOperation | None = None
    termination: asyncio.Task[None] | None = None
    interrupted: bool = False
    released: bool = False
    source_path: Path | None = None


def _observe_completion(task: asyncio.Task[Any]) -> None:
    """取回后台停止异常；显式等待者仍会收到同一个失败。"""
    if not task.cancelled():
        task.exception()


async def _launch_windows_shell(
    loop: asyncio.AbstractEventLoop,
    protocol: _ProcessProtocol,
    command: str,
    *,
    cwd: Path,
    stdin: BinaryIO | int,
) -> None:
    """禁用 cmd AutoRun，避免宿主启动脚本提前消费本次文件 stdin。"""
    executable = str(Path(os.environ["SystemRoot"]) / "System32" / "cmd.exe")
    # subprocess_shell 固定使用 /c；直接复用其 transport 层才能加入 /d，
    # 同时保留 shell 原始命令文本，避免 subprocess_exec 的 argv 二次转义。
    await loop._make_subprocess_transport(  # type: ignore[attr-defined]
        protocol=protocol,
        args=f'"{executable}" /d /s /c "{command}"',
        shell=False,
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        cwd=cwd,
        executable=executable,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


class _NativeStopOperation:
    """固定本次 session 调用集合，物理停止不等待 body 自身返回。"""

    def __init__(self, service: NativeCommandService, scope: CommandScope) -> None:
        self.service = service
        self.scope = scope
        self.calls = tuple(
            call for call in service._calls.values() if call.scope.session_id == scope.session_id
        )
        self.receipt = CommandStopReceipt(service._service_id, uuid4().hex)
        for call in self.calls:
            call.stop_operation = self
            call.stop_requested.set()
        self._stopped = asyncio.create_task(self._stop())
        self._drained = asyncio.create_task(self._drain())
        self._drained.add_done_callback(_observe_completion)

    def retry(self) -> None:
        """显式重试失败的停止，保留同一操作与已确认事实。"""
        if self._drained.done() and self._drained.exception() is not None:
            if self._stopped.exception() is not None:
                self._stopped = asyncio.create_task(self._stop())
            self._drained = asyncio.create_task(self._drain())
            self._drained.add_done_callback(_observe_completion)

    async def _stop(self) -> CommandStopReceipt:
        results = await asyncio.gather(
            *(self.service._stop_call(call) for call in self.calls), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return self.receipt

    async def _drain(self) -> CommandStopReceipt:
        await asyncio.shield(self._stopped)
        await asyncio.gather(*(call.done.wait() for call in self.calls))
        for call in self.calls:
            if not call.released:
                await self.service._release_call(call)
        self.service._stops.pop(self.scope.session_id, None)
        self.service._operations.pop(self.receipt.stop_id, None)
        return self.receipt

    async def wait_stopped(self) -> CommandStopReceipt:
        """等待已持有前台退出，不等待调用者自己的输出收尾。"""
        return await asyncio.shield(self._stopped)

    async def wait_drained(self) -> CommandStopReceipt:
        """等待本次全部 body 与输出资源收尾。"""
        return await asyncio.shield(self._drained)


class NativeCommandService:
    """运行宿主 shell，持有当前调用并支持可重试的有限清理。

    Args:
        workspace_root (Path): 用于结果 cwd 投影的已解析 root workspace。
    """

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root
        self._service_id = uuid4().hex
        self._calls: dict[tuple[str, str], _Call] = {}
        self._stops: dict[str, _NativeStopOperation] = {}
        self._operations: dict[str, _NativeStopOperation] = {}
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    async def prepare(self) -> None:
        """Native 无外部资源或可选驱动需要准备。"""
        if self._closing:
            raise IrisCommandError("Native 执行服务正在关闭")

    async def execute(self, scope: CommandScope, request: CommandRequest) -> CommandOutcome:
        """保护启动与关键收尾，外层重复取消加入同一 session 停止。"""
        if self._closing:
            raise IrisCommandError("Native 执行服务正在关闭", started=False)
        started = asyncio.get_running_loop().time()
        pending = self._stops.get(scope.session_id)
        if pending is not None:
            try:
                await pending.wait_drained()
            except asyncio.CancelledError:
                return self._outcome(
                    request,
                    CommandStatus.CANCELLED,
                    started,
                    stderr="命令尚未启动，等待停止时被取消",
                )
        if self._closing:
            raise IrisCommandError("Native 执行服务正在关闭", started=False)
        call = _Call(scope, request)
        self._calls[(scope.run_id, request.call_id)] = call
        body = asyncio.create_task(self._execute_call(call, started))
        while True:
            try:
                return await asyncio.shield(body)
            except asyncio.CancelledError:
                if body.done():
                    return body.result()
                self.stop(scope)

    def stop(self, scope: CommandScope) -> _NativeStopOperation:
        """调度或加入该 session 的停止，不影响独立 session。"""
        operation = self._stops.get(scope.session_id)
        if operation is not None:
            operation.retry()
            return operation
        operation = _NativeStopOperation(self, scope)
        self._stops[scope.session_id] = operation
        self._operations[operation.receipt.stop_id] = operation
        return operation

    async def wait_drained(self, receipt: CommandStopReceipt) -> None:
        """等待原停止操作；已完成的收据不会重新停止任何调用。"""
        operation = self._operations.get(receipt.stop_id)
        if operation is not None:
            operation.retry()
            await operation.wait_drained()

    async def aclose(self) -> None:
        """禁止新调用，并等待所有已持有前台与 transport 收尾；失败可重试。"""
        if self._closed:
            return
        self._closing = True
        if self._close_task is None or (
            self._close_task.done() and self._close_task.exception() is not None
        ):
            self._close_task = asyncio.create_task(self._close())
        while True:
            try:
                await asyncio.shield(self._close_task)
                return
            except asyncio.CancelledError:
                if self._close_task.done():
                    return self._close_task.result()

    async def _close(self) -> None:
        scopes = {call.scope.session_id: call.scope for call in self._calls.values()}
        scopes.update({key: operation.scope for key, operation in self._stops.items()})
        results = await asyncio.gather(
            *(self.stop(scope).wait_drained() for scope in scopes.values()), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        self._closed = True

    async def _execute_call(self, call: _Call, started: float) -> CommandOutcome:
        receipt = None
        try:
            try:
                if isinstance(call.request.payload, PythonCode):
                    call.source_path = await asyncio.to_thread(
                        write_python_source, call.request.payload.code
                    )
                if call.stop_operation is None:
                    call.protocol = await self._launch(call.request, call.source_path)
            except (OSError, NotImplementedError) as error:
                raise IrisCommandError(
                    "Native 程序无法准备或启动，请确认宿主支持 asyncio subprocess",
                    started=False,
                    error=str(error),
                ) from error
            finally:
                call.ready.set()
            protocol = call.protocol
            if protocol is None:
                receipt = await cast(_NativeStopOperation, call.stop_operation).wait_stopped()
                await self._release_call(call)
                return self._outcome(
                    call.request, CommandStatus.CANCELLED, started, receipt=receipt
                )
            status = CommandStatus.EXITED
            stop_requested = asyncio.create_task(call.stop_requested.wait())
            try:
                completed, _pending = await asyncio.wait(
                    (protocol.exited, stop_requested),
                    timeout=call.request.timeout_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not completed:
                    status = CommandStatus.TIMED_OUT
                    await self._terminate(call)
            finally:
                stop_requested.cancel()
                await asyncio.gather(stop_requested, return_exceptions=True)
            if call.stop_operation is not None:
                receipt = await call.stop_operation.wait_stopped()
            if call.interrupted:
                status = CommandStatus.CANCELLED
            await self._release_call(call)
            return self._outcome(
                call.request,
                status,
                started,
                exit_code=protocol.exited.result() if status is CommandStatus.EXITED else None,
                output=protocol.output,
                receipt=receipt,
            )
        except IrisCommandCleanupError as error:
            raise IrisToolOutcomeUnknownError(
                "Native 前台或必要收尾状态无法确认", stop_receipt=receipt, **error.context
            ) from error
        finally:
            await self._remove_source(call)
            call.done.set()
            if call.protocol is None:
                self._calls.pop((call.scope.run_id, call.request.call_id), None)
                call.released = True

    async def _launch(self, request: CommandRequest, source_path: Path | None) -> _ProcessProtocol:
        loop = asyncio.get_running_loop()
        protocol = _ProcessProtocol()
        input_file: BinaryIO | None = None
        try:
            if request.stdin is not None:
                input_file = TemporaryFile()
                input_file.write(request.stdin)
                input_file.seek(0)
            stdin = subprocess.DEVNULL if input_file is None else input_file
            if isinstance(request.payload, PythonCode):
                options: dict[str, Any] = (
                    {"creationflags": subprocess.CREATE_NO_WINDOW}
                    if os.name == "nt"
                    else {"start_new_session": True}
                )
                await loop.subprocess_exec(
                    lambda: protocol,
                    sys.executable,
                    "-X",
                    "utf8",
                    "-u",
                    "-c",
                    PYTHON_LOADER_SOURCE,
                    str(source_path),
                    stdin=stdin,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=request.cwd,
                    **options,
                )
            elif os.name == "nt":
                await _launch_windows_shell(
                    loop,
                    protocol,
                    request.payload.command,
                    stdin=stdin,
                    cwd=request.cwd,
                )
            else:
                await loop.subprocess_exec(
                    lambda: protocol,
                    "/bin/sh",
                    "-c",
                    cast(ShellCommand, request.payload).command,
                    stdin=stdin,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=request.cwd,
                    start_new_session=True,
                )
        finally:
            if input_file is not None:
                input_file.close()
        return protocol

    async def _remove_source(self, call: _Call) -> None:
        """清理临时源码；文件删除失败不能覆盖已知进程结果。"""
        if call.source_path is not None:
            try:
                await asyncio.to_thread(call.source_path.unlink, missing_ok=True)
            except OSError:
                logger.debug("Native 临时 Python 源码删除失败", exc_info=True)

    async def _stop_call(self, call: _Call) -> None:
        await call.ready.wait()
        if call.protocol is not None and not call.protocol.exited.done():
            call.interrupted = True
            await self._terminate(call)

    async def _terminate(self, call: _Call) -> None:
        if call.termination is None or (
            call.termination.done() and call.termination.exception() is not None
        ):
            call.termination = asyncio.create_task(
                self._terminate_process(cast(_ProcessProtocol, call.protocol))
            )
        try:
            await asyncio.shield(call.termination)
        except (OSError, TimeoutError) as error:
            raise IrisCommandCleanupError("Native 停止操作未完成", error=str(error)) from error

    async def _terminate_process(self, protocol: _ProcessProtocol) -> None:
        if protocol.exited.done():
            return
        pid = protocol.transport.get_pid()
        if os.name == "nt":
            helper = None
            try:
                helper = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/PID",
                    str(pid),
                    "/T",
                    "/F",
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                await asyncio.wait_for(helper.wait(), _CONTROL_SECONDS)
            except (OSError, TimeoutError):
                if helper is not None and helper.returncode is None:
                    helper.kill()
                    await asyncio.wait_for(helper.wait(), _CONTROL_SECONDS)
            if not protocol.exited.done():
                try:
                    protocol.transport.kill()
                except ProcessLookupError:
                    pass
        else:
            signalled_at = asyncio.get_running_loop().time()
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(asyncio.shield(protocol.exited), _TERM_GRACE_SECONDS)
            except TimeoutError:
                pass
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                pass
            else:
                # shell 先退出不代表本次进程组已空，仍给存活后代完整 grace 再尝试 KILL。
                remaining = _TERM_GRACE_SECONDS - (asyncio.get_running_loop().time() - signalled_at)
                if remaining > 0:
                    await asyncio.sleep(remaining)
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        try:
            await asyncio.wait_for(asyncio.shield(protocol.exited), _CONTROL_SECONDS)
        except TimeoutError as error:
            raise IrisCommandCleanupError("无法确认 Native 前台退出", pid=pid) from error

    async def _release_call(self, call: _Call) -> None:
        protocol = call.protocol
        if protocol is not None:
            try:
                await asyncio.wait_for(asyncio.shield(protocol.pipes_closed), _DRAIN_SECONDS)
            except TimeoutError:
                protocol.output.mark_truncated("drain_timeout")
            protocol.transport.close()
            try:
                await asyncio.wait_for(asyncio.shield(protocol.closed), _CONTROL_SECONDS)
            except TimeoutError as error:
                raise IrisCommandCleanupError("Native 输出 transport 无法关闭") from error
        call.released = True
        self._calls.pop((call.scope.run_id, call.request.call_id), None)

    def _outcome(
        self,
        request: CommandRequest,
        status: CommandStatus,
        started: float,
        *,
        exit_code: int | None = None,
        output: OutputBuffer | None = None,
        stderr: str = "",
        receipt: CommandStopReceipt | None = None,
    ) -> CommandOutcome:
        return CommandOutcome(
            mode=CommandMode.NATIVE,
            status=status,
            exit_code=exit_code,
            stdout="" if output is None else output.stdout,
            stderr=stderr if output is None else output.stderr,
            output_stats=(output or OutputBuffer()).stats,
            duration_seconds=asyncio.get_running_loop().time() - started,
            cwd=request.cwd.relative_to(self._workspace_root).as_posix(),
            stop_receipt=receipt,
        )


__all__ = ["NativeCommandService"]
