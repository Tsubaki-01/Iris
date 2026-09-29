"""共享 Docker 沙箱中的受控命令调用与停止结算。"""

from __future__ import annotations

import asyncio
import io
import logging
import tarfile
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..exceptions import (
    IrisCommandCleanupError,
    IrisCommandError,
    IrisSandboxError,
    IrisToolOutcomeUnknownError,
)
from ..sandbox import DockerConfig
from ..sandbox.docker import CONTROL_SECONDS, DockerSandbox
from ._output import OutputBuffer
from ._python import PYTHON_LOADER_SOURCE
from .models import (
    CommandMode,
    CommandOutcome,
    CommandRequest,
    CommandScope,
    CommandStatus,
    CommandStopReceipt,
    PythonCode,
)

if TYPE_CHECKING:
    from aiodocker.containers import DockerContainer
    from aiodocker.execs import Exec
    from aiodocker.stream import Stream

_DRAIN_SECONDS = 1.0
_POLL_SECONDS = 0.05
_HELPER_GRACE_SECONDS = 1.0
_READ_RESULT = "import pathlib,sys;sys.stdout.write(pathlib.Path(sys.argv[1]).read_text())"
_DELETE_RESULT = "import pathlib,sys;pathlib.Path(sys.argv[1]).unlink(missing_ok=True)"
logger = logging.getLogger(__name__)


class _CommandResult(BaseModel):
    """在临时 IPC 首次返回宿主时验证一次真实命令结果。"""

    model_config = ConfigDict(extra="forbid")

    reason: Literal["exited", "timed_out"]
    returncode: int = Field(strict=True)


@dataclass(slots=True)
class _Call:
    """本服务当前持有的一个调用及其有限收尾。"""

    scope: CommandScope
    request: CommandRequest
    output: OutputBuffer = field(default_factory=OutputBuffer)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    stop_operation: _DockerStopOperation | None = None
    stream: Stream | None = None
    reader: asyncio.Task[None] | None = None
    admitted: bool = False
    dispatched: bool = False
    cancelled: bool = False
    released: bool = False
    result: _CommandResult | None = None
    source_path: str | None = None


def _observe_completion(task: asyncio.Task[Any]) -> None:
    """停止由服务拥有，即使调用方退出也收取其完成异常。"""
    if not task.cancelled():
        task.exception()


class _DockerStopOperation:
    """固定本轮调用集合；物理停止不等待调用 body 自己结束。"""

    def __init__(self, service: DockerCommandService, scope: CommandScope) -> None:
        self.service = service
        self.receipt = CommandStopReceipt(service._service_id, uuid4().hex)
        self.calls = tuple(service._calls.values())
        for call in self.calls:
            call.stop_operation = self
        self.mark_cancelled(scope)
        self._stopped = asyncio.create_task(self._stop())
        self._drained = asyncio.create_task(self._drain())
        self._drained.add_done_callback(_observe_completion)

    def mark_cancelled(self, scope: CommandScope) -> None:
        """同 session 发起停止的调用与其他 session 的连带中断分开。"""
        for call in self.calls:
            if call.scope.session_id == scope.session_id:
                call.cancelled = True

    def retry(self) -> None:
        """显式清理重试复用原收据和固定调用集合。"""
        if self._drained.done() and self._drained.exception() is not None:
            if self._stopped.done() and self._stopped.exception() is not None:
                self._stopped = asyncio.create_task(self._stop())
            self._drained = asyncio.create_task(self._drain())
            self._drained.add_done_callback(_observe_completion)

    async def _stop(self) -> CommandStopReceipt:
        await self.service._physical_stop()
        return self.receipt

    async def _drain(self) -> CommandStopReceipt:
        await asyncio.shield(self._stopped)
        await asyncio.gather(*(call.done.wait() for call in self.calls))
        for call in self.calls:
            if not call.released:
                await self.service._release_call(call)
        self.service._stop_operation = None
        self.service._operations.pop(self.receipt.stop_id, None)
        return self.receipt

    async def wait_stopped(self) -> CommandStopReceipt:
        """仅等待共享容器已停止，避免 body 等待自身排空。"""
        return await asyncio.shield(self._stopped)

    async def wait_drained(self) -> CommandStopReceipt:
        """等待本轮物理停止和旧调用收尾后再开放准入。"""
        return await asyncio.shield(self._drained)


class DockerCommandService:
    """一个 live root 的惰性共享容器，不连接远程 daemon 或回退宿主。

    Args:
        workspace_root (Path): 已解析的 root 挂载目录。
        config (DockerConfig): root 唯一拥有的已解析 Docker 配置。
        workspace_writable (bool): root 工作区挂载是否可写。
    """

    def __init__(
        self, workspace_root: Path, config: DockerConfig, *, workspace_writable: bool
    ) -> None:
        self._workspace_root = workspace_root
        self._service_id = uuid4().hex
        self._sandbox = DockerSandbox(
            workspace_root, config, workspace_writable=workspace_writable, owner_id=self._service_id
        )
        self._prepare_task: asyncio.Task[None] | None = None
        self._control_lock = asyncio.Lock()
        self._calls: dict[tuple[str, str], _Call] = {}
        self._stop_operation: _DockerStopOperation | None = None
        self._operations: dict[str, _DockerStopOperation] = {}
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._helper_source = (
            files("iris.command").joinpath("_container_helper.py").read_text(encoding="utf-8")
        )

    async def prepare(self) -> None:
        """只准备一次 driver、Linux engine 与现有镜像，不创建容器。"""
        if self._closing:
            raise IrisCommandError("Docker 执行服务正在关闭", started=False)
        if self._prepare_task is None:
            self._prepare_task = asyncio.create_task(self._prepare())
            self._prepare_task.add_done_callback(_observe_completion)
        await asyncio.shield(self._prepare_task)

    async def _prepare(self) -> None:
        try:
            await self._sandbox.prepare()
        except IrisSandboxError as error:
            raise IrisCommandError(error.message, started=False, **error.context) from error

    async def execute(self, scope: CommandScope, request: CommandRequest) -> CommandOutcome:
        """并发运行命令，外层取消只登记一次共享停止并保护必要收尾。"""
        started = asyncio.get_running_loop().time()
        await self.prepare()
        if self._closing:
            raise IrisCommandError("Docker 执行服务正在关闭", started=False)
        call = _Call(scope, request)
        body = asyncio.create_task(self._execute_call(call, started))
        while True:
            try:
                return await asyncio.shield(body)
            except asyncio.CancelledError:
                if body.done():
                    return body.result()
                if call.admitted:
                    self.stop(scope)
                else:
                    # 纯排队 body 尚无控制 I/O，可以取消；已准入的 body 必须收回启动事实。
                    body.cancel()

    def stop(self, scope: CommandScope) -> _DockerStopOperation:
        """同步关闭命令准入；独立 session 加入同一个物理停止。"""
        operation = self._stop_operation
        if operation is not None:
            operation.mark_cancelled(scope)
            operation.retry()
            return operation
        operation = _DockerStopOperation(self, scope)
        self._stop_operation = operation
        self._operations[operation.receipt.stop_id] = operation
        return operation

    async def wait_drained(self, receipt: CommandStopReceipt) -> None:
        """消费原停止操作；已完成的旧收据不会再次停止当前容器。"""
        operation = self._operations.get(receipt.stop_id)
        if operation is not None:
            operation.retry()
            await operation.wait_drained()

    async def aclose(self) -> None:
        """停止、删除本实例唯一容器并关闭 client；清理失败可重试。"""
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
        if self._prepare_task is not None:
            await asyncio.gather(asyncio.shield(self._prepare_task), return_exceptions=True)
        await self.stop(CommandScope("", "")).wait_drained()
        try:
            await self._sandbox.aclose()
        except IrisSandboxError as error:
            raise IrisCommandCleanupError(error.message, **error.context) from error
        self._closed = True

    async def _launch(self, call: _Call, result_path: str) -> Exec | None:
        while True:
            while (pending := self._stop_operation) is not None:
                await pending.wait_drained()
            await self._control_lock.acquire()
            if self._stop_operation is None:
                break
            self._control_lock.release()
        try:
            if self._closing:
                raise IrisCommandError("Docker 执行服务正在关闭", started=False)
            call.admitted = True
            self._calls[(call.scope.run_id, call.request.call_id)] = call
            await self._sandbox.create()
            if call.stop_operation is not None:
                return None
            await self._sandbox.start()
            if call.stop_operation is not None:
                return None
            container = cast("DockerContainer", self._sandbox.container)
            payload = call.request.payload
            if isinstance(payload, PythonCode):
                name = f"iris-python-{uuid4().hex}.py"
                call.source_path = f"/tmp/{name}"
                source = payload.code.encode("utf-8")
                archive_bytes = io.BytesIO()
                with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
                    entry = tarfile.TarInfo(name)
                    entry.size = len(source)
                    entry.mode = 0o444
                    entry.uid, entry.gid = (int(value) for value in self._sandbox.user.split(":"))
                    archive.addfile(entry, io.BytesIO(source))
                async with asyncio.timeout(CONTROL_SECONDS):
                    await container.put_archive("/tmp", archive_bytes.getvalue())
                if call.stop_operation is not None:
                    return None
                kind, value = "python", call.source_path
            else:
                kind, value = "shell", payload.command
            cwd = "/workspace"
            relative = call.request.cwd.relative_to(self._workspace_root).as_posix()
            if relative != ".":
                cwd += f"/{relative}"
            async with asyncio.timeout(CONTROL_SECONDS):
                execution = await container.exec(
                    [
                        "python",
                        "-c",
                        self._helper_source,
                        PYTHON_LOADER_SOURCE,
                        kind,
                        value,
                        str(call.request.timeout_seconds),
                        result_path,
                    ],
                    stdin=False,
                    tty=False,
                    workdir=cwd,
                    user=self._sandbox.user,
                )
                if call.stop_operation is not None:
                    return None
                call.stream = execution.start(detach=False, timeout=self._sandbox.control_timeout)
                call.dispatched = True
                await call.stream.__aenter__()
            call.reader = asyncio.create_task(self._consume(call.stream, call.output))
            return execution
        finally:
            self._control_lock.release()

    async def _execute_call(self, call: _Call, started: float) -> CommandOutcome:
        receipt = None
        try:
            result_path = f"/tmp/iris-command-{uuid4().hex}.json"
            execution = await self._launch(call, result_path)
            if execution is not None and call.stop_operation is None:
                supervision = (
                    call.request.timeout_seconds + _HELPER_GRACE_SECONDS + 2 * CONTROL_SECONDS
                )
                async with asyncio.timeout(supervision):
                    await self._monitor(execution, call)
                if call.stop_operation is None:
                    await self._drain_output(call)
                    payload = await self._control_exec(_READ_RESULT, result_path)
                    if payload is not None:
                        call.result = _CommandResult.model_validate_json(payload)
                        deletion_errors: tuple[type[Exception], ...] = (
                            *self._sandbox.driver_errors,
                            IrisCommandError,
                        )
                        try:
                            await self._control_exec(_DELETE_RESULT, result_path)
                        except deletion_errors:
                            logger.debug("已知命令结果的临时文件删除失败", exc_info=True)
            if call.stop_operation is not None:
                receipt = await call.stop_operation.wait_stopped()
            await self._release_call(call)
            return self._call_outcome(call, started, receipt)
        except asyncio.CancelledError:
            # execute 只取消尚未准入的 body；此时没有用户 exec 或待收回控制操作。
            return self._outcome(call.request, CommandStatus.CANCELLED, started)
        except (
            *self._sandbox.driver_errors,
            ValidationError,
            IrisCommandError,
            IrisSandboxError,
        ) as error:
            if not call.admitted:
                raise
            owns_stop = call.stop_operation is None
            operation = call.stop_operation or self.stop(call.scope)
            cleanup_error = None
            try:
                receipt = await operation.wait_stopped()
            except IrisCommandCleanupError as failure:
                if owns_stop:
                    cleanup_error = failure
            try:
                await self._release_call(call)
            except IrisCommandCleanupError as failure:
                cleanup_error = failure
            if cleanup_error is not None and (call.result is not None or not call.dispatched):
                known = (
                    self._call_outcome(call, started, receipt) if call.result is not None else None
                )
                details = dict(cleanup_error.context)
                if not call.dispatched:
                    details["started"] = False
                raise IrisCommandCleanupError(
                    cleanup_error.message, command_outcome=known, **details
                ) from cleanup_error
            if call.result is not None:
                return self._call_outcome(call, started, receipt)
            if call.dispatched:
                raise IrisToolOutcomeUnknownError(
                    "Docker 命令执行结果无法确认", stop_receipt=receipt, error=str(error)
                ) from error
            raise IrisCommandError(
                "Docker 命令尚未启动，环境控制失败", started=False, error=str(error)
            ) from error
        finally:
            call.done.set()

    async def _monitor(self, execution: Exec, call: _Call) -> None:
        while call.stop_operation is None:
            reader = cast("asyncio.Task[None]", call.reader)
            if reader.done():
                reader.result()
            async with asyncio.timeout(CONTROL_SECONDS):
                state = await execution.inspect()
            if not state["Running"]:
                return
            await asyncio.sleep(_POLL_SECONDS)

    async def _consume(self, stream: Stream, output: OutputBuffer) -> None:
        try:
            while (message := await stream.read_out()) is not None:
                output.append(message.stream, message.data)
        except self._sandbox.driver_errors:
            output.mark_truncated("stream_error")
            raise

    async def _drain_output(self, call: _Call) -> None:
        if call.reader is not None:
            try:
                await asyncio.wait_for(asyncio.shield(call.reader), _DRAIN_SECONDS)
            except TimeoutError:
                call.output.mark_truncated("drain_timeout")
                call.reader.cancel()
                await asyncio.gather(call.reader, return_exceptions=True)

    async def _control_exec(self, source: str, path: str) -> str | None:
        async with self._control_lock:
            if self._stop_operation is not None:
                return None
            stream = None
            try:
                async with asyncio.timeout(CONTROL_SECONDS):
                    execution = await cast("DockerContainer", self._sandbox.container).exec(
                        ["python", "-c", source, path],
                        stdin=False,
                        tty=False,
                        user=self._sandbox.user,
                    )
                    if self._stop_operation is not None:
                        return None
                    stream = execution.start(detach=False, timeout=self._sandbox.control_timeout)
                    await stream.__aenter__()
                    output = OutputBuffer()
                    await self._consume(stream, output)
                    state = await execution.inspect()
                    while state["Running"]:
                        await asyncio.sleep(_POLL_SECONDS)
                        state = await execution.inspect()
                    if state["ExitCode"] != 0:
                        raise IrisCommandError("Docker 临时结果控制失败", error=output.stderr)
                    return output.stdout
            finally:
                if stream is not None:
                    async with asyncio.timeout(CONTROL_SECONDS):
                        await stream.close()

    async def _release_call(self, call: _Call) -> None:
        if call.reader is not None:
            if not call.reader.done():
                call.output.mark_truncated("stream_closed")
                call.reader.cancel()
            await asyncio.gather(call.reader, return_exceptions=True)
        if call.stream is not None:
            try:
                async with asyncio.timeout(CONTROL_SECONDS):
                    await call.stream.close()
            except self._sandbox.driver_errors as error:
                raise IrisCommandCleanupError("Docker 输出流关闭失败", error=str(error)) from error
        call.released = True
        self._calls.pop((call.scope.run_id, call.request.call_id), None)

    async def _physical_stop(self) -> None:
        async with self._control_lock:
            try:
                await self._sandbox.stop()
            except IrisSandboxError as error:
                raise IrisCommandCleanupError(error.message, **error.context) from error

    def _call_outcome(
        self, call: _Call, started: float, receipt: CommandStopReceipt | None
    ) -> CommandOutcome:
        if call.result is not None:
            status = (
                CommandStatus.EXITED if call.result.reason == "exited" else CommandStatus.TIMED_OUT
            )
            exit_code = call.result.returncode if status is CommandStatus.EXITED else None
        else:
            status = (
                CommandStatus.CANCELLED if call.cancelled else CommandStatus.ENVIRONMENT_INTERRUPTED
            )
            exit_code = None
        return self._outcome(
            call.request, status, started, output=call.output, exit_code=exit_code, receipt=receipt
        )

    def _outcome(
        self,
        request: CommandRequest,
        status: CommandStatus,
        started: float,
        *,
        output: OutputBuffer | None = None,
        exit_code: int | None = None,
        receipt: CommandStopReceipt | None = None,
    ) -> CommandOutcome:
        return CommandOutcome(
            mode=CommandMode.DOCKER,
            status=status,
            exit_code=exit_code,
            stdout="" if output is None else output.stdout,
            stderr="" if output is None else output.stderr,
            output_stats=(output or OutputBuffer()).stats,
            duration_seconds=asyncio.get_running_loop().time() - started,
            cwd=request.cwd.relative_to(self._workspace_root).as_posix(),
            stop_receipt=receipt,
        )


__all__ = ["DockerCommandService"]
