"""Native 服务级停止只影响目标 session，旧停止证明不影响后续调用。"""

import asyncio
from pathlib import Path

import pytest

from iris.exceptions import IrisExecutionError
from iris.execution.models import CommandStatus, ExecutionScope
from iris.execution.native import NativeCommandService

from .test_native import _alive, _request, _wait_file


@pytest.mark.asyncio
async def test_stop_drains_one_session_and_old_receipt_does_not_stop_restart(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "first.pid"
    scope = ExecutionScope("first", "session-a")
    service = NativeCommandService(tmp_path)
    first = asyncio.create_task(
        service.execute(
            scope,
            _request(
                tmp_path,
                "import os, pathlib, time\n"
                f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
                "time.sleep(30)\n",
                call_id="first",
            ),
        )
    )
    other = asyncio.create_task(
        service.execute(
            ExecutionScope("other", "session-b"),
            _request(tmp_path, "import time; time.sleep(1); print('other')", call_id="other"),
        )
    )
    try:
        pid = int(await _wait_file(marker))
        operation = service.stop(scope)
        assert service.stop(scope) is operation
        receipt = await asyncio.wait_for(operation.wait_stopped(), 5)
        assert not _alive(pid)
        result = await first
        assert result.status is CommandStatus.CANCELLED
        assert result.stop_receipt == receipt
        assert await operation.wait_drained() == receipt

        next_marker = tmp_path / "next.pid"
        subsequent = asyncio.create_task(
            service.execute(
                ExecutionScope("next", "session-a"),
                _request(
                    tmp_path,
                    "import os, pathlib, time\n"
                    f"pathlib.Path({str(next_marker)!r}).write_text(str(os.getpid()))\n"
                    "time.sleep(0.5); print('next')\n",
                    call_id="next",
                ),
            )
        )
        await _wait_file(next_marker)
        await service.wait_drained(receipt)
        next_result, other_result = await asyncio.gather(subsequent, other)
        assert next_result.status is CommandStatus.EXITED
        assert next_result.stdout.strip() == "next"
        assert other_result.status is CommandStatus.EXITED
        assert other_result.stdout.strip() == "other"
    finally:
        await service.aclose()
        await asyncio.gather(first, other, return_exceptions=True)


@pytest.mark.asyncio
async def test_close_stops_live_process_and_refuses_new_commands(tmp_path: Path) -> None:
    marker = tmp_path / "closing.pid"
    service = NativeCommandService(tmp_path)
    scope = ExecutionScope("run", "session")
    request = _request(
        tmp_path,
        "import os, pathlib, time\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n",
    )
    task = asyncio.create_task(service.execute(scope, request))
    try:
        pid = int(await _wait_file(marker))
        await asyncio.wait_for(service.aclose(), 5)
        assert not _alive(pid)
        assert (await task).status is CommandStatus.CANCELLED
        with pytest.raises(IrisExecutionError):
            await service.execute(scope, request)
        await service.aclose()
    finally:
        await service.aclose()
        await asyncio.gather(task, return_exceptions=True)
