"""Python 载荷复用 Native 命令资源和停止边界。"""

import asyncio
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from iris.command import CommandRequest, CommandScope, CommandStatus, PythonCode, native
from iris.command.native import NativeCommandService
from iris.exceptions import IrisCommandError


@pytest.mark.asyncio
async def test_python_direct_launch_large_source_unicode_and_project_cwd(tmp_path: Path) -> None:
    (tmp_path / "project_module.py").write_text("VALUE = 42", encoding="utf-8")
    code = (
        "# padding " + "x" * 100_000 + "\n"
        "import project_module, pathlib, sys\n"
        f"assert sys.executable == {sys.executable!r}\n"
        "pathlib.Path('report.txt').write_text('中文', encoding='utf-8')\n"
        "print(project_module.VALUE, 'quoted \\\" value')\n"
    )
    service = NativeCommandService(tmp_path)
    try:
        outcome = await service.execute(
            CommandScope("r", "s"), CommandRequest("py", PythonCode(code), tmp_path, 5)
        )
        assert outcome.exit_code == 0, outcome.stderr
        assert "42" in outcome.stdout
        assert (tmp_path / "report.txt").read_text(encoding="utf-8") == "中文"
    finally:
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("code,exit_code", [("raise SystemExit(124)", 124), ("1 / 0", 1)])
async def test_python_preserves_real_exit(tmp_path: Path, code: str, exit_code: int) -> None:
    service = NativeCommandService(tmp_path)
    try:
        result = await service.execute(
            CommandScope("r", "s"), CommandRequest("py", PythonCode(code), tmp_path, 5)
        )
        assert result.status is CommandStatus.EXITED and result.exit_code == exit_code
        if exit_code == 1:
            assert "1 / 0" in result.stderr and "<iris-python>" in result.stderr
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_cancel_during_source_preparation_does_not_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = threading.Event(), threading.Event()
    paths: list[Path] = []
    original = native.write_python_source

    def prepare(code: str) -> Path:
        path = original(code)
        paths.append(path)
        entered.set()
        assert release.wait(3)
        return path

    async def no_spawn(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("取消后的源码准备不应启动进程")

    monkeypatch.setattr(native, "write_python_source", prepare)
    monkeypatch.setattr(asyncio.get_running_loop(), "subprocess_exec", no_spawn)
    service = NativeCommandService(tmp_path)
    task = asyncio.create_task(
        service.execute(
            CommandScope("r", "s"), CommandRequest("py", PythonCode("print('no')"), tmp_path, 5)
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        result = await asyncio.wait_for(task, 3)
        assert result.status is CommandStatus.CANCELLED
        assert result.stop_receipt is not None
        assert paths and not paths[0].exists()
    finally:
        release.set()
        await service.aclose()


@pytest.mark.asyncio
async def test_python_timeout_keeps_other_session_running(tmp_path: Path) -> None:
    service = NativeCommandService(tmp_path)
    try:
        short, other = await asyncio.gather(
            service.execute(
                CommandScope("a", "a"),
                CommandRequest("short", PythonCode("import time; time.sleep(10)"), tmp_path, 0.2),
            ),
            service.execute(
                CommandScope("b", "b"),
                CommandRequest(
                    "other", PythonCode("import time; time.sleep(0.5); print('kept')"), tmp_path, 3
                ),
            ),
        )
        assert short.status is CommandStatus.TIMED_OUT
        assert other.status is CommandStatus.EXITED and other.stdout.strip() == "kept"
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_source_preparation_failure_is_known_not_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_prepare(code: str) -> Path:
        raise OSError("cannot create temporary source")

    monkeypatch.setattr(native, "write_python_source", fail_prepare)
    service = NativeCommandService(tmp_path)
    try:
        with pytest.raises(IrisCommandError) as caught:
            await service.execute(
                CommandScope("r", "s"),
                CommandRequest("py", PythonCode("print(1)"), tmp_path, 5),
            )
        assert caught.value.context["started"] is False
        assert not service._calls
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_source_delete_failure_preserves_known_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths: list[Path] = []
    prepare = native.write_python_source
    unlink = Path.unlink

    def capture_source(code: str) -> Path:
        path = prepare(code)
        paths.append(path)
        return path

    def fail_source_delete(path: Path, missing_ok: bool = False) -> None:
        if path in paths:
            raise OSError("source deletion failed")
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(native, "write_python_source", capture_source)
    monkeypatch.setattr(Path, "unlink", fail_source_delete)
    service = NativeCommandService(tmp_path)
    try:
        result = await service.execute(
            CommandScope("r", "s"),
            CommandRequest("py", PythonCode("raise SystemExit(7)"), tmp_path, 5),
        )
        assert result.status is CommandStatus.EXITED and result.exit_code == 7
        assert paths[0].exists()
    finally:
        await service.aclose()
        for path in paths:
            unlink(path, missing_ok=True)
