"""不同文件服务共享真实路径锁，串行检查与修改且不阻塞独立文件。"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Literal

import pytest

from iris.exceptions import IrisCancellationRequestedError, IrisToolExecutionError
from iris.message import ToolUseBlock
from iris.tools import (
    DefaultPermissionPolicy,
    ReadFileState,
    ToolExecutionContext,
    ToolExecutor,
    WorkspaceFileService,
    WriteFileInput,
    register_file_tools,
)
from iris.tools.builtin.file import EditFileInput


class _BlockingService(WorkspaceFileService):
    """在实际写入点等待，允许另一个服务确定进入同一路径。"""

    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        super().__init__()
        self.entered = entered
        self.release = release

    def atomic_write(self, path: Path, content: str) -> None:
        """保持 worker 在临界区内，直到测试允许实际替换。"""
        self.entered.set()
        assert self.release.wait(3)
        super().atomic_write(path, content)


class _AnnouncedService(WorkspaceFileService):
    """通知第二个 worker 已完成真实路径解析。"""

    def __init__(self, resolved: threading.Event) -> None:
        super().__init__()
        self.resolved = resolved

    def resolve_path(
        self, path: str, context: ToolExecutionContext, *, write: bool = False
    ) -> Path:
        """沿用真实路径边界，只额外提供可控测试时序。"""
        result = super().resolve_path(path, context, write=write)
        self.resolved.set()
        return result


def _context(root: Path, target: Path) -> ToolExecutionContext:
    """取得当前文件的独立读观测，模拟不同 session。"""
    state = ReadFileState()
    if target.exists():
        state.update(target)
    return ToolExecutionContext(workspace_root=root, read_state=state)


def _modify(
    service: WorkspaceFileService,
    name: Literal["write", "edit"],
    target: Path,
    context: ToolExecutionContext,
    content: str,
) -> str:
    """两个同步修改入口都必须经同一个路径锁。"""
    if name == "edit":
        return service.edit_file(
            EditFileInput(file_path=str(target), old_string="before", new_string=content), context
        )
    return service.write_file(WriteFileInput(file_path=str(target), content=content), context)


@pytest.mark.parametrize(
    "first,second", [("write", "write"), ("write", "edit"), ("edit", "write"), ("edit", "edit")]
)
def test_distinct_services_recheck_freshness_after_same_path_wait(
    tmp_path: Path, first: Literal["write", "edit"], second: Literal["write", "edit"]
) -> None:
    """两个 v1 观测竞争时，后进入者读取锁内的 v2 状态并拒绝覆盖。"""
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    context_a, context_b = _context(tmp_path, target), _context(tmp_path, target)
    entered, release, resolved = threading.Event(), threading.Event(), threading.Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(
            _modify,
            _BlockingService(entered, release),
            first,
            target,
            context_a,
            "first-completed-content",
        )
        try:
            assert entered.wait(1)
            future_b = pool.submit(
                _modify, _AnnouncedService(resolved), second, target, context_b, "second"
            )
            assert resolved.wait(1)
            assert not wait([future_b], timeout=0.1).done
        finally:
            release.set()
        assert future_a.result().startswith(("WROTE:", "EDITED:"))
        with pytest.raises(IrisToolExecutionError, match="STALE_FILE_STATE"):
            future_b.result()
    assert target.read_text(encoding="utf-8") == "first-completed-content"
    assert context_b.read_state.get(target).size_bytes == len("before")


def test_new_file_existence_is_checked_after_waiting_for_other_writer(tmp_path: Path) -> None:
    """等待期间被另一服务创建的目标，不能按等待前的不存在状态覆盖。"""
    target = tmp_path / "created.txt"
    context_a, context_b = _context(tmp_path, target), _context(tmp_path, target)
    entered, release, resolved = threading.Event(), threading.Event(), threading.Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(
            _modify, _BlockingService(entered, release), "write", target, context_a, "first"
        )
        try:
            assert entered.wait(1)
            future_b = pool.submit(
                _modify, _AnnouncedService(resolved), "write", target, context_b, "second"
            )
            assert resolved.wait(1)
            assert not wait([future_b], timeout=0.1).done
        finally:
            release.set()
        future_a.result()
        with pytest.raises(IrisToolExecutionError, match="FILE_NOT_READ"):
            future_b.result()
    assert target.read_text(encoding="utf-8") == "first"


def test_different_paths_enter_io_independently(tmp_path: Path) -> None:
    """全局表 mutex 不覆盖真实文件 I/O，不同路径可以同时持锁。"""
    entered_a, entered_b, release = threading.Event(), threading.Event(), threading.Event()
    target_a, target_b = tmp_path / "a.txt", tmp_path / "b.txt"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                _modify,
                _BlockingService(event, release),
                "write",
                target,
                _context(tmp_path, target),
                "new",
            )
            for event, target in ((entered_a, target_a), (entered_b, target_b))
        ]
        try:
            assert entered_a.wait(1) and entered_b.wait(1)
        finally:
            release.set()
        assert all(future.result().startswith("WROTE:") for future in futures)


def test_failed_edit_releases_path_for_next_valid_edit(tmp_path: Path) -> None:
    """检查异常不能永久占据路径锁或阻止下一次合法修改。"""
    target = tmp_path / "a.txt"
    target.write_text("before", encoding="utf-8")
    context = _context(tmp_path, target)
    service = WorkspaceFileService()
    with pytest.raises(IrisToolExecutionError, match="MATCH_NOT_FOUND"):
        service.edit_file(
            EditFileInput(file_path="a.txt", old_string="absent", new_string="x"), context
        )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_modify, WorkspaceFileService(), "edit", target, context, "after")
        assert future.result(timeout=1) == "EDITED: a.txt"


class _Cancellation:
    """由测试在 worker 等待路径锁时请求业务取消。"""

    requested = False

    def raise_if_requested(self) -> None:
        """复用工具外层的业务取消检查。"""
        if self.requested:
            raise IrisCancellationRequestedError("cancelled")


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["write_file", "edit_file"])
async def test_waiting_mutation_cancels_without_writing_or_unknown(
    tmp_path: Path, name: str
) -> None:
    """低层锁等待者交回取消，真实文件与读取状态证明第二次修改未发生。"""
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    context_a, context_b = _context(tmp_path, target), _context(tmp_path, target)
    cancellation = _Cancellation()
    context_b.cancellation = cancellation
    entered, release, resolved = threading.Event(), threading.Event(), threading.Event()
    task_a = asyncio.create_task(
        asyncio.to_thread(
            _modify,
            _BlockingService(entered, release),
            "write",
            target,
            context_a,
            "first-completed-content",
        )
    )
    executor = ToolExecutor(
        register_file_tools(file_service=_AnnouncedService(resolved)),
        permission_policy=DefaultPermissionPolicy(write_mode="allow"),
    )
    arguments = {
        "file_path": "notes.txt",
        **(
            {"content": "second"}
            if name == "write_file"
            else {"old_string": "before", "new_string": "second"}
        ),
    }
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        task_b = asyncio.create_task(
            executor.execute_one(ToolUseBlock(id="waiting", name=name, input=arguments), context_b)
        )
        assert await asyncio.to_thread(resolved.wait, 1)
        cancellation.requested = True
    finally:
        release.set()
    await task_a
    with pytest.raises(IrisCancellationRequestedError):
        await task_b
    assert target.read_text(encoding="utf-8") == "first-completed-content"
    assert context_b.read_state.get(target).size_bytes == len("before")
