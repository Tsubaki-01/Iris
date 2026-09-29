"""有限工具 IO 在线程执行并收回确定结果，共享读取状态仍由 loop 更新。"""

from __future__ import annotations

import asyncio
import shutil
import threading
from pathlib import Path
from typing import BinaryIO

import pytest

from iris.exceptions import IrisToolExecutionError
from iris.message import ToolUseBlock
from iris.tools import (
    PublishArtifactTool,
    ReadFileRecord,
    ReadFileState,
    ToolArtifact,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    WorkspaceFileService,
    register_file_tools,
)
from iris.tools.artifacts import ToolArtifactStore


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["write_file", "edit_file"])
async def test_file_mutation_runs_in_worker_and_merges_only_its_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """取消等待不丢已完成写入，worker 不接触共享 read_state。"""
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    other = tmp_path / "other.txt"
    other.write_text("other", encoding="utf-8")
    state = ReadFileState()
    state.update(target)
    before = dict(state.files)
    service = WorkspaceFileService()
    context = ToolExecutionContext(workspace_root=tmp_path, read_state=state)
    loop_thread = threading.get_ident()
    entered = threading.Event()
    release = threading.Event()
    writes: list[int] = []
    merges: list[int] = []
    original_write = service.atomic_write
    original_merge = ReadFileState.merge
    waiting_service = WorkspaceFileService()
    waiting_context = ToolExecutionContext(
        workspace_root=tmp_path, read_state=ReadFileState(files=dict(state.files))
    )
    waiting_resolved = threading.Event()
    original_resolve = waiting_service.resolve_path

    def resolve_waiter(path: str, context: ToolExecutionContext, *, write: bool = False) -> Path:
        resolved = original_resolve(path, context, write=write)
        waiting_resolved.set()
        return resolved

    def write(path: Path, content: str) -> None:
        writes.append(threading.get_ident())
        entered.set()
        assert release.wait(2)
        assert state.files == {**before, str(other.resolve()): state.get(other)}
        original_write(path, content)

    def merge(self: ReadFileState, record: ReadFileRecord) -> None:
        if self is state:
            merges.append(threading.get_ident())
        original_merge(self, record)

    monkeypatch.setattr(service, "atomic_write", write)
    monkeypatch.setattr(ReadFileState, "merge", merge)
    monkeypatch.setattr(waiting_service, "resolve_path", resolve_waiter)
    tool = register_file_tools(file_service=service).get(name)
    params = (
        {"file_path": "notes.txt", "content": "after"}
        if name == "write_file"
        else {"file_path": "notes.txt", "old_string": "before", "new_string": "after"}
    )
    execution = asyncio.create_task(tool.arun(tool.validate_input(params), context))
    waiting: asyncio.Task[ToolResult] | None = None
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        assert writes == [writes[0]] and writes[0] != loop_thread
        assert not execution.done() and state.files == before
        waiting_tool = register_file_tools(file_service=waiting_service).get("write_file")
        waiting = asyncio.create_task(
            waiting_tool.arun(
                waiting_tool.validate_input({"file_path": "notes.txt", "content": "second"}),
                waiting_context,
            )
        )
        assert await asyncio.to_thread(waiting_resolved.wait, 1)
        state.update(other)
        execution.cancel()
        await asyncio.sleep(0)
        execution.cancel()
        await asyncio.sleep(0)
        assert not execution.done()
        assert not waiting.done()
    finally:
        release.set()
        results = await asyncio.gather(
            execution, *([waiting] if waiting is not None else []), return_exceptions=True
        )

    result, waiting_result = results
    assert not isinstance(result, BaseException)
    assert not result.is_error
    assert target.read_text(encoding="utf-8") == "after"
    assert len(state.files) == 2
    assert state.get(target).size_bytes == 5
    assert merges == [loop_thread, loop_thread]
    assert len(writes) == 1
    assert isinstance(waiting_result, IrisToolExecutionError)
    assert "STALE_FILE_STATE" in waiting_result.message
    if name == "edit_file":
        assert "+after\n" in result.data["file_change"]["patch"]


@pytest.mark.asyncio
async def test_publish_copy_runs_once_and_returns_result_after_repeated_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """发布复制在线程中执行，重复取消等待不能遗留未收回的副本写入。"""
    source = tmp_path / "report.csv"
    source.write_bytes(b"value\n42\n")
    entered, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()
    workers: list[int] = []
    copy = shutil.copyfileobj

    def blocked_copy(original: BinaryIO, output: BinaryIO, length: int) -> None:
        workers.append(threading.get_ident())
        entered.set()
        assert release.wait(2)
        copy(original, output, length=length)

    monkeypatch.setattr("iris.tools.artifacts.shutil.copyfileobj", blocked_copy)
    tool = PublishArtifactTool()
    context = ToolExecutionContext(workspace_root=tmp_path, call_id="publish")
    operation = asyncio.create_task(
        tool.arun(tool.validate_input({"file_path": "report.csv"}), context)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        operation.cancel()
        await asyncio.sleep(0)
        operation.cancel()
        await asyncio.sleep(0)
        assert not operation.done()
    finally:
        release.set()
        [result] = await asyncio.gather(operation, return_exceptions=True)
    assert not isinstance(result, BaseException)
    assert result.artifact is not None
    assert result.artifact.path.read_bytes() == b"value\n42\n"
    assert len(workers) == 1 and workers[0] != loop_thread
    assert context.read_state is None


@pytest.mark.asyncio
async def test_artifact_finalization_runs_in_worker_and_drains_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """工具返回后本地 artifact 工作不占 loop，重复取消不重放写入。"""
    loop_thread = threading.get_ident()
    entered = threading.Event()
    release = threading.Event()
    writes: list[int] = []
    persist = ToolArtifactStore._persist_text

    def blocked_persist(self: ToolArtifactStore, *args: object, **kwargs: object) -> ToolArtifact:
        writes.append(threading.get_ident())
        entered.set()
        assert release.wait(2)
        return persist(self, *args, **kwargs)

    monkeypatch.setattr(ToolArtifactStore, "_persist_text", blocked_persist)
    registry = ToolRegistry()
    tool = registry.register_function(lambda: "long" * 1000, name="large")
    tool.definition = tool.definition.model_copy(update={"max_result_chars": 400})
    execution = asyncio.create_task(
        ToolExecutor(registry).execute_one(
            ToolUseBlock(id="artifact", name="large"),
            ToolExecutionContext(workspace_root=tmp_path),
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        assert writes[0] != loop_thread
        assert not execution.done()
        execution.cancel()
        await asyncio.sleep(0)
        execution.cancel()
    finally:
        release.set()
        [result] = await asyncio.gather(execution, return_exceptions=True)
    assert not isinstance(result, BaseException)
    assert result.artifact.path.read_text(encoding="utf-8") == "long" * 1000
    assert len(writes) == 1
