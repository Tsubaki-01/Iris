"""有限工具 IO 在线程执行并收回确定结果，共享读取状态仍由 loop 更新。"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from iris.message import ToolUseBlock
from iris.tools import (
    ReadFileRecord,
    ReadFileState,
    ToolArtifact,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
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
    tool = register_file_tools(file_service=service).get(name)
    params = (
        {"file_path": "notes.txt", "content": "after"}
        if name == "write_file"
        else {"file_path": "notes.txt", "old_string": "before", "new_string": "after"}
    )
    execution = asyncio.create_task(tool.arun(tool.validate_input(params), context))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        assert writes == [writes[0]] and writes[0] != loop_thread
        assert not execution.done() and state.files == before
        state.update(other)
        execution.cancel()
        await asyncio.sleep(0)
        execution.cancel()
        await asyncio.sleep(0)
        assert not execution.done()
    finally:
        release.set()
        [result] = await asyncio.gather(execution, return_exceptions=True)

    assert not isinstance(result, BaseException)
    assert not result.is_error
    assert target.read_text(encoding="utf-8") == "after"
    assert len(state.files) == 2
    assert state.get(target).size_bytes == 5
    assert merges == [loop_thread, loop_thread]
    assert len(writes) == 1


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
