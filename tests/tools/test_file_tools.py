from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from iris.exceptions import IrisToolValidationError
from iris.message import ToolUseBlock
from iris.tools import (
    DefaultPermissionPolicy,
    GrepSearchInput,
    ListFilesInput,
    ReadFileRecord,
    ReadFileState,
    ToolExecutionContext,
    ToolExecutor,
    WorkspaceFileService,
    WorkspacePolicy,
    register_file_tools,
)


class _ScandirEntries:
    """为测试提供可控顺序的 scandir context manager。"""

    def __init__(self, entries: list[os.DirEntry[str]]) -> None:
        self._entries = entries

    def __enter__(self) -> _ScandirEntries:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def __iter__(self) -> Iterator[os.DirEntry[str]]:
        if not self._entries:
            return
        yield self._entries[0]
        if len(self._entries) > 1:
            raise AssertionError("达到 limit 后仍消费下一个目录项")

    def close(self) -> None:
        """与 os.ScandirIterator 保持最小关闭接口。"""


class _FirstLineOnlyReader:
    """只允许逐行消费首行，拒绝整文件 read。"""

    def __init__(self, handle) -> None:
        self._handle = handle
        self._read_first = False

    def __enter__(self) -> _FirstLineOnlyReader:
        self._handle.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._handle.__exit__(*args)

    def __iter__(self) -> _FirstLineOnlyReader:
        return self

    def __next__(self) -> str:
        if self._read_first:
            raise AssertionError("达到全局 limit 后仍读取后续行")
        self._read_first = True
        return next(self._handle)

    def read(self, *args: object, **kwargs: object) -> str:
        del args, kwargs
        raise AssertionError("grep 不应整文件读取")

    def __getattr__(self, name: str):
        return getattr(self._handle, name)


def _file_executor(
    *,
    write_mode: Literal["confirm", "allow", "deny"] = "confirm",
) -> ToolExecutor:
    registry = register_file_tools()
    return ToolExecutor(
        registry,
        permission_policy=DefaultPermissionPolicy(write_mode=write_mode),
    )


def test_read_file_record_is_frozen_and_forbids_extra_fields(tmp_path: Path) -> None:
    """worker observation 不能被共享调用方原地改写或扩展。"""
    record = ReadFileRecord(
        path=(tmp_path / "notes.txt").resolve(),
        mtime_ns=1,
        size_bytes=2,
    )

    with pytest.raises(ValidationError):
        record.mtime_ns = 3
    with pytest.raises(ValidationError):
        ReadFileRecord.model_validate(
            {
                "path": str(record.path),
                "mtime_ns": 1,
                "size_bytes": 2,
                "unexpected": True,
            }
        )


def test_read_file_state_merge_does_not_stat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """loop 直接合并 worker observation，不重复访问 filesystem。"""
    path = (tmp_path / "notes.txt").resolve()
    record = ReadFileRecord(path=path, mtime_ns=10, size_bytes=20)
    state = ReadFileState()
    original_stat = Path.stat

    def fail_stat(
        self: Path,
        *,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        if self == path:
            raise AssertionError("merge 不应重新 stat")
        return original_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", fail_stat)

    state.merge(record)
    state.merge(record)

    assert state.files == {str(path): record}


def test_read_file_state_merge_rejects_non_absolute_observation() -> None:
    """loop merge 只接受 worker 已解析完成的绝对路径观测。"""
    state = ReadFileState()
    record = ReadFileRecord(path=Path("notes.txt"), mtime_ns=10, size_bytes=20)

    with pytest.raises(IrisToolValidationError, match="绝对路径"):
        state.merge(record)

    assert state.files == {}


@pytest.mark.asyncio
async def test_read_file_observation_runs_in_worker_and_merges_on_loop(
    tmp_path: Path,
) -> None:
    """文件读取在线程执行，共享 read_state 只由 loop continuation 修改。"""
    loop_thread_id = threading.get_ident()
    target = tmp_path / "notes.txt"
    target.write_text("content", encoding="utf-8")
    started = threading.Event()
    release = threading.Event()
    worker_thread_ids: list[int] = []

    class ObservedService(WorkspaceFileService):
        def read_file_observed(self, params, context):
            del params, context
            worker_thread_ids.append(threading.get_ident())
            started.set()
            release.wait(timeout=2)
            stat = target.stat()
            return (
                "content",
                ReadFileRecord(
                    path=target.resolve(),
                    mtime_ns=stat.st_mtime_ns,
                    size_bytes=stat.st_size,
                ),
            )

    context = ToolExecutionContext(workspace_root=tmp_path, read_state=ReadFileState())
    tool = register_file_tools(file_service=ObservedService()).get("read_file")
    execution = asyncio.create_task(tool.arun({"file_path": "notes.txt"}, context))

    try:
        assert await asyncio.to_thread(started.wait, 1)
        assert context.read_state.files == {}
        release.set()
        result = await execution
    finally:
        release.set()
        if not execution.done():
            await execution

    assert result.model_content == "content"
    assert worker_thread_ids and worker_thread_ids[0] != loop_thread_id
    assert str(target.resolve()) in context.read_state.files


def test_list_files_max_results_zero_does_not_touch_path(tmp_path: Path) -> None:
    """零结果请求在 resolve/stat/walk 前直接结束。"""

    class NoTouchService(WorkspaceFileService):
        def resolve_path(self, path, context):
            del path, context
            raise AssertionError("max_results=0 不应解析路径")

    result = NoTouchService().list_files(
        ListFilesInput(path="missing", max_results=0),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result == ""


def test_list_files_first_result_does_not_enter_remaining_subtree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """达到 limit 后不再下降到尚未访问的目录。"""
    first = tmp_path / "first.txt"
    first.write_text("first", encoding="utf-8")
    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()
    (sentinel / "late.txt").write_text("late", encoding="utf-8")
    original_scandir = os.scandir

    def ordered_scandir(path):
        resolved = Path(path).resolve()
        if resolved == sentinel.resolve():
            raise AssertionError("达到 limit 后仍进入哨兵子树")
        iterator = original_scandir(path)
        try:
            entries = list(iterator)
        finally:
            iterator.close()
        entries.sort(key=lambda entry: entry.name != first.name)
        return _ScandirEntries(entries)

    monkeypatch.setattr(os, "scandir", ordered_scandir)

    result = WorkspaceFileService().list_files(
        ListFilesInput(max_results=1),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result == "first.txt"


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        (
            "**/*.py",
            {
                "root.py",
                "schema.py",
                str(Path("sub") / "child.py"),
                str(Path("sub/deep.py")),
                str(Path("sub") / "schema.py"),
            },
        ),
        (
            "sub/**/*.py",
            {
                str(Path("sub") / "child.py"),
                str(Path("sub/deep.py")),
                str(Path("sub") / "schema.py"),
            },
        ),
        ("**/schema.py", {"schema.py", str(Path("sub") / "schema.py")}),
    ],
)
def test_list_files_preserves_recursive_glob_zero_segment_matches(
    tmp_path: Path,
    pattern: str,
    expected: set[str],
) -> None:
    """`**` 与旧 rglob 一样可匹配零个或多个目录段。"""
    (tmp_path / "root.py").write_text("root", encoding="utf-8")
    (tmp_path / "schema.py").write_text("schema", encoding="utf-8")
    subdirectory = tmp_path / "sub"
    subdirectory.mkdir()
    (subdirectory / "child.py").write_text("child", encoding="utf-8")
    (subdirectory / "deep.py").write_text("deep", encoding="utf-8")
    (subdirectory / "schema.py").write_text("schema", encoding="utf-8")

    result = WorkspaceFileService().list_files(
        ListFilesInput(pattern=pattern),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert set(result.splitlines()) == expected


def test_grep_first_match_does_not_read_remaining_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """首行达到全局 limit 后立即关闭文件。"""
    target = tmp_path / "notes.txt"
    target.write_text("needle\nlate\n", encoding="utf-8")
    original_open = Path.open

    def first_line_only_open(path: Path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path.resolve() == target.resolve():
            return _FirstLineOnlyReader(handle)
        return handle

    monkeypatch.setattr(Path, "open", first_line_only_open)

    result = WorkspaceFileService().grep_search(
        GrepSearchInput(pattern="needle", max_results=1),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result == "notes.txt:1: needle"


def test_grep_missing_root_preserves_empty_result(tmp_path: Path) -> None:
    """旧 rglob 对缺失搜索根返回空，streaming 实现保持该兼容语义。"""
    result = WorkspaceFileService().grep_search(
        GrepSearchInput(pattern="needle", path="not-created"),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result == ""


def test_grep_skips_iris_directory_before_descending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """grep 不枚举明确排除的 .iris 子树。"""
    (tmp_path / "visible.txt").write_text("ordinary\n", encoding="utf-8")
    hidden = tmp_path / ".iris"
    hidden.mkdir()
    (hidden / "secret.txt").write_text("needle\n", encoding="utf-8")
    original_scandir = os.scandir

    def guarded_scandir(path):
        if Path(path).resolve() == hidden.resolve():
            raise AssertionError("不应下降到 .iris")
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", guarded_scandir)

    result = WorkspaceFileService().grep_search(
        GrepSearchInput(pattern="needle"),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result == ""


@pytest.mark.asyncio
async def test_read_file_inside_workspace_updates_read_state(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    context = ToolExecutionContext(workspace_root=tmp_path, read_state=ReadFileState())

    result = await _file_executor().execute_one(
        ToolUseBlock(id="call_1", name="read_file", input={"file_path": "notes.txt"}),
        context,
    )

    resolved = path.resolve()
    assert result.is_error is False
    assert result.model_content == "alpha\nbeta"
    assert "1: alpha" not in result.model_content
    assert "L0001 | alpha" not in result.model_content
    assert str(resolved) in context.read_state.files


@pytest.mark.asyncio
async def test_read_file_can_include_line_numbers_when_requested(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    context = ToolExecutionContext(workspace_root=tmp_path, read_state=ReadFileState())

    result = await _file_executor().execute_one(
        ToolUseBlock(
            id="call_1",
            name="read_file",
            input={"file_path": "notes.txt", "with_line_numbers": True},
        ),
        context,
    )

    resolved = path.resolve()
    assert result.is_error is False
    assert result.model_content == "L0001 | alpha\nL0002 | beta"
    assert str(resolved) in context.read_state.files


@pytest.mark.asyncio
async def test_read_file_accepts_absolute_path_inside_workspace(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("alpha\n", encoding="utf-8")

    result = await _file_executor().execute_one(
        ToolUseBlock(id="call_1", name="read_file", input={"file_path": str(path)}),
        ToolExecutionContext(workspace_root=tmp_path, read_state=ReadFileState()),
    )

    assert result.is_error is False
    assert result.model_content == "alpha"


@pytest.mark.asyncio
async def test_file_tools_reject_parent_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    result = await _file_executor().execute_one(
        ToolUseBlock(
            id="call_1",
            name="read_file",
            input={"file_path": f"..{os.sep}{outside.name}"},
        ),
        ToolExecutionContext(workspace_root=tmp_path, read_state=ReadFileState()),
    )

    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "VALIDATION_ERROR"
    assert "PATH_OUTSIDE_WORKSPACE" in result.model_content


@pytest.mark.asyncio
async def test_file_tools_reject_symlink_escape(tmp_path: Path) -> None:
    outside_dir = tmp_path.parent / f"{tmp_path.name}_outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "secret.txt"
    outside_file.write_text("secret", encoding="utf-8")
    link = tmp_path / "link"
    try:
        link.symlink_to(outside_dir, target_is_directory=True)
    except OSError:
        pytest.skip("当前平台不支持创建符号链接")

    result = await _file_executor().execute_one(
        ToolUseBlock(
            id="call_1",
            name="read_file",
            input={"file_path": "link/secret.txt"},
        ),
        ToolExecutionContext(workspace_root=tmp_path, read_state=ReadFileState()),
    )

    assert result.is_error is True
    assert result.error is not None
    assert "PATH_OUTSIDE_WORKSPACE" in result.model_content


@pytest.mark.asyncio
async def test_write_file_refuses_existing_unread_file(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("old", encoding="utf-8")

    result = await _file_executor(write_mode="allow").execute_one(
        ToolUseBlock(
            id="call_1",
            name="write_file",
            input={"file_path": "notes.txt", "content": "new"},
        ),
        ToolExecutionContext(workspace_root=tmp_path, read_state=ReadFileState()),
    )

    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "FILE_NOT_READ"
    assert "FILE_NOT_READ" in result.model_content
    assert path.read_text(encoding="utf-8") == "old"


@pytest.mark.asyncio
async def test_write_file_reports_workspace_relative_path(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "notes.txt"

    result = await _file_executor(write_mode="allow").execute_one(
        ToolUseBlock(
            id="call_1",
            name="write_file",
            input={"file_path": str(path), "content": "new"},
        ),
        ToolExecutionContext(workspace_root=tmp_path, read_state=ReadFileState()),
    )

    assert result.is_error is False
    assert result.model_content == "WROTE: nested/notes.txt"
    assert path.read_text(encoding="utf-8") == "new"


@pytest.mark.asyncio
async def test_edit_file_reports_workspace_relative_posix_path(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "notes.txt"
    path.parent.mkdir()
    path.write_text("hello old\n", encoding="utf-8")
    context = ToolExecutionContext(workspace_root=tmp_path, read_state=ReadFileState())
    executor = _file_executor(write_mode="allow")
    await executor.execute_one(
        ToolUseBlock(id="read_1", name="read_file", input={"file_path": str(path)}),
        context,
    )

    result = await executor.execute_one(
        ToolUseBlock(
            id="edit_1",
            name="edit_file",
            input={"file_path": str(path), "old_string": "old", "new_string": "new"},
        ),
        context,
    )

    assert result.is_error is False
    assert result.model_content == "EDITED: nested/notes.txt"
    assert path.read_text(encoding="utf-8") == "hello new\n"


@pytest.mark.asyncio
async def test_read_then_edit_works_without_preseeded_read_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello old\n", encoding="utf-8")
    context = ToolExecutionContext(workspace_root=tmp_path)
    executor = _file_executor(write_mode="allow")

    await executor.execute_one(
        ToolUseBlock(id="read_1", name="read_file", input={"file_path": "notes.txt"}),
        context,
    )
    result = await executor.execute_one(
        ToolUseBlock(
            id="edit_1",
            name="edit_file",
            input={"file_path": "notes.txt", "old_string": "old", "new_string": "new"},
        ),
        context,
    )

    assert result.is_error is False
    assert path.read_text(encoding="utf-8") == "hello new\n"


@pytest.mark.asyncio
async def test_execute_many_read_then_edit_shares_read_state(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello old\n", encoding="utf-8")
    executor = _file_executor(write_mode="allow")

    results = await executor.execute_many(
        [
            ToolUseBlock(
                id="read_1",
                name="read_file",
                input={"file_path": "notes.txt"},
            ),
            ToolUseBlock(
                id="edit_1",
                name="edit_file",
                input={
                    "file_path": "notes.txt",
                    "old_string": "old",
                    "new_string": "new",
                },
            ),
        ],
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert [result.is_error for result in results] == [False, False]
    assert path.read_text(encoding="utf-8") == "hello new\n"


@pytest.mark.asyncio
async def test_edit_file_refuses_stale_read_state(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("old", encoding="utf-8")
    context = ToolExecutionContext(workspace_root=tmp_path, read_state=ReadFileState())
    executor = _file_executor(write_mode="allow")
    await executor.execute_one(
        ToolUseBlock(id="read_1", name="read_file", input={"file_path": "notes.txt"}),
        context,
    )
    path.write_text("changed", encoding="utf-8")

    result = await executor.execute_one(
        ToolUseBlock(
            id="edit_1",
            name="edit_file",
            input={"file_path": "notes.txt", "old_string": "old", "new_string": "new"},
        ),
        context,
    )

    assert result.is_error is True
    assert result.error is not None
    assert "STALE_FILE_STATE" in result.model_content


@pytest.mark.asyncio
async def test_edit_file_reports_missing_and_ambiguous_matches(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("same\nsame\n", encoding="utf-8")
    context = ToolExecutionContext(workspace_root=tmp_path, read_state=ReadFileState())
    executor = _file_executor(write_mode="allow")
    await executor.execute_one(
        ToolUseBlock(id="read_1", name="read_file", input={"file_path": "notes.txt"}),
        context,
    )

    missing = await executor.execute_one(
        ToolUseBlock(
            id="edit_1",
            name="edit_file",
            input={
                "file_path": "notes.txt",
                "old_string": "absent",
                "new_string": "new",
            },
        ),
        context,
    )
    ambiguous = await executor.execute_one(
        ToolUseBlock(
            id="edit_2",
            name="edit_file",
            input={"file_path": "notes.txt", "old_string": "same", "new_string": "new"},
        ),
        context,
    )

    assert "MATCH_NOT_FOUND" in missing.model_content
    assert "AMBIGUOUS_MATCH" in ambiguous.model_content


@pytest.mark.asyncio
async def test_successful_edit_updates_file_and_read_state(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello old\n", encoding="utf-8")
    context = ToolExecutionContext(workspace_root=tmp_path, read_state=ReadFileState())
    executor = _file_executor(write_mode="allow")
    await executor.execute_one(
        ToolUseBlock(id="read_1", name="read_file", input={"file_path": "notes.txt"}),
        context,
    )

    result = await executor.execute_one(
        ToolUseBlock(
            id="edit_1",
            name="edit_file",
            input={"file_path": "notes.txt", "old_string": "old", "new_string": "new"},
        ),
        context,
    )

    assert result.is_error is False
    assert path.read_text(encoding="utf-8") == "hello new\n"
    assert context.read_state.files[str(path.resolve())].size_bytes == path.stat().st_size


@pytest.mark.asyncio
async def test_large_grep_result_creates_artifact(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("# existing\n", encoding="utf-8")
    path = tmp_path / "log.txt"
    path.write_text("\n".join(f"needle {index}" for index in range(80)), encoding="utf-8")
    context = ToolExecutionContext(
        workspace_root=tmp_path,
        session_id="session_1",
        read_state=ReadFileState(),
    )

    result = await ToolExecutor(
        register_file_tools(max_result_chars=120),
        artifact_preview_chars=80,
    ).execute_one(
        ToolUseBlock(id="grep_1", name="grep_search", input={"pattern": "needle"}),
        context,
    )

    assert result.artifact is not None
    assert result.artifact.path.exists()
    assert result.artifact.path == tmp_path / ".iris" / "tool-results" / "session_1" / "grep_1.txt"
    assert result.artifact.size_bytes > 120
    assert ".iris/" in result.model_content
    assert "建议将 .iris/ 加入 .gitignore" in result.model_content


@pytest.mark.asyncio
async def test_artifact_store_sanitizes_session_and_tool_ids(tmp_path: Path) -> None:
    path = tmp_path / "log.txt"
    path.write_text("\n".join(f"needle {index}" for index in range(80)), encoding="utf-8")
    context = ToolExecutionContext(
        workspace_root=tmp_path,
        session_id="../escape",
        read_state=ReadFileState(),
    )

    result = await ToolExecutor(
        register_file_tools(max_result_chars=120),
        artifact_preview_chars=80,
    ).execute_one(
        ToolUseBlock(id="../owned", name="grep_search", input={"pattern": "needle"}),
        context,
    )

    assert result.artifact is not None
    artifact_root = (tmp_path / ".iris" / "tool-results").resolve()
    assert result.artifact.path.resolve().relative_to(artifact_root)
    assert not (tmp_path / ".iris" / "owned.txt").exists()


@pytest.mark.asyncio
async def test_recursive_file_tools_skip_symlinked_files_outside_workspace(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}_outside_recursive.txt"
    outside.write_text("needle secret\n", encoding="utf-8")
    link = tmp_path / "outside-link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("当前平台不支持创建符号链接")

    context = ToolExecutionContext(workspace_root=tmp_path, read_state=ReadFileState())
    executor = _file_executor()

    listed = await executor.execute_one(
        ToolUseBlock(id="list_1", name="list_files", input={}),
        context,
    )
    grepped = await executor.execute_one(
        ToolUseBlock(id="grep_1", name="grep_search", input={"pattern": "needle"}),
        context,
    )

    assert "outside-link.txt" not in listed.model_content
    assert "needle secret" not in grepped.model_content


@pytest.mark.asyncio
async def test_read_file_rejects_unbounded_limit(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("alpha\n", encoding="utf-8")

    result = await _file_executor().execute_one(
        ToolUseBlock(
            id="read_1",
            name="read_file",
            input={"file_path": "notes.txt", "limit": 10001},
        ),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.is_error is True
    assert result.error is not None
    assert result.error.code == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_grep_search_max_results_zero_returns_no_matches(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")

    result = await _file_executor().execute_one(
        ToolUseBlock(
            id="grep_1",
            name="grep_search",
            input={"pattern": "needle", "max_results": 0},
        ),
        ToolExecutionContext(workspace_root=tmp_path),
    )

    assert result.is_error is False
    assert result.model_content == ""


def test_workspace_policy_resolves_inside_paths_and_rejects_outside(
    tmp_path: Path,
) -> None:
    policy = WorkspacePolicy()

    assert policy.resolve_path("a.txt", workspace_root=tmp_path) == (tmp_path / "a.txt").resolve()

    with pytest.raises(Exception, match="PATH_OUTSIDE_WORKSPACE"):
        policy.resolve_path("../outside.txt", workspace_root=tmp_path)
