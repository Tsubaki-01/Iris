from __future__ import annotations

import os
from pathlib import Path

import pytest

from iris.message import ToolUseBlock
from iris.tools import (
    DefaultPermissionPolicy,
    ReadFileState,
    ToolExecutionContext,
    ToolExecutor,
    WorkspacePolicy,
    register_file_tools,
)


def _file_executor(*, allow_writes: bool = False) -> ToolExecutor:
    registry = register_file_tools()
    return ToolExecutor(
        registry,
        permission_policy=DefaultPermissionPolicy(allow_writes=allow_writes),
    )


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
    assert "1: alpha" in result.model_content()
    assert "2: beta" in result.model_content()
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
    assert "1: alpha" in result.model_content()


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
    assert "PATH_OUTSIDE_WORKSPACE" in result.model_content()


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
    assert "PATH_OUTSIDE_WORKSPACE" in result.model_content()


@pytest.mark.asyncio
async def test_write_file_refuses_existing_unread_file(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("old", encoding="utf-8")

    result = await _file_executor(allow_writes=True).execute_one(
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
    assert "FILE_NOT_READ" in result.model_content()
    assert path.read_text(encoding="utf-8") == "old"


@pytest.mark.asyncio
async def test_read_then_edit_works_without_preseeded_read_state(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello old\n", encoding="utf-8")
    context = ToolExecutionContext(workspace_root=tmp_path)
    executor = _file_executor(allow_writes=True)

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
async def test_edit_file_refuses_stale_read_state(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("old", encoding="utf-8")
    context = ToolExecutionContext(workspace_root=tmp_path, read_state=ReadFileState())
    executor = _file_executor(allow_writes=True)
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
    assert "STALE_FILE_STATE" in result.model_content()


@pytest.mark.asyncio
async def test_edit_file_reports_missing_and_ambiguous_matches(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("same\nsame\n", encoding="utf-8")
    context = ToolExecutionContext(workspace_root=tmp_path, read_state=ReadFileState())
    executor = _file_executor(allow_writes=True)
    await executor.execute_one(
        ToolUseBlock(id="read_1", name="read_file", input={"file_path": "notes.txt"}),
        context,
    )

    missing = await executor.execute_one(
        ToolUseBlock(
            id="edit_1",
            name="edit_file",
            input={"file_path": "notes.txt", "old_string": "absent", "new_string": "new"},
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

    assert "MATCH_NOT_FOUND" in missing.model_content()
    assert "AMBIGUOUS_MATCH" in ambiguous.model_content()


@pytest.mark.asyncio
async def test_successful_edit_updates_file_and_read_state(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello old\n", encoding="utf-8")
    context = ToolExecutionContext(workspace_root=tmp_path, read_state=ReadFileState())
    executor = _file_executor(allow_writes=True)
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
    assert ".iris/" in result.model_content()
    assert "建议将 .iris/ 加入 .gitignore" in result.model_content()


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

    assert "outside-link.txt" not in listed.model_content()
    assert "needle secret" not in grepped.model_content()


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
    assert result.model_content() == ""


def test_workspace_policy_resolves_inside_paths_and_rejects_outside(tmp_path: Path) -> None:
    policy = WorkspacePolicy()

    assert policy.resolve_path("a.txt", workspace_root=tmp_path) == (tmp_path / "a.txt").resolve()

    with pytest.raises(Exception, match="PATH_OUTSIDE_WORKSPACE"):
        policy.resolve_path("../outside.txt", workspace_root=tmp_path)
