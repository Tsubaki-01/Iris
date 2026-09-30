"""Todo 文档诊断与领域异常的不同语义。"""

from pathlib import Path

import pytest

from iris.exceptions import IrisError, IrisTodoError
from iris.todo.document import read_todo


@pytest.mark.asyncio
async def test_invalid_utf8_is_diagnostic_without_partial_items(tmp_path: Path) -> None:
    """无法解码的文件可由普通文件操作修复，不误报为空清单。"""
    path = tmp_path / ".iris/todos/776f726b.md"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"- [x] done\n- [ ] \xff")
    snapshot = await read_todo(tmp_path, "work")
    assert snapshot.path == path
    assert snapshot.items == ()
    assert snapshot.error is not None
    assert "UTF-8" in snapshot.error
    assert "编码" in snapshot.error


@pytest.mark.asyncio
async def test_io_failure_is_todo_error_with_path_and_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实读取故障保留异常链、路径和原因，不当作文件不存在。"""
    failure = PermissionError("read denied")

    def fail_read(path: Path, *, encoding: str) -> str:
        raise failure

    monkeypatch.setattr(Path, "read_text", fail_read)
    with pytest.raises(IrisTodoError) as caught:
        await read_todo(tmp_path, "work")
    error = caught.value
    assert isinstance(error, IrisError)
    assert error.__cause__ is failure
    assert error.context["path"] == str(tmp_path / ".iris/todos/776f726b.md")
    assert error.context["error"] == "read denied"
    assert error.runtime_source == "runtime"
    assert error.runtime_code == "TODO_ERROR"
