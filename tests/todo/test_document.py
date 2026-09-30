"""会话 Todo Markdown 的只读解析契约。"""

from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import get_ident

import pytest

from iris.todo import TodoItem, TodoSnapshot, TodoStatus
from iris.todo.document import read_todo


def _write_todo(workspace: Path, session_id: str, content: bytes) -> Path:
    path = workspace / ".iris" / "todos" / f"{session_id.encode('utf-8').hex()}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.mark.asyncio
async def test_missing_document_is_empty_without_creating_directories(tmp_path: Path) -> None:
    """首次读取只返回位置，不创建工作目录或文件。"""
    snapshot = await read_todo(tmp_path, "work")
    assert snapshot == TodoSnapshot(tmp_path / ".iris/todos/776f726b.md", (), None)
    assert not (tmp_path / ".iris").exists()


@pytest.mark.asyncio
async def test_session_identity_selects_exact_document(tmp_path: Path) -> None:
    """读取器保留大小写、空格和 Unicode 身份，并稳定定位单个文件。"""
    session_ids = ("work", "Work", " work ", "work/child", "会话")
    paths = [
        _write_todo(tmp_path, session_id, f"- [ ] {session_id}".encode())
        for session_id in session_ids
    ]
    for session_id, path in zip(session_ids, paths, strict=True):
        first = await read_todo(tmp_path, session_id)
        second = await read_todo(tmp_path, session_id)
        assert (
            first
            == second
            == TodoSnapshot(path, (TodoItem(session_id.strip(), TodoStatus.PENDING),), None)
        )
    assert len(set(paths)) == len(session_ids)


@pytest.mark.parametrize(
    "content",
    [b"", b" \t\n\r\n", b"#\n ## title\r\n  ### title\n   ######\t heading\n"],
)
@pytest.mark.asyncio
async def test_empty_and_heading_only_documents_are_empty(tmp_path: Path, content: bytes) -> None:
    """允许的空白与 ATX 标题不产生任务。"""
    path = _write_todo(tmp_path, "work", content)
    assert await read_todo(tmp_path, "work") == TodoSnapshot(path, (), None)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("bom", ["", "\ufeff"])
@pytest.mark.asyncio
async def test_status_order_duplicates_and_content_are_preserved(
    tmp_path: Path, newline: str, bom: str
) -> None:
    """三态、多个进行中与重复内容均原样保留，不重排或截断。"""
    content = newline.join(
        [
            "# 清单",
            "- [ ]  重复内容  ",
            "- [-]\t内部  空格\t保留 ",
            "- [-] 另一个进行中",
            "- [x] 重复内容",
            "- [X] 完成",
            "- [ ] 重复内容",
        ]
    )
    path = _write_todo(tmp_path, "work", (bom + content).encode())
    assert await read_todo(tmp_path, "work") == TodoSnapshot(
        path,
        (
            TodoItem("重复内容", TodoStatus.PENDING),
            TodoItem("内部  空格\t保留", TodoStatus.IN_PROGRESS),
            TodoItem("另一个进行中", TodoStatus.IN_PROGRESS),
            TodoItem("重复内容", TodoStatus.COMPLETED),
            TodoItem("完成", TodoStatus.COMPLETED),
            TodoItem("重复内容", TodoStatus.PENDING),
        ),
        None,
    )


@pytest.mark.asyncio
async def test_all_completed_items_remain_in_snapshot(tmp_path: Path) -> None:
    """全完成清单仍可查询且 DTO 不可变。"""
    _write_todo(tmp_path, "work", b"- [x] done\n- [X] also done")
    snapshot = await read_todo(tmp_path, "work")
    assert snapshot.items == (
        TodoItem("done", TodoStatus.COMPLETED),
        TodoItem("also done", TodoStatus.COMPLETED),
    )
    with pytest.raises(FrozenInstanceError):
        snapshot.items = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        snapshot.items[0].content = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "invalid_line",
    [
        "正文",
        "- [?] unknown",
        "- [done] unknown",
        "- [ ]",
        "- [ ] \t ",
        " - [ ] nested",
        "\t- [ ] nested",
        "-\t[ ] prefix",
        "- [ ]missing separator",
        "* [ ] alternate list",
        "####### too deep",
        "#missing separator",
        "    # indented heading",
        "```",
        "---",
    ],
)
@pytest.mark.asyncio
async def test_first_invalid_line_rejects_entire_document_and_can_be_repaired(
    tmp_path: Path, invalid_line: str
) -> None:
    """诊断包含一基行号，不泄露有效前缀；外部修复后立即读取新内容。"""
    content = f"# Tasks\n- [x] prefix\n{invalid_line}\n- [ ] suffix\ninvalid again"
    path = _write_todo(tmp_path, "work", content.encode())
    snapshot = await read_todo(tmp_path, "work")
    assert snapshot.path == path
    assert snapshot.items == ()
    assert snapshot.error is not None
    assert "第 3 行" in snapshot.error
    assert path.read_bytes() == content.encode()
    path.write_text("- [ ] repaired", encoding="utf-8")
    assert await read_todo(tmp_path, "work") == TodoSnapshot(
        path, (TodoItem("repaired", TodoStatus.PENDING),), None
    )


@pytest.mark.asyncio
async def test_read_occurs_off_event_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """磁盘读取位于有限的后台线程操作，不阻塞运行循环。"""
    _write_todo(tmp_path, "work", b"- [ ] task")
    original = Path.read_text
    read_threads: list[int] = []

    def record_read(path: Path, *, encoding: str) -> str:
        read_threads.append(get_ident())
        return original(path, encoding=encoding)

    monkeypatch.setattr(Path, "read_text", record_read)
    snapshot = await read_todo(tmp_path, "work")
    assert snapshot.items == (TodoItem("task", TodoStatus.PENDING),)
    assert len(read_threads) == 1
    assert read_threads[0] != get_ident()
