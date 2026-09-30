"""定位并只读解析当前会话的 Markdown 工作清单。"""

import asyncio
import re
from pathlib import Path

from ..exceptions import IrisTodoError
from .models import TodoItem, TodoSnapshot, TodoStatus

_HEADING = re.compile(r" {0,3}#{1,6}(?:\s|$)")
_ITEM = re.compile(r"- \[([ \-xX])\][ \t]+(.*)")
_STATUSES = {
    " ": TodoStatus.PENDING,
    "-": TodoStatus.IN_PROGRESS,
    "x": TodoStatus.COMPLETED,
    "X": TodoStatus.COMPLETED,
}


def _read_document(path: Path) -> TodoSnapshot:
    try:
        content = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return TodoSnapshot(path, (), None)
    except UnicodeDecodeError:
        return TodoSnapshot(path, (), "Todo 文件编码错误，必须使用 UTF-8。")
    except OSError as exc:
        raise IrisTodoError("Todo 文件读取失败", path=str(path), error=str(exc)) from exc

    items: list[TodoItem] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip() or _HEADING.match(line):
            continue
        match = _ITEM.fullmatch(line)
        if match is None or not match[2].strip():
            return TodoSnapshot(
                path,
                (),
                f"Todo 第 {line_number} 行格式错误："
                "应为标题或带非空内容的 - [ ]、- [-]、- [x] 条目。",
            )
        items.append(TodoItem(match[2].strip(), _STATUSES[match[1]]))
    return TodoSnapshot(path, tuple(items), None)


async def read_todo(workspace_root: Path, session_id: str) -> TodoSnapshot:
    """在线程中读取当前会话清单，不创建目录、缓存或修改文件。

    Args:
        workspace_root: 已解析的工作区绝对路径。
        session_id: 调用方已确定的会话身份。

    Returns:
        当前文件内容或格式诊断的不可变快照。

    Raises:
        IrisTodoError: 除文件不存在之外的操作系统读取失败。
    """
    path = workspace_root / ".iris" / "todos" / f"{session_id.encode('utf-8').hex()}.md"
    return await asyncio.to_thread(_read_document, path)
