"""会话 Todo 文件的不可变只读投影。"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class TodoStatus(StrEnum):
    """清单条目的三种工作状态。"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class TodoItem:
    """保留原始顺序与内容的单个任务。"""

    content: str
    status: TodoStatus


@dataclass(frozen=True, slots=True)
class TodoSnapshot:
    """一次读取结果；格式错误时仅包含诊断而无有效条目。"""

    path: Path
    items: tuple[TodoItem, ...]
    error: str | None


__all__ = ["TodoStatus", "TodoItem", "TodoSnapshot"]
