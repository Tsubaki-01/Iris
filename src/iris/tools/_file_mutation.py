"""文件服务共用的进程内真实路径锁，覆盖实际 worker 的完整修改。"""

from _thread import LockType
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from ..exceptions import IrisToolExecutionError
from .base import CancellationSignal


@dataclass(slots=True)
class _LockEntry:
    """一个路径的锁及仍持有或等待该条目的引用数。"""

    lock: LockType = field(default_factory=Lock)
    references: int = 0


_entries: dict[Path, _LockEntry] = {}
_entries_lock = Lock()


@contextmanager
def file_mutation(path: Path, *, cancellation: CancellationSignal | None) -> Iterator[None]:
    """借用已解析路径的锁，修改开始前检查一次业务取消，退出时收回引用。"""
    with _entries_lock:
        entry = _entries.get(path)
        if entry is None:
            entry = _LockEntry()
            _entries[path] = entry
        entry.references += 1
    try:
        with entry.lock:
            if cancellation is not None and cancellation.requested:
                raise IrisToolExecutionError("FILE_OPERATION_CANCELLED: 文件修改开始前已取消")
            yield
    finally:
        with _entries_lock:
            entry.references -= 1
            if entry.references == 0:
                del _entries[path]
