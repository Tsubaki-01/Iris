"""文件读后写入的乐观锁状态。"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..exceptions import IrisToolValidationError


class ReadFileRecord(BaseModel):
    """已读取文件的乐观锁记录。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    mtime_ns: int
    size_bytes: int
    digest: str | None = None


class ReadFileState(BaseModel):
    """当前上下文中已读取文件状态。"""

    files: dict[str, ReadFileRecord] = Field(default_factory=dict)

    def merge(self, record: ReadFileRecord) -> None:
        """合并已完成的文件读取观测，不再次访问文件系统。

        Args:
            record (ReadFileRecord): worker 对已解析文件生成的不可变观测。

        Raises:
            IrisToolValidationError: observation 未使用绝对路径。
        """
        if not record.path.is_absolute():
            raise IrisToolValidationError("read observation 必须使用绝对路径")
        self.files[str(record.path)] = record

    def update(self, path: Path) -> None:
        """用当前文件 stat 刷新记录。"""
        stat = path.stat()
        resolved = path.resolve()
        self.merge(
            ReadFileRecord(
                path=resolved,
                mtime_ns=stat.st_mtime_ns,
                size_bytes=stat.st_size,
            )
        )

    def get(self, path: Path) -> ReadFileRecord | None:
        """按 resolve 后路径获取读取记录。"""
        return self.files.get(str(path.resolve()))
