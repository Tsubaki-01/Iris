"""通用文件工具消费的记忆投影视图边界。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ...exceptions import IrisToolValidationError


class MemoryFileView(Protocol):
    """由 memory 层提供的路径、来源版本和同步状态。"""

    @property
    def root(self) -> Path:
        """返回派生文件根目录。"""

    @property
    def namespaces(self) -> tuple[str, ...]:
        """返回当前允许的 namespace。"""

    @property
    def document_paths(self) -> tuple[Path, ...]:
        """返回当前范围内全部正式文档路径。"""

    def namespace_directory(self, namespace: str) -> Path:
        """返回 namespace 的派生目录。"""

    def source_revision(self, first_line: str) -> int | None:
        """解析实际打开文件的第一行来源标记。"""

    def projection_warning(self, namespace: str) -> str | None:
        """返回完整正文尚未同步的提示，不需要已经存在文件。"""

    def warning(
        self, namespace: str, source_revision: int | None, *, overview: bool = False
    ) -> str | None:
        """根据读后数据库状态返回陈旧说明。"""


class MemoryFileBoundary:
    """在记忆根内只允许访问当前范围的正式投影文件。"""

    def __init__(self, view: MemoryFileView) -> None:
        """消费构造层提供的路径视图，不读取记忆正文。"""
        self.view = view
        self.root = view.root
        self.documents = frozenset(view.document_paths)
        self.directories = frozenset(
            parent
            for path in self.documents
            for parent in path.parents
            if parent.is_relative_to(self.root)
        )
        self.namespace_directories = {
            namespace: view.namespace_directory(namespace) for namespace in view.namespaces
        }

    def permits(self, path: Path) -> bool:
        """用于直接访问和遍历剪枝；普通 workspace 路径沿用原策略。"""
        return (
            not path.is_relative_to(self.root)
            or path in self.documents
            or path in self.directories
            or path == self.root
        )

    def check(self, path: Path, *, write: bool) -> None:
        """在实际文件操作边界应用投影读写约定。"""
        if write and path.is_relative_to(self.root):
            raise IrisToolValidationError("记忆投影文件应通过 memory 工具或 SDK 修改")
        if not self.permits(path):
            raise IrisToolValidationError("文件不在当前允许读取的记忆投影范围", path=str(path))

    def warning(self, path: Path, source_revision: int | None) -> str | None:
        """读取内容后查询相应 namespace 的同步状态。"""
        for namespace, directory in self.namespace_directories.items():
            if path.is_relative_to(directory):
                return self.view.warning(
                    namespace, source_revision, overview=path.name == "Memory.md"
                )
        return None

    def projection_warnings(self, path: Path) -> list[str]:
        """查询读取目标覆盖的namespace，包含尚未创建正文目录的情况。"""
        warnings: list[str] = []
        for namespace, directory in self.namespace_directories.items():
            if path.is_relative_to(directory) or directory.is_relative_to(path):
                warning = self.view.projection_warning(namespace)
                if warning is not None:
                    warnings.append(warning)
        return warnings
