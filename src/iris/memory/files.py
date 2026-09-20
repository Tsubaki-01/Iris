"""供通用文件工具消费的只读记忆路径与新鲜度接口。"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path

from .models import MemoryNamespaceState
from .store import MemoryStore

BODY_PATHS: tuple[str, ...] = (
    "User/user.md",
    "User/preferences.md",
    "Feedback/feedback.md",
    "Feedback/corrections.md",
    "Reference/notes.md",
    "Tasks/task.md",
    "Sessions/session_items.md",
)
DOCUMENT_PATHS: tuple[str, ...] = ("Memory.md", *BODY_PATHS)
SOURCE_REVISION_PATTERN = re.compile(r"<!-- iris-memory source_revision: (\d+) -->")


def namespace_key(namespace: str) -> str:
    """将不透明 namespace 编码为可逆且独立的目录 key。"""
    return "ns_" + base64.urlsafe_b64encode(namespace.encode("utf-8")).decode("ascii").rstrip("=")


def source_revision(first_line: str) -> int | None:
    """解析实际读取文件的首行版本，不以读前状态推断文件来源。"""
    match = SOURCE_REVISION_PATTERN.fullmatch(first_line.strip())
    return int(match.group(1)) if match else None


def freshness_warning(
    state: MemoryNamespaceState, revision: int | None, *, overview: bool = False
) -> str | None:
    """根据实际文件版本与读后 I/P 组合可见的陈旧提示。"""
    warnings: list[str] = []
    label = "概览" if overview else "正文"
    if revision != state.item_revision:
        warnings.append(
            f"{label}陈旧：source_revision={revision}，当前 item_revision={state.item_revision}。"
        )
    if state.item_revision > 0 and state.projection_revision != state.item_revision:
        warnings.append(
            "正文文件未同步："
            f"projection_revision={state.projection_revision}，"
            f"item_revision={state.item_revision}；数据库已保存，现有文件可能不完整。"
        )
    return " ".join(warnings) or None


@dataclass(frozen=True, slots=True)
class MemoryFileAccess:
    """当前读取范围内的正式文件路径和读后版本查询。"""

    root: Path
    namespaces: tuple[str, ...]
    store: MemoryStore

    def namespace_directory(self, namespace: str) -> Path:
        """返回 namespace 的独立绝对目录。"""
        return self.root / "namespaces" / namespace_key(namespace)

    @property
    def document_paths(self) -> tuple[Path, ...]:
        """列出当前 namespace 范围内全部正式 Markdown 路径。"""
        return tuple(
            self.namespace_directory(namespace) / relative
            for namespace in self.namespaces
            for relative in DOCUMENT_PATHS
        )

    def source_revision(self, first_line: str) -> int | None:
        """解析同一个已打开文件实际读取到的首行。"""
        return source_revision(first_line)

    def read_namespace_state(self, namespace: str) -> MemoryNamespaceState:
        """读取文件之后查询当前 namespace 版本。"""
        return self.store.read_namespace_state(namespace)

    def projection_warning(self, namespace: str) -> str | None:
        """没有可读文件时仍报告数据库与完整正文的同步状态。"""
        state = self.read_namespace_state(namespace)
        return freshness_warning(state, state.item_revision)

    def warning(
        self, namespace: str, source_revision: int | None, *, overview: bool = False
    ) -> str | None:
        """返回当前文件的陈旧状态，允许正文继续读取。"""
        return freshness_warning(
            self.read_namespace_state(namespace), source_revision, overview=overview
        )
