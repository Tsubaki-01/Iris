"""SQLite 权威记忆的确定性 Markdown 分类投影与显式概览发布。"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from html import escape
from itertools import groupby
from pathlib import Path

from ..exceptions import IrisMemoryError
from .files import BODY_PATHS, namespace_key, source_revision
from .models import (
    MemoryCategory,
    MemoryItem,
    MemoryItemKind,
    MemoryNamespaceSnapshot,
    MemoryNamespaceState,
    MemoryOverviewContent,
)
from .store import MemoryStore

KNOWLEDGE_SCOPE_MARKER = "<!-- iris-memory-knowledge-scope -->"


class FileMemoryMirror:
    """唯一的记忆文件发布 owner；SQLite 为锁与内容版本提供权威边界。"""

    def __init__(self, root: Path, *, workspace_root: Path | None = None) -> None:
        """绑定正文根目录，以及文档相对路径使用的可选 workspace。"""
        self.root = root.resolve(strict=False)
        self.workspace_root = workspace_root.resolve(strict=False) if workspace_root else None

    def initialize_layout(self) -> None:
        """只初始化目录，正文首次发布仍走 store 的完整事务。"""
        try:
            (self.root / "namespaces").mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise IrisMemoryError("memory mirror 初始化失败", root=str(self.root)) from exc

    def namespace_directory(self, namespace: str) -> Path:
        """返回 namespace 的独立目录，并保留 workspace containment。"""
        return self._resolve_relative(f"namespaces/{namespace_key(namespace)}")

    def document_path(self, namespace: str, relative_path: str = "Memory.md") -> Path:
        """返回正式文档的 workspace 相对路径或独立绝对路径。"""
        path = self._resolve_relative(f"namespaces/{namespace_key(namespace)}/{relative_path}")
        if self.workspace_root is not None:
            return path.relative_to(self.workspace_root)
        return path

    def rebuild_from_store(self, store: MemoryStore, namespace: str) -> MemoryNamespaceState:
        """在 store 的短发布事务内现读并发布全部分类正文。"""
        return store.publish_projection(namespace, self._publish_snapshot)

    def _publish_snapshot(self, snapshot: MemoryNamespaceSnapshot) -> None:
        """发布当前锁内完整快照，每个分类文件携带同一来源版本。"""
        namespace = snapshot.state.namespace
        prefix = f"namespaces/{namespace_key(namespace)}"
        for target in BODY_PATHS:
            items = tuple(item for item in snapshot.items if target_for_item(item) == target)
            content = self._render_body(namespace, snapshot.state.item_revision, target, items)
            self._atomic_replace(f"{prefix}/{target}", content)

    def read_overview(self, namespace: str) -> tuple[int, str, str] | None:
        """读回规定格式的完整概览和知识范围；缺少文件返回 None。"""
        path = self.namespace_directory(namespace) / "Memory.md"
        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError) as exc:
            raise IrisMemoryError("memory 概览读取失败", path=str(path)) from exc
        first_line, _, body = content.partition("\n")
        revision = source_revision(first_line)
        facts, marker, navigation = body.partition(KNOWLEDGE_SCOPE_MARKER)
        scope_header = "\n## 可查询的知识\n\n"
        if (
            revision is None
            or not facts.startswith(f"# Memory：{namespace}\n\n## 核心事实\n\n")
            or not marker
            or not navigation.startswith(scope_header)
            or not navigation[len(scope_header):].strip()
        ):
            raise IrisMemoryError("memory 概览格式不完整，请显式刷新概览", path=str(path))
        return revision, content, navigation.strip() + "\n"

    def publish_overview(
        self,
        store: MemoryStore,
        snapshot: MemoryNamespaceSnapshot,
        content: MemoryOverviewContent,
    ) -> tuple[bool, MemoryNamespaceState]:
        """短发布锁内比较来源版本，原子发布完整的双段概览。"""
        namespace = snapshot.state.namespace
        revision = snapshot.state.item_revision
        rendered = (
            f"<!-- iris-memory source_revision: {revision} -->\n"
            f"# Memory：{namespace}\n\n"
            f"## 核心事实\n\n{content.core_facts.strip()}\n\n"
            f"{KNOWLEDGE_SCOPE_MARKER}\n"
            f"## 可查询的知识\n\n{content.knowledge_scope.strip()}\n"
        )

        def publish(state: MemoryNamespaceState) -> tuple[bool, MemoryNamespaceState]:
            path = self.namespace_directory(namespace) / "Memory.md"
            try:
                with path.open(encoding="utf-8") as previous:
                    previous_revision = source_revision(previous.readline())
            except FileNotFoundError:
                previous_revision = None
            except (OSError, UnicodeError) as exc:
                raise IrisMemoryError("memory 概览来源读取失败", path=str(path)) from exc
            if previous_revision is not None and previous_revision > revision:
                return False, state
            self._atomic_replace(f"namespaces/{namespace_key(namespace)}/Memory.md", rendered)
            return True, state

        return store.publish_overview(namespace, publish)

    def _render_body(
        self, namespace: str, revision: int, target: str, items: Sequence[MemoryItem]
    ) -> str:
        """渲染分类正文，按 kind 保留所有条目及其完整文本。"""
        lines = [
            f"<!-- iris-memory source_revision: {revision} -->",
            f"# {target}",
            "",
            f"- namespace: {namespace}",
            "",
        ]
        if not items:
            lines.append("当前无记忆。")
        for kind, members in groupby(items, key=lambda item: item.kind):
            lines.extend([f"## {kind.value}", ""])
            for item in members:
                metadata = item.model_dump(mode="json", exclude={"text"}, exclude_none=True)
                information = []
                for name, value in metadata.items():
                    if name in {"artifacts", "metadata"} and not value:
                        continue
                    display = (
                        json.dumps(value, ensure_ascii=False, indent=2)
                        if name in {"artifacts", "metadata"}
                        else str(value)
                    )
                    information.append(f"{name}: {display}")
                lines.extend(
                    [
                        item.text,
                        "",
                        "<details>",
                        "<summary>记录信息</summary>",
                        "",
                        "<pre>",
                        escape("\n".join(information), quote=False),
                        "</pre>",
                        "</details>",
                        "",
                        "---",
                        "",
                    ]
                )
        return "\n".join(lines).rstrip() + "\n"

    def _resolve_relative(self, relative_path: str) -> Path:
        """解析生成路径并保留文件副作用前的 root containment。"""
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise IrisMemoryError("memory mirror 路径必须是相对路径", path=relative_path)
        resolved = (self.root / candidate).resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise IrisMemoryError("memory mirror 路径不能逃逸 root", path=relative_path) from exc
        return resolved

    def _atomic_replace(self, relative_path: str, content: str) -> None:
        """先写入本次临时文件，再原子替换单份完整文档。"""
        path = self._resolve_relative(relative_path)
        temp_path: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temp_name = tempfile.mkstemp(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
            )
            os.close(descriptor)
            temp_path = Path(temp_name)
            temp_path.write_text(content, encoding="utf-8")
            os.replace(temp_path, path)
        except OSError as exc:
            raise IrisMemoryError("memory mirror 写入失败", path=str(path)) from exc
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass


def target_for_item(item: MemoryItem) -> str:
    """沿用 category/kind 到分类正文的单一映射。"""
    if item.category == MemoryCategory.USER:
        return "User/preferences.md" if item.kind == MemoryItemKind.PREFERENCE else "User/user.md"
    if item.category == MemoryCategory.FEEDBACK:
        return (
            "Feedback/corrections.md"
            if item.kind == MemoryItemKind.CORRECTION
            else "Feedback/feedback.md"
        )
    if item.category == MemoryCategory.REFERENCE:
        return "Reference/notes.md"
    if item.category == MemoryCategory.TASK:
        return "Tasks/task.md"
    return "Sessions/session_items.md"
