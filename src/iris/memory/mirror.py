"""本地记忆 mirror 文件投影。

Mirror 是 SQLite 权威数据的只读投影，便于人工审查和 diff；本模块不支持从文件反向导入。

Example:
    mirror = FileMemoryMirror(Path(".iris/memory"))
    mirror.initialize_layout()
"""

# region imports
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..exceptions import IrisMemoryError
from .models import (
    MemoryCategory,
    MemoryEvent,
    MemoryItem,
    MemoryItemKind,
    MemoryItemStatus,
    MemoryScope,
)
from .store import MemoryStore

# endregion

# ==========================================
#                 Constants
# ==========================================
# region constants
LAYOUT_DIRECTORIES: tuple[str, ...] = (
    "User",
    "Feedback",
    "Reference",
    "Reference/docs",
    "Tasks",
    "Tasks/plans",
    "Sessions",
    "Sessions/session_summaries",
)
LAYOUT_FILES: tuple[str, ...] = (
    "Memory.md",
    "User/user.md",
    "User/profile.json",
    "User/preferences.md",
    "Feedback/feedback.md",
    "Feedback/corrections.md",
    "Reference/notes.md",
    "Reference/links.json",
    "Tasks/task.md",
    "Tasks/task.json",
    "Sessions/recent_events.md",
)
GENERATED_MARKDOWN_FILES: tuple[str, ...] = (
    "User/user.md",
    "User/preferences.md",
    "Feedback/feedback.md",
    "Feedback/corrections.md",
    "Reference/notes.md",
    "Tasks/task.md",
    "Sessions/recent_events.md",
)
JSON_DEFAULTS: dict[str, dict[str, Any]] = {
    "User/profile.json": {},
    "Reference/links.json": {"links": []},
    "Tasks/task.json": {"items": []},
}
# endregion


class FileMemoryMirror:
    """将权威记忆数据投影到 `.iris/memory/` 文件树。"""

    def __init__(self, root: Path) -> None:
        """初始化 mirror 根目录。"""
        self.root = root

    def initialize_layout(self) -> None:
        """创建固定目录和缺失的初始文件。"""
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            for directory in LAYOUT_DIRECTORIES:
                self._resolve_relative(directory).mkdir(parents=True, exist_ok=True)
            for relative_path in LAYOUT_FILES:
                path = self._resolve_relative(relative_path)
                if path.exists():
                    continue
                if relative_path.endswith(".json"):
                    path.write_text(
                        _dump_pretty_json(JSON_DEFAULTS.get(relative_path, {})),
                        encoding="utf-8",
                    )
                else:
                    path.write_text("", encoding="utf-8")
        except OSError as exc:
            raise IrisMemoryError("memory mirror 初始化失败", root=str(self.root)) from exc

    def mirror_item(self, item: MemoryItem) -> None:
        """将一个 active item 投影到对应 Markdown/JSON 文件。"""
        if item.status != MemoryItemStatus.ACTIVE:
            return
        self.initialize_layout()
        target = self._target_for_item(item)
        block = self._render_item_markdown(item)
        self._upsert_markdown_block(target, item.id, block, marker_type="item")
        if item.category == MemoryCategory.TASK and item.kind == MemoryItemKind.TASK_STATE:
            self._upsert_task_state(item)

    def mirror_event(self, event: MemoryEvent) -> None:
        """将审计事件追加到最近事件 mirror。"""
        self.initialize_layout()
        block = self._render_event_markdown(event)
        self._upsert_markdown_block(
            "Sessions/recent_events.md",
            event.id,
            block,
            marker_type="event",
        )

    def rebuild_from_store(self, store: MemoryStore, scope: MemoryScope) -> None:
        """从权威 store 确定性重建 active mirror 文件。"""
        self.initialize_layout()
        try:
            for relative_path in GENERATED_MARKDOWN_FILES:
                self._resolve_relative(relative_path).write_text("", encoding="utf-8")
            self._resolve_relative("Tasks/task.json").write_text(
                _dump_pretty_json({"items": []}),
                encoding="utf-8",
            )
            items = sorted(
                store.list_items(scope, limit=100),
                key=lambda item: (
                    item.category.value,
                    item.kind.value,
                    item.created_at,
                    item.id,
                ),
            )
            for item in items:
                self.mirror_item(item)
        except OSError as exc:
            raise IrisMemoryError("memory mirror 重建失败", root=str(self.root)) from exc

    def _resolve_relative(self, relative_path: str) -> Path:
        """解析系统生成的相对路径，并拒绝逃逸 root。"""
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise IrisMemoryError("memory mirror 路径必须是相对路径", path=relative_path)
        root = self.root.resolve(strict=False)
        resolved = (root / candidate).resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise IrisMemoryError("memory mirror 路径不能逃逸 root", path=relative_path) from exc
        return resolved

    def _render_item_markdown(self, item: MemoryItem) -> str:
        """将记忆条目渲染为稳定 Markdown block。"""
        lines = [
            f"### Memory Item {item.id}",
            "",
            f"- id: {item.id}",
            f"- category: {item.category.value}",
            f"- kind: {item.kind.value}",
            f"- scope: {_scope_summary(item.scope)}",
            f"- created_at: {item.created_at}",
            f"- updated_at: {item.updated_at}",
        ]
        if item.confidence is not None:
            lines.append(f"- confidence: {item.confidence}")
        if item.importance is not None:
            lines.append(f"- importance: {item.importance}")
        lines.extend(["", item.text, ""])
        return "\n".join(lines)

    def _render_event_markdown(self, event: MemoryEvent) -> str:
        """将审计事件渲染为最近事件 Markdown block。"""
        lines = [
            f"### Memory Event {event.id}",
            "",
            f"- id: {event.id}",
            f"- event_type: {event.event_type.value}",
            f"- actor: {event.actor.value}",
            f"- scope: {_scope_summary(event.scope)}",
            f"- created_at: {event.created_at}",
        ]
        if event.item_id:
            lines.append(f"- item_id: {event.item_id}")
        if event.episode_id:
            lines.append(f"- episode_id: {event.episode_id}")
        if event.reason:
            lines.append(f"- reason: {event.reason}")
        lines.append("")
        return "\n".join(lines)

    def _target_for_item(self, item: MemoryItem) -> str:
        """根据 category/kind 映射 mirror 文件。"""
        if item.category == MemoryCategory.USER:
            if item.kind == MemoryItemKind.PREFERENCE:
                return "User/preferences.md"
            return "User/user.md"
        if item.category == MemoryCategory.FEEDBACK:
            if item.kind == MemoryItemKind.CORRECTION:
                return "Feedback/corrections.md"
            return "Feedback/feedback.md"
        if item.category == MemoryCategory.REFERENCE:
            return "Reference/notes.md"
        if item.category == MemoryCategory.TASK:
            return "Tasks/task.md"
        return "Sessions/recent_events.md"

    def _upsert_markdown_block(
        self,
        relative_path: str,
        entity_id: str,
        block: str,
        *,
        marker_type: str,
    ) -> None:
        """按 marker 替换或追加一段生成内容。"""
        path = self._resolve_relative(relative_path)
        try:
            content = path.read_text(encoding="utf-8") if path.exists() else ""
            pattern = _block_pattern(marker_type, entity_id)
            generated = _wrap_block(marker_type, entity_id, block)
            if re.search(pattern, content, flags=re.DOTALL):
                new_content = re.sub(pattern, generated, content, flags=re.DOTALL)
            else:
                prefix = content.rstrip()
                new_content = f"{prefix}\n\n{generated}" if prefix else generated
            path.write_text(f"{new_content.rstrip()}\n", encoding="utf-8")
        except OSError as exc:
            raise IrisMemoryError("memory mirror 写入失败", path=str(path)) from exc

    def _upsert_task_state(self, item: MemoryItem) -> None:
        """更新结构化 task mirror。"""
        path = self._resolve_relative("Tasks/task.json")
        payload = {
            "id": item.id,
            "text": item.text,
            "metadata": item.metadata,
            "updated_at": item.updated_at,
        }
        try:
            current = _load_json_object(path)
            items = [
                entry
                for entry in current.get("items", [])
                if isinstance(entry, dict) and entry.get("id") != item.id
            ]
            items.append(payload)
            current["items"] = sorted(items, key=lambda entry: str(entry.get("id", "")))
            path.write_text(_dump_pretty_json(current), encoding="utf-8")
        except (OSError, TypeError) as exc:
            raise IrisMemoryError("memory mirror task.json 写入失败", path=str(path)) from exc


def _scope_summary(scope: MemoryScope) -> str:
    """生成稳定可读的 scope 摘要。"""
    parts = [
        f"workspace={scope.workspace_id}",
        f"agent={scope.agent_id}",
        f"collection={scope.collection}",
        f"visibility={scope.visibility.value}",
    ]
    if scope.session_id:
        parts.append(f"session={scope.session_id}")
    return ", ".join(parts)


def _wrap_block(marker_type: str, entity_id: str, block: str) -> str:
    """包裹生成内容，便于后续稳定替换。"""
    return (
        f"<!-- iris-memory-{marker_type}:{entity_id} -->\n"
        f"{block.rstrip()}\n"
        f"<!-- /iris-memory-{marker_type}:{entity_id} -->"
    )


def _block_pattern(marker_type: str, entity_id: str) -> str:
    """生成匹配指定生成块的正则。"""
    escaped_id = re.escape(entity_id)
    return (
        rf"<!-- iris-memory-{marker_type}:{escaped_id} -->.*?"
        rf"<!-- /iris-memory-{marker_type}:{escaped_id} -->"
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    """读取 JSON object；空文件按空对象处理。"""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("memory mirror JSON 文件必须是 object")
    return value


def _dump_pretty_json(value: dict[str, Any]) -> str:
    """序列化人工可读 JSON mirror。"""
    try:
        return f"{json.dumps(value, ensure_ascii=False, indent=2)}\n"
    except TypeError as exc:
        raise IrisMemoryError("memory mirror JSON 必须可序列化") from exc
