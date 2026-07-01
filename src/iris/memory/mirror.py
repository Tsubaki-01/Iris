"""本地记忆 mirror 文件投影。

Mirror 是 SQLite 权威数据的只读投影，便于人工审查和 diff；本模块不支持从文件反向导入。

Mirror 文件大致长这样：

```text
.iris/memory/
  Memory.md
  User/
    user.md
    profile.json
    preferences.md
  Feedback/
    feedback.md
    corrections.md
  Reference/
    notes.md
    docs/
    links.json
  Tasks/
    task.md
    task.json
    plans/
  Sessions/
    session_items.md
    recent_events.md
    session_summaries/
```

`User/user.md`、`User/preferences.md`、`Feedback/feedback.md`、
`Feedback/corrections.md`、`Reference/notes.md`、`Tasks/task.md`
与 `Sessions/session_items.md`
是按记忆类别投影的 Markdown 文件。每条记忆由稳定 marker 包裹，便于后续覆盖更新：

```markdown
<!-- iris-memory-item:scopehash12345678:mem_123 -->
### Memory Item mem_123

- id: mem_123
- category: user
- kind: preference
- scope: workspace=workspace, agent=agent, collection=default, visibility=agent
- created_at: 2026-06-03T10:00:00
- updated_at: 2026-06-03T10:00:00
- confidence: 0.8
- importance: 0.7

用户偏好简洁中文回答
<!-- /iris-memory-item:scopehash12345678:mem_123 -->
```

`Sessions/recent_events.md` 记录每个 scope 最近 100 条审计事件：

```markdown
# Recent Memory Events

This file is a generated recent projection. It keeps only the latest 100 events per scope.
The complete audit logs shall be subject to SQLite memory_events.

<!-- iris-memory-event:scopehash12345678:evt_123 -->
### Memory Event evt_123

- id: evt_123
- event_type: observe
- actor: agent
- scope: workspace=workspace, agent=agent, collection=default, visibility=agent
- created_at: 2026-06-03T10:00:00
- reason: user message observed
<!-- /iris-memory-event:scopehash12345678:evt_123 -->
```

`Tasks/task.json` 是任务状态的结构化投影：

```json
{
  "items": [
    {
      "id": "mem_123",
      "text": "阶段二实现 mirror",
      "metadata": {
        "stage": 2,
        "status": "in_progress"
      },
      "updated_at": "2026-06-03T10:00:00"
    }
  ]
}
```

`User/profile.json` 默认是 `{}`，`Reference/links.json` 默认是
`{"links": []}`；这些 JSON 文件当前只由 mirror 初始化或特定投影逻辑写入。

Example:
    mirror = FileMemoryMirror(Path(".iris/memory"))
    mirror.initialize_layout()
"""

# region imports
from __future__ import annotations

import hashlib
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
    "Sessions/session_items.md",
    "Sessions/recent_events.md",
)
GENERATED_MARKDOWN_FILES: tuple[str, ...] = (
    "User/user.md",
    "User/preferences.md",
    "Feedback/feedback.md",
    "Feedback/corrections.md",
    "Reference/notes.md",
    "Tasks/task.md",
    "Sessions/session_items.md",
    "Sessions/recent_events.md",
)
GENERATED_ITEM_MARKDOWN_FILES: tuple[str, ...] = tuple(
    relative_path
    for relative_path in GENERATED_MARKDOWN_FILES
    if relative_path != "Sessions/recent_events.md"
)
JSON_DEFAULTS: dict[str, dict[str, Any]] = {
    "User/profile.json": {},
    "Reference/links.json": {"links": []},
    "Tasks/task.json": {"items": []},
}
RECENT_EVENTS_LIMIT = 100
RECENT_EVENTS_PATH = "Sessions/recent_events.md"
RECENT_EVENTS_HEADER = (
    "# Recent Memory Events\n\n"
    "This file is a generated recent projection. It keeps only the latest "
    f"{RECENT_EVENTS_LIMIT} events per scope.\n"
    "The complete audit logs shall be subject to SQLite memory_events.\n"
)
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
                if relative_path == RECENT_EVENTS_PATH:
                    path.write_text(RECENT_EVENTS_HEADER, encoding="utf-8")
                elif relative_path.endswith(".json"):
                    path.write_text(
                        _dump_pretty_json(JSON_DEFAULTS.get(relative_path, {})),
                        encoding="utf-8",
                    )
                else:
                    path.write_text("", encoding="utf-8")
            self._ensure_recent_events_header()
        except OSError as exc:
            raise IrisMemoryError(
                "memory mirror 初始化失败", root=str(self.root)
            ) from exc

    def mirror_item(self, item: MemoryItem) -> None:
        """将一个 active item 投影到对应 Markdown/JSON 文件。"""
        if item.status != MemoryItemStatus.ACTIVE:
            return
        self.initialize_layout()
        target = self._target_for_item(item)
        block = self._render_item_markdown(item)
        self._upsert_markdown_block(
            target,
            item.id,
            block,
            marker_type="item",
            scope=item.scope,
        )
        if (
            item.category == MemoryCategory.TASK
            and item.kind == MemoryItemKind.TASK_STATE
        ):
            self._upsert_task_state(item)

    def mirror_event(self, event: MemoryEvent) -> None:
        """将审计事件追加到最近事件 mirror。"""
        self.initialize_layout()
        block = self._render_event_markdown(event)
        self._upsert_markdown_block(
            RECENT_EVENTS_PATH,
            event.id,
            block,
            marker_type="event",
            scope=event.scope,
        )
        self._trim_recent_events_for_scope(event.scope)

    def rebuild_from_store(self, store: MemoryStore, scope: MemoryScope) -> None:
        """从权威 store 确定性重建 active mirror 文件。"""
        self.initialize_layout()
        try:
            for relative_path in GENERATED_ITEM_MARKDOWN_FILES:
                self._remove_scope_markdown_blocks(relative_path, "item", scope)
            self._remove_scope_markdown_blocks(RECENT_EVENTS_PATH, "event", scope)
            self._remove_task_states_for_scope(scope)
            items = sorted(
                store.list_items(scope, limit=None),
                key=lambda item: (
                    item.category.value,
                    item.kind.value,
                    item.created_at,
                    item.id,
                ),
            )
            for item in items:
                self.mirror_item(item)
            events = sorted(
                store.list_events(scope, limit=RECENT_EVENTS_LIMIT),
                key=lambda event: (event.created_at, event.id),
            )
            for event in events:
                self.mirror_event(event)
        except OSError as exc:
            raise IrisMemoryError(
                "memory mirror 重建失败", root=str(self.root)
            ) from exc

    def _resolve_relative(self, relative_path: str) -> Path:
        """解析系统生成的相对路径，并拒绝逃逸 root。"""
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise IrisMemoryError(
                "memory mirror 路径必须是相对路径", path=relative_path
            )
        root = self.root.resolve(strict=False)
        resolved = (root / candidate).resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise IrisMemoryError(
                "memory mirror 路径不能逃逸 root", path=relative_path
            ) from exc
        return resolved

    def _ensure_recent_events_header(self) -> None:
        """确保 recent events 文件带有 recent-only 投影说明。"""
        path = self._resolve_relative(RECENT_EVENTS_PATH)
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        if content.startswith(RECENT_EVENTS_HEADER):
            return
        stripped = content.lstrip()
        if stripped.startswith(RECENT_EVENTS_HEADER.rstrip()):
            path.write_text(_normalize_markdown_content(stripped), encoding="utf-8")
            return
        if stripped:
            path.write_text(
                _normalize_markdown_content(f"{RECENT_EVENTS_HEADER}\n{stripped}"),
                encoding="utf-8",
            )
            return
        path.write_text(RECENT_EVENTS_HEADER, encoding="utf-8")

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
        return "Sessions/session_items.md"

    def _upsert_markdown_block(
        self,
        relative_path: str,
        entity_id: str,
        block: str,
        *,
        marker_type: str,
        scope: MemoryScope,
    ) -> None:
        """按 marker 替换或追加一段生成内容。"""
        path = self._resolve_relative(relative_path)
        try:
            content = path.read_text(encoding="utf-8") if path.exists() else ""
            scope_hash = _scope_hash(scope)
            pattern = _block_pattern(marker_type, scope_hash, entity_id)
            generated = _wrap_block(marker_type, scope_hash, entity_id, block)
            if re.search(pattern, content, flags=re.DOTALL):
                new_content = re.sub(
                    pattern, lambda _: generated, content, flags=re.DOTALL
                )
            else:
                prefix = content.rstrip()
                new_content = f"{prefix}\n\n{generated}" if prefix else generated
            path.write_text(f"{new_content.rstrip()}\n", encoding="utf-8")
        except OSError as exc:
            raise IrisMemoryError("memory mirror 写入失败", path=str(path)) from exc

    def _remove_scope_markdown_blocks(
        self,
        relative_path: str,
        marker_type: str,
        scope: MemoryScope,
    ) -> None:
        """删除指定文件中属于当前 scope 的 generated Markdown blocks。"""
        path = self._resolve_relative(relative_path)
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        pattern = _scope_blocks_pattern(marker_type, scope)
        cleaned = re.sub(pattern, lambda _: "\n\n", content, flags=re.DOTALL)
        if relative_path == RECENT_EVENTS_PATH:
            cleaned = _with_recent_events_header(cleaned)
        path.write_text(_normalize_markdown_content(cleaned), encoding="utf-8")

    def _trim_recent_events_for_scope(self, scope: MemoryScope) -> None:
        """只保留指定 scope 的最近 N 条事件投影。"""
        path = self._resolve_relative(RECENT_EVENTS_PATH)
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        matches = list(
            re.finditer(_scope_blocks_pattern("event", scope), content, re.DOTALL)
        )
        overflow = len(matches) - RECENT_EVENTS_LIMIT
        if overflow <= 0:
            self._ensure_recent_events_header()
            return
        remove_spans = [match.span() for match in matches[:overflow]]
        trimmed = _remove_spans(content, remove_spans)
        path.write_text(
            _normalize_markdown_content(_with_recent_events_header(trimmed)),
            encoding="utf-8",
        )

    def _upsert_task_state(self, item: MemoryItem) -> None:
        """更新结构化 task mirror。"""
        path = self._resolve_relative("Tasks/task.json")
        payload = {
            "id": item.id,
            "scope_hash": _scope_hash(item.scope),
            "scope": _scope_summary(item.scope),
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
            current["items"] = sorted(
                items,
                key=lambda entry: (
                    str(entry.get("scope_hash", "")),
                    str(entry.get("id", "")),
                ),
            )
            path.write_text(_dump_pretty_json(current), encoding="utf-8")
        except (OSError, TypeError) as exc:
            raise IrisMemoryError(
                "memory mirror task.json 写入失败", path=str(path)
            ) from exc

    def _remove_task_states_for_scope(self, scope: MemoryScope) -> None:
        """删除 task.json 中属于当前 scope 的结构化投影。"""
        path = self._resolve_relative("Tasks/task.json")
        scope_hash = _scope_hash(scope)
        try:
            current = _load_json_object(path)
            current["items"] = [
                entry
                for entry in current.get("items", [])
                if not isinstance(entry, dict) or entry.get("scope_hash") != scope_hash
            ]
            path.write_text(_dump_pretty_json(current), encoding="utf-8")
        except (OSError, TypeError) as exc:
            raise IrisMemoryError(
                "memory mirror task.json 重建失败", path=str(path)
            ) from exc


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


def _scope_key(scope: MemoryScope) -> str:
    """生成稳定 scope key，用于派生 marker hash。"""
    session_id = scope.session_id or ""
    return (
        f"workspace={scope.workspace_id}|agent={scope.agent_id}|"
        f"collection={scope.collection}|visibility={scope.visibility.value}|"
        f"session={session_id}"
    )


def _scope_hash(scope: MemoryScope) -> str:
    """生成短 hash，避免把长 scope 直接塞进 marker。"""
    return hashlib.sha256(_scope_key(scope).encode("utf-8")).hexdigest()[:16]


def _wrap_block(marker_type: str, scope_hash: str, entity_id: str, block: str) -> str:
    """包裹生成内容，便于后续稳定替换。"""
    return (
        f"<!-- iris-memory-{marker_type}:{scope_hash}:{entity_id} -->\n"
        f"{block.rstrip()}\n"
        f"<!-- /iris-memory-{marker_type}:{scope_hash}:{entity_id} -->"
    )


def _block_pattern(marker_type: str, scope_hash: str, entity_id: str) -> str:
    """生成匹配指定生成块的正则。"""
    escaped_scope_hash = re.escape(scope_hash)
    escaped_id = re.escape(entity_id)
    return (
        rf"<!-- iris-memory-{marker_type}:{escaped_scope_hash}:{escaped_id} -->.*?"
        rf"<!-- /iris-memory-{marker_type}:{escaped_scope_hash}:{escaped_id} -->"
    )


def _scope_blocks_pattern(marker_type: str, scope: MemoryScope) -> str:
    """生成匹配同一 scope 下全部生成块的正则。"""
    escaped_scope_hash = re.escape(_scope_hash(scope))
    return (
        rf"(?:\r?\n)*<!-- iris-memory-{marker_type}:{escaped_scope_hash}:[^>\n]+ -->.*?"
        rf"<!-- /iris-memory-{marker_type}:{escaped_scope_hash}:[^>\n]+ -->(?:\r?\n)*"
    )


def _normalize_markdown_content(content: str) -> str:
    """清理生成块删除后留下的首尾空白。"""
    stripped = content.strip()
    return f"{stripped}\n" if stripped else ""


def _with_recent_events_header(content: str) -> str:
    """给 recent events 内容补上固定文件头。"""
    stripped = content.lstrip()
    if stripped.startswith(RECENT_EVENTS_HEADER.rstrip()):
        return stripped
    if stripped:
        return f"{RECENT_EVENTS_HEADER}\n{stripped}"
    return RECENT_EVENTS_HEADER


def _remove_spans(content: str, spans: list[tuple[int, int]]) -> str:
    """按 span 删除文本片段。"""
    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        pieces.append(content[cursor:start])
        cursor = end
    pieces.append(content[cursor:])
    return "".join(pieces)


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
