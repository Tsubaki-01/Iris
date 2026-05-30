"""SQLite 记忆存储实现。

SQLite 是 Stage 1 的权威存储；Markdown/JSON mirror、YAML 配置与工具层由后续阶段实现。

Example:
    store = SQLiteMemoryStore(".iris/memory/memory.db")
    store.initialize_schema()
"""

# region imports
from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from ..exceptions import IrisMemoryError
from .models import (
    MemoryActor,
    MemoryArtifactRef,
    MemoryCategory,
    MemoryEpisode,
    MemoryEvent,
    MemoryEventType,
    MemoryItem,
    MemoryItemKind,
    MemoryItemPatch,
    MemoryItemStatus,
    MemoryLevel,
    MemoryQuery,
    MemoryScope,
    MemorySearchResult,
    MemorySourceType,
    MemoryVisibility,
    _now_iso,
)

# endregion


class SQLiteMemoryStore:
    """基于本地 SQLite 文件的长期记忆权威 store。

    Args:
        path: SQLite 数据库文件路径。
        use_fts: 是否尝试启用 FTS5 搜索索引；不可用时自动降级为 LIKE fallback。
    """

    def __init__(self, path: str | Path, *, use_fts: bool = True) -> None:
        """初始化 SQLite store 并创建表结构。"""
        self.path = Path(path)
        self.use_fts = use_fts
        self._fts_enabled = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise IrisMemoryError("SQLite memory 目录创建失败", path=str(self.path)) from exc
        self.initialize_schema()

    @property
    def fts_enabled(self) -> bool:
        """返回当前 store 是否实际启用了 FTS5。"""
        return self._fts_enabled

    def initialize_schema(self) -> None:
        """创建或补齐 Stage 1 记忆表结构。"""
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_schema (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_episodes (
                        id TEXT PRIMARY KEY,
                        scope_workspace_id TEXT NOT NULL,
                        scope_agent_id TEXT NOT NULL,
                        scope_collection TEXT NOT NULL,
                        scope_visibility TEXT NOT NULL,
                        scope_session_id TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        source_id TEXT NOT NULL,
                        text TEXT NOT NULL,
                        category TEXT NOT NULL,
                        artifacts_json TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_items (
                        id TEXT PRIMARY KEY,
                        scope_workspace_id TEXT NOT NULL,
                        scope_agent_id TEXT NOT NULL,
                        scope_collection TEXT NOT NULL,
                        scope_visibility TEXT NOT NULL,
                        scope_session_id TEXT NOT NULL,
                        episode_id TEXT,
                        level TEXT NOT NULL,
                        category TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        text TEXT NOT NULL,
                        status TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        source_id TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        confidence REAL,
                        importance REAL,
                        artifacts_json TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        deleted_at TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_events (
                        id TEXT PRIMARY KEY,
                        scope_workspace_id TEXT NOT NULL,
                        scope_agent_id TEXT NOT NULL,
                        scope_collection TEXT NOT NULL,
                        scope_visibility TEXT NOT NULL,
                        scope_session_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        actor TEXT NOT NULL,
                        item_id TEXT,
                        episode_id TEXT,
                        reason TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO memory_schema (key, value)
                    VALUES ('schema_version', '1')
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """
                )
                self._fts_enabled = False
                if self.use_fts:
                    try:
                        connection.execute(
                            """
                            CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_fts
                            USING fts5(item_id UNINDEXED, text)
                            """
                        )
                    except sqlite3.Error:
                        self._fts_enabled = False
                    else:
                        self._fts_enabled = True
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory 初始化失败", path=str(self.path)) from exc

    def rebuild_index(self) -> None:
        """重建 FTS 索引；未启用 FTS 时该方法不产生副作用。"""
        if not self._fts_enabled:
            return
        try:
            with self._connect() as connection:
                connection.execute("DELETE FROM memory_items_fts")
                rows = connection.execute(
                    """
                    SELECT id, text FROM memory_items
                    WHERE status = ?
                    """,
                    (MemoryItemStatus.ACTIVE.value,),
                ).fetchall()
                connection.executemany(
                    "INSERT INTO memory_items_fts (item_id, text) VALUES (?, ?)",
                    [(row["id"], row["text"]) for row in rows],
                )
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory 索引重建失败", path=str(self.path)) from exc

    def add_episode(self, episode: MemoryEpisode, *, event: MemoryEvent) -> MemoryEpisode:
        """保存 L1 片段记忆和对应审计事件。"""
        try:
            with self._connect() as connection:
                self._insert_episode(connection, episode)
                self._insert_event(connection, event)
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory episode 写入失败", path=str(self.path)) from exc
        return episode

    def add_item(self, item: MemoryItem, *, event: MemoryEvent) -> MemoryItem:
        """保存 L2 长期记忆条目和对应审计事件。"""
        try:
            with self._connect() as connection:
                self._upsert_item(connection, item)
                self._refresh_fts_row(connection, item)
                self._insert_event(connection, event)
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory item 写入失败", path=str(self.path)) from exc
        return item

    def update_item(
        self,
        item_id: str,
        scope: MemoryScope,
        patch: MemoryItemPatch,
        *,
        event: MemoryEvent,
    ) -> MemoryItem:
        """更新长期记忆条目并记录审计事件。"""
        current = self.get_item(item_id, scope)
        if current is None:
            raise IrisMemoryError("记忆条目不存在", item_id=item_id)
        updates = patch.model_dump(exclude_unset=True)
        if not updates:
            return current
        updates["updated_at"] = _now_iso()
        updated = current.model_copy(update=updates)
        try:
            with self._connect() as connection:
                self._upsert_item(connection, updated)
                self._refresh_fts_row(connection, updated)
                self._insert_event(connection, event)
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory item 更新失败", path=str(self.path)) from exc
        return updated

    def delete_item(self, item_id: str, scope: MemoryScope, *, event: MemoryEvent) -> None:
        """将长期记忆条目标记为删除并记录审计事件。"""
        try:
            with self._connect() as connection:
                current = self._fetch_item(connection, item_id, scope, include_deleted=False)
                if current is None:
                    return
                deleted = current.model_copy(
                    update={
                        "status": MemoryItemStatus.DELETED,
                        "updated_at": _now_iso(),
                        "deleted_at": _now_iso(),
                    }
                )
                self._upsert_item(connection, deleted)
                self._delete_fts_row(connection, item_id)
                self._insert_event(connection, event)
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory item 删除失败", path=str(self.path)) from exc

    def get_item(self, item_id: str, scope: MemoryScope) -> MemoryItem | None:
        """读取指定 scope 下的活跃长期记忆条目。"""
        try:
            with self._connect() as connection:
                return self._fetch_item(connection, item_id, scope, include_deleted=False)
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory item 读取失败", path=str(self.path)) from exc

    def search(self, query: MemoryQuery) -> list[MemorySearchResult]:
        """按查询条件召回长期记忆。"""
        if query.text and self._fts_enabled and not query.item_ids:
            fts_results = self._search_fts(query)
            if fts_results:
                return fts_results
        return self._search_fallback(query)

    def list_items(
        self,
        scope: MemoryScope,
        *,
        limit: int = 50,
        include_deleted: bool = False,
    ) -> list[MemoryItem]:
        """列出指定 scope 下的长期记忆条目。"""
        safe_limit = _safe_limit(limit)
        clause, params = _scope_clause(scope)
        sql = f"SELECT * FROM memory_items WHERE {clause}"
        if not include_deleted:
            sql += " AND status = ?"
            params.append(MemoryItemStatus.ACTIVE.value)
        sql += " ORDER BY updated_at DESC, id DESC LIMIT ?"
        params.append(safe_limit)
        try:
            with self._connect() as connection:
                rows = connection.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory item 列表读取失败", path=str(self.path)) from exc
        return [_row_to_item(row) for row in rows]

    def list_events(
        self,
        scope: MemoryScope,
        *,
        item_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        """列出指定 scope 下的审计事件。"""
        safe_limit = _safe_limit(limit)
        clause, params = _scope_clause(scope)
        sql = f"SELECT * FROM memory_events WHERE {clause}"
        if item_id is not None:
            sql += " AND item_id = ?"
            params.append(item_id)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(safe_limit)
        try:
            with self._connect() as connection:
                rows = connection.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory event 列表读取失败", path=str(self.path)) from exc
        return [_row_to_event(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        """创建启用 row factory 的 SQLite 连接。"""
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _insert_episode(self, connection: sqlite3.Connection, episode: MemoryEpisode) -> None:
        """插入 L1 片段记忆。"""
        scope_values = _scope_values(episode.scope)
        connection.execute(
            """
            INSERT INTO memory_episodes (
                id,
                scope_workspace_id,
                scope_agent_id,
                scope_collection,
                scope_visibility,
                scope_session_id,
                source_type,
                source_id,
                text,
                category,
                artifacts_json,
                metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode.id,
                *scope_values,
                episode.source_type.value,
                episode.source_id,
                episode.text,
                episode.category.value,
                _dump_json([artifact.model_dump(mode="json") for artifact in episode.artifacts]),
                _dump_json(episode.metadata),
                episode.created_at,
            ),
        )

    def _upsert_item(self, connection: sqlite3.Connection, item: MemoryItem) -> None:
        """插入或替换 L2 长期记忆条目。"""
        scope_values = _scope_values(item.scope)
        connection.execute(
            """
            INSERT INTO memory_items (
                id,
                scope_workspace_id,
                scope_agent_id,
                scope_collection,
                scope_visibility,
                scope_session_id,
                episode_id,
                level,
                category,
                kind,
                text,
                status,
                source_type,
                source_id,
                reason,
                confidence,
                importance,
                artifacts_json,
                metadata_json,
                created_at,
                updated_at,
                deleted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                episode_id = excluded.episode_id,
                level = excluded.level,
                category = excluded.category,
                kind = excluded.kind,
                text = excluded.text,
                status = excluded.status,
                source_type = excluded.source_type,
                source_id = excluded.source_id,
                reason = excluded.reason,
                confidence = excluded.confidence,
                importance = excluded.importance,
                artifacts_json = excluded.artifacts_json,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at,
                deleted_at = excluded.deleted_at
            """,
            (
                item.id,
                *scope_values,
                item.episode_id,
                item.level.value,
                item.category.value,
                item.kind.value,
                item.text,
                item.status.value,
                item.source_type.value,
                item.source_id,
                item.reason,
                item.confidence,
                item.importance,
                _dump_json([artifact.model_dump(mode="json") for artifact in item.artifacts]),
                _dump_json(item.metadata),
                item.created_at,
                item.updated_at,
                item.deleted_at,
            ),
        )

    def _insert_event(self, connection: sqlite3.Connection, event: MemoryEvent) -> None:
        """插入审计事件。"""
        scope_values = _scope_values(event.scope)
        connection.execute(
            """
            INSERT INTO memory_events (
                id,
                scope_workspace_id,
                scope_agent_id,
                scope_collection,
                scope_visibility,
                scope_session_id,
                event_type,
                actor,
                item_id,
                episode_id,
                reason,
                metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                *scope_values,
                event.event_type.value,
                event.actor.value,
                event.item_id,
                event.episode_id,
                event.reason,
                _dump_json(event.metadata),
                event.created_at,
            ),
        )

    def _fetch_item(
        self,
        connection: sqlite3.Connection,
        item_id: str,
        scope: MemoryScope,
        *,
        include_deleted: bool,
    ) -> MemoryItem | None:
        """在指定 scope 下读取一个条目。"""
        clause, params = _scope_clause(scope)
        sql = f"SELECT * FROM memory_items WHERE id = ? AND {clause}"
        query_params: list[Any] = [item_id, *params]
        if not include_deleted:
            sql += " AND status = ?"
            query_params.append(MemoryItemStatus.ACTIVE.value)
        row = connection.execute(sql, query_params).fetchone()
        if row is None:
            return None
        return _row_to_item(row)

    def _refresh_fts_row(self, connection: sqlite3.Connection, item: MemoryItem) -> None:
        """刷新单条 FTS 索引。"""
        if not self._fts_enabled:
            return
        self._delete_fts_row(connection, item.id)
        if item.status == MemoryItemStatus.ACTIVE:
            connection.execute(
                "INSERT INTO memory_items_fts (item_id, text) VALUES (?, ?)",
                (item.id, item.text),
            )

    def _delete_fts_row(self, connection: sqlite3.Connection, item_id: str) -> None:
        """删除单条 FTS 索引。"""
        if not self._fts_enabled:
            return
        connection.execute("DELETE FROM memory_items_fts WHERE item_id = ?", (item_id,))

    def _search_fts(self, query: MemoryQuery) -> list[MemorySearchResult] | None:
        """使用 FTS5 搜索；查询语法不兼容时返回 None 交给 fallback。"""
        clause, params = _query_clause(query, item_alias="i")
        sql = f"""
            SELECT i.*, bm25(memory_items_fts) AS rank
            FROM memory_items_fts
            JOIN memory_items i ON i.id = memory_items_fts.item_id
            WHERE memory_items_fts MATCH ? AND {clause}
            ORDER BY rank ASC
            LIMIT ?
        """
        try:
            with self._connect() as connection:
                rows = connection.execute(sql, [query.text, *params, query.limit]).fetchall()
        except sqlite3.Error:
            return None
        return [
            MemorySearchResult(
                item=_row_to_item(row),
                score=float(row["rank"]),
                source="sqlite_fts",
                matched_text=row["text"],
            )
            for row in rows
        ]

    def _search_fallback(self, query: MemoryQuery) -> list[MemorySearchResult]:
        """使用 item id、LIKE 或 recent ordering 的 SQLite fallback 搜索。"""
        clause, params = _query_clause(query)
        sql = f"SELECT * FROM memory_items WHERE {clause}"
        if query.item_ids:
            placeholders = _placeholders(query.item_ids)
            sql += f" AND id IN ({placeholders})"
            params.extend(query.item_ids)
        elif query.text:
            sql += " AND text LIKE ?"
            params.append(f"%{query.text}%")
        sql += " ORDER BY updated_at DESC, id DESC LIMIT ?"
        params.append(query.limit)
        try:
            with self._connect() as connection:
                rows = connection.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory 搜索失败", path=str(self.path)) from exc
        return [
            MemorySearchResult(
                item=_row_to_item(row),
                score=1.0 if query.text and query.text in row["text"] else 0.0,
                source="sqlite_fallback",
                matched_text=row["text"],
            )
            for row in rows
        ]


def _query_clause(query: MemoryQuery, *, item_alias: str = "") -> tuple[str, list[Any]]:
    """生成查询 SQL 条件。"""
    clause, params = _scope_clause(query.scope, alias=item_alias)
    if not query.include_deleted:
        clause += f" AND {_column('status', item_alias)} = ?"
        params.append(MemoryItemStatus.ACTIVE.value)
    if query.categories:
        clause += f" AND {_column('category', item_alias)} IN ({_placeholders(query.categories)})"
        params.extend(category.value for category in query.categories)
    if query.kinds:
        clause += f" AND {_column('kind', item_alias)} IN ({_placeholders(query.kinds)})"
        params.extend(kind.value for kind in query.kinds)
    return clause, params


def _scope_clause(scope: MemoryScope, *, alias: str = "") -> tuple[str, list[Any]]:
    """生成完整 scope 隔离 SQL 条件。"""
    columns = [
        "scope_workspace_id",
        "scope_agent_id",
        "scope_collection",
        "scope_visibility",
        "scope_session_id",
    ]
    clause = " AND ".join(f"{_column(column, alias)} = ?" for column in columns)
    return clause, _scope_values(scope)


def _column(name: str, alias: str) -> str:
    """按需生成带表别名的列名。"""
    if not alias:
        return name
    return f"{alias}.{name}"


def _scope_values(scope: MemoryScope) -> list[str]:
    """将 scope 转换为 SQLite 存储值。"""
    return [
        scope.workspace_id,
        scope.agent_id,
        scope.collection,
        scope.visibility.value,
        scope.session_id or "",
    ]


def _row_to_item(row: sqlite3.Row) -> MemoryItem:
    """将 SQLite row 转换为长期记忆条目。"""
    return MemoryItem(
        id=row["id"],
        scope=_row_to_scope(row),
        episode_id=row["episode_id"],
        level=MemoryLevel(row["level"]),
        category=MemoryCategory(row["category"]),
        kind=MemoryItemKind(row["kind"]),
        text=row["text"],
        status=MemoryItemStatus(row["status"]),
        source_type=MemorySourceType(row["source_type"]),
        source_id=row["source_id"],
        reason=row["reason"],
        confidence=row["confidence"],
        importance=row["importance"],
        artifacts=_load_artifacts(row["artifacts_json"]),
        metadata=_load_metadata(row["metadata_json"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        deleted_at=row["deleted_at"],
    )


def _row_to_event(row: sqlite3.Row) -> MemoryEvent:
    """将 SQLite row 转换为审计事件。"""
    return MemoryEvent(
        id=row["id"],
        scope=_row_to_scope(row),
        event_type=MemoryEventType(row["event_type"]),
        actor=MemoryActor(row["actor"]),
        item_id=row["item_id"],
        episode_id=row["episode_id"],
        reason=row["reason"],
        metadata=_load_metadata(row["metadata_json"]),
        created_at=row["created_at"],
    )


def _row_to_scope(row: sqlite3.Row) -> MemoryScope:
    """从 row 的 scope 字段恢复 MemoryScope。"""
    session_id = row["scope_session_id"] or None
    return MemoryScope(
        workspace_id=row["scope_workspace_id"],
        agent_id=row["scope_agent_id"],
        collection=row["scope_collection"],
        visibility=MemoryVisibility(row["scope_visibility"]),
        session_id=session_id,
    )


def _load_artifacts(value: str) -> list[MemoryArtifactRef]:
    """反序列化 artifact 引用列表。"""
    raw = cast(list[dict[str, Any]], json.loads(value))
    return [MemoryArtifactRef.model_validate(item) for item in raw]


def _load_metadata(value: str) -> dict[str, Any]:
    """反序列化 metadata 字典。"""
    return cast(dict[str, Any], json.loads(value))


def _dump_json(value: Any) -> str:
    """序列化 JSON 字段并统一转换错误类型。"""
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError as exc:
        raise IrisMemoryError("Memory 数据必须可 JSON 序列化") from exc


def _safe_limit(limit: int) -> int:
    """限制列表读取的最大返回数量。"""
    return min(max(limit, 1), 100)


def _placeholders(values: Sequence[object]) -> str:
    """生成 SQL IN 子句占位符。"""
    return ", ".join("?" for _ in values)
