"""SQLite 记忆存储实现。

SQLite 是长期记忆的权威存储；schema v4 保留 FTS5 并保存 namespace 文件投影版本。

Example:
    store = SQLiteMemoryStore(".iris/memory/memory.db")
    results = store.search(MemorySearchQuery(query="用户偏好"), ["project"])
"""

# region imports
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar, cast

from ..exceptions import IrisMemoryError
from ._query import make_snippet, prepare_fts_query, tokenize_text
from .models import (
    MemoryActor,
    MemoryArtifactRef,
    MemoryCandidate,
    MemoryCandidateStatus,
    MemoryCategory,
    MemoryEpisode,
    MemoryEvent,
    MemoryEventType,
    MemoryItem,
    MemoryItemKind,
    MemoryItemPatch,
    MemoryItemStatus,
    MemoryLevel,
    MemoryNamespaceSnapshot,
    MemoryNamespaceState,
    MemorySearchHit,
    MemorySearchQuery,
    MemorySearchResponse,
    MemorySourceType,
    _now_iso,
)

# endregion

PublicationT = TypeVar("PublicationT")


class SQLiteMemoryStore:
    """基于本地 SQLite 文件的长期记忆权威 store。

    Args:
        path: SQLite 数据库文件路径。
    """

    def __init__(self, path: str | Path) -> None:
        """初始化 SQLite store；只接受新空库或 memory schema v4。"""
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise IrisMemoryError("SQLite memory 目录创建失败", path=str(self.path)) from exc
        self.initialize_schema()

    def initialize_schema(self) -> None:
        """创建 schema v4；旧版本在任何结构写入前明确拒绝。"""
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                tables = {
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                if tables:
                    if "memory_schema" not in tables:
                        raise IrisMemoryError("SQLite memory 缺少版本记录", path=str(self.path))
                    version = connection.execute(
                        "SELECT value FROM memory_schema WHERE key = 'schema_version'"
                    ).fetchone()
                    if version is None or version["value"] != "4":
                        raise IrisMemoryError(
                            "SQLite memory 版本不受支持，要求 schema version 4",
                            path=str(self.path),
                            version=None if version is None else version["value"],
                        )
                    return
                connection.execute("""
                    CREATE TABLE memory_schema (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                    """)
                connection.execute("""
                    CREATE TABLE memory_episodes (
                        id TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        source_id TEXT NOT NULL,
                        text TEXT NOT NULL,
                        category TEXT NOT NULL,
                        artifacts_json TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """)
                connection.execute("""
                    CREATE TABLE memory_items (
                        id TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL,
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
                    """)
                connection.execute("""
                    CREATE TABLE memory_events (
                        id TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        actor TEXT NOT NULL,
                        item_id TEXT,
                        episode_id TEXT,
                        reason TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """)
                connection.execute("""
                    CREATE TABLE memory_candidates (
                        id TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        episode_ids_json TEXT NOT NULL,
                        category TEXT NOT NULL,
                        suggested_level TEXT NOT NULL,
                        text TEXT NOT NULL,
                        confidence REAL,
                        importance REAL,
                        reason TEXT NOT NULL,
                        status TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """)
                connection.execute("""
                    CREATE INDEX idx_memory_candidates_namespace_status
                    ON memory_candidates (
                        namespace,
                        status,
                        created_at
                    )
                    """)
                connection.execute("""
                    CREATE INDEX idx_memory_items_namespace_status_updated
                    ON memory_items (
                        namespace,
                        status,
                        updated_at DESC,
                        id DESC
                    )
                    """)
                connection.execute("""
                    CREATE INDEX idx_memory_events_namespace_created
                    ON memory_events (
                        namespace,
                        created_at DESC,
                        id DESC
                    )
                    """)
                connection.execute(
                    "INSERT INTO memory_schema (key, value) VALUES ('schema_version', '4')"
                )
                connection.execute("""
                    CREATE VIRTUAL TABLE memory_items_fts USING fts5(
                        item_id UNINDEXED,
                        text,
                        tokenize='unicode61 remove_diacritics 0'
                    )
                    """)
                connection.execute("""
                    CREATE TABLE memory_namespace_state (
                        namespace TEXT PRIMARY KEY,
                        item_revision INTEGER NOT NULL,
                        projection_revision INTEGER
                    )
                    """)
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory 初始化失败", path=str(self.path)) from exc

    def rebuild_index(self) -> None:
        """用统一词法重建全部条目的 FTS 索引。"""
        try:
            with self._connection() as connection:
                self._rebuild_fts(connection)
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory 索引重建失败", path=str(self.path)) from exc

    def add_episode(self, episode: MemoryEpisode, *, event: MemoryEvent) -> MemoryEpisode:
        """保存 L1 片段记忆和对应审计事件。"""
        try:
            with self._connection() as connection:
                self._insert_episode(connection, episode)
                self._insert_event(connection, event)
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory episode 写入失败", path=str(self.path)) from exc
        return episode

    def add_item(self, item: MemoryItem, *, event: MemoryEvent) -> MemoryItem:
        """保存 L2 长期记忆条目和对应审计事件。"""
        try:
            with self._connection() as connection:
                self._ensure_new_item_id(connection, item)
                self._upsert_item(connection, item)
                if item.level == MemoryLevel.SEMANTIC and item.status == MemoryItemStatus.ACTIVE:
                    self._advance_item_revision(connection, item.namespace)
                self._insert_event(connection, event)
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory item 写入失败", path=str(self.path)) from exc
        return item

    def update_item(
        self,
        item_id: str,
        namespace: str,
        patch: MemoryItemPatch,
        *,
        event: MemoryEvent,
    ) -> MemoryItem:
        """在同一写事务中读取、更新长期记忆条目并记录审计事件。"""
        updates = {field: getattr(patch, field) for field in patch.model_fields_set}
        try:
            with self._connection() as connection:
                # 读取前取得写事务，避免不同连接用旧快照覆盖彼此的字段修改。
                connection.execute("BEGIN IMMEDIATE")
                current = self._fetch_item(connection, item_id, namespace, include_deleted=False)
                if current is None:
                    raise IrisMemoryError("记忆条目不存在", item_id=item_id)
                if not updates or all(
                    getattr(current, key) == value for key, value in updates.items()
                ):
                    return current
                updates["updated_at"] = _now_iso()
                updated = current.model_copy(update=updates)
                self._upsert_item(connection, updated)
                if updated.level == MemoryLevel.SEMANTIC:
                    self._advance_item_revision(connection, namespace)
                self._insert_event(connection, event)
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory item 更新失败", path=str(self.path)) from exc
        return updated

    def delete_item(self, item_id: str, namespace: str, *, event: MemoryEvent) -> bool:
        """将长期记忆条目标记为删除并记录审计事件，返回是否实际删除。"""
        try:
            with self._connection() as connection:
                # 先串行化写入，避免软删除用旧正文覆盖更新或重复记录实际删除。
                connection.execute("BEGIN IMMEDIATE")
                current = self._fetch_item(connection, item_id, namespace, include_deleted=False)
                if current is None:
                    return False
                deleted = current.model_copy(
                    update={
                        "status": MemoryItemStatus.DELETED,
                        "updated_at": _now_iso(),
                        "deleted_at": _now_iso(),
                    }
                )
                self._upsert_item(connection, deleted)
                if current.level == MemoryLevel.SEMANTIC:
                    self._advance_item_revision(connection, namespace)
                self._insert_event(connection, event)
                return True
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory item 删除失败", path=str(self.path)) from exc

    def get_item(self, item_id: str, namespaces: Sequence[str]) -> MemoryItem | None:
        """在联合 namespace 范围内读取指定活跃条目。"""
        try:
            with self._connection() as connection:
                clause, params = _namespaces_clause(namespaces)
                row = connection.execute(
                    f"SELECT * FROM memory_items WHERE id = ? AND {clause} AND status = ?",
                    [item_id, *params, MemoryItemStatus.ACTIVE.value],
                ).fetchone()
                return None if row is None else _row_to_item(row)
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory item 读取失败", path=str(self.path)) from exc

    def search(
        self, query: MemorySearchQuery, namespaces: Sequence[str]
    ) -> MemorySearchResponse:
        """在允许范围内用完整词法检索，并以多读一条判断剩余候选。"""
        terms = tokenize_text(query.query)
        if not namespaces or not terms:
            return MemorySearchResponse((), False)
        clause, params = _query_clause(query, namespaces, item_alias="i")
        sql = f"""
            SELECT i.*, bm25(memory_items_fts) AS rank
            FROM memory_items_fts
            JOIN memory_items i ON i.id = memory_items_fts.item_id
            WHERE memory_items_fts MATCH ? AND {clause}
            ORDER BY rank ASC, i.updated_at DESC, i.id DESC
            LIMIT ?
        """
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    sql,
                    [prepare_fts_query(terms, query.required_terms), *params, query.limit + 1],
                ).fetchall()
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory 搜索失败", path=str(self.path)) from exc
        query_terms = frozenset(terms)
        hits: list[MemorySearchHit] = []
        for row in rows[: query.limit]:
            item = _row_to_item(row)
            snippet, is_complete = make_snippet(item.text, query_terms)
            hits.append(
                MemorySearchHit(
                    item.id, item.namespace, item.category, item.kind, snippet, is_complete
                )
            )
        return MemorySearchResponse(tuple(hits), len(rows) > query.limit)

    def read_namespace_state(self, namespace: str) -> MemoryNamespaceState:
        """读取 namespace 的当前条目与完整投影版本。"""
        try:
            with self._connection() as connection:
                return self._read_namespace_state(connection, namespace)
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory 版本读取失败", path=str(self.path)) from exc

    def read_namespace_snapshot(self, namespace: str) -> MemoryNamespaceSnapshot:
        """在同一读事务内取得完整 active L2 条目及其版本。"""
        try:
            with self._connection() as connection:
                connection.execute("BEGIN")
                return self._read_namespace_snapshot(connection, namespace)
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory 快照读取失败", path=str(self.path)) from exc

    def publish_projection(
        self, namespace: str, publish: Callable[[MemoryNamespaceSnapshot], None]
    ) -> MemoryNamespaceState:
        """在短写事务内现读并发布全部正文，成功后推进投影版本。"""
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                snapshot = self._read_namespace_snapshot(connection, namespace)
                publish(snapshot)
                revision = snapshot.state.item_revision
                connection.execute(
                    """
                    INSERT INTO memory_namespace_state
                        (namespace, item_revision, projection_revision)
                    VALUES (?, ?, ?)
                    ON CONFLICT(namespace) DO UPDATE SET
                        projection_revision = excluded.projection_revision
                    """,
                    (namespace, revision, revision),
                )
                return MemoryNamespaceState(namespace, revision, revision)
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory 正文发布失败", path=str(self.path)) from exc

    def publish_overview(  # noqa: UP047
        self, namespace: str, publish: Callable[[MemoryNamespaceState], PublicationT]
    ) -> PublicationT:
        """串行化完整概览文件的版本比较与发布，不覆盖模型调用。"""
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                return publish(self._read_namespace_state(connection, namespace))
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory 概览发布失败", path=str(self.path)) from exc

    def list_items(
        self,
        namespaces: Sequence[str],
        *,
        limit: int | None = 50,
        include_deleted: bool = False,
        categories: Sequence[MemoryCategory] | None = None,
        kinds: Sequence[MemoryItemKind] | None = None,
    ) -> list[MemoryItem]:
        """列出指定 namespace 下的长期记忆条目。"""
        clause, params = _namespaces_clause(namespaces)
        sql = f"SELECT * FROM memory_items WHERE {clause}"
        if not include_deleted:
            sql += " AND status = ?"
            params.append(MemoryItemStatus.ACTIVE.value)
        if categories:
            sql += f" AND category IN ({_placeholders(categories)})"
            params.extend(category.value for category in categories)
        if kinds:
            sql += f" AND kind IN ({_placeholders(kinds)})"
            params.extend(kind.value for kind in kinds)
        sql += " ORDER BY updated_at DESC, id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(_validated_list_limit(limit))
        try:
            with self._connection() as connection:
                rows = connection.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory item 列表读取失败", path=str(self.path)) from exc
        return [_row_to_item(row) for row in rows]

    def list_events(
        self,
        namespace: str,
        *,
        item_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        """列出指定 namespace 下的审计事件。"""
        safe_limit = _validated_list_limit(limit)
        clause, params = _namespace_clause(namespace)
        sql = f"SELECT * FROM memory_events WHERE {clause}"
        if item_id is not None:
            sql += " AND item_id = ?"
            params.append(item_id)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(safe_limit)
        try:
            with self._connection() as connection:
                rows = connection.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory event 列表读取失败", path=str(self.path)) from exc
        return [_row_to_event(row) for row in rows]

    def add_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        event: MemoryEvent,
    ) -> MemoryCandidate:
        """保存候选记忆和对应审计事件。"""
        try:
            with self._connection() as connection:
                self._ensure_new_candidate_id(connection, candidate)
                self._upsert_candidate(connection, candidate)
                self._insert_event(connection, event)
        except sqlite3.Error as exc:
            raise IrisMemoryError(
                "SQLite memory candidate 写入失败",
                path=str(self.path),
            ) from exc
        return candidate

    def list_candidates(
        self,
        namespace: str,
        *,
        status: MemoryCandidateStatus | None = None,
        limit: int = 50,
    ) -> list[MemoryCandidate]:
        """列出指定 namespace 下的候选记忆。"""
        safe_limit = _validated_list_limit(limit)
        clause, params = _namespace_clause(namespace)
        sql = f"SELECT * FROM memory_candidates WHERE {clause}"
        if status is not None:
            sql += " AND status = ?"
            params.append(status.value)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(safe_limit)
        try:
            with self._connection() as connection:
                rows = connection.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise IrisMemoryError(
                "SQLite memory candidate 列表读取失败",
                path=str(self.path),
            ) from exc
        return [_row_to_candidate(row) for row in rows]

    def update_candidate_status(
        self,
        candidate_id: str,
        namespace: str,
        status: MemoryCandidateStatus,
        *,
        event: MemoryEvent,
    ) -> MemoryCandidate:
        """更新候选记忆状态并记录审计事件。"""
        try:
            with self._connection() as connection:
                current = self._fetch_candidate(connection, candidate_id, namespace)
                if current is None:
                    raise IrisMemoryError("候选记忆不存在", candidate_id=candidate_id)
                updated = current.model_copy(update={"status": status})
                self._upsert_candidate(connection, updated)
                self._insert_event(connection, event)
        except sqlite3.Error as exc:
            raise IrisMemoryError(
                "SQLite memory candidate 状态更新失败",
                path=str(self.path),
            ) from exc
        return updated

    def promote_candidate(
        self,
        candidate_id: str,
        namespace: str,
        *,
        kind: MemoryItemKind,
        actor: MemoryActor,
        reason: str,
    ) -> MemoryItem:
        """在串行写事务中将 pending candidate 晋升为 L2 item。"""
        try:
            with self._connection() as connection:
                # 同一候选的并发晋升必须在读取状态前排队，后继调用复用已晋升条目。
                connection.execute("BEGIN IMMEDIATE")
                candidate = self._fetch_candidate(connection, candidate_id, namespace)
                if candidate is None:
                    raise IrisMemoryError("候选记忆不存在", candidate_id=candidate_id)
                if candidate.status == MemoryCandidateStatus.ACCEPTED:
                    existing = self._fetch_item_by_source_id(connection, candidate_id, namespace)
                    if existing is None:
                        raise IrisMemoryError("已接受候选缺少晋升条目", candidate_id=candidate_id)
                    return existing
                if candidate.status != MemoryCandidateStatus.PENDING:
                    raise IrisMemoryError(
                        "候选记忆不可晋升",
                        candidate_id=candidate_id,
                        status=candidate.status.value,
                    )

                item = _item_from_candidate(candidate, kind=kind)
                accepted = candidate.model_copy(
                    update={"status": MemoryCandidateStatus.ACCEPTED},
                )
                add_event = MemoryEvent(
                    namespace=namespace,
                    event_type=MemoryEventType.ADD,
                    actor=actor,
                    item_id=item.id,
                    episode_id=item.episode_id,
                    reason=candidate.reason,
                    metadata={
                        "candidate_id": candidate.id,
                        "episode_ids": candidate.episode_ids,
                    },
                )
                accept_event = MemoryEvent(
                    namespace=namespace,
                    event_type=MemoryEventType.CANDIDATE_ACCEPT,
                    actor=actor,
                    item_id=item.id,
                    episode_id=item.episode_id,
                    reason=reason,
                    metadata={
                        "candidate_id": candidate.id,
                        "candidate_status": MemoryCandidateStatus.ACCEPTED.value,
                        "promoted_item_id": item.id,
                    },
                )

                self._ensure_new_item_id(connection, item)
                self._upsert_item(connection, item)
                self._advance_item_revision(connection, namespace)
                self._upsert_candidate(connection, accepted)
                self._insert_event(connection, add_event)
                self._insert_event(connection, accept_event)
                return item
        except sqlite3.Error as exc:
            raise IrisMemoryError(
                "SQLite memory candidate 晋升失败",
                path=str(self.path),
                candidate_id=candidate_id,
            ) from exc

    def _connect(self) -> sqlite3.Connection:
        """创建启用 row factory 的 SQLite 连接。"""
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _advance_item_revision(self, connection: sqlite3.Connection, namespace: str) -> None:
        """与有效 L2 写入在同一事务内推进 namespace 版本。"""
        connection.execute(
            """
            INSERT INTO memory_namespace_state (namespace, item_revision, projection_revision)
            VALUES (?, 1, NULL)
            ON CONFLICT(namespace) DO UPDATE SET item_revision = item_revision + 1
            """,
            (namespace,),
        )

    def _read_namespace_state(
        self, connection: sqlite3.Connection, namespace: str
    ) -> MemoryNamespaceState:
        """读取当前事务中的版本状态，尚无条目时返回初始版本。"""
        row = connection.execute(
            "SELECT item_revision, projection_revision FROM memory_namespace_state "
            "WHERE namespace = ?",
            (namespace,),
        ).fetchone()
        if row is None:
            return MemoryNamespaceState(namespace)
        return MemoryNamespaceState(namespace, row["item_revision"], row["projection_revision"])

    def _read_namespace_snapshot(
        self, connection: sqlite3.Connection, namespace: str
    ) -> MemoryNamespaceSnapshot:
        """在调用方持有的事务内按分类和类型读取全部有效 L2 条目。"""
        state = self._read_namespace_state(connection, namespace)
        rows = connection.execute(
            """
            SELECT * FROM memory_items WHERE namespace = ? AND status = ? AND level = ?
            ORDER BY category, kind, created_at, id
            """,
            (namespace, MemoryItemStatus.ACTIVE.value, MemoryLevel.SEMANTIC.value),
        ).fetchall()
        return MemoryNamespaceSnapshot(state, tuple(_row_to_item(row) for row in rows))

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """创建带事务边界且退出时显式关闭的 SQLite 连接。"""
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _insert_episode(self, connection: sqlite3.Connection, episode: MemoryEpisode) -> None:
        """插入 L1 片段记忆。"""
        connection.execute(
            """
            INSERT INTO memory_episodes (
                id,
                namespace,
                source_type,
                source_id,
                text,
                category,
                artifacts_json,
                metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode.id,
                episode.namespace,
                episode.source_type.value,
                episode.source_id,
                episode.text,
                episode.category.value,
                _dump_json([artifact.model_dump(mode="json") for artifact in episode.artifacts]),
                _dump_json(episode.metadata),
                episode.created_at,
            ),
        )

    def _ensure_new_item_id(
        self,
        connection: sqlite3.Connection,
        item: MemoryItem,
    ) -> None:
        """确保新增 item 使用的全局 ID 未被任何 namespace 占用。"""
        existing = self._fetch_item_by_id(connection, item.id)
        if existing is None:
            return
        raise IrisMemoryError(
            "记忆条目 id 已存在",
            item_id=item.id,
            existing_namespace=existing.namespace,
            requested_namespace=item.namespace,
        )

    def _ensure_new_candidate_id(
        self,
        connection: sqlite3.Connection,
        candidate: MemoryCandidate,
    ) -> None:
        """确保新增 candidate 使用的全局 ID 未被任何 namespace 占用。"""
        existing = self._fetch_candidate_by_id(connection, candidate.id)
        if existing is None:
            return
        raise IrisMemoryError(
            "候选记忆 id 已存在",
            candidate_id=candidate.id,
            existing_namespace=existing.namespace,
            requested_namespace=candidate.namespace,
        )

    def _upsert_item(self, connection: sqlite3.Connection, item: MemoryItem) -> None:
        """插入或替换 L2 长期记忆条目。"""
        connection.execute(
            """
            INSERT INTO memory_items (
                id,
                namespace,
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                item.namespace,
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
        self._refresh_fts_row(connection, item)

    def _upsert_candidate(
        self,
        connection: sqlite3.Connection,
        candidate: MemoryCandidate,
    ) -> None:
        """插入或替换候选记忆。"""
        connection.execute(
            """
            INSERT INTO memory_candidates (
                id,
                namespace,
                episode_ids_json,
                category,
                suggested_level,
                text,
                confidence,
                importance,
                reason,
                status,
                metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                episode_ids_json = excluded.episode_ids_json,
                category = excluded.category,
                suggested_level = excluded.suggested_level,
                text = excluded.text,
                confidence = excluded.confidence,
                importance = excluded.importance,
                reason = excluded.reason,
                status = excluded.status,
                metadata_json = excluded.metadata_json
            """,
            (
                candidate.id,
                candidate.namespace,
                _dump_json(candidate.episode_ids),
                candidate.category.value,
                candidate.suggested_level.value,
                candidate.text,
                candidate.confidence,
                candidate.importance,
                candidate.reason,
                candidate.status.value,
                _dump_json(candidate.metadata),
                candidate.created_at,
            ),
        )

    def _insert_event(self, connection: sqlite3.Connection, event: MemoryEvent) -> None:
        """插入审计事件。"""
        connection.execute(
            """
            INSERT INTO memory_events (
                id,
                namespace,
                event_type,
                actor,
                item_id,
                episode_id,
                reason,
                metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.namespace,
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
        namespace: str,
        *,
        include_deleted: bool,
    ) -> MemoryItem | None:
        """在指定 namespace 下读取一个条目。"""
        clause, params = _namespace_clause(namespace)
        sql = f"SELECT * FROM memory_items WHERE id = ? AND {clause}"
        query_params: list[Any] = [item_id, *params]
        if not include_deleted:
            sql += " AND status = ?"
            query_params.append(MemoryItemStatus.ACTIVE.value)
        row = connection.execute(sql, query_params).fetchone()
        if row is None:
            return None
        return _row_to_item(row)

    def _fetch_item_by_id(
        self,
        connection: sqlite3.Connection,
        item_id: str,
    ) -> MemoryItem | None:
        """不带 namespace 地按全局 ID 读取 item，用于新增前冲突检测。"""
        row = connection.execute(
            "SELECT * FROM memory_items WHERE id = ? LIMIT 1",
            (item_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_item(row)

    def _fetch_item_by_source_id(
        self,
        connection: sqlite3.Connection,
        source_id: str,
        namespace: str,
    ) -> MemoryItem | None:
        """按 source_id 在指定 namespace 下查找活跃 item，用于 promotion 重试。"""
        clause, params = _namespace_clause(namespace)
        row = connection.execute(
            f"""
            SELECT * FROM memory_items
            WHERE source_id = ? AND {clause} AND status = ?
            ORDER BY created_at ASC, id ASC
            LIMIT 1
            """,
            [source_id, *params, MemoryItemStatus.ACTIVE.value],
        ).fetchone()
        if row is None:
            return None
        return _row_to_item(row)

    def _fetch_candidate(
        self,
        connection: sqlite3.Connection,
        candidate_id: str,
        namespace: str,
    ) -> MemoryCandidate | None:
        """在指定 namespace 下读取一个候选记忆。"""
        clause, params = _namespace_clause(namespace)
        sql = f"SELECT * FROM memory_candidates WHERE id = ? AND {clause}"
        row = connection.execute(sql, [candidate_id, *params]).fetchone()
        if row is None:
            return None
        return _row_to_candidate(row)

    def _fetch_candidate_by_id(
        self,
        connection: sqlite3.Connection,
        candidate_id: str,
    ) -> MemoryCandidate | None:
        """不带 namespace 地按全局 ID 读取 candidate，用于新增前冲突检测。"""
        row = connection.execute(
            "SELECT * FROM memory_candidates WHERE id = ? LIMIT 1",
            (candidate_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_candidate(row)

    def _rebuild_fts(self, connection: sqlite3.Connection) -> None:
        """保留全部条目与词项频次，状态过滤由查询负责。"""
        connection.execute("DELETE FROM memory_items_fts")
        rows = connection.execute("SELECT id, text FROM memory_items").fetchall()
        connection.executemany(
            "INSERT INTO memory_items_fts (item_id, text) VALUES (?, ?)",
            [(row["id"], " ".join(tokenize_text(row["text"]))) for row in rows],
        )

    def _refresh_fts_row(self, connection: sqlite3.Connection, item: MemoryItem) -> None:
        """在条目写事务中刷新同一份词法索引，包括已删除状态。"""
        connection.execute("DELETE FROM memory_items_fts WHERE item_id = ?", (item.id,))
        connection.execute(
            "INSERT INTO memory_items_fts (item_id, text) VALUES (?, ?)",
            (item.id, " ".join(tokenize_text(item.text))),
        )


def _query_clause(
    query: MemorySearchQuery, namespaces: Sequence[str], *, item_alias: str = ""
) -> tuple[str, list[Any]]:
    """生成查询 SQL 条件。"""
    clause, params = _namespaces_clause(namespaces, alias=item_alias)
    clause += f" AND {_column('status', item_alias)} = ?"
    params.append(MemoryItemStatus.ACTIVE.value)
    if query.categories:
        clause += f" AND {_column('category', item_alias)} IN ({_placeholders(query.categories)})"
        params.extend(category.value for category in query.categories)
    if query.kinds:
        clause += f" AND {_column('kind', item_alias)} IN ({_placeholders(query.kinds)})"
        params.extend(kind.value for kind in query.kinds)
    return clause, params


def _namespaces_clause(namespaces: Sequence[str], *, alias: str = "") -> tuple[str, list[Any]]:
    """一次 SQL 联合全部读取 namespace。"""
    return f"{_column('namespace', alias)} IN ({_placeholders(namespaces)})", list(namespaces)


def _namespace_clause(namespace: str) -> tuple[str, list[Any]]:
    """生成写入或候选操作绑定的单 namespace 条件。"""
    return "namespace = ?", [namespace]


def _column(name: str, alias: str) -> str:
    """按需生成带表别名的列名。"""
    if not alias:
        return name
    return f"{alias}.{name}"


def _row_to_item(row: sqlite3.Row) -> MemoryItem:
    """将 SQLite row 转换为长期记忆条目。"""
    return MemoryItem(
        id=row["id"],
        namespace=row["namespace"],
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


def _item_from_candidate(candidate: MemoryCandidate, *, kind: MemoryItemKind) -> MemoryItem:
    """从 pending candidate 构造晋升后的 L2 item。"""
    metadata = {
        **candidate.metadata,
        "candidate_id": candidate.id,
        "episode_ids": candidate.episode_ids,
    }
    return MemoryItem(
        namespace=candidate.namespace,
        episode_id=candidate.episode_ids[0],
        category=candidate.category,
        kind=kind,
        text=candidate.text,
        source_type=MemorySourceType.SDK,
        source_id=candidate.id,
        reason=candidate.reason,
        confidence=candidate.confidence,
        importance=candidate.importance,
        metadata=metadata,
    )


def _row_to_candidate(row: sqlite3.Row) -> MemoryCandidate:
    """将 SQLite row 转换为候选记忆。"""
    episode_ids = cast(list[str], json.loads(row["episode_ids_json"]))
    return MemoryCandidate(
        id=row["id"],
        namespace=row["namespace"],
        episode_ids=episode_ids,
        category=MemoryCategory(row["category"]),
        suggested_level=MemoryLevel(row["suggested_level"]),
        text=row["text"],
        confidence=row["confidence"],
        importance=row["importance"],
        reason=row["reason"],
        status=MemoryCandidateStatus(row["status"]),
        metadata=_load_metadata(row["metadata_json"]),
        created_at=row["created_at"],
    )


def _row_to_event(row: sqlite3.Row) -> MemoryEvent:
    """将 SQLite row 转换为审计事件。"""
    return MemoryEvent(
        id=row["id"],
        namespace=row["namespace"],
        event_type=MemoryEventType(row["event_type"]),
        actor=MemoryActor(row["actor"]),
        item_id=row["item_id"],
        episode_id=row["episode_id"],
        reason=row["reason"],
        metadata=_load_metadata(row["metadata_json"]),
        created_at=row["created_at"],
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


def _validated_list_limit(limit: int) -> int:
    """校验公开列表读取的返回数量。"""
    if not 1 <= limit <= 100:
        raise IrisMemoryError("limit 必须在 1 到 100 之间", limit=limit)
    return limit


def _placeholders(values: Sequence[object]) -> str:
    """生成 SQL IN 子句占位符。"""
    return ", ".join("?" for _ in values)
