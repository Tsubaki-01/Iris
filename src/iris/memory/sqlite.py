"""SQLite 记忆权威存储：正式知识、不可变材料和生成阶段原子提交。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from ..exceptions import IrisMemoryError
from ._query import make_snippet, prepare_fts_query, tokenize_text
from .generation_models import (
    DreamChange,
    DreamPlan,
    DreamSnapshot,
    EpisodeCursor,
    EpisodeProgress,
    FlushCommit,
    GenerationResult,
    GenerationState,
    MemoryCaptureSource,
    ObservationState,
)
from .models import (
    MemoryActor,
    MemoryCategory,
    MemoryEpisode,
    MemoryEvent,
    MemoryEventType,
    MemoryEvidenceRef,
    MemoryItem,
    MemoryItemKind,
    MemoryItemPatch,
    MemoryItemStatus,
    MemoryNamespaceSnapshot,
    MemoryNamespaceState,
    MemoryObservation,
    MemorySearchHit,
    MemorySearchQuery,
    MemorySearchResponse,
    MemorySourceType,
    _now_iso,
)

PublicationT = TypeVar("PublicationT")


class SQLiteMemoryStore:
    """保留 FTS 读取面、使用 schema v5 的本地记忆存储。"""

    def __init__(self, path: str | Path) -> None:
        """只接受新空库或当前 schema，不迁移已有数据库。"""
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise IrisMemoryError("SQLite memory 目录创建失败", path=str(self.path)) from exc
        self.initialize_schema()

    def initialize_schema(self) -> None:
        """在写入任何结构前拒绝旧 schema。"""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            tables = {
                row["name"]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if tables:
                if "memory_schema" not in tables:
                    raise IrisMemoryError("SQLite memory 缺少版本记录", path=str(self.path))
                row = connection.execute(
                    "SELECT value FROM memory_schema WHERE key='schema_version'"
                ).fetchone()
                if row is None or row["value"] != "5":
                    raise IrisMemoryError(
                        "SQLite memory 版本不受支持，要求 schema version 5",
                        path=str(self.path),
                        version=None if row is None else row["value"],
                    )
                return
            statements = (
                "CREATE TABLE memory_schema (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
                "INSERT INTO memory_schema VALUES ('schema_version','5')",
                (
                    "CREATE TABLE memory_items (id TEXT PRIMARY KEY, namespace "
                    "TEXT NOT NULL, category TEXT NOT NULL, kind TEXT NOT NULL,"
                    " text TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT"
                    " NOT NULL, updated_at TEXT NOT NULL, payload TEXT NOT "
                    "NULL)"
                ),
                (
                    "CREATE INDEX idx_memory_items_namespace_status_updated ON "
                    "memory_items(namespace,status,updated_at DESC,id DESC)"
                ),
                (
                    "CREATE VIRTUAL TABLE memory_items_fts USING fts5(item_id "
                    "UNINDEXED,text,tokenize='unicode61 remove_diacritics 0')"
                ),
                (
                    "CREATE TABLE memory_events (id TEXT PRIMARY KEY, namespace"
                    " TEXT NOT NULL, item_id TEXT, created_at TEXT NOT NULL, "
                    "payload TEXT NOT NULL)"
                ),
                (
                    "CREATE INDEX idx_memory_events_namespace_created ON "
                    "memory_events(namespace,created_at DESC,id DESC)"
                ),
                (
                    "CREATE TABLE memory_namespace_state (namespace TEXT "
                    "PRIMARY KEY,item_revision INTEGER NOT "
                    "NULL,projection_revision INTEGER)"
                ),
                (
                    "CREATE TABLE memory_episodes (id TEXT PRIMARY "
                    "KEY,namespace TEXT NOT NULL,payload TEXT NOT "
                    "NULL,created_at TEXT NOT NULL,record_index INTEGER NOT "
                    "NULL DEFAULT 0,text_offset INTEGER NOT NULL DEFAULT "
                    "0,flushed INTEGER NOT NULL DEFAULT 0)"
                ),
                (
                    "CREATE TABLE memory_observations (id TEXT PRIMARY "
                    "KEY,namespace TEXT NOT NULL,payload TEXT NOT "
                    "NULL,created_at TEXT NOT NULL,status TEXT NOT NULL DEFAULT"
                    " 'pending',item_id TEXT,reason TEXT NOT NULL DEFAULT "
                    "'',blocked_budget INTEGER,dependencies TEXT NOT NULL "
                    "DEFAULT '[]')"
                ),
                (
                    "CREATE INDEX idx_memory_observations_pending ON "
                    "memory_observations(namespace,status,created_at,id)"
                ),
                (
                    "CREATE TABLE memory_item_evidence (item_id TEXT NOT "
                    "NULL,kind TEXT NOT NULL,source_id TEXT NOT NULL,record_id "
                    "TEXT,start INTEGER NOT NULL,end INTEGER)"
                ),
                (
                    "CREATE INDEX idx_memory_evidence_source ON "
                    "memory_item_evidence(kind,source_id,record_id)"
                ),
                (
                    "CREATE TABLE memory_dream_changes (event_id TEXT PRIMARY "
                    "KEY,namespace TEXT NOT NULL,item_id TEXT NOT NULL,status "
                    "TEXT NOT NULL DEFAULT 'pending',reason TEXT NOT NULL "
                    "DEFAULT '',blocked_budget INTEGER,dependencies TEXT NOT "
                    "NULL DEFAULT '[]')"
                ),
                (
                    "CREATE TABLE memory_capture_sources (lifecycle_source_id "
                    "TEXT NOT NULL,run_id TEXT NOT NULL,namespace TEXT NOT "
                    "NULL,captured_until INTEGER NOT "
                    "NULL,terminal_message_count INTEGER,payload TEXT NOT "
                    "NULL,PRIMARY KEY(lifecycle_source_id,run_id,namespace))"
                ),
                (
                    "CREATE TABLE memory_generation_results (id TEXT PRIMARY "
                    "KEY,namespace TEXT NOT NULL,stage TEXT NOT NULL,created_at"
                    " TEXT NOT NULL,payload TEXT NOT NULL)"
                ),
            )
            for statement in statements:
                connection.execute(statement)

    def rebuild_index(self) -> None:
        """使用既有词法重新生成派生 FTS 索引。"""
        with self._connection() as connection:
            self._rebuild_fts(connection)

    def add_episode(self, episode: MemoryEpisode, *, event: MemoryEvent) -> MemoryEpisode:
        """保存不可变经历及真实写入事件。"""
        with self._connection() as connection:
            self._insert_episode(connection, episode)
            self._insert_event(connection, event)
        return episode

    def add_item(self, item: MemoryItem, *, event: MemoryEvent) -> MemoryItem:
        """显式写入在同一事务保存知识、当前证据、事件和 DreamChange。"""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._ensure_new_item_id(connection, item)
            self._validate_evidence(connection, item.namespace, item.evidence)
            stored = item.model_copy(
                update={
                    "evidence": (
                        MemoryEvidenceRef(kind="event", source_id=event.id),
                        *item.evidence,
                    )
                }
            )
            self._upsert_item(connection, stored)
            self._insert_explicit_event(connection, event, before=None, after=stored)
            self._advance_item_revision(connection, item.namespace)
        return stored

    def update_item(
        self, item_id: str, namespace: str, patch: MemoryItemPatch, *, event: MemoryEvent
    ) -> MemoryItem:
        """正文改变时原子替换当前支持，旧支持只保存在历史中。"""
        updates = {field: getattr(patch, field) for field in patch.model_fields_set}
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._fetch_item(connection, item_id, namespace, include_deleted=False)
            if current is None:
                raise IrisMemoryError("记忆条目不存在", item_id=item_id)
            if not updates or all(getattr(current, key) == value for key, value in updates.items()):
                return current
            if "evidence" in updates:
                self._validate_evidence(connection, namespace, updates["evidence"])
            if "text" in updates and updates["text"] != current.text:
                updates["evidence"] = (
                    MemoryEvidenceRef(kind="event", source_id=event.id),
                    *(patch.evidence or ()),
                )
            updates["source_type"] = event.source_type
            updates["source_id"] = event.source_id
            updates["updated_at"] = _now_iso()
            updated = current.model_copy(update=updates)
            self._upsert_item(connection, updated)
            self._insert_explicit_event(connection, event, before=current, after=updated)
            self._advance_item_revision(connection, namespace)
            self._wake_dependencies(connection, namespace, {item_id})
            return updated

    def delete_item(self, item_id: str, namespace: str, *, event: MemoryEvent) -> bool:
        """软删除当前知识，保留来源与历史，并登记待整理纠正。"""
        with self._connection() as connection:
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
            self._insert_explicit_event(connection, event, before=current, after=deleted)
            self._advance_item_revision(connection, namespace)
            self._wake_dependencies(connection, namespace, {item_id})
            return True

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

    def search(self, query: MemorySearchQuery, namespaces: Sequence[str]) -> MemorySearchResponse:
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
        """在同一读事务内取得完整 active 正式条目及其版本。"""
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

    def get_episode(self, episode_id: str, namespace: str) -> MemoryEpisode | None:
        """按 namespace 读取原始材料，供证据定位而非普通搜索使用。"""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM memory_episodes WHERE id=? AND namespace=?",
                (episode_id, namespace),
            ).fetchone()
            return None if row is None else MemoryEpisode.model_validate_json(row["payload"])

    def list_pending_episodes(self, namespace: str, *, limit: int = 100) -> list[EpisodeProgress]:
        """读取尚未完整 flush 的经历与精确处理游标。"""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT e.*,s.payload AS source_payload FROM memory_episodes e "
                "LEFT JOIN memory_capture_sources s ON s.namespace=e.namespace "
                "AND s.run_id=json_extract(e.payload,'$.source_id') "
                "AND s.lifecycle_source_id="
                "json_extract(e.payload,'$.metadata.lifecycle_source_id') "
                "WHERE e.namespace=? AND e.flushed=0 ORDER BY e.created_at,e.id LIMIT ?",
                (namespace, _validated_list_limit(limit)),
            ).fetchall()
            return [
                EpisodeProgress(
                    MemoryEpisode.model_validate_json(row["payload"]),
                    EpisodeCursor(row["record_index"], row["text_offset"]),
                    None
                    if row["source_payload"] is None
                    else MemoryCaptureSource.model_validate_json(row["source_payload"]).outcome,
                )
                for row in rows
            ]

    def register_source(self, source: MemoryCaptureSource) -> MemoryCaptureSource:
        """首次登记 run；重复登记保留已持久化的捕获水位。"""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload FROM memory_capture_sources WHERE "
                "lifecycle_source_id=? AND run_id=? AND namespace=?",
                (source.lifecycle_source_id, source.run_id, source.namespace),
            ).fetchone()
            if row is not None:
                return MemoryCaptureSource.model_validate_json(row["payload"])
            source = source.model_copy(
                update={
                    "registered_revision": self._read_namespace_state(
                        connection, source.namespace
                    ).item_revision
                }
            )
            connection.execute(
                "INSERT INTO memory_capture_sources VALUES (?,?,?,?,?,?)",
                (
                    source.lifecycle_source_id,
                    source.run_id,
                    source.namespace,
                    source.captured_until,
                    source.terminal_message_count,
                    _dump_model(source),
                ),
            )
            return source

    def list_capture_sources(
        self, lifecycle_source_id: str, namespace: str
    ) -> list[MemoryCaptureSource]:
        """读取同一 lifecycle store 中尚未封口的登记来源。"""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM memory_capture_sources WHERE "
                "lifecycle_source_id=? AND namespace=? AND "
                "(terminal_message_count IS NULL OR captured_until < "
                "terminal_message_count)",
                (lifecycle_source_id, namespace),
            ).fetchall()
            return [MemoryCaptureSource.model_validate_json(row["payload"]) for row in rows]

    def commit_capture(
        self,
        source: MemoryCaptureSource,
        *,
        expected_captured_until: int,
        episode: MemoryEpisode | None,
    ) -> bool:
        """以捕获水位 CAS 原子保存未捕获后缀或零增量终态封口。"""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload FROM memory_capture_sources WHERE "
                "lifecycle_source_id=? AND run_id=? AND namespace=?",
                (source.lifecycle_source_id, source.run_id, source.namespace),
            ).fetchone()
            if row is None:
                raise IrisMemoryError("记忆来源尚未登记", run_id=source.run_id)
            current = MemoryCaptureSource.model_validate_json(row["payload"])
            if current.captured_until != expected_captured_until:
                return False
            if source.captured_until < current.captured_until:
                raise IrisMemoryError("记忆捕获水位不能倒退", run_id=source.run_id)
            if episode is not None:
                if episode.namespace != source.namespace:
                    raise IrisMemoryError("捕获材料 namespace 不匹配")
                self._insert_episode(connection, episode)
            connection.execute(
                "UPDATE memory_capture_sources SET "
                "captured_until=?,terminal_message_count=?,payload=? WHERE "
                "lifecycle_source_id=? AND run_id=? AND namespace=?",
                (
                    source.captured_until,
                    source.terminal_message_count,
                    _dump_model(source),
                    source.lifecycle_source_id,
                    source.run_id,
                    source.namespace,
                ),
            )
            return True

    def commit_flush(self, commit: FlushCommit) -> bool:
        """一次消费精确原文区间，并原子保存观察和实际阶段成本。"""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            positions: dict[str, tuple[MemoryEpisode, int, int]] = {}
            for piece in commit.slices:
                if piece.episode_id not in positions:
                    row = connection.execute(
                        "SELECT * FROM memory_episodes WHERE id=? AND namespace=?",
                        (piece.episode_id, commit.namespace),
                    ).fetchone()
                    if row is None:
                        raise IrisMemoryError("flush 经历不存在", episode_id=piece.episode_id)
                    if row["flushed"]:
                        return False
                    positions[piece.episode_id] = (
                        MemoryEpisode.model_validate_json(row["payload"]),
                        row["record_index"],
                        row["text_offset"],
                    )
                episode, index, offset = positions[piece.episode_id]
                if index >= len(episode.records):
                    return False
                record = episode.records[index]
                if record.id != piece.record_id or piece.start != offset:
                    return False
                if (
                    piece.end < piece.start
                    or piece.end > len(record.text)
                    or piece.text != record.text[piece.start : piece.end]
                ):
                    raise IrisMemoryError("flush 原文区间不匹配", episode_id=episode.id)
                if piece.end == len(record.text):
                    index += 1
                    offset = 0
                else:
                    offset = piece.end
                positions[episode.id] = (episode, index, offset)
            for observation in commit.observations:
                if observation.namespace != commit.namespace:
                    raise IrisMemoryError("观察 namespace 不匹配")
                for evidence in observation.evidence:
                    if evidence.kind != "episode" or not any(
                        piece.episode_id == evidence.source_id
                        and piece.record_id == evidence.record_id
                        and evidence.start >= piece.start
                        and evidence.end is not None
                        and evidence.end <= piece.end
                        for piece in commit.slices
                    ):
                        raise IrisMemoryError(
                            "观察证据必须属于本次 flush 原文片段", observation_id=observation.id
                        )
                connection.execute(
                    "INSERT INTO memory_observations "
                    "(id,namespace,payload,created_at) VALUES (?,?,?,?)",
                    (
                        observation.id,
                        observation.namespace,
                        _dump_model(observation),
                        observation.created_at,
                    ),
                )
            for episode, index, offset in positions.values():
                connection.execute(
                    "UPDATE memory_episodes SET record_index=?,text_offset=?,flushed=? WHERE id=?",
                    (index, offset, int(index == len(episode.records)), episode.id),
                )
            self._insert_result(connection, commit.result)
            return True

    def list_observations(
        self, namespace: str, *, status: str | None = None, limit: int = 100
    ) -> list[ObservationState]:
        """读取观察及其独立处理结果，不将它们暴露为正式知识。"""
        with self._connection() as connection:
            sql = "SELECT * FROM memory_observations WHERE namespace=?"
            params: list[Any] = [namespace]
            if status is not None:
                sql += " AND status=?"
                params.append(status)
            sql += " ORDER BY created_at,id LIMIT ?"
            params.append(_validated_list_limit(limit))
            return [_row_to_observation_state(row) for row in connection.execute(sql, params)]

    def read_dream_snapshot(
        self,
        namespace: str,
        *,
        observation_ids: Sequence[str] | None = None,
        change_ids: Sequence[str] | None = None,
        limit: int = 16,
        related_limit: int = 8,
    ) -> DreamSnapshot:
        """同一连接读取固定输入、目标、已知关联、相关 tombstone 和纠正。"""
        with self._connection() as connection:
            connection.execute("BEGIN")
            revision = self._read_namespace_state(connection, namespace).item_revision
            observations = tuple(
                MemoryObservation.model_validate_json(row["payload"])
                for row in self._pending_rows(
                    connection, "memory_observations", "id", namespace, observation_ids, limit
                )
            )
            changes = tuple(
                _row_to_change(row)
                for row in self._pending_rows(
                    connection, "memory_dream_changes", "event_id", namespace, change_ids, limit
                )
            )
            target_ids = {change.item_id for change in changes}
            target_ids.update(
                item_id for observation in observations for item_id in observation.target_item_ids
            )
            # 原文已被某个条目使用时，其当前条目与历史处理结果都属于必要上下文。
            for observation in observations:
                for evidence in observation.evidence:
                    rows = connection.execute(
                        "SELECT i.id FROM memory_item_evidence e JOIN "
                        "memory_items i ON i.id=e.item_id WHERE "
                        "i.namespace=? AND e.kind=? AND e.source_id=? AND "
                        "e.record_id IS ? AND (?='event' OR (e.start<? AND e.end>?))",
                        (
                            namespace,
                            evidence.kind,
                            evidence.source_id,
                            evidence.record_id,
                            evidence.kind,
                            evidence.end,
                            evidence.start,
                        ),
                    ).fetchall()
                    target_ids.update(row["id"] for row in rows)
                    historical = connection.execute(
                        "SELECT DISTINCT o.item_id FROM memory_observations o, "
                        "json_each(o.payload,'$.evidence') e WHERE o.namespace=? "
                        "AND o.status='processed' AND o.item_id IS NOT NULL "
                        "AND json_extract(e.value,'$.kind')=? "
                        "AND json_extract(e.value,'$.source_id')=? "
                        "AND json_extract(e.value,'$.record_id') IS ? "
                        "AND (?='event' OR (json_extract(e.value,'$.start')<? "
                        "AND json_extract(e.value,'$.end')>?))",
                        (
                            namespace,
                            evidence.kind,
                            evidence.source_id,
                            evidence.record_id,
                            evidence.kind,
                            evidence.end,
                            evidence.start,
                        ),
                    ).fetchall()
                    target_ids.update(row["item_id"] for row in historical)
                    # 直接写入也可能曾使用原文，不能只依赖当前支持集合。
                    historical_events = connection.execute(
                        "SELECT DISTINCT h.item_id FROM memory_events h, "
                        "json_each(h.payload,'$.after.evidence') e WHERE h.namespace=? "
                        "AND h.item_id IS NOT NULL AND json_extract(e.value,'$.kind')=? "
                        "AND json_extract(e.value,'$.source_id')=? "
                        "AND json_extract(e.value,'$.record_id') IS ? "
                        "AND (?='event' OR (json_extract(e.value,'$.start')<? "
                        "AND json_extract(e.value,'$.end')>?))",
                        (
                            namespace,
                            evidence.kind,
                            evidence.source_id,
                            evidence.record_id,
                            evidence.kind,
                            evidence.end,
                            evidence.start,
                        ),
                    ).fetchall()
                    target_ids.update(row["item_id"] for row in historical_events)
            # 显式写入也要找到相似知识，不能只给 Dream 看它自己。
            query_texts = [observation.text for observation in observations]
            query_texts.extend(
                item.text for item in self._fetch_items(connection, namespace, target_ids)
            )
            # FTS 只选择比较材料；维护查询包含 deleted/superseded，普通查询仍只返回 active。
            for text in query_texts:
                terms = tokenize_text(text)
                if terms:
                    rows = connection.execute(
                        "SELECT i.id FROM memory_items_fts JOIN "
                        "memory_items i ON i.id=memory_items_fts.item_id "
                        "WHERE memory_items_fts MATCH ? AND i.namespace=? "
                        "ORDER BY bm25(memory_items_fts),i.updated_at DESC "
                        "LIMIT ?",
                        (prepare_fts_query(terms, []), namespace, related_limit),
                    ).fetchall()
                    target_ids.update(row["id"] for row in rows)
            related_items = self._fetch_items(connection, namespace, target_ids)
            # 历史整理结果仍指向旧条目；合并后的 keeper 必须同包读取。
            frontier = {
                item.superseded_by
                for item in related_items
                if item.superseded_by is not None and item.superseded_by not in target_ids
            }
            while frontier:
                target_ids.update(frontier)
                keepers = self._fetch_items(connection, namespace, frontier)
                related_items.extend(keepers)
                frontier = {
                    item.superseded_by
                    for item in keepers
                    if item.superseded_by is not None and item.superseded_by not in target_ids
                }
            items = tuple(related_items)
            # 当前支持、最近显式纠正和未处理的 blocked 纠正是必要材料。
            event_ids = {change.event_id for change in changes}
            event_ids.update(
                ref.source_id for item in items for ref in item.evidence if ref.kind == "event"
            )
            latest = connection.execute(
                "SELECT id FROM (SELECT id,row_number() OVER "
                "(PARTITION BY item_id ORDER BY created_at DESC,rowid DESC) n "
                "FROM memory_events WHERE namespace=? AND item_id IN "
                "(SELECT value FROM json_each(?)) AND json_extract(payload,'$.event_type') "
                "IN ('update','delete','supersede')) WHERE n=1",
                (namespace, _dump_json(sorted(target_ids))),
            ).fetchall()
            event_ids.update(row["id"] for row in latest)
            blocked = connection.execute(
                "SELECT event_id FROM memory_dream_changes WHERE namespace=? "
                "AND status='blocked' AND item_id IN (SELECT value FROM json_each(?))",
                (namespace, _dump_json(sorted(target_ids))),
            ).fetchall()
            event_ids.update(row["event_id"] for row in blocked)
            event_rows = connection.execute(
                "SELECT payload FROM memory_events WHERE namespace=? "
                "AND id IN (SELECT value FROM json_each(?)) ORDER BY created_at,rowid",
                (namespace, _dump_json(sorted(event_ids))),
            ).fetchall()
            events = tuple(MemoryEvent.model_validate_json(row["payload"]) for row in event_rows)
            return DreamSnapshot(namespace, revision, observations, changes, items, events)

    def commit_dream(
        self, snapshot: DreamSnapshot, plan: DreamPlan, *, result: GenerationResult
    ) -> bool:
        """原子应用最终知识计划、当前证据及固定输入处理结果。"""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if not self._snapshot_current(connection, snapshot):
                return False
            self._validate_plan(snapshot, plan)
            changed_ids: set[str] = set()
            for operation in plan.operations:
                current = (
                    None
                    if operation.action == "add"
                    else self._fetch_item(
                        connection, operation.target_id, snapshot.namespace, include_deleted=True
                    )
                )
                if operation.action == "add":
                    item = MemoryItem(
                        id=operation.new_id,
                        namespace=snapshot.namespace,
                        text=operation.text,
                        category=operation.category,
                        kind=operation.kind,
                        evidence=operation.evidence,
                        source_type=MemorySourceType.GENERATION,
                        source_id=result.id,
                        reason=operation.reason,
                    )
                    self._ensure_new_item_id(connection, item)
                else:
                    if current is None:
                        raise IrisMemoryError("Dream 修改目标不存在", item_id=operation.target_id)
                    updates: dict[str, Any] = {"reason": operation.reason}
                    if operation.action in {"update", "merge"}:
                        updates.update(
                            text=operation.text,
                            category=operation.category,
                            kind=operation.kind,
                            evidence=operation.evidence,
                            source_type=MemorySourceType.GENERATION,
                            source_id=result.id,
                        )
                    elif operation.action == "support":
                        updates["evidence"] = operation.evidence
                    else:
                        updates.update(status=MemoryItemStatus.DELETED, deleted_at=_now_iso())
                    item = current.model_copy(update=updates)
                self._validate_evidence(connection, snapshot.namespace, item.evidence)
                if current is None or item != current:
                    item = item.model_copy(update={"updated_at": _now_iso()})
                    self._upsert_item(connection, item)
                    self._insert_dream_event(connection, current, item, operation.reason)
                    changed_ids.add(item.id)
                for merged_id in operation.merge_ids:
                    merged = self._fetch_item(
                        connection, merged_id, snapshot.namespace, include_deleted=True
                    )
                    if merged is None:
                        raise IrisMemoryError("Dream 合并目标不存在", item_id=merged_id)
                    retired = merged.model_copy(
                        update={
                            "status": MemoryItemStatus.SUPERSEDED,
                            "superseded_by": item.id,
                            "reason": operation.reason,
                            "source_type": MemorySourceType.GENERATION,
                            "source_id": result.id,
                            "updated_at": _now_iso(),
                        }
                    )
                    self._upsert_item(connection, retired)
                    self._insert_dream_event(connection, merged, retired, operation.reason)
                    changed_ids.add(merged_id)
            for resolution in plan.resolutions:
                connection.execute(
                    "UPDATE memory_observations SET "
                    "status='processed',item_id=?,reason=?,blocked_budget=NULL,dependencies='[]'"
                    " WHERE id=?",
                    (resolution.item_id, resolution.reason, resolution.observation_id),
                )
            for change in snapshot.changes:
                connection.execute(
                    "UPDATE memory_dream_changes SET "
                    "status='processed',blocked_budget=NULL,dependencies='[]'"
                    " WHERE event_id=?",
                    (change.event_id,),
                )
            if changed_ids:
                self._advance_item_revision(connection, snapshot.namespace)
                self._wake_dependencies(connection, snapshot.namespace, changed_ids)
            self._insert_result(
                connection,
                result.model_copy(
                    update={
                        "item_revision": self._read_namespace_state(
                            connection, snapshot.namespace
                        ).item_revision
                    }
                ),
            )
            return True

    def block_dream(
        self,
        snapshot: DreamSnapshot,
        *,
        reason: str,
        budget: int,
        dependency_item_ids: Sequence[str],
    ) -> bool:
        """只在快照仍有效时标记容量受阻，不消费观察或显式修订。"""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if not self._snapshot_current(connection, snapshot):
                return False
            dependencies = _dump_json(list(dependency_item_ids))
            for observation in snapshot.observations:
                connection.execute(
                    "UPDATE memory_observations SET "
                    "status='blocked',reason=?,blocked_budget=?,dependencies=?"
                    " WHERE id=?",
                    (reason, budget, dependencies, observation.id),
                )
            for change in snapshot.changes:
                connection.execute(
                    "UPDATE memory_dream_changes SET "
                    "status='blocked',reason=?,blocked_budget=?,dependencies=?"
                    " WHERE event_id=?",
                    (reason, budget, dependencies, change.event_id),
                )
            return True

    def retry_blocked(self, namespace: str, *, budget: int | None = None) -> int:
        """预算改变或调用方显式要求时，重新开放对应受阻输入。"""
        with self._connection() as connection:
            changed = 0
            for table in ("memory_observations", "memory_dream_changes"):
                sql = (
                    f"UPDATE {table} SET "
                    f"status='pending',reason='',blocked_budget=NULL,dependencies='[]'"
                    f" WHERE namespace=? AND status='blocked'"
                )
                params: list[Any] = [namespace]
                if budget is not None:
                    sql += " AND blocked_budget != ?"
                    params.append(budget)
                changed += connection.execute(sql, params).rowcount
            return changed

    def record_generation_result(self, result: GenerationResult) -> None:
        """保存模型失败、取消或冲突时仍然已产生的实际用量。"""
        with self._connection() as connection:
            self._insert_result(connection, result)

    def generation_state(self, namespace: str) -> GenerationState:
        """读取当前积压及各阶段最近一次实际结果。"""
        with self._connection() as connection:
            connection.execute("BEGIN")
            state = self._read_namespace_state(connection, namespace)
            episodes = connection.execute(
                "SELECT count(*) FROM memory_episodes WHERE namespace=? AND flushed=0", (namespace,)
            ).fetchone()[0]
            counts: dict[str, int] = {}
            for table in ("memory_observations", "memory_dream_changes"):
                for row in connection.execute(
                    f"SELECT status,count(*) AS count FROM {table} WHERE "
                    f"namespace=? GROUP BY status",
                    (namespace,),
                ):
                    counts[f"{table}:{row['status']}"] = row["count"]
            rows = connection.execute(
                "SELECT payload FROM memory_generation_results WHERE id IN "
                "(SELECT id FROM (SELECT id,row_number() OVER (PARTITION BY"
                " stage ORDER BY created_at DESC,id DESC) AS n FROM "
                "memory_generation_results WHERE namespace=?) WHERE n=1) "
                "ORDER BY created_at DESC",
                (namespace,),
            ).fetchall()
            return GenerationState(
                namespace,
                episodes,
                counts.get("memory_observations:pending", 0),
                counts.get("memory_observations:blocked", 0),
                counts.get("memory_dream_changes:pending", 0),
                counts.get("memory_dream_changes:blocked", 0),
                state.item_revision,
                state.projection_revision,
                tuple(GenerationResult.model_validate_json(row["payload"]) for row in rows),
            )

    def _connect(self) -> sqlite3.Connection:
        """为每次完整操作建立独立连接。"""
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """拥有提交、回滚与关闭，统一转换 SQLite 底层错误。"""
        connection = self._connect()
        try:
            with connection:
                yield connection
        except sqlite3.Error as exc:
            raise IrisMemoryError("SQLite memory 操作失败", path=str(self.path)) from exc
        finally:
            connection.close()

    def _advance_item_revision(self, connection: sqlite3.Connection, namespace: str) -> None:
        """正式条目或当前证据发生实质变化时推进唯一版本。"""
        connection.execute(
            "INSERT INTO memory_namespace_state VALUES (?,1,NULL) ON "
            "CONFLICT(namespace) DO UPDATE SET "
            "item_revision=item_revision+1",
            (namespace,),
        )

    def _read_namespace_state(
        self, connection: sqlite3.Connection, namespace: str
    ) -> MemoryNamespaceState:
        """读取当前事务中的正式知识与投影版本。"""
        row = connection.execute(
            "SELECT item_revision,projection_revision FROM "
            "memory_namespace_state WHERE namespace=?",
            (namespace,),
        ).fetchone()
        return (
            MemoryNamespaceState(namespace)
            if row is None
            else MemoryNamespaceState(namespace, row["item_revision"], row["projection_revision"])
        )

    def _read_namespace_snapshot(
        self, connection: sqlite3.Connection, namespace: str
    ) -> MemoryNamespaceSnapshot:
        """读取同一事务中的全部 active 正式知识。"""
        state = self._read_namespace_state(connection, namespace)
        rows = connection.execute(
            "SELECT * FROM memory_items WHERE namespace=? AND status=? "
            "ORDER BY category,kind,created_at,id",
            (namespace, MemoryItemStatus.ACTIVE.value),
        ).fetchall()
        return MemoryNamespaceSnapshot(state, tuple(_row_to_item(row) for row in rows))

    def _insert_episode(self, connection: sqlite3.Connection, episode: MemoryEpisode) -> None:
        """保存不可变材料内容，初始游标指向首条记录。"""
        connection.execute(
            "INSERT INTO memory_episodes (id,namespace,payload,created_at) VALUES (?,?,?,?)",
            (episode.id, episode.namespace, _dump_model(episode), episode.created_at),
        )

    def _ensure_new_item_id(self, connection: sqlite3.Connection, item: MemoryItem) -> None:
        """新增 ID 不得占用其他条目或 namespace。"""
        row = connection.execute(
            "SELECT namespace FROM memory_items WHERE id=?", (item.id,)
        ).fetchone()
        if row is not None:
            raise IrisMemoryError(
                "记忆条目 id 已存在",
                item_id=item.id,
                existing_namespace=row["namespace"],
                requested_namespace=item.namespace,
            )

    def _upsert_item(self, connection: sqlite3.Connection, item: MemoryItem) -> None:
        """同事务维护知识正文、当前证据关系和派生词法索引。"""
        connection.execute(
            "INSERT INTO memory_items VALUES (?,?,?,?,?,?,?,?,?) ON "
            "CONFLICT(id) DO UPDATE SET "
            "category=excluded.category,kind=excluded.kind,text=excluded.text,status=excluded.status,updated_at=excluded.updated_at,payload=excluded.payload",
            (
                item.id,
                item.namespace,
                item.category.value,
                item.kind.value,
                item.text,
                item.status.value,
                item.created_at,
                item.updated_at,
                _dump_model(item),
            ),
        )
        connection.execute("DELETE FROM memory_item_evidence WHERE item_id=?", (item.id,))
        connection.executemany(
            "INSERT INTO memory_item_evidence VALUES (?,?,?,?,?,?)",
            [
                (item.id, ref.kind, ref.source_id, ref.record_id, ref.start, ref.end)
                for ref in item.evidence
            ],
        )
        self._refresh_fts_row(connection, item)

    def _insert_event(self, connection: sqlite3.Connection, event: MemoryEvent) -> None:
        """保存不可变操作事实与前后证据。"""
        connection.execute(
            "INSERT INTO memory_events VALUES (?,?,?,?,?)",
            (event.id, event.namespace, event.item_id, event.created_at, _dump_model(event)),
        )

    def _insert_explicit_event(
        self,
        connection: sqlite3.Connection,
        event: MemoryEvent,
        *,
        before: MemoryItem | None,
        after: MemoryItem,
    ) -> None:
        """补齐本次真实有效正文，并登记且仅登记显式变更。"""
        event = event.model_copy(
            update={
                "namespace": after.namespace,
                "item_id": after.id,
                "before": None if before is None else before.model_dump(mode="python"),
                "after": after.model_dump(mode="python"),
            }
        )
        self._insert_event(connection, event)
        connection.execute(
            "INSERT INTO memory_dream_changes (event_id,namespace,item_id) VALUES (?,?,?)",
            (event.id, after.namespace, after.id),
        )

    def _insert_dream_event(
        self,
        connection: sqlite3.Connection,
        before: MemoryItem | None,
        after: MemoryItem,
        reason: str,
    ) -> None:
        """Dream 自己的修订不再次登记 DreamChange。"""
        self._insert_event(
            connection,
            MemoryEvent(
                namespace=after.namespace,
                item_id=after.id,
                event_type=MemoryEventType.DREAM,
                actor=MemoryActor.SYSTEM,
                reason=reason,
                before=None if before is None else before.model_dump(mode="python"),
                after=after.model_dump(mode="python"),
            ),
        )

    def _fetch_item(
        self, connection: sqlite3.Connection, item_id: str, namespace: str, *, include_deleted: bool
    ) -> MemoryItem | None:
        """按稳定身份读取本 namespace 中的条目。"""
        sql = "SELECT * FROM memory_items WHERE id=? AND namespace=?"
        params: list[Any] = [item_id, namespace]
        if not include_deleted:
            sql += " AND status=?"
            params.append(MemoryItemStatus.ACTIVE.value)
        row = connection.execute(sql, params).fetchone()
        return None if row is None else _row_to_item(row)

    def _fetch_items(
        self, connection: sqlite3.Connection, namespace: str, item_ids: set[str]
    ) -> list[MemoryItem]:
        """同一快照读取相关正式条目，包括 tombstone。"""
        rows = connection.execute(
            "SELECT * FROM memory_items WHERE namespace=? AND id IN (SELECT"
            " value FROM json_each(?)) ORDER BY created_at,id",
            (namespace, _dump_json(sorted(item_ids))),
        ).fetchall()
        return [_row_to_item(row) for row in rows]

    def _pending_rows(
        self,
        connection: sqlite3.Connection,
        table: str,
        id_column: str,
        namespace: str,
        ids: Sequence[str] | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        """在单一读事务中选择 pending 输入；显式空列表代表不选该类。"""
        sql = f"SELECT * FROM {table} WHERE namespace=? AND status='pending'"
        params: list[Any] = [namespace]
        if ids is not None:
            sql += f" AND {id_column} IN (SELECT value FROM json_each(?))"
            params.append(_dump_json(list(ids)))
        sql += " ORDER BY rowid LIMIT ?"
        params.append(_validated_list_limit(limit))
        return connection.execute(sql, params).fetchall()

    def _snapshot_current(self, connection: sqlite3.Connection, snapshot: DreamSnapshot) -> bool:
        """唯一提交 owner 验证正式版本及固定输入尚未被消费。"""
        if (
            self._read_namespace_state(connection, snapshot.namespace).item_revision
            != snapshot.item_revision
        ):
            return False
        for table, column, ids in (
            (
                "memory_observations",
                "id",
                [observation.id for observation in snapshot.observations],
            ),
            ("memory_dream_changes", "event_id", [change.event_id for change in snapshot.changes]),
        ):
            count = connection.execute(
                f"SELECT count(*) FROM {table} WHERE namespace=? AND "
                f"status='pending' AND {column} IN (SELECT value FROM "
                f"json_each(?))",
                (snapshot.namespace, _dump_json(ids)),
            ).fetchone()[0]
            if count != len(ids):
                return False
        return True

    def _validate_plan(self, snapshot: DreamSnapshot, plan: DreamPlan) -> None:
        """在副作用边界确认操作身份和固定输入的最终去向。"""
        observation_ids = {observation.id for observation in snapshot.observations}
        resolved_ids = [resolution.observation_id for resolution in plan.resolutions]
        if set(resolved_ids) != observation_ids or len(resolved_ids) != len(observation_ids):
            raise IrisMemoryError("Dream 必须且只能处理快照中的每个观察")
        targets = {item.id: item for item in snapshot.items}
        touched: set[str] = set()
        added: set[str] = set()
        for operation in plan.operations:
            if operation.action == "add":
                if (
                    not operation.new_id
                    or operation.target_id is not None
                    or not operation.text
                    or not operation.evidence
                ):
                    raise IrisMemoryError("Dream 新增必须包含程序分配 ID、正文与证据")
                if operation.new_id in added or operation.new_id in targets:
                    raise IrisMemoryError("Dream 新增 ID 重复")
                added.add(operation.new_id)
            else:
                if operation.target_id not in targets or operation.target_id in touched:
                    raise IrisMemoryError("Dream 目标必须来自快照且仅有一个最终操作")
                if targets[operation.target_id].status != MemoryItemStatus.ACTIVE:
                    raise IrisMemoryError("Dream 不得改写或复活已退役条目")
                touched.add(operation.target_id)
                if operation.action in {"update", "merge"} and (
                    not operation.text or not operation.evidence
                ):
                    raise IrisMemoryError("Dream 更新必须包含完整正文和当前证据")
                if operation.action == "support" and not operation.evidence:
                    raise IrisMemoryError("Dream 支持操作必须包含当前证据")
            if operation.merge_ids and operation.action != "merge":
                raise IrisMemoryError("仅 merge 操作可以退役合并条目")
            for item_id in operation.merge_ids:
                if (
                    item_id not in targets
                    or item_id in touched
                    or targets[item_id].status != MemoryItemStatus.ACTIVE
                ):
                    raise IrisMemoryError("Dream 合并目标重复、无效或不在快照中")
                touched.add(item_id)
        final_ids = {
            item.id for item in snapshot.items if item.status == MemoryItemStatus.ACTIVE
        } | added
        final_ids.difference_update(
            item_id for operation in plan.operations for item_id in operation.merge_ids
        )
        final_ids.difference_update(
            operation.target_id for operation in plan.operations if operation.action == "delete"
        )
        for resolution in plan.resolutions:
            if resolution.item_id is not None and resolution.item_id not in final_ids:
                raise IrisMemoryError("观察必须关联本次最终仍有效的条目")

    def _validate_evidence(
        self, connection: sqlite3.Connection, namespace: str, refs: Sequence[MemoryEvidenceRef]
    ) -> None:
        """在引用副作用边界确认来源确实存在于同一 namespace。"""
        for ref in refs:
            if ref.kind == "event":
                row = connection.execute(
                    "SELECT 1 FROM memory_events WHERE id=? AND namespace=?",
                    (ref.source_id, namespace),
                ).fetchone()
                if row is None:
                    raise IrisMemoryError("显式记忆证据不存在", source_id=ref.source_id)
            else:
                row = connection.execute(
                    "SELECT payload FROM memory_episodes WHERE id=? AND namespace=?",
                    (ref.source_id, namespace),
                ).fetchone()
                if row is None:
                    raise IrisMemoryError("经历证据不存在", source_id=ref.source_id)
                episode = MemoryEpisode.model_validate_json(row["payload"])
                record = next(
                    (record for record in episode.records if record.id == ref.record_id), None
                )
                if (
                    record is None
                    or ref.end is None
                    or not 0 <= ref.start <= ref.end <= len(record.text)
                ):
                    raise IrisMemoryError("经历证据范围不存在", source_id=ref.source_id)

    def _wake_dependencies(
        self, connection: sqlite3.Connection, namespace: str, item_ids: set[str]
    ) -> None:
        """只恢复依赖本次变化条目的 blocked 输入，且与变化同事务。"""
        for table in ("memory_observations", "memory_dream_changes"):
            connection.execute(
                f"UPDATE {table} SET "
                f"status='pending',reason='',blocked_budget=NULL,dependencies='[]'"
                f" WHERE namespace=? AND status='blocked' AND EXISTS (SELECT"
                f" 1 FROM json_each(dependencies) WHERE value IN (SELECT "
                f"value FROM json_each(?)))",
                (namespace, _dump_json(sorted(item_ids))),
            )

    def _insert_result(self, connection: sqlite3.Connection, result: GenerationResult) -> None:
        """持久保存阶段成本；一次结果只记一次。"""
        connection.execute(
            "INSERT INTO memory_generation_results VALUES (?,?,?,?,?) ON CONFLICT(id) DO NOTHING",
            (result.id, result.namespace, result.stage, result.created_at, _dump_model(result)),
        )

    def _rebuild_fts(self, connection: sqlite3.Connection) -> None:
        """派生索引保留所有正文，普通查询负责 active 过滤。"""
        connection.execute("DELETE FROM memory_items_fts")
        rows = connection.execute("SELECT id,text FROM memory_items").fetchall()
        connection.executemany(
            "INSERT INTO memory_items_fts VALUES (?,?)",
            [(row["id"], " ".join(tokenize_text(row["text"]))) for row in rows],
        )

    def _refresh_fts_row(self, connection: sqlite3.Connection, item: MemoryItem) -> None:
        """与正式写入在同一事务刷新现有词法索引。"""
        connection.execute("DELETE FROM memory_items_fts WHERE item_id=?", (item.id,))
        connection.execute(
            "INSERT INTO memory_items_fts VALUES (?,?)",
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
    """数据库边界完整解析正式条目。"""
    return MemoryItem.model_validate_json(row["payload"])


def _row_to_event(row: sqlite3.Row) -> MemoryEvent:
    """数据库边界完整解析写入事件。"""
    return MemoryEvent.model_validate_json(row["payload"])


def _row_to_observation_state(row: sqlite3.Row) -> ObservationState:
    """组合不可变观察内容和持久处理元数据。"""
    return ObservationState(
        MemoryObservation.model_validate_json(row["payload"]),
        row["status"],
        row["item_id"],
        row["reason"],
        row["blocked_budget"],
        tuple(json.loads(row["dependencies"])),
    )


def _row_to_change(row: sqlite3.Row) -> DreamChange:
    """从持久状态读取显式变更待办。"""
    return DreamChange(
        row["event_id"],
        row["item_id"],
        row["status"],
        row["reason"],
        row["blocked_budget"],
        tuple(json.loads(row["dependencies"])),
    )


def _dump_model(value: Any) -> str:
    """严格编码持久边界模型，保留中文证据。"""
    return _dump_json(value.model_dump(mode="python"))


def _dump_json(value: Any) -> str:
    """JSON 编码的唯一错误转换入口。"""
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise IrisMemoryError("Memory 数据必须可 JSON 序列化") from exc


def _validated_list_limit(limit: int) -> int:
    """公开列表读取边界的数量约束。"""
    if not 1 <= limit <= 100:
        raise IrisMemoryError("limit 必须在 1 到 100 之间", limit=limit)
    return limit


def _placeholders(values: Sequence[object]) -> str:
    """生成 SQL IN 子句占位符。"""
    return ", ".join("?" for _ in values)
