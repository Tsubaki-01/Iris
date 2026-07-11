"""SQLite 会话存储实现。"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from ..exceptions import HITLConflictError, HITLResponseMismatchError, IrisSessionError
from ..hitl.models import (
    HumanInteraction,
    HumanInteractionResponse,
    InteractionResumePhase,
)


class SQLiteSessionStore:
    """使用本地 SQLite 文件保存 session JSON 数据。

    Args:
        path (str | Path): SQLite 数据库文件路径。
    """

    def __init__(self, path: str | Path) -> None:
        """初始化 SQLite store 并创建必要表结构。"""
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise IrisSessionError("SQLite session 目录创建失败", path=str(self.path)) from exc
        self._initialize_schema()

    def save_messages(self, session_id: str, messages: list[dict[str, object]]) -> None:
        """保存会话消息列表。

        Args:
            session_id (str): 会话标识。
            messages (list[dict[str, object]]): 可 JSON 序列化的消息列表。

        Raises:
            IrisSessionError: JSON 序列化或 SQLite 写入失败时抛出。
        """
        self._upsert_column(session_id, "messages_json", _dump_json(messages))

    def load_messages(self, session_id: str) -> list[dict[str, object]]:
        """读取会话消息列表。"""
        value = self._load_column(session_id, "messages_json", "[]")
        return cast(list[dict[str, object]], json.loads(value))

    def save_run_metadata(self, session_id: str, metadata: dict[str, object]) -> None:
        """保存运行元数据。"""
        self._upsert_column(session_id, "run_metadata_json", _dump_json(metadata))

    def load_run_metadata(self, session_id: str) -> dict[str, object]:
        """读取运行元数据。"""
        value = self._load_column(session_id, "run_metadata_json", "{}")
        return cast(dict[str, object], json.loads(value))

    def append_tool_event(self, session_id: str, event: dict[str, object]) -> None:
        """追加工具调用或结果摘要。"""
        events = self.load_tool_events(session_id)
        events.append(event)
        self._upsert_column(session_id, "tool_events_json", _dump_json(events))

    def load_tool_events(self, session_id: str) -> list[dict[str, object]]:
        """读取工具调用或结果摘要列表。"""
        value = self._load_column(session_id, "tool_events_json", "[]")
        return cast(list[dict[str, object]], json.loads(value))

    def create_interaction(self, interaction: HumanInteraction) -> None:
        """原子创建一条 HITL interaction。"""
        values = _interaction_values(interaction)
        try:
            with sqlite3.connect(self.path) as connection:
                connection.execute(
                    """
                    INSERT INTO human_interactions (
                        interaction_id, session_id, run_id, step_index, tool_call_id,
                        kind, status, resume_phase, request_json, response_json,
                        checkpoint_json, version, created_at, resolved_at, consumed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
        except sqlite3.IntegrityError as exc:
            raise HITLConflictError(
                "HITL interaction 已存在或 session 已有 active interaction",
                interaction_id=interaction.interaction_id,
                session_id=interaction.session_id,
            ) from exc
        except sqlite3.Error as exc:
            raise IrisSessionError("SQLite HITL interaction 写入失败", path=str(self.path)) from exc

    def load_interaction(self, interaction_id: str) -> HumanInteraction | None:
        """按 ID 读取 HITL interaction。"""
        try:
            with sqlite3.connect(self.path) as connection:
                connection.row_factory = sqlite3.Row
                row = connection.execute(
                    "SELECT * FROM human_interactions WHERE interaction_id = ?",
                    (interaction_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise IrisSessionError(
                "SQLite HITL interaction 读取失败",
                path=str(self.path),
                interaction_id=interaction_id,
            ) from exc
        return _row_to_interaction(row, path=self.path) if row is not None else None

    def list_pending_interactions(self, session_id: str | None = None) -> list[HumanInteraction]:
        """按创建时间列出 pending HITL interaction。"""
        sql = "SELECT * FROM human_interactions WHERE status = 'pending'"
        parameters: tuple[str, ...] = ()
        if session_id is not None:
            sql += " AND session_id = ?"
            parameters = (session_id,)
        sql += " ORDER BY created_at, interaction_id"
        try:
            with sqlite3.connect(self.path) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(sql, parameters).fetchall()
        except sqlite3.Error as exc:
            raise IrisSessionError(
                "SQLite HITL interaction 列表读取失败", path=str(self.path)
            ) from exc
        return [_row_to_interaction(row, path=self.path) for row in rows]

    def resolve_interaction(
        self,
        interaction_id: str,
        response: HumanInteractionResponse,
        *,
        expected_version: int,
    ) -> HumanInteraction:
        """以单事务 compare-and-set 写入人工响应。"""
        response_json = _dump_json(response.model_dump(mode="json"))
        resolved_at = datetime.now(UTC).isoformat()
        try:
            with sqlite3.connect(self.path) as connection:
                connection.row_factory = sqlite3.Row
                current = _load_interaction_row(connection, interaction_id)
                if current is None or current["kind"] != response.kind.value:
                    raise HITLResponseMismatchError(
                        "HITL response kind 与 interaction 不匹配",
                        interaction_id=interaction_id,
                    )
                cursor = connection.execute(
                    """
                    UPDATE human_interactions
                    SET response_json = ?, status = 'resolved', resolved_at = ?,
                        version = version + 1
                    WHERE interaction_id = ? AND status = 'pending' AND version = ?
                    """,
                    (response_json, resolved_at, interaction_id, expected_version),
                )
                if cursor.rowcount != 1:
                    raise HITLConflictError(
                        "HITL resolve compare-and-set 失败", interaction_id=interaction_id
                    )
                row = _load_interaction_row(connection, interaction_id)
        except sqlite3.Error as exc:
            raise IrisSessionError("SQLite HITL resolve 失败", path=str(self.path)) from exc
        assert row is not None
        return _row_to_interaction(row, path=self.path)

    def claim_interaction(
        self,
        interaction_id: str,
        checkpoint: dict[str, Any],
        *,
        expected_version: int,
    ) -> HumanInteraction:
        """以单事务 compare-and-set 领取已响应 interaction。"""
        checkpoint_json = _dump_json(checkpoint)
        consumed_at = datetime.now(UTC).isoformat()
        try:
            with sqlite3.connect(self.path) as connection:
                connection.row_factory = sqlite3.Row
                cursor = connection.execute(
                    """
                    UPDATE human_interactions
                    SET checkpoint_json = ?, status = 'consumed', resume_phase = 'claimed',
                        consumed_at = ?, version = version + 1
                    WHERE interaction_id = ? AND status = 'resolved'
                        AND resume_phase = 'waiting' AND version = ?
                    """,
                    (checkpoint_json, consumed_at, interaction_id, expected_version),
                )
                if cursor.rowcount != 1:
                    raise HITLConflictError(
                        "HITL claim compare-and-set 失败", interaction_id=interaction_id
                    )
                row = _load_interaction_row(connection, interaction_id)
        except sqlite3.Error as exc:
            raise IrisSessionError("SQLite HITL claim 失败", path=str(self.path)) from exc
        assert row is not None
        return _row_to_interaction(row, path=self.path)

    def update_consumed_interaction(
        self,
        interaction_id: str,
        *,
        resume_phase: InteractionResumePhase,
        checkpoint: dict[str, Any],
        expected_version: int,
    ) -> HumanInteraction:
        """以单事务 compare-and-set 更新已消费 interaction。"""
        if resume_phase is InteractionResumePhase.WAITING:
            raise HITLConflictError(
                "consumed interaction 不能回到 waiting", interaction_id=interaction_id
            )
        checkpoint_json = _dump_json(checkpoint)
        try:
            with sqlite3.connect(self.path) as connection:
                connection.row_factory = sqlite3.Row
                cursor = connection.execute(
                    """
                    UPDATE human_interactions
                    SET checkpoint_json = ?, resume_phase = ?, version = version + 1
                    WHERE interaction_id = ? AND status = 'consumed' AND version = ?
                    """,
                    (checkpoint_json, resume_phase.value, interaction_id, expected_version),
                )
                if cursor.rowcount != 1:
                    raise HITLConflictError(
                        "HITL update compare-and-set 失败", interaction_id=interaction_id
                    )
                row = _load_interaction_row(connection, interaction_id)
        except sqlite3.Error as exc:
            raise IrisSessionError("SQLite HITL update 失败", path=str(self.path)) from exc
        assert row is not None
        return _row_to_interaction(row, path=self.path)

    def _initialize_schema(self) -> None:
        """创建 session 和 HITL interaction 表。"""
        try:
            with sqlite3.connect(self.path) as connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        messages_json TEXT NOT NULL DEFAULT '[]',
                        run_metadata_json TEXT NOT NULL DEFAULT '{}',
                        tool_events_json TEXT NOT NULL DEFAULT '[]',
                        updated_at TEXT NOT NULL
                    )
                    """)
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS human_interactions (
                        interaction_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        step_index INTEGER NOT NULL,
                        tool_call_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        status TEXT NOT NULL,
                        resume_phase TEXT NOT NULL,
                        request_json TEXT NOT NULL,
                        response_json TEXT,
                        checkpoint_json TEXT NOT NULL,
                        version INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        resolved_at TEXT,
                        consumed_at TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_human_interactions_session_status_phase
                    ON human_interactions (session_id, status, resume_phase)
                    """
                )
                connection.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_human_interactions_active_session
                    ON human_interactions (session_id)
                    WHERE status IN ('pending', 'resolved')
                        OR (status = 'consumed' AND resume_phase != 'result_committed')
                    """
                )
        except sqlite3.Error as exc:
            raise IrisSessionError("SQLite session 初始化失败", path=str(self.path)) from exc

    def _upsert_column(self, session_id: str, column: str, value: str) -> None:
        """更新单个 JSON 字段。"""
        updated_at = datetime.now().isoformat()
        sql = f"""
            INSERT INTO sessions (session_id, {column}, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                {column} = excluded.{column},
                updated_at = excluded.updated_at
        """
        try:
            with sqlite3.connect(self.path) as connection:
                connection.execute(sql, (session_id, value, updated_at))
        except sqlite3.Error as exc:
            raise IrisSessionError(
                "SQLite session 写入失败",
                path=str(self.path),
                session_id=session_id,
            ) from exc

    def _load_column(self, session_id: str, column: str, default: str) -> str:
        """读取单个 JSON 字段。"""
        try:
            with sqlite3.connect(self.path) as connection:
                row = connection.execute(
                    f"SELECT {column} FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise IrisSessionError(
                "SQLite session 读取失败",
                path=str(self.path),
                session_id=session_id,
            ) from exc
        if row is None:
            return default
        return cast(str, row[0])


def _dump_json(value: Any) -> str:
    """序列化 JSON 值。"""
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise IrisSessionError("Session 数据必须可 JSON 序列化") from exc


def _interaction_values(interaction: HumanInteraction) -> tuple[object, ...]:
    return (
        interaction.interaction_id,
        interaction.session_id,
        interaction.run_id,
        interaction.step_index,
        interaction.tool_call_id,
        interaction.kind.value,
        interaction.status.value,
        interaction.resume_phase.value,
        _dump_json(interaction.request.model_dump(mode="json")),
        _dump_json(interaction.response.model_dump(mode="json"))
        if interaction.response is not None
        else None,
        _dump_json(interaction.checkpoint),
        interaction.version,
        interaction.created_at.isoformat(),
        interaction.resolved_at.isoformat() if interaction.resolved_at is not None else None,
        interaction.consumed_at.isoformat() if interaction.consumed_at is not None else None,
    )


def _load_interaction_row(
    connection: sqlite3.Connection,
    interaction_id: str,
) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM human_interactions WHERE interaction_id = ?",
        (interaction_id,),
    ).fetchone()


def _row_to_interaction(row: sqlite3.Row, *, path: Path) -> HumanInteraction:
    try:
        return HumanInteraction.model_validate(
            {
                "interaction_id": row["interaction_id"],
                "session_id": row["session_id"],
                "run_id": row["run_id"],
                "step_index": row["step_index"],
                "tool_call_id": row["tool_call_id"],
                "kind": row["kind"],
                "status": row["status"],
                "resume_phase": row["resume_phase"],
                "request": json.loads(row["request_json"]),
                "response": json.loads(row["response_json"])
                if row["response_json"] is not None
                else None,
                "checkpoint": json.loads(row["checkpoint_json"]),
                "version": row["version"],
                "created_at": row["created_at"],
                "resolved_at": row["resolved_at"],
                "consumed_at": row["consumed_at"],
            }
        )
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        raise IrisSessionError("SQLite HITL interaction 数据无效", path=str(path)) from exc


__all__ = ["SQLiteSessionStore"]
