"""SQLite 会话存储实现。"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from ..exceptions import IrisExecutionError


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
            raise IrisExecutionError("SQLite session 目录创建失败", path=str(self.path)) from exc
        self._initialize_schema()

    def save_messages(self, session_id: str, messages: list[dict[str, object]]) -> None:
        """保存会话消息列表。

        Args:
            session_id (str): 会话标识。
            messages (list[dict[str, object]]): 可 JSON 序列化的消息列表。

        Raises:
            IrisExecutionError: JSON 序列化或 SQLite 写入失败时抛出。
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

    def _initialize_schema(self) -> None:
        """创建 session 表。"""
        try:
            with sqlite3.connect(self.path) as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        messages_json TEXT NOT NULL DEFAULT '[]',
                        run_metadata_json TEXT NOT NULL DEFAULT '{}',
                        tool_events_json TEXT NOT NULL DEFAULT '[]',
                        updated_at TEXT NOT NULL
                    )
                    """
                )
        except sqlite3.Error as exc:
            raise IrisExecutionError("SQLite session 初始化失败", path=str(self.path)) from exc

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
            raise IrisExecutionError(
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
            raise IrisExecutionError(
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
        return json.dumps(value, ensure_ascii=False)
    except TypeError as exc:
        raise IrisExecutionError("Session 数据必须可 JSON 序列化") from exc


__all__ = ["SQLiteSessionStore"]
