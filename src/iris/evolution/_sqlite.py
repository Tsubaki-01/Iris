"""Evolution SQLite 的连接、唯一序列化边界和当前数据库结构。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import ValidationError

from ..exceptions import IrisEvolutionError


@contextmanager
def connection(path: Path, *, write: bool = False) -> Iterator[sqlite3.Connection]:
    """每次操作持有一个一致性快照，写操作串行准入并在异常时回滚。"""
    database = None
    try:
        database = sqlite3.connect(path, timeout=30)
        database.row_factory = sqlite3.Row
        database.execute("PRAGMA foreign_keys=ON")
        database.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        yield database
        database.commit()
    except (sqlite3.Error, ValidationError, json.JSONDecodeError) as exc:
        raise IrisEvolutionError(
            "Evolution 数据库操作失败", path=str(path), error=str(exc)
        ) from exc
    finally:
        if database is not None:
            database.close()


def encode(value: object) -> str:
    """仅在进入持久存储时严格编码完整对象图。"""
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise IrisEvolutionError("Evolution 数据序列化失败", error=str(exc)) from exc


def initialize(path: Path) -> None:
    """只创建当前 schema，不读取或迁移旧 JSON 存储。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise IrisEvolutionError("Evolution 数据库目录创建失败", path=str(path)) from exc
    with connection(path, write=True) as database:
        tables = database.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        version = database.execute("PRAGMA user_version").fetchone()[0]
        if tables:
            if version != 3:
                raise IrisEvolutionError("Evolution 数据库要求 schema 3", version=version)
            return
        statements = (
            "CREATE TABLE sources (source_key TEXT PRIMARY KEY, registration_json TEXT NOT NULL, "
            "consumed_until INTEGER NOT NULL, lifecycle_source_id TEXT NOT NULL, "
            "captured_until INTEGER NOT NULL, observed_terminal INTEGER, observed_outcome TEXT)",
            "CREATE INDEX sources_capture ON sources(lifecycle_source_id) WHERE "
            "observed_terminal IS NULL OR captured_until < observed_terminal",
            "CREATE INDEX sources_pending ON sources(source_key) WHERE "
            "captured_until=observed_terminal AND consumed_until < observed_terminal",
            "CREATE TABLE captures (id TEXT PRIMARY KEY, source_key TEXT NOT NULL REFERENCES "
            "sources(source_key), start_count INTEGER NOT NULL, end_count INTEGER NOT NULL, "
            "receipt_json TEXT NOT NULL, "
            "body_json TEXT)",
            "CREATE INDEX captures_source ON captures(source_key,end_count,start_count)",
            "CREATE TABLE requests (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
            "summary_json TEXT NOT NULL, payload TEXT NOT NULL, result_json TEXT, "
            "routing_json TEXT NOT NULL)",
            "CREATE INDEX requests_history ON requests(created_at,id)",
            "CREATE INDEX requests_pending ON requests(created_at,id) WHERE result_json IS NULL",
            "CREATE TABLE publications (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
            "settled INTEGER NOT NULL, summary_json TEXT NOT NULL, state_json TEXT NOT NULL)",
            "CREATE TABLE publication_details (publication_id TEXT PRIMARY KEY REFERENCES "
            "publications(id), detail_json TEXT NOT NULL)",
            "CREATE INDEX publications_history ON publications(created_at,id)",
            "CREATE INDEX publications_unsettled ON publications(created_at,id) WHERE settled=0",
            "CREATE TABLE consumed_publications (publication_id TEXT PRIMARY KEY)",
            "CREATE TABLE progress (id INTEGER PRIMARY KEY CHECK(id=1), latest_step TEXT NOT NULL)",
            "PRAGMA user_version=3",
        )
        for statement in statements:
            database.execute(statement)
