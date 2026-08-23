"""Lifecycle SQLite 当前 schema 的精确定义、检查与创建。"""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from pathlib import Path

from ..exceptions import IrisLifecycleSchemaError

_SCHEMA_VERSION = 2

_IDENTITY_STATEMENT = """
CREATE TABLE lifecycle_schema (
    component TEXT PRIMARY KEY,
    version INTEGER NOT NULL
)
"""

_SESSIONS_STATEMENT = """
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
    message_count INTEGER NOT NULL DEFAULT 0 CHECK (message_count >= 0),
    updated_at TEXT NOT NULL
)
"""

_SESSION_MESSAGES_STATEMENT = """
CREATE TABLE session_messages (
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 1),
    message_json TEXT NOT NULL,
    PRIMARY KEY (session_id, ordinal)
)
"""

_COMMON_STATEMENTS = (
    """
    CREATE TABLE agent_runs (
        run_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        phase TEXT NOT NULL CHECK (phase IN ('active', 'waiting', 'terminal')),
        stop_reason TEXT,
        request_json TEXT NOT NULL,
        options_json TEXT NOT NULL,
        environment_fingerprint TEXT NOT NULL,
        session_revision INTEGER NOT NULL,
        run_revision INTEGER NOT NULL CHECK (run_revision >= 1),
        current_activation_id TEXT,
        pending_interaction_id TEXT,
        cancellation_requested_at TEXT,
        cancellation_reason TEXT,
        model_steps_reserved INTEGER NOT NULL DEFAULT 0,
        model_steps_committed INTEGER NOT NULL DEFAULT 0,
        tool_calls_committed INTEGER NOT NULL DEFAULT 0,
        usage_json TEXT NOT NULL DEFAULT '{}',
        assistant_message_json TEXT,
        error_json TEXT,
        checkpoint_sequence INTEGER NOT NULL DEFAULT 0,
        last_event_sequence INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        started_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        finished_at TEXT,
        CHECK ((phase = 'terminal') = (stop_reason IS NOT NULL)),
        CHECK (model_steps_committed <= model_steps_reserved)
    )
    """,
    """
    CREATE TABLE session_run_lanes (
        session_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL UNIQUE,
        revision INTEGER NOT NULL,
        acquired_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE run_activations (
        activation_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES agent_runs(run_id),
        ordinal INTEGER NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ('start', 'resume', 'recover')),
        status TEXT NOT NULL CHECK (status IN ('active', 'settled', 'abandoned')),
        outcome TEXT CHECK (
            outcome IN (
                'completed', 'suspended', 'failed', 'cancelled', 'recovered',
                'outcome_unknown'
            )
        ),
        started_at TEXT NOT NULL,
        ended_at TEXT,
        UNIQUE (run_id, ordinal)
    )
    """,
    """
    CREATE TABLE run_checkpoints (
        run_id TEXT PRIMARY KEY REFERENCES agent_runs(run_id),
        sequence INTEGER NOT NULL,
        activation_id TEXT NOT NULL,
        checkpoint_version INTEGER NOT NULL,
        cursor_json TEXT NOT NULL,
        session_revision INTEGER NOT NULL,
        model_steps_reserved INTEGER NOT NULL,
        model_steps_committed INTEGER NOT NULL,
        environment_fingerprint TEXT NOT NULL,
        resumability TEXT NOT NULL CHECK (
            resumability IN ('safe', 'outcome_ready', 'blocked_unknown')
        ),
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE run_tool_calls (
        run_id TEXT NOT NULL REFERENCES agent_runs(run_id),
        tool_call_id TEXT NOT NULL,
        step_index INTEGER NOT NULL,
        ordinal INTEGER NOT NULL,
        tool_name TEXT NOT NULL,
        arguments_json TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        interaction_id TEXT,
        phase TEXT NOT NULL CHECK (
            phase IN ('prepared', 'claimed', 'committed', 'outcome_unknown')
        ),
        claim_activation_id TEXT,
        result_json TEXT,
        version INTEGER NOT NULL,
        prepared_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        claimed_at TEXT,
        committed_at TEXT,
        PRIMARY KEY (run_id, tool_call_id)
    )
    """,
    """
    CREATE TABLE run_interactions (
        interaction_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES agent_runs(run_id),
        session_id TEXT NOT NULL,
        step_index INTEGER NOT NULL,
        tool_call_id TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('pending', 'resolved', 'closed')),
        request_json TEXT NOT NULL,
        response_json TEXT,
        version INTEGER NOT NULL,
        expires_at TEXT,
        created_at TEXT NOT NULL,
        resolved_at TEXT,
        closed_at TEXT,
        close_reason TEXT
    )
    """,
    """
    CREATE UNIQUE INDEX one_open_interaction_per_run
    ON run_interactions(run_id)
    WHERE status IN ('pending', 'resolved')
    """,
    """
    CREATE TABLE run_events (
        run_id TEXT NOT NULL REFERENCES agent_runs(run_id),
        sequence INTEGER NOT NULL,
        session_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        occurred_at TEXT NOT NULL,
        activation_id TEXT,
        step_index INTEGER,
        correlation_id TEXT,
        payload_json TEXT NOT NULL,
        PRIMARY KEY (run_id, sequence)
    )
    """,
)

SCHEMA_STATEMENTS = (
    _IDENTITY_STATEMENT,
    _SESSIONS_STATEMENT,
    _SESSION_MESSAGES_STATEMENT,
    *_COMMON_STATEMENTS,
)


def _object_name(sql: str) -> str:
    match = re.search(r"CREATE\s+(?:UNIQUE\s+)?(?:TABLE|INDEX)\s+([a-z_]+)", sql, re.I)
    if match is None:
        raise ValueError("无法解析 schema object name")
    return match.group(1)


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def _expected_objects(statements: tuple[str, ...]) -> dict[str, str]:
    return {_object_name(statement): _normalize_sql(statement) for statement in statements}


_EXPECTED_OBJECTS = _expected_objects(SCHEMA_STATEMENTS)


def require_exact_schema(path: str | Path) -> None:
    """要求文件精确匹配当前 lifecycle schema。"""
    resolved = Path(path).resolve()
    uri = f"{resolved.as_uri()}?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            rows = connection.execute(
                """
                SELECT name, sql
                FROM sqlite_master
                WHERE type IN ('table', 'index')
                  AND name NOT LIKE 'sqlite_%'
                """
            ).fetchall()
            actual = {name: _normalize_sql(sql) for name, sql in rows if sql is not None}
            identity = (
                connection.execute("SELECT component, version FROM lifecycle_schema").fetchall()
                if "lifecycle_schema" in actual
                else []
            )
    except sqlite3.Error as exc:
        raise IrisLifecycleSchemaError(
            "无法只读检查 lifecycle SQLite schema",
            path=str(resolved),
        ) from exc

    if actual != _EXPECTED_OBJECTS or identity != [("agent_lifecycle", _SCHEMA_VERSION)]:
        raise IrisLifecycleSchemaError(
            "SQLite 文件不是当前 lifecycle schema",
            path=str(resolved),
            expected_version=_SCHEMA_VERSION,
        )


def create_schema(connection: sqlite3.Connection) -> None:
    """在 caller 提供的空 SQLite connection 中创建当前 schema。"""
    try:
        connection.execute("BEGIN IMMEDIATE")
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.execute(
            "INSERT INTO lifecycle_schema(component, version) VALUES (?, ?)",
            ("agent_lifecycle", _SCHEMA_VERSION),
        )
        connection.commit()
    except sqlite3.Error:
        connection.rollback()
        raise


__all__ = [
    "SCHEMA_STATEMENTS",
    "create_schema",
    "require_exact_schema",
]
