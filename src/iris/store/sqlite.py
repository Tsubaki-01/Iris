"""精确 lifecycle schema v1 的同步 SQLite store。"""

from __future__ import annotations

import json
import re
import sqlite3
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from ..exceptions import (
    IrisLifecycleSchemaError,
    IrisRunConflictError,
    IrisRunPersistenceError,
    IrisRunStateError,
)
from ..hitl.models import HumanInteraction, HumanInteractionRequest, HumanInteractionResponse
from ..lifecycle.models import (
    ActivationOutcome,
    ActivationRecord,
    ActivationStatus,
    AgentRunOptions,
    AgentRunRequest,
    RunCheckpoint,
    RunErrorInfo,
    RunEvent,
    RunPhase,
    RunRecord,
    RunStopReason,
    RunToolCallRecord,
    RunUsage,
    SessionSnapshot,
    project_result,
)
from ..lifecycle.store import (
    BeginActivation,
    ClaimToolCall,
    CommitModelStep,
    CommitToolResult,
    CreateRun,
    FinishRun,
    RecoverActiveRun,
    RequestCancellation,
    ReserveModelStep,
    ResolveInteraction,
    RunCommit,
    SuspendRun,
)
from ..message.message import Msg
from ..tools.base import ToolResult
from .in_memory import InMemoryLifecycleStore

_CommandT = TypeVar("_CommandT", bound=BaseModel)
_RESPONSE_ADAPTER = TypeAdapter(HumanInteractionResponse)
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC).isoformat()


def _object_name(sql: str) -> str:
    match = re.search(r"CREATE\s+(?:UNIQUE\s+)?(?:TABLE|INDEX)\s+([a-z_]+)", sql, re.I)
    if match is None:
        raise ValueError("无法解析 schema object name")
    return match.group(1)


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE lifecycle_schema (
        component TEXT PRIMARY KEY,
        version INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE sessions (
        session_id TEXT PRIMARY KEY,
        revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
        messages_json TEXT NOT NULL DEFAULT '[]',
        updated_at TEXT NOT NULL
    )
    """,
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
        outcome TEXT CHECK (outcome IN ('waiting', 'terminal')),
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

_EXPECTED_OBJECT_SQL = {
    _object_name(statement): _normalize_sql(statement) for statement in _SCHEMA_STATEMENTS
}


class SQLiteStore:
    """使用一个 SQLite 文件持久化完整 ``LifecycleStore`` aggregate。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()
        self._replays: dict[str, RunCommit] = {}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            is_empty = not self.path.exists() or self.path.stat().st_size == 0
            if is_empty:
                with self._connect() as connection:
                    _create_schema(connection)
            _inspect_schema_read_only(self.path)
        except IrisLifecycleSchemaError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise IrisRunPersistenceError(
                "无法初始化 lifecycle SQLite store",
                path=str(self.path),
            ) from exc

    def create_run(self, command: CreateRun) -> RunCommit:
        return self._mutate("create_run", command)

    def begin_activation(self, command: BeginActivation) -> RunCommit:
        return self._mutate("begin_activation", command)

    def reserve_model_step(self, command: ReserveModelStep) -> RunCommit:
        return self._mutate("reserve_model_step", command)

    def commit_model_step(self, command: CommitModelStep) -> RunCommit:
        return self._mutate("commit_model_step", command)

    def claim_tool_call(self, command: ClaimToolCall) -> RunCommit:
        return self._mutate("claim_tool_call", command)

    def commit_tool_result(self, command: CommitToolResult) -> RunCommit:
        return self._mutate("commit_tool_result", command)

    def suspend_run(self, command: SuspendRun) -> RunCommit:
        return self._mutate("suspend_run", command)

    def resolve_interaction(self, command: ResolveInteraction) -> RunCommit:
        return self._mutate("resolve_interaction", command)

    def request_cancellation(self, command: RequestCancellation) -> RunCommit:
        return self._mutate("request_cancellation", command)

    def finish_run(self, command: FinishRun) -> RunCommit:
        return self._mutate("finish_run", command)

    def recover_active_run(self, command: RecoverActiveRun) -> RunCommit:
        return self._mutate("recover_active_run", command)

    def load_run(self, run_id: str) -> RunRecord | None:
        return self._read("load_run", run_id)

    def load_session(self, session_id: str) -> SessionSnapshot:
        return self._read("load_session", session_id)

    def load_interaction(self, interaction_id: str) -> HumanInteraction | None:
        return self._read("load_interaction", interaction_id)

    def load_checkpoint(self, run_id: str) -> RunCheckpoint | None:
        return self._read("load_checkpoint", run_id)

    def list_tool_calls(self, run_id: str) -> list[RunToolCallRecord]:
        return self._read("list_tool_calls", run_id)

    def load_result(self, run_id: str) -> Any:
        return self._read("load_result", run_id)

    def list_events(self, run_id: str, after_sequence: int = 0) -> list[RunEvent]:
        return self._read("list_events", run_id, after_sequence)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _mutate(self, method_name: str, command: _CommandT) -> RunCommit:
        with self._lock:
            try:
                with self._connect() as connection:
                    _execute(connection, "BEGIN IMMEDIATE")
                    memory = _load_memory(
                        connection,
                        self._replays,
                        path=self.path,
                        operation=method_name,
                    )
                    method = getattr(memory, method_name)
                    commit: RunCommit = method(deepcopy(command))
                    _replace_all(connection, memory)
                    connection.commit()
                    self._replays = deepcopy(memory._replays)
                    return deepcopy(commit)
            except (IrisRunConflictError, IrisRunPersistenceError):
                raise
            except sqlite3.IntegrityError as exc:
                raise IrisRunConflictError(
                    "lifecycle SQLite constraint 冲突",
                    path=str(self.path),
                    operation=method_name,
                ) from exc
            except sqlite3.Error as exc:
                raise IrisRunPersistenceError(
                    "lifecycle SQLite transaction 失败",
                    path=str(self.path),
                    operation=method_name,
                ) from exc

    def _read(self, method_name: str, *args: object) -> Any:
        with self._lock:
            try:
                with self._connect() as connection:
                    memory = _load_memory(
                        connection,
                        self._replays,
                        path=self.path,
                        operation=method_name,
                    )
                    return getattr(memory, method_name)(*args)
            except IrisRunPersistenceError:
                raise
            except sqlite3.Error as exc:
                raise IrisRunPersistenceError(
                    "lifecycle SQLite read 失败",
                    path=str(self.path),
                    operation=method_name,
                ) from exc


def _inspect_schema_read_only(path: Path) -> None:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
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
            path=str(path),
        ) from exc
    if actual != _EXPECTED_OBJECT_SQL or identity != [("agent_lifecycle", 1)]:
        raise IrisLifecycleSchemaError(
            "SQLite 文件不是精确 lifecycle schema v1",
            path=str(path),
        )


def _create_schema(connection: sqlite3.Connection) -> None:
    try:
        _execute(connection, "BEGIN IMMEDIATE")
        for statement in _SCHEMA_STATEMENTS:
            _execute(connection, statement)
        _execute(
            connection,
            "INSERT INTO lifecycle_schema(component, version) VALUES (?, ?)",
            ("agent_lifecycle", 1),
        )
        connection.commit()
    except sqlite3.Error:
        connection.rollback()
        raise


def _execute(
    connection: sqlite3.Connection,
    sql: str,
    params: tuple[object, ...] = (),
) -> sqlite3.Cursor:
    """执行一条 mutation statement；测试可在此注入故障。"""
    return connection.execute(sql, params)


def _load_memory(
    connection: sqlite3.Connection,
    replays: dict[str, RunCommit],
    *,
    path: Path,
    operation: str,
) -> InMemoryLifecycleStore:
    try:
        memory = InMemoryLifecycleStore()
        memory._sessions = {
            row["session_id"]: SessionSnapshot(
                session_id=row["session_id"],
                revision=row["revision"],
                messages=[Msg.from_dict(item) for item in _load_json(row["messages_json"])],
            )
            for row in connection.execute("SELECT * FROM sessions")
        }
        memory._runs = {
            row["run_id"]: _row_to_run(row)
            for row in connection.execute("SELECT * FROM agent_runs")
        }
        memory._lanes = {
            row["session_id"]: row["run_id"]
            for row in connection.execute("SELECT session_id, run_id FROM session_run_lanes")
        }
        memory._activations = {
            row["activation_id"]: _row_to_activation(row, memory._runs)
            for row in connection.execute("SELECT * FROM run_activations")
        }
        memory._checkpoints = {
            row["run_id"]: _row_to_checkpoint(row)
            for row in connection.execute("SELECT * FROM run_checkpoints")
        }
        memory._tool_calls = {
            (row["run_id"], row["tool_call_id"]): _row_to_tool_call(row)
            for row in connection.execute("SELECT * FROM run_tool_calls")
        }
        memory._interactions = {
            row["interaction_id"]: _row_to_interaction(row)
            for row in connection.execute("SELECT * FROM run_interactions")
        }
        memory._events = {run_id: [] for run_id in memory._runs}
        for row in connection.execute("SELECT * FROM run_events ORDER BY run_id, sequence"):
            memory._events[row["run_id"]].append(_row_to_event(row))
        memory._results = {}
        for run in memory._runs.values():
            if run.phase is RunPhase.ACTIVE:
                continue
            interaction = (
                memory._interactions.get(run.pending_interaction_id)
                if run.pending_interaction_id is not None
                else None
            )
            memory._results[run.run_id] = project_result(run, interaction)
        memory._replays = deepcopy(replays)
        return memory
    except (
        ValidationError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        IndexError,
        IrisRunStateError,
        json.JSONDecodeError,
    ) as exc:
        raise IrisRunPersistenceError(
            "lifecycle SQLite durable row 无法验证",
            path=str(path),
            operation=operation,
        ) from exc


def _row_to_run(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        session_id=row["session_id"],
        agent_id=row["agent_id"],
        request=AgentRunRequest.model_validate(_load_json(row["request_json"])),
        options=AgentRunOptions.model_validate(_load_json(row["options_json"])),
        phase=row["phase"],
        stop_reason=row["stop_reason"],
        revision=row["run_revision"],
        current_activation_id=row["current_activation_id"],
        pending_interaction_id=row["pending_interaction_id"],
        cancellation_requested_at=row["cancellation_requested_at"],
        cancellation_reason=row["cancellation_reason"],
        usage=RunUsage.model_validate(_load_json(row["usage_json"])),
        environment_fingerprint=row["environment_fingerprint"],
        assistant_message=_load_message(row["assistant_message_json"]),
        error=(
            RunErrorInfo.model_validate(_load_json(row["error_json"]))
            if row["error_json"] is not None
            else None
        ),
        checkpoint_sequence=row["checkpoint_sequence"],
        last_event_sequence=row["last_event_sequence"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        updated_at=row["updated_at"],
        finished_at=row["finished_at"],
    )


def _row_to_activation(
    row: sqlite3.Row,
    runs: dict[str, RunRecord],
) -> ActivationRecord:
    status = ActivationStatus(row["status"])
    stored_outcome = row["outcome"]
    if stored_outcome is None:
        outcome = None
    elif stored_outcome == "waiting":
        outcome = ActivationOutcome.SUSPENDED
    elif status is ActivationStatus.ABANDONED:
        outcome = ActivationOutcome.RECOVERED
    else:
        stop_reason = runs[row["run_id"]].stop_reason
        outcome = {
            RunStopReason.COMPLETED: ActivationOutcome.COMPLETED,
            RunStopReason.CANCELLED: ActivationOutcome.CANCELLED,
            RunStopReason.OUTCOME_UNKNOWN: ActivationOutcome.OUTCOME_UNKNOWN,
        }.get(stop_reason, ActivationOutcome.FAILED)
    return ActivationRecord(
        activation_id=row["activation_id"],
        run_id=row["run_id"],
        ordinal=row["ordinal"],
        kind=row["kind"],
        status=status,
        outcome=outcome,
        started_at=row["started_at"],
        ended_at=row["ended_at"],
    )


def _row_to_checkpoint(row: sqlite3.Row) -> RunCheckpoint:
    return RunCheckpoint(
        run_id=row["run_id"],
        sequence=row["sequence"],
        activation_id=row["activation_id"],
        checkpoint_version=row["checkpoint_version"],
        engine_cursor=_load_json(row["cursor_json"]),
        session_revision=row["session_revision"],
        model_steps_reserved=row["model_steps_reserved"],
        model_steps_committed=row["model_steps_committed"],
        environment_fingerprint=row["environment_fingerprint"],
        resumability=row["resumability"],
    )


def _row_to_tool_call(row: sqlite3.Row) -> RunToolCallRecord:
    updated_at = row["committed_at"] or row["claimed_at"] or row["prepared_at"]
    return RunToolCallRecord(
        run_id=row["run_id"],
        tool_call_id=row["tool_call_id"],
        step_index=row["step_index"],
        ordinal=row["ordinal"],
        tool_name=row["tool_name"],
        arguments=_load_json(row["arguments_json"]),
        fingerprint=row["fingerprint"],
        interaction_id=row["interaction_id"],
        phase=row["phase"],
        claim_activation_id=row["claim_activation_id"],
        result=(
            ToolResult.model_validate(_load_json(row["result_json"]))
            if row["result_json"] is not None
            else None
        ),
        version=row["version"],
        created_at=row["prepared_at"],
        updated_at=updated_at,
        claimed_at=row["claimed_at"],
        committed_at=row["committed_at"],
    )


def _row_to_interaction(row: sqlite3.Row) -> HumanInteraction:
    response = (
        _RESPONSE_ADAPTER.validate_python(_load_json(row["response_json"]))
        if row["response_json"] is not None
        else None
    )
    return HumanInteraction(
        interaction_id=row["interaction_id"],
        run_id=row["run_id"],
        session_id=row["session_id"],
        step_index=row["step_index"],
        tool_call_id=row["tool_call_id"],
        status=row["status"],
        request=HumanInteractionRequest.model_validate(_load_json(row["request_json"])),
        response=response,
        version=row["version"],
        expires_at=row["expires_at"],
        created_at=row["created_at"],
        resolved_at=row["resolved_at"],
        closed_at=row["closed_at"],
        close_reason=row["close_reason"],
    )


def _row_to_event(row: sqlite3.Row) -> RunEvent:
    return RunEvent(
        run_id=row["run_id"],
        sequence=row["sequence"],
        session_id=row["session_id"],
        kind=row["kind"],
        occurred_at=row["occurred_at"],
        activation_id=row["activation_id"],
        step_index=row["step_index"],
        correlation_id=row["correlation_id"],
        payload=_load_json(row["payload_json"]),
    )


def _replace_all(connection: sqlite3.Connection, memory: InMemoryLifecycleStore) -> None:
    for table in (
        "run_events",
        "run_interactions",
        "run_tool_calls",
        "run_checkpoints",
        "run_activations",
        "session_run_lanes",
        "agent_runs",
        "sessions",
    ):
        _execute(connection, f"DELETE FROM {table}")

    for session in memory._sessions.values():
        updated_at = max(
            (
                run.updated_at.isoformat()
                for run in memory._runs.values()
                if run.session_id == session.session_id
            ),
            default=_EPOCH,
        )
        _execute(
            connection,
            """INSERT INTO sessions(
                session_id, revision, messages_json, updated_at
            ) VALUES (?, ?, ?, ?)""",
            (session.session_id, session.revision, _dump_json(session.messages), updated_at),
        )
    for run in memory._runs.values():
        checkpoint = memory._checkpoints.get(run.run_id)
        session_revision = (
            checkpoint.session_revision
            if checkpoint is not None
            else memory._sessions[run.session_id].revision
        )
        _execute(connection, _INSERT_RUN, _run_values(run, session_revision))
    for session_id, run_id in memory._lanes.items():
        run = memory._runs[run_id]
        _execute(
            connection,
            """INSERT INTO session_run_lanes(
                session_id, run_id, revision, acquired_at
            ) VALUES (?, ?, ?, ?)""",
            (session_id, run_id, run.revision, run.created_at.isoformat()),
        )
    for activation in memory._activations.values():
        _execute(
            connection,
            """INSERT INTO run_activations(
                activation_id, run_id, ordinal, kind, status, outcome, started_at, ended_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                activation.activation_id,
                activation.run_id,
                activation.ordinal,
                activation.kind.value,
                activation.status.value,
                _stored_activation_outcome(activation),
                activation.started_at.isoformat(),
                _iso(activation.ended_at),
            ),
        )
    for checkpoint in memory._checkpoints.values():
        _execute(
            connection,
            """INSERT INTO run_checkpoints(
                run_id, sequence, activation_id, checkpoint_version, cursor_json,
                session_revision, model_steps_reserved, model_steps_committed,
                environment_fingerprint, resumability, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                checkpoint.run_id,
                checkpoint.sequence,
                checkpoint.activation_id,
                checkpoint.checkpoint_version,
                _dump_json(checkpoint.engine_cursor),
                checkpoint.session_revision,
                checkpoint.model_steps_reserved,
                checkpoint.model_steps_committed,
                checkpoint.environment_fingerprint,
                checkpoint.resumability.value,
                memory._runs[checkpoint.run_id].updated_at.isoformat(),
            ),
        )
    for call in memory._tool_calls.values():
        _execute(connection, _INSERT_TOOL_CALL, _tool_call_values(call))
    for interaction in memory._interactions.values():
        _execute(connection, _INSERT_INTERACTION, _interaction_values(interaction))
    for events in memory._events.values():
        for event in events:
            _execute(connection, _INSERT_EVENT, _event_values(event))


_INSERT_RUN = """INSERT INTO agent_runs(
    run_id, session_id, agent_id, phase, stop_reason, request_json, options_json,
    environment_fingerprint, session_revision, run_revision, current_activation_id,
    pending_interaction_id, cancellation_requested_at, cancellation_reason,
    model_steps_reserved, model_steps_committed, tool_calls_committed, usage_json,
    assistant_message_json, error_json, checkpoint_sequence, last_event_sequence,
    created_at, started_at, updated_at, finished_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

_INSERT_TOOL_CALL = """INSERT INTO run_tool_calls(
    run_id, tool_call_id, step_index, ordinal, tool_name, arguments_json, fingerprint,
    interaction_id, phase, claim_activation_id, result_json, version, prepared_at,
    claimed_at, committed_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

_INSERT_INTERACTION = """INSERT INTO run_interactions(
    interaction_id, run_id, session_id, step_index, tool_call_id, status,
    request_json, response_json, version, expires_at, created_at, resolved_at,
    closed_at, close_reason
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

_INSERT_EVENT = """INSERT INTO run_events(
    run_id, sequence, session_id, kind, occurred_at, activation_id, step_index,
    correlation_id, payload_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""


def _run_values(run: RunRecord, session_revision: int) -> tuple[object, ...]:
    usage = run.usage
    return (
        run.run_id,
        run.session_id,
        run.agent_id,
        run.phase.value,
        run.stop_reason.value if run.stop_reason is not None else None,
        _dump_json(run.request),
        _dump_json(run.options),
        run.environment_fingerprint,
        session_revision,
        run.revision,
        run.current_activation_id,
        run.pending_interaction_id,
        _iso(run.cancellation_requested_at),
        run.cancellation_reason,
        usage.model_steps_reserved,
        usage.model_steps_committed,
        usage.tool_calls_committed,
        _dump_json(usage),
        _dump_json(run.assistant_message) if run.assistant_message is not None else None,
        _dump_json(run.error) if run.error is not None else None,
        run.checkpoint_sequence,
        run.last_event_sequence,
        run.created_at.isoformat(),
        run.started_at.isoformat(),
        run.updated_at.isoformat(),
        _iso(run.finished_at),
    )


def _tool_call_values(call: RunToolCallRecord) -> tuple[object, ...]:
    return (
        call.run_id,
        call.tool_call_id,
        call.step_index,
        call.ordinal,
        call.tool_name,
        _dump_json(call.arguments),
        call.fingerprint,
        call.interaction_id,
        call.phase.value,
        call.claim_activation_id,
        _dump_json(call.result) if call.result is not None else None,
        call.version,
        call.created_at.isoformat(),
        _iso(call.claimed_at),
        _iso(call.committed_at),
    )


def _interaction_values(interaction: HumanInteraction) -> tuple[object, ...]:
    return (
        interaction.interaction_id,
        interaction.run_id,
        interaction.session_id,
        interaction.step_index,
        interaction.tool_call_id,
        interaction.status.value,
        _dump_json(interaction.request),
        _dump_json(interaction.response) if interaction.response is not None else None,
        interaction.version,
        _iso(interaction.expires_at),
        interaction.created_at.isoformat(),
        _iso(interaction.resolved_at),
        _iso(interaction.closed_at),
        interaction.close_reason,
    )


def _event_values(event: RunEvent) -> tuple[object, ...]:
    return (
        event.run_id,
        event.sequence,
        event.session_id,
        event.kind.value,
        event.occurred_at.isoformat(),
        event.activation_id,
        event.step_index,
        event.correlation_id,
        _dump_json(event.payload),
    )


def _stored_activation_outcome(activation: ActivationRecord) -> str | None:
    if activation.outcome is None:
        return None
    if activation.outcome is ActivationOutcome.SUSPENDED:
        return "waiting"
    return "terminal"


def _dump_json(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
        value = [item.model_dump(mode="json") for item in value]
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _load_json(value: str) -> Any:
    return json.loads(value)


def _load_message(value: str | None) -> Msg | None:
    return None if value is None else Msg.from_dict(_load_json(value))


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


__all__ = ["SQLiteStore"]
