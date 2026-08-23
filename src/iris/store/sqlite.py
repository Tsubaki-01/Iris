"""精确 lifecycle schema v1 的同步 SQLite store。"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from ..exceptions import (
    IrisLifecycleSchemaError,
    IrisRunConflictError,
    IrisRunNotFoundError,
    IrisRunPersistenceError,
    IrisRunRecoveryError,
    IrisRunStateError,
)
from ..hitl.models import (
    HumanInteraction,
    HumanInteractionRequest,
    HumanInteractionResponse,
    InteractionStatus,
    PermissionInteractionResponse,
    QuestionInteractionResponse,
)
from ..lifecycle.models import (
    ActivationKind,
    ActivationOutcome,
    ActivationRecord,
    ActivationStatus,
    AgentRunOptions,
    AgentRunRequest,
    CheckpointResumability,
    RecoveryDisposition,
    RunCheckpoint,
    RunControlSnapshot,
    RunErrorInfo,
    RunEvent,
    RunEventKind,
    RunPhase,
    RunRecord,
    RunResult,
    RunStopReason,
    RunToolCallRecord,
    RunUsage,
    SessionSnapshot,
    ToolCallPhase,
    project_result,
)
from ..lifecycle.store import (
    ClaimToolCall,
    CommitModelStep,
    CommitToolResult,
    CreateRun,
    FinishRun,
    RecoverActiveRun,
    RequestCancellation,
    ReserveModelStep,
    ResolveInteraction,
    ResumeWaitingRun,
    RunCommit,
    SuspendRun,
)
from ..message.message import Msg, TextBlock
from ..tools.base import ToolErrorInfo, ToolResult
from ._terminal_closure import build_terminal_tool_closure

_CommandT = TypeVar("_CommandT", bound=BaseModel)
_ReadT = TypeVar("_ReadT")
_RESPONSE_ADAPTER = TypeAdapter(HumanInteractionResponse)


class _ActiveCommand(Protocol):
    """Active mutation 共用的 CAS 与 activation fence 输入。"""

    run_id: str
    expected_run_revision: int
    activation_id: str


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
        return self._mutate(
            "create_run",
            command,
            self._create_run,
            replayable=False,
        )

    def resume_waiting_run(self, command: ResumeWaitingRun) -> RunCommit:
        return self._mutate(
            "resume_waiting_run",
            command,
            self._resume_waiting_run,
        )

    def reserve_model_step(self, command: ReserveModelStep) -> RunCommit:
        return self._mutate(
            "reserve_model_step",
            command,
            self._reserve_model_step,
        )

    def commit_model_step(self, command: CommitModelStep) -> RunCommit:
        return self._mutate(
            "commit_model_step",
            command,
            self._commit_model_step,
        )

    def claim_tool_call(self, command: ClaimToolCall) -> RunCommit:
        return self._mutate(
            "claim_tool_call",
            command,
            self._claim_tool_call,
        )

    def commit_tool_result(self, command: CommitToolResult) -> RunCommit:
        return self._mutate(
            "commit_tool_result",
            command,
            self._commit_tool_result,
        )

    def suspend_run(self, command: SuspendRun) -> RunCommit:
        return self._mutate(
            "suspend_run",
            command,
            self._suspend_run,
        )

    def resolve_interaction(self, command: ResolveInteraction) -> RunCommit:
        return self._mutate(
            "resolve_interaction",
            command,
            self._resolve_interaction,
        )

    def request_cancellation(self, command: RequestCancellation) -> RunCommit:
        return self._mutate(
            "request_cancellation",
            command,
            self._request_cancellation,
        )

    def finish_run(self, command: FinishRun) -> RunCommit:
        return self._mutate(
            "finish_run",
            command,
            self._finish_run,
        )

    def recover_active_run(self, command: RecoverActiveRun) -> RunCommit:
        return self._mutate(
            "recover_active_run",
            command,
            self._recover_active_run,
        )

    def load_run(self, run_id: str) -> RunRecord | None:
        return self._read(
            "load_run",
            lambda connection: self._select_run(
                connection,
                run_id,
                operation="load_run",
            ),
        )

    def load_run_control(self, run_id: str) -> RunControlSnapshot | None:
        """只读取 activation/cancellation 判断所需的八个 run 字段。"""

        def read(connection: sqlite3.Connection) -> RunControlSnapshot | None:
            row = connection.execute(
                """SELECT run_id, phase, run_revision, current_activation_id,
                    cancellation_requested_at, cancellation_reason,
                    last_event_sequence, updated_at
                FROM agent_runs WHERE run_id = ?""",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return _decode_row(
                _row_to_run_control,
                row,
                path=self.path,
                operation="load_run_control",
            )

        return self._read("load_run_control", read)

    def load_session(self, session_id: str) -> SessionSnapshot:
        return self._read(
            "load_session",
            lambda connection: self._select_session(
                connection,
                session_id,
                operation="load_session",
            ),
        )

    def load_session_lane(self, session_id: str) -> str | None:
        def read(connection: sqlite3.Connection) -> str | None:
            row = connection.execute(
                "SELECT run_id FROM session_run_lanes WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return None if row is None else str(row["run_id"])

        return self._read("load_session_lane", read)

    def load_interaction(self, interaction_id: str) -> HumanInteraction | None:
        return self._read(
            "load_interaction",
            lambda connection: self._select_interaction(
                connection,
                interaction_id,
                operation="load_interaction",
            ),
        )

    def load_checkpoint(self, run_id: str) -> RunCheckpoint | None:
        return self._read(
            "load_checkpoint",
            lambda connection: self._select_checkpoint(
                connection,
                run_id,
                operation="load_checkpoint",
            ),
        )

    def load_tool_call(
        self,
        run_id: str,
        tool_call_id: str,
    ) -> RunToolCallRecord | None:
        """使用现有 composite primary key 读取 exact tool call。"""
        return self._read(
            "load_tool_call",
            lambda connection: self._select_tool_call(
                connection,
                run_id,
                tool_call_id,
                operation="load_tool_call",
            ),
        )

    def list_tool_calls(self, run_id: str) -> list[RunToolCallRecord]:
        def read(connection: sqlite3.Connection) -> list[RunToolCallRecord]:
            if self._select_run(connection, run_id, operation="list_tool_calls") is None:
                raise IrisRunNotFoundError("run 不存在", run_id=run_id)
            return [
                _decode_row(
                    _row_to_tool_call,
                    row,
                    path=self.path,
                    operation="list_tool_calls",
                )
                for row in connection.execute(
                    """SELECT * FROM run_tool_calls
                    WHERE run_id = ?
                    ORDER BY step_index, ordinal""",
                    (run_id,),
                )
            ]

        return self._read("list_tool_calls", read)

    def load_result(self, run_id: str) -> RunResult | None:
        def read(connection: sqlite3.Connection) -> RunResult | None:
            run = self._select_run(connection, run_id, operation="load_result")
            if run is None or run.phase is RunPhase.ACTIVE:
                return None
            interaction = (
                self._select_interaction(
                    connection,
                    run.pending_interaction_id,
                    operation="load_result",
                )
                if run.phase is RunPhase.WAITING
                else None
            )
            return _project_durable_result(
                run,
                interaction,
                path=self.path,
                operation="load_result",
            )

        return self._read("load_result", read)

    def list_events(self, run_id: str, after_sequence: int = 0) -> list[RunEvent]:
        if after_sequence < 0:
            raise IrisRunStateError("after_sequence 不能小于 0", after_sequence=after_sequence)

        def read(connection: sqlite3.Connection) -> list[RunEvent]:
            if self._select_run(connection, run_id, operation="list_events") is None:
                raise IrisRunNotFoundError("run 不存在", run_id=run_id)
            return [
                _decode_row(
                    _row_to_event,
                    row,
                    path=self.path,
                    operation="list_events",
                )
                for row in connection.execute(
                    """SELECT * FROM run_events
                    WHERE run_id = ? AND sequence > ?
                    ORDER BY sequence""",
                    (run_id, after_sequence),
                )
            ]

        return self._read("list_events", read)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _mutate(
        self,
        operation: str,
        command: _CommandT,
        handler: Callable[[sqlite3.Connection, _CommandT], RunCommit],
        *,
        replayable: bool = True,
    ) -> RunCommit:
        with self._lock:
            try:
                with self._connect() as connection:
                    _execute(connection, "BEGIN IMMEDIATE")
                    if replayable:
                        replay = self._load_replay(connection, operation, command)
                        if replay is not None:
                            connection.commit()
                            return replay
                    commit = handler(connection, deepcopy(command))
                    connection.commit()
                    if replayable:
                        self._replays[_replay_key(operation, command)] = deepcopy(commit)
                    return deepcopy(commit)
            except (IrisRunConflictError, IrisRunPersistenceError):
                raise
            except sqlite3.IntegrityError as exc:
                raise IrisRunConflictError(
                    "lifecycle SQLite constraint 冲突",
                    path=str(self.path),
                    operation=operation,
                ) from exc
            except sqlite3.Error as exc:
                raise IrisRunPersistenceError(
                    "lifecycle SQLite transaction 失败",
                    path=str(self.path),
                    operation=operation,
                ) from exc

    def _load_replay(
        self,
        connection: sqlite3.Connection,
        operation: str,
        command: BaseModel,
    ) -> RunCommit | None:
        """把 process-local replay 刷新为当前 durable facts。"""
        replay = self._replays.get(_replay_key(operation, command))
        if replay is None:
            return None
        run = self._require_run(connection, replay.run.run_id, operation=operation)
        session = (
            self._select_session(
                connection,
                run.session_id,
                operation=operation,
            )
            if replay.session is not None
            else None
        )
        checkpoint = self._select_checkpoint(
            connection,
            run.run_id,
            operation=operation,
        )
        interaction = (
            self._select_interaction(
                connection,
                replay.interaction.interaction_id,
                operation=operation,
            )
            if replay.interaction is not None
            else None
        )
        return replay.model_copy(
            deep=True,
            update={
                "run": run,
                "session": session,
                "checkpoint": checkpoint,
                "interaction": interaction,
                "events": (),
                "result": self._select_result(connection, run, operation=operation),
            },
        )

    def _require_run(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        *,
        operation: str,
    ) -> RunRecord:
        """读取目标 run，不存在时抛出领域 not-found。"""
        run = self._select_run(connection, run_id, operation=operation)
        if run is None:
            raise IrisRunNotFoundError("run 不存在", run_id=run_id)
        return run

    def _require_checkpoint(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        *,
        operation: str,
    ) -> RunCheckpoint:
        """读取 current checkpoint，不存在时抛出 state error。"""
        checkpoint = self._select_checkpoint(connection, run_id, operation=operation)
        if checkpoint is None:
            raise IrisRunStateError("run 不存在 current checkpoint", run_id=run_id)
        return checkpoint

    def _require_activation(
        self,
        connection: sqlite3.Connection,
        activation_id: str | None,
        *,
        operation: str,
    ) -> ActivationRecord:
        """读取 exact activation fence。"""
        activation = self._select_activation(connection, activation_id, operation=operation)
        if activation is None:
            raise IrisRunConflictError(
                "activation 不存在或 fence 已变化",
                activation_id=activation_id,
            )
        return activation

    def _require_interaction(
        self,
        connection: sqlite3.Connection,
        interaction_id: str | None,
        *,
        operation: str,
    ) -> HumanInteraction:
        """读取 exact interaction，不存在时保留领域 conflict 语义。"""
        interaction = self._select_interaction(
            connection,
            interaction_id,
            operation=operation,
        )
        if interaction is None:
            raise IrisRunConflictError(
                "interaction 不存在或已变化",
                interaction_id=interaction_id,
            )
        return interaction

    def _require_tool_call(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        tool_call_id: str,
        *,
        operation: str,
    ) -> RunToolCallRecord:
        """读取 exact tool call，不存在时抛出领域 not-found。"""
        tool_call = self._select_tool_call(
            connection,
            run_id,
            tool_call_id,
            operation=operation,
        )
        if tool_call is None:
            raise IrisRunNotFoundError(
                "tool call 不存在",
                run_id=run_id,
                tool_call_id=tool_call_id,
            )
        return tool_call

    @staticmethod
    def _require_revision(run: RunRecord, expected: int) -> None:
        """校验 run revision CAS。"""
        if run.revision != expected:
            raise IrisRunConflictError(
                "run revision 已变化",
                run_id=run.run_id,
                expected=expected,
                actual=run.revision,
            )

    def _require_lane(
        self,
        connection: sqlite3.Connection,
        run: RunRecord,
    ) -> None:
        """校验 session lane 仍由目标 run 持有。"""
        row = connection.execute(
            "SELECT run_id FROM session_run_lanes WHERE session_id = ?",
            (run.session_id,),
        ).fetchone()
        if row is None or row["run_id"] != run.run_id:
            raise IrisRunConflictError("session lane owner 已变化", run_id=run.run_id)

    def _require_active(
        self,
        connection: sqlite3.Connection,
        command: _ActiveCommand,
        *,
        operation: str,
    ) -> RunRecord:
        """校验 active run、revision、activation fence 与 lane。"""
        run = self._require_run(connection, command.run_id, operation=operation)
        self._require_revision(run, command.expected_run_revision)
        if run.phase is RunPhase.TERMINAL:
            raise IrisRunStateError("terminal run 不接受进一步 mutation", run_id=run.run_id)
        if run.phase is not RunPhase.ACTIVE:
            raise IrisRunStateError("command 要求 active run", run_id=run.run_id)
        if run.current_activation_id != command.activation_id:
            raise IrisRunConflictError("activation fence 已变化", run_id=run.run_id)
        self._require_lane(connection, run)
        return run

    def _require_history_preconditions(
        self,
        connection: sqlite3.Connection,
        run: RunRecord,
        expected_session_revision: int,
        *,
        operation: str,
    ) -> SessionSnapshot:
        """校验 lane 与 session history revision。"""
        self._require_lane(connection, run)
        session = self._select_session(
            connection,
            run.session_id,
            operation=operation,
        )
        if session.revision != expected_session_revision:
            raise IrisRunConflictError(
                "session revision 已变化",
                session_id=run.session_id,
                expected=expected_session_revision,
                actual=session.revision,
            )
        return session

    def _select_result(
        self,
        connection: sqlite3.Connection,
        run: RunRecord,
        *,
        operation: str,
    ) -> RunResult | None:
        """从当前连接读取并投影目标 run 的 durable result。"""
        if run.phase is RunPhase.ACTIVE:
            return None
        interaction = (
            self._select_interaction(
                connection,
                run.pending_interaction_id,
                operation=operation,
            )
            if run.phase is RunPhase.WAITING
            else None
        )
        return _project_durable_result(
            run,
            interaction,
            path=self.path,
            operation=operation,
        )

    def _resume_waiting_run(
        self,
        connection: sqlite3.Connection,
        command: ResumeWaitingRun,
    ) -> RunCommit:
        """从 resolved waiting run 增量建立新的 active fence。"""
        operation = "resume_waiting_run"
        run = self._require_run(connection, command.run_id, operation=operation)
        self._require_revision(run, command.expected_run_revision)
        if run.phase is not RunPhase.WAITING:
            raise IrisRunStateError("只有 waiting run 可以开始新 activation", run_id=run.run_id)
        checkpoint = self._require_checkpoint(connection, run.run_id, operation=operation)
        if checkpoint.sequence != command.expected_checkpoint_sequence:
            raise IrisRunConflictError("checkpoint sequence 已变化", run_id=run.run_id)
        if (
            self._select_activation(
                connection,
                command.new_activation_id,
                operation=operation,
            )
            is not None
        ):
            raise IrisRunConflictError(
                "activation_id 已存在",
                activation_id=command.new_activation_id,
            )
        interaction = self._require_interaction(
            connection,
            run.pending_interaction_id,
            operation=operation,
        )
        if interaction.status is not InteractionStatus.RESOLVED:
            raise IrisRunStateError(
                "interaction 尚未 resolved",
                interaction_id=interaction.interaction_id,
            )
        self._require_lane(connection, run)

        row = connection.execute(
            """SELECT COALESCE(MAX(ordinal), 0) AS max_ordinal
            FROM run_activations WHERE run_id = ?""",
            (run.run_id,),
        ).fetchone()
        ordinal = int(row["max_ordinal"]) + 1
        activation = ActivationRecord(
            activation_id=command.new_activation_id,
            run_id=run.run_id,
            ordinal=ordinal,
            kind=command.kind,
            status=ActivationStatus.ACTIVE,
            started_at=command.now,
        )
        rebound = RunCheckpoint.model_validate(
            checkpoint.model_dump()
            | {
                "sequence": checkpoint.sequence + 1,
                "activation_id": activation.activation_id,
            }
        )
        closed = interaction.model_copy(
            deep=True,
            update={
                "status": InteractionStatus.CLOSED,
                "version": interaction.version + 1,
                "closed_at": command.now,
                "close_reason": "resumed",
            },
        )
        event_sequence = run.last_event_sequence + 1
        updated = _replace_run(
            run,
            phase=RunPhase.ACTIVE,
            revision=run.revision + 1,
            current_activation_id=activation.activation_id,
            pending_interaction_id=None,
            checkpoint_sequence=rebound.sequence,
            last_event_sequence=event_sequence,
            updated_at=command.now,
        )
        event = _make_event(
            updated,
            RunEventKind.ACTIVATION_STARTED,
            command.now,
            sequence=event_sequence,
            activation_id=activation.activation_id,
        )
        self._touch_session(
            connection,
            run.session_id,
            checkpoint.session_revision,
            command.now,
        )
        self._insert_activation(connection, activation)
        self._update_run(connection, run, updated, checkpoint.session_revision)
        self._update_checkpoint(connection, checkpoint, rebound, command.now)
        self._update_interaction(connection, interaction, closed)
        self._insert_event(connection, event)
        return RunCommit(
            run=updated,
            checkpoint=rebound,
            interaction=closed,
            events=(event,),
        )

    def _reserve_model_step(
        self,
        connection: sqlite3.Connection,
        command: ReserveModelStep,
    ) -> RunCommit:
        """增量提交 model-step reservation 或预算耗尽结算。"""
        operation = "reserve_model_step"
        run = self._require_active(connection, command, operation=operation)
        checkpoint = self._require_checkpoint(connection, run.run_id, operation=operation)
        if run.usage.model_steps_reserved >= run.options.limits.max_model_steps:
            activation = self._require_activation(
                connection,
                run.current_activation_id,
                operation=operation,
            )
            settled = ActivationRecord.model_validate(
                activation.model_dump()
                | {
                    "status": ActivationStatus.SETTLED,
                    "outcome": ActivationOutcome.FAILED,
                    "ended_at": command.now,
                }
            )
            sequence = run.last_event_sequence + 1
            updated = _replace_run(
                run,
                phase=RunPhase.TERMINAL,
                stop_reason=RunStopReason.BUDGET_EXHAUSTED,
                revision=run.revision + 1,
                current_activation_id=None,
                last_event_sequence=sequence,
                updated_at=command.now,
                finished_at=command.now,
            )
            event = _make_event(
                updated,
                RunEventKind.RUN_TERMINAL,
                command.now,
                sequence=sequence,
                activation_id=activation.activation_id,
                payload={"stop_reason": RunStopReason.BUDGET_EXHAUSTED.value},
            )
            self._touch_session(
                connection,
                run.session_id,
                checkpoint.session_revision,
                command.now,
            )
            self._update_run(connection, run, updated, checkpoint.session_revision)
            self._update_activation(connection, activation, settled)
            self._delete_lane(connection, run, require_match=True)
            self._insert_event(connection, event)
            return RunCommit(
                run=updated,
                checkpoint=checkpoint,
                events=(event,),
                result=project_result(updated),
            )

        usage = RunUsage.model_validate(
            run.usage.model_dump() | {"model_steps_reserved": run.usage.model_steps_reserved + 1}
        )
        updated_checkpoint = checkpoint.model_copy(
            update={"model_steps_reserved": usage.model_steps_reserved}
        )
        sequence = run.last_event_sequence + 1
        updated = _replace_run(
            run,
            revision=run.revision + 1,
            usage=usage,
            last_event_sequence=sequence,
            updated_at=command.now,
        )
        event = _make_event(
            updated,
            RunEventKind.MODEL_STEP_RESERVED,
            command.now,
            sequence=sequence,
            activation_id=command.activation_id,
            step_index=usage.model_steps_reserved - 1,
        )
        self._touch_session(
            connection,
            run.session_id,
            checkpoint.session_revision,
            command.now,
        )
        self._update_run(connection, run, updated, checkpoint.session_revision)
        self._update_checkpoint(connection, checkpoint, updated_checkpoint, command.now)
        self._insert_event(connection, event)
        return RunCommit(
            run=updated,
            checkpoint=updated_checkpoint,
            events=(event,),
        )

    def _commit_model_step(
        self,
        connection: sqlite3.Connection,
        command: CommitModelStep,
    ) -> RunCommit:
        """增量提交模型响应、history、tool intents 与 checkpoint。"""
        operation = "commit_model_step"
        run = self._require_active(connection, command, operation=operation)
        session = self._require_history_preconditions(
            connection,
            run,
            command.expected_session_revision,
            operation=operation,
        )
        current_checkpoint = self._require_checkpoint(
            connection,
            run.run_id,
            operation=operation,
        )
        next_session = _append_messages(session, command.message_delta)
        _validate_checkpoint_replacement(
            run,
            current_checkpoint,
            command.checkpoint,
            command.activation_id,
            next_session.revision,
            command.usage,
        )
        if command.usage.model_steps_reserved != run.usage.model_steps_reserved:
            raise IrisRunConflictError("model-step reserved counter 已变化", run_id=run.run_id)
        if command.usage.model_steps_committed != run.usage.model_steps_committed + 1:
            raise IrisRunStateError("model-step commit 必须恰好推进一个 committed counter")
        prepared = self._validate_prepared_calls(
            connection,
            run,
            command.prepared_tool_calls,
            operation=operation,
        )
        sequence = run.last_event_sequence + 1
        updated = _replace_run(
            run,
            revision=run.revision + 1,
            usage=command.usage,
            assistant_message=command.assistant_message,
            checkpoint_sequence=command.checkpoint.sequence,
            last_event_sequence=sequence,
            updated_at=command.now,
        )
        event = _make_event(
            updated,
            RunEventKind.MODEL_STEP_COMMITTED,
            command.now,
            sequence=sequence,
            activation_id=command.activation_id,
            step_index=command.usage.model_steps_committed - 1,
        )
        if command.message_delta:
            self._update_session(connection, session, next_session, command.now)
        else:
            self._touch_session(
                connection,
                session.session_id,
                session.revision,
                command.now,
            )
        self._update_run(connection, run, updated, next_session.revision)
        self._update_checkpoint(
            connection,
            current_checkpoint,
            command.checkpoint,
            command.now,
        )
        for tool_call in prepared:
            self._insert_tool_call(connection, tool_call)
        self._insert_event(connection, event)
        return RunCommit(
            run=updated,
            session=next_session if command.message_delta else None,
            checkpoint=command.checkpoint,
            events=(event,),
        )

    def _claim_tool_call(
        self,
        connection: sqlite3.Connection,
        command: ClaimToolCall,
    ) -> RunCommit:
        """将一条 prepared tool call 增量转为 claimed。"""
        operation = "claim_tool_call"
        run = self._require_active(connection, command, operation=operation)
        tool_call = self._require_tool_call(
            connection,
            run.run_id,
            command.tool_call_id,
            operation=operation,
        )
        if tool_call.version != command.expected_tool_version:
            raise IrisRunConflictError(
                "tool call version 已变化",
                tool_call_id=command.tool_call_id,
            )
        if tool_call.fingerprint != command.fingerprint:
            raise IrisRunConflictError(
                "tool call fingerprint 不匹配",
                tool_call_id=command.tool_call_id,
            )
        if tool_call.phase is not ToolCallPhase.PREPARED:
            raise IrisRunStateError("只有 prepared tool call 可以 claim")
        if run.cancellation_requested_at is not None:
            raise IrisRunStateError("已请求取消的 run 不接受新的 tool call claim")
        checkpoint = self._require_checkpoint(connection, run.run_id, operation=operation)
        claimed = RunToolCallRecord.model_validate(
            tool_call.model_dump()
            | {
                "phase": ToolCallPhase.CLAIMED,
                "claim_activation_id": command.activation_id,
                "version": tool_call.version + 1,
                "updated_at": command.now,
                "claimed_at": command.now,
            }
        )
        sequence = run.last_event_sequence + 1
        updated = _replace_run(
            run,
            revision=run.revision + 1,
            last_event_sequence=sequence,
            updated_at=command.now,
        )
        event = _make_event(
            updated,
            RunEventKind.TOOL_CALL_CLAIMED,
            command.now,
            sequence=sequence,
            activation_id=command.activation_id,
            step_index=tool_call.step_index,
            correlation_id=tool_call.tool_call_id,
        )
        self._touch_session(
            connection,
            run.session_id,
            checkpoint.session_revision,
            command.now,
        )
        self._update_run(connection, run, updated, checkpoint.session_revision)
        self._update_tool_call(connection, tool_call, claimed)
        self._insert_event(connection, event)
        return RunCommit(
            run=updated,
            checkpoint=checkpoint,
            events=(event,),
        )

    def _commit_tool_result(
        self,
        connection: sqlite3.Connection,
        command: CommitToolResult,
    ) -> RunCommit:
        """增量提交精确工具结果、history 与 checkpoint。"""
        operation = "commit_tool_result"
        run = self._require_active(connection, command, operation=operation)
        session = self._require_history_preconditions(
            connection,
            run,
            command.expected_session_revision,
            operation=operation,
        )
        tool_call = self._require_tool_call(
            connection,
            run.run_id,
            command.tool_call_id,
            operation=operation,
        )
        if tool_call.version != command.expected_tool_version:
            raise IrisRunConflictError(
                "tool call version 已变化",
                tool_call_id=command.tool_call_id,
            )
        if command.result.tool_use_id != tool_call.tool_call_id:
            raise IrisRunConflictError(
                "tool result identity 不匹配",
                tool_call_id=command.tool_call_id,
            )
        if command.result.tool_name != tool_call.tool_name:
            raise IrisRunConflictError(
                "tool result name 不匹配",
                tool_call_id=command.tool_call_id,
            )
        if (
            tool_call.phase is ToolCallPhase.PREPARED
            and not _is_preflight_result(command.result)
            and not self._is_interaction_result(
                connection,
                tool_call,
                command.result,
                operation=operation,
            )
        ):
            raise IrisRunStateError("可能包含副作用的工具结果必须先 claim")
        if tool_call.phase not in {ToolCallPhase.PREPARED, ToolCallPhase.CLAIMED}:
            raise IrisRunStateError("当前 tool call phase 不能提交 result")

        next_session = _append_messages(session, command.message_delta)
        checkpoint = self._require_checkpoint(connection, run.run_id, operation=operation)
        _validate_checkpoint_replacement(
            run,
            checkpoint,
            command.checkpoint,
            command.activation_id,
            next_session.revision,
            run.usage,
        )
        committed_call = RunToolCallRecord.model_validate(
            tool_call.model_dump()
            | {
                "phase": ToolCallPhase.COMMITTED,
                "result": command.result,
                "version": tool_call.version + 1,
                "updated_at": command.now,
                "committed_at": command.now,
            }
        )
        usage = RunUsage.model_validate(
            run.usage.model_dump() | {"tool_calls_committed": run.usage.tool_calls_committed + 1}
        )
        sequence = run.last_event_sequence + 1
        updated = _replace_run(
            run,
            revision=run.revision + 1,
            usage=usage,
            checkpoint_sequence=command.checkpoint.sequence,
            last_event_sequence=sequence,
            updated_at=command.now,
        )
        event = _make_event(
            updated,
            RunEventKind.TOOL_CALL_COMMITTED,
            command.now,
            sequence=sequence,
            activation_id=command.activation_id,
            step_index=tool_call.step_index,
            correlation_id=tool_call.tool_call_id,
        )
        if command.message_delta:
            self._update_session(connection, session, next_session, command.now)
        else:
            self._touch_session(
                connection,
                session.session_id,
                session.revision,
                command.now,
            )
        self._update_run(connection, run, updated, next_session.revision)
        self._update_checkpoint(connection, checkpoint, command.checkpoint, command.now)
        self._update_tool_call(connection, tool_call, committed_call)
        self._insert_event(connection, event)
        return RunCommit(
            run=updated,
            session=next_session if command.message_delta else None,
            checkpoint=command.checkpoint,
            events=(event,),
        )

    def _suspend_run(
        self,
        connection: sqlite3.Connection,
        command: SuspendRun,
    ) -> RunCommit:
        """原子提交当前 facts 并把 active run 增量转为 waiting。"""
        operation = "suspend_run"
        run = self._require_active(connection, command, operation=operation)
        session = self._require_history_preconditions(
            connection,
            run,
            command.expected_session_revision,
            operation=operation,
        )
        checkpoint = self._require_checkpoint(connection, run.run_id, operation=operation)
        next_session = _append_messages(session, command.message_delta)
        _validate_checkpoint_replacement(
            run,
            checkpoint,
            command.checkpoint,
            command.activation_id,
            next_session.revision,
            command.usage,
        )
        interaction = command.pending_interaction
        _validate_pending_interaction(run, interaction)
        open_row = connection.execute(
            """SELECT interaction_id FROM run_interactions
            WHERE run_id = ? AND status IN ('pending', 'resolved')
            LIMIT 1""",
            (run.run_id,),
        ).fetchone()
        if open_row is not None:
            raise IrisRunConflictError("run 已存在 open interaction", run_id=run.run_id)
        prepared = self._validate_prepared_calls(
            connection,
            run,
            command.prepared_tool_calls,
            operation=operation,
        )
        interaction_tool = next(
            (item for item in prepared if item.tool_call_id == interaction.tool_call_id),
            None,
        )
        if interaction_tool is None:
            interaction_tool = self._select_tool_call(
                connection,
                run.run_id,
                interaction.tool_call_id,
                operation=operation,
            )
        if interaction_tool is None or interaction_tool.phase is not ToolCallPhase.PREPARED:
            raise IrisRunConflictError("interaction 缺少对应 prepared tool call")
        subject = interaction.request.tool_call
        if (
            interaction.tool_call_id != interaction_tool.tool_call_id
            or interaction.step_index != interaction_tool.step_index
            or subject.tool_call_id != interaction_tool.tool_call_id
            or subject.tool_name != interaction_tool.tool_name
            or subject.arguments != interaction_tool.arguments
            or subject.fingerprint != interaction_tool.fingerprint
        ):
            raise IrisRunConflictError("interaction 与 prepared tool call subject 不匹配")
        if interaction_tool.interaction_id not in {None, interaction.interaction_id}:
            raise IrisRunConflictError("prepared tool call 已绑定其他 interaction")
        bound_interaction_tool = interaction_tool.model_copy(
            update={"interaction_id": interaction.interaction_id}
        )
        activation = self._require_activation(
            connection,
            command.activation_id,
            operation=operation,
        )
        settled = ActivationRecord.model_validate(
            activation.model_dump()
            | {
                "status": ActivationStatus.SETTLED,
                "outcome": ActivationOutcome.SUSPENDED,
                "ended_at": command.now,
            }
        )
        sequence = run.last_event_sequence + 1
        updated = _replace_run(
            run,
            phase=RunPhase.WAITING,
            revision=run.revision + 1,
            current_activation_id=None,
            pending_interaction_id=interaction.interaction_id,
            usage=command.usage,
            assistant_message=command.assistant_message,
            checkpoint_sequence=command.checkpoint.sequence,
            last_event_sequence=sequence,
            updated_at=command.now,
        )
        event = _make_event(
            updated,
            RunEventKind.INTERACTION_SUSPENDED,
            command.now,
            sequence=sequence,
            activation_id=command.activation_id,
            step_index=interaction.step_index,
            correlation_id=interaction.interaction_id,
        )
        result = project_result(updated, interaction)
        if command.message_delta:
            self._update_session(connection, session, next_session, command.now)
        else:
            self._touch_session(
                connection,
                session.session_id,
                session.revision,
                command.now,
            )
        self._update_run(connection, run, updated, next_session.revision)
        self._update_activation(connection, activation, settled)
        self._update_checkpoint(connection, checkpoint, command.checkpoint, command.now)
        self._insert_interaction(connection, interaction)
        for tool_call in prepared:
            self._insert_tool_call(
                connection,
                bound_interaction_tool
                if tool_call.tool_call_id == interaction.tool_call_id
                else tool_call,
            )
        if all(tool_call.tool_call_id != interaction.tool_call_id for tool_call in prepared):
            self._update_tool_call(connection, interaction_tool, bound_interaction_tool)
        self._insert_event(connection, event)
        return RunCommit(
            run=updated,
            session=next_session if command.message_delta else None,
            checkpoint=command.checkpoint,
            interaction=interaction,
            events=(event,),
            result=result,
        )

    def _resolve_interaction(
        self,
        connection: sqlite3.Connection,
        command: ResolveInteraction,
    ) -> RunCommit:
        """使用 version、kind 与 fingerprint CAS 增量写入人工响应。"""
        operation = "resolve_interaction"
        run = self._require_run(connection, command.run_id, operation=operation)
        if run.phase is not RunPhase.WAITING:
            raise IrisRunStateError("只有 waiting run 可以 resolve interaction")
        interaction = self._require_interaction(
            connection,
            command.interaction_id,
            operation=operation,
        )
        if run.pending_interaction_id != interaction.interaction_id:
            raise IrisRunConflictError("interaction 已不再属于 run 当前等待")
        if interaction.request.tool_call.fingerprint != command.expected_fingerprint:
            raise IrisRunConflictError("interaction fingerprint 不匹配")
        if interaction.request.prompt.kind != command.response.kind:
            raise IrisRunConflictError("interaction response kind 不匹配")
        checkpoint = self._require_checkpoint(connection, run.run_id, operation=operation)
        if interaction.status is InteractionStatus.RESOLVED:
            if interaction.response != command.response:
                raise IrisRunConflictError("interaction 已保存不同 response")
            return RunCommit(
                run=run,
                checkpoint=checkpoint,
                interaction=interaction,
                events=(),
                result=project_result(run, interaction),
            )
        self._require_revision(run, command.expected_run_revision)
        if interaction.version != command.expected_interaction_version:
            raise IrisRunConflictError("interaction version 已变化")
        if interaction.status is not InteractionStatus.PENDING:
            raise IrisRunStateError("只有 pending interaction 可以 resolve")
        if interaction.expires_at is not None and command.now >= interaction.expires_at:
            raise IrisRunStateError(
                "interaction 已过期",
                interaction_id=interaction.interaction_id,
            )
        resolved = interaction.model_copy(
            deep=True,
            update={
                "status": InteractionStatus.RESOLVED,
                "response": command.response,
                "version": interaction.version + 1,
                "resolved_at": command.now,
            },
        )
        sequence = run.last_event_sequence + 1
        updated = _replace_run(
            run,
            revision=run.revision + 1,
            last_event_sequence=sequence,
            updated_at=command.now,
        )
        event = _make_event(
            updated,
            RunEventKind.INTERACTION_RESOLVED,
            command.now,
            sequence=sequence,
            correlation_id=interaction.interaction_id,
        )
        result = project_result(updated, resolved)
        self._touch_session(
            connection,
            run.session_id,
            checkpoint.session_revision,
            command.now,
        )
        self._update_run(connection, run, updated, checkpoint.session_revision)
        self._update_interaction(connection, interaction, resolved)
        self._insert_event(connection, event)
        return RunCommit(
            run=updated,
            checkpoint=checkpoint,
            interaction=resolved,
            events=(event,),
            result=result,
        )

    def _is_interaction_result(
        self,
        connection: sqlite3.Connection,
        tool_call: RunToolCallRecord,
        result: ToolResult,
        *,
        operation: str,
    ) -> bool:
        """验证 closed interaction 是否精确授权当前无 claim 结果。"""
        if tool_call.interaction_id is None:
            return False
        interaction = self._select_interaction(
            connection,
            tool_call.interaction_id,
            operation=operation,
        )
        if (
            interaction is None
            or interaction.status is not InteractionStatus.CLOSED
            or interaction.response is None
            or interaction.close_reason != "resumed"
        ):
            return False
        response = interaction.response
        subject = interaction.request.tool_call
        if (
            interaction.run_id != tool_call.run_id
            or interaction.tool_call_id != tool_call.tool_call_id
            or interaction.step_index != tool_call.step_index
            or subject.tool_call_id != tool_call.tool_call_id
            or subject.tool_name != tool_call.tool_name
            or subject.arguments != tool_call.arguments
            or subject.fingerprint != tool_call.fingerprint
        ):
            return False
        if isinstance(response, QuestionInteractionResponse):
            expected = ToolResult(
                tool_use_id=subject.tool_call_id,
                tool_name=subject.tool_name,
                content=[TextBlock(text=response.answer)],
                data={"answer": response.answer},
            )
            return result == expected
        if not isinstance(response, PermissionInteractionResponse) or response.decision != "reject":
            return False
        expected = ToolResult(
            tool_use_id=subject.tool_call_id,
            tool_name=subject.tool_name,
            is_error=True,
            error=ToolErrorInfo(
                code="USER_REJECTED",
                message="用户拒绝了工具调用",
            ),
        )
        return result == expected

    def _request_cancellation(
        self,
        connection: sqlite3.Connection,
        command: RequestCancellation,
    ) -> RunCommit:
        """记录首次 cancellation request，并可结算 waiting run。"""
        operation = "request_cancellation"
        run = self._require_run(connection, command.run_id, operation=operation)
        if run.phase is RunPhase.TERMINAL:
            raise IrisRunStateError("terminal run 不接受 cancellation request")
        if run.phase is RunPhase.ACTIVE:
            if command.activation_id != run.current_activation_id:
                raise IrisRunConflictError("activation fence 已变化", run_id=run.run_id)
        elif command.activation_id is not None:
            raise IrisRunConflictError("waiting run 不应携带 activation fence")
        checkpoint = self._require_checkpoint(connection, run.run_id, operation=operation)
        if run.cancellation_requested_at is not None:
            if run.cancellation_reason == command.reason and not command.settle_waiting:
                replay_interaction = (
                    self._require_interaction(
                        connection,
                        run.pending_interaction_id,
                        operation=operation,
                    )
                    if run.phase is RunPhase.WAITING
                    else None
                )
                return RunCommit(
                    run=run,
                    checkpoint=checkpoint,
                    interaction=replay_interaction,
                    events=(),
                    result=(
                        project_result(run, replay_interaction)
                        if run.phase is RunPhase.WAITING
                        else None
                    ),
                )
            raise IrisRunConflictError(
                "cancellation 已由其他 command 请求",
                run_id=run.run_id,
            )
        self._require_revision(run, command.expected_run_revision)
        sequence = run.last_event_sequence + 1
        events = [
            _make_event(
                run,
                RunEventKind.CANCELLATION_REQUESTED,
                command.now,
                sequence=sequence,
                activation_id=command.activation_id,
                payload={"reason": command.reason},
            )
        ]
        interaction: HumanInteraction | None = None
        current_session: SessionSnapshot | None = None
        updated_session: SessionSnapshot | None = None
        updated_checkpoint = checkpoint
        unknown_pairs: list[tuple[RunToolCallRecord, RunToolCallRecord]] = []
        if run.phase is RunPhase.WAITING and command.settle_waiting:
            interaction = self._close_interaction(
                connection,
                run,
                command.now,
                command.reason,
                operation=operation,
            )
            closures = self._terminal_tool_closures(
                connection,
                run,
                command.now,
                operation=operation,
            )
            closure_messages = [message for _, _, message in closures]
            current_session = self._select_session(
                connection,
                run.session_id,
                operation=operation,
            )
            appended_session = _append_messages(current_session, closure_messages)
            if closure_messages:
                updated_session = appended_session
                updated_checkpoint = checkpoint.model_copy(
                    deep=True,
                    update={"session_revision": appended_session.revision},
                )
            unknown_pairs = [
                (current_call, updated_call)
                for current_call, updated_call, _ in closures
                if current_call.phase is ToolCallPhase.CLAIMED
            ]
            for index, (_, record) in enumerate(unknown_pairs, start=1):
                events.append(
                    _make_event(
                        run,
                        RunEventKind.TOOL_CALL_OUTCOME_UNKNOWN,
                        command.now,
                        sequence=sequence + index,
                        activation_id=record.claim_activation_id,
                        step_index=record.step_index,
                        correlation_id=record.tool_call_id,
                    )
                )
            sequence += len(unknown_pairs) + 1
            updated = _replace_run(
                run,
                phase=RunPhase.TERMINAL,
                stop_reason=RunStopReason.CANCELLED,
                revision=run.revision + 1,
                pending_interaction_id=None,
                cancellation_requested_at=command.now,
                cancellation_reason=command.reason,
                last_event_sequence=sequence,
                updated_at=command.now,
                finished_at=command.now,
            )
            events.append(
                _make_event(
                    updated,
                    RunEventKind.RUN_TERMINAL,
                    command.now,
                    sequence=sequence,
                )
            )
            result = project_result(updated)
        else:
            updated = _replace_run(
                run,
                revision=run.revision + 1,
                cancellation_requested_at=command.now,
                cancellation_reason=command.reason,
                last_event_sequence=sequence,
                updated_at=command.now,
            )
            if run.phase is RunPhase.WAITING:
                waiting_interaction = self._require_interaction(
                    connection,
                    run.pending_interaction_id,
                    operation=operation,
                )
                result = project_result(updated, waiting_interaction)
            else:
                result = None

        if updated_session is not None and current_session is not None:
            self._update_session(connection, current_session, updated_session, command.now)
            self._update_checkpoint(connection, checkpoint, updated_checkpoint, command.now)
        else:
            self._touch_session(
                connection,
                run.session_id,
                checkpoint.session_revision,
                command.now,
            )
        self._update_run(connection, run, updated, updated_checkpoint.session_revision)
        if interaction is not None:
            current_interaction = self._require_interaction(
                connection,
                interaction.interaction_id,
                operation=operation,
            )
            self._update_interaction(connection, current_interaction, interaction)
            self._delete_lane(connection, run, require_match=False)
        for current_call, updated_call in unknown_pairs:
            self._update_tool_call(connection, current_call, updated_call)
        for event in events:
            self._insert_event(connection, event)
        return RunCommit(
            run=updated,
            session=updated_session,
            checkpoint=updated_checkpoint,
            interaction=interaction,
            events=tuple(events),
            result=result,
        )

    def _finish_run(
        self,
        connection: sqlite3.Connection,
        command: FinishRun,
    ) -> RunCommit:
        """将 active 或 waiting run 增量结算为 terminal。"""
        operation = "finish_run"
        run = self._require_run(connection, command.run_id, operation=operation)
        self._require_revision(run, command.expected_run_revision)
        if run.phase is RunPhase.TERMINAL:
            raise IrisRunStateError("terminal run 不接受进一步 mutation", run_id=run.run_id)
        self._require_lane(connection, run)
        if run.phase is RunPhase.ACTIVE:
            if command.activation_id != run.current_activation_id:
                raise IrisRunConflictError("activation fence 已变化", run_id=run.run_id)
        elif command.activation_id is not None:
            raise IrisRunConflictError("waiting run 不应携带 activation fence")
        if (
            command.stop_reason in {RunStopReason.FAILED, RunStopReason.OUTCOME_UNKNOWN}
            and command.error is None
        ):
            raise IrisRunStateError("failed/outcome_unknown finish 必须包含 error")
        if command.stop_reason is RunStopReason.COMPLETED and command.error is not None:
            raise IrisRunStateError("completed finish 不能包含 error")

        checkpoint = self._require_checkpoint(connection, run.run_id, operation=operation)
        current_interaction: HumanInteraction | None = None
        interaction: HumanInteraction | None = None
        if run.phase is RunPhase.WAITING:
            current_interaction = self._require_interaction(
                connection,
                run.pending_interaction_id,
                operation=operation,
            )
            close_reason = command.interaction_close_reason or command.stop_reason.value
            interaction = _closed_interaction(current_interaction, command.now, close_reason)
        if run.phase is RunPhase.ACTIVE:
            activation = self._require_activation(
                connection,
                run.current_activation_id,
                operation=operation,
            )
            settled = ActivationRecord.model_validate(
                activation.model_dump()
                | {
                    "status": ActivationStatus.SETTLED,
                    "outcome": _activation_outcome(command.stop_reason),
                    "ended_at": command.now,
                }
            )
        else:
            activation = None
            settled = None
        closures = self._terminal_tool_closures(
            connection,
            run,
            command.now,
            operation=operation,
        )
        closure_messages = [message for _, _, message in closures]
        current_session = self._select_session(
            connection,
            run.session_id,
            operation=operation,
        )
        updated_session = _append_messages(current_session, closure_messages)
        updated_checkpoint = (
            checkpoint.model_copy(
                deep=True,
                update={"session_revision": updated_session.revision},
            )
            if closure_messages
            else checkpoint
        )
        unknown_calls = [
            (current_call, updated_call)
            for current_call, updated_call, _ in closures
            if current_call.phase is ToolCallPhase.CLAIMED
        ]
        sequence = run.last_event_sequence + len(unknown_calls) + 1
        updated = _replace_run(
            run,
            phase=RunPhase.TERMINAL,
            stop_reason=command.stop_reason,
            revision=run.revision + 1,
            current_activation_id=None,
            pending_interaction_id=None,
            assistant_message=command.assistant_message,
            error=command.error,
            last_event_sequence=sequence,
            updated_at=command.now,
            finished_at=command.now,
        )
        unknown_events = tuple(
            _make_event(
                updated,
                RunEventKind.TOOL_CALL_OUTCOME_UNKNOWN,
                command.now,
                sequence=run.last_event_sequence + index,
                activation_id=unknown_call.claim_activation_id,
                step_index=unknown_call.step_index,
                correlation_id=unknown_call.tool_call_id,
            )
            for index, (_, unknown_call) in enumerate(unknown_calls, start=1)
        )
        terminal_event = _make_event(
            updated,
            RunEventKind.RUN_TERMINAL,
            command.now,
            sequence=sequence,
            activation_id=command.activation_id,
            payload={"stop_reason": command.stop_reason.value},
        )
        result = project_result(updated)

        if closure_messages:
            self._update_session(connection, current_session, updated_session, command.now)
            self._update_checkpoint(connection, checkpoint, updated_checkpoint, command.now)
        else:
            self._touch_session(
                connection,
                run.session_id,
                checkpoint.session_revision,
                command.now,
            )
        self._update_run(connection, run, updated, updated_checkpoint.session_revision)
        self._delete_lane(connection, run, require_match=True)
        if activation is not None and settled is not None:
            self._update_activation(connection, activation, settled)
        if current_interaction is not None and interaction is not None:
            self._update_interaction(connection, current_interaction, interaction)
        for current_call, unknown_call in unknown_calls:
            self._update_tool_call(connection, current_call, unknown_call)
        for event in (*unknown_events, terminal_event):
            self._insert_event(connection, event)
        return RunCommit(
            run=updated,
            session=updated_session if closure_messages else None,
            checkpoint=updated_checkpoint,
            interaction=interaction,
            events=(*unknown_events, terminal_event),
            result=result,
        )

    def _recover_active_run(
        self,
        connection: sqlite3.Connection,
        command: RecoverActiveRun,
    ) -> RunCommit:
        """按 durable checkpoint/tool facts 增量恢复或终止旧 activation。"""
        operation = "recover_active_run"
        run = self._require_run(connection, command.run_id, operation=operation)
        self._require_revision(run, command.expected_run_revision)
        if run.phase is not RunPhase.ACTIVE:
            raise IrisRunStateError("只有 active run 可以执行 active recovery")
        if run.current_activation_id != command.expected_activation_id:
            raise IrisRunConflictError("activation fence 已变化", run_id=run.run_id)
        checkpoint = self._require_checkpoint(connection, run.run_id, operation=operation)
        if checkpoint.sequence != command.expected_checkpoint_sequence:
            raise IrisRunConflictError("checkpoint sequence 已变化", run_id=run.run_id)
        activation = self._require_activation(
            connection,
            command.expected_activation_id,
            operation=operation,
        )
        claimed = self._select_claimed_tool_calls(
            connection,
            run.run_id,
            operation=operation,
        )
        if command.recovery_disposition is RecoveryDisposition.RESUME and claimed:
            raise IrisRunRecoveryError(
                "safe recovery 不能重放 unresolved durable claim",
                run_id=run.run_id,
            )
        abandoned_outcome = (
            ActivationOutcome.OUTCOME_UNKNOWN
            if command.recovery_disposition is RecoveryDisposition.OUTCOME_UNKNOWN
            else ActivationOutcome.RECOVERED
        )
        abandoned = ActivationRecord.model_validate(
            activation.model_dump()
            | {
                "status": ActivationStatus.ABANDONED,
                "outcome": abandoned_outcome,
                "ended_at": command.now,
            }
        )
        first_sequence = run.last_event_sequence + 1
        abandoned_event = _make_event(
            run,
            RunEventKind.ACTIVATION_ABANDONED,
            command.now,
            sequence=first_sequence,
            activation_id=activation.activation_id,
        )

        terminal_closures = (
            self._terminal_tool_closures(
                connection,
                run,
                command.now,
                operation=operation,
            )
            if command.recovery_disposition
            in {RecoveryDisposition.OUTCOME_UNKNOWN, RecoveryDisposition.FINALIZE}
            else []
        )
        closure_messages = [message for _, _, message in terminal_closures]
        current_session: SessionSnapshot | None = None
        updated_session: SessionSnapshot | None = None
        terminal_checkpoint = checkpoint
        if closure_messages:
            current_session = self._select_session(
                connection,
                run.session_id,
                operation=operation,
            )
            updated_session = _append_messages(current_session, closure_messages)
            terminal_checkpoint = checkpoint.model_copy(
                deep=True,
                update={"session_revision": updated_session.revision},
            )
        unknown_pairs: list[tuple[RunToolCallRecord, RunToolCallRecord]] = []
        activation_next: ActivationRecord | None = None
        rebound: RunCheckpoint | None = None
        delete_lane = False
        if command.recovery_disposition is RecoveryDisposition.OUTCOME_UNKNOWN:
            if not claimed:
                raise IrisRunRecoveryError(
                    "outcome_unknown recovery 缺少 unresolved durable claim",
                    run_id=run.run_id,
                )
            unknown_pairs = [
                (current_call, updated_call)
                for current_call, updated_call, _ in terminal_closures
                if current_call.phase is ToolCallPhase.CLAIMED
            ]
            terminal_sequence = first_sequence + len(unknown_pairs) + 1
            updated = _replace_run(
                run,
                phase=RunPhase.TERMINAL,
                stop_reason=RunStopReason.OUTCOME_UNKNOWN,
                revision=run.revision + 1,
                current_activation_id=None,
                error=RunErrorInfo(
                    code="TOOL_OUTCOME_UNKNOWN",
                    message="工具 claim 缺少可证明的 durable result",
                    source="tool",
                    details={"tool_call_ids": [item.tool_call_id for item in claimed]},
                ),
                last_event_sequence=terminal_sequence,
                updated_at=command.now,
                finished_at=command.now,
            )
            unknown_events = tuple(
                _make_event(
                    updated,
                    RunEventKind.TOOL_CALL_OUTCOME_UNKNOWN,
                    command.now,
                    sequence=first_sequence + index,
                    activation_id=record.claim_activation_id,
                    step_index=record.step_index,
                    correlation_id=record.tool_call_id,
                )
                for index, (_, record) in enumerate(unknown_pairs, start=1)
            )
            terminal_event = _make_event(
                updated,
                RunEventKind.RUN_TERMINAL,
                command.now,
                sequence=terminal_sequence,
                payload={"stop_reason": RunStopReason.OUTCOME_UNKNOWN.value},
            )
            events = (abandoned_event, *unknown_events, terminal_event)
            result = project_result(updated)
            output_checkpoint = terminal_checkpoint
            delete_lane = True
        elif command.recovery_disposition is RecoveryDisposition.FINALIZE:
            if claimed:
                raise IrisRunRecoveryError(
                    "outcome-ready recovery 不能忽略 unresolved durable claim",
                    run_id=run.run_id,
                )
            if checkpoint.resumability is not CheckpointResumability.OUTCOME_READY:
                raise IrisRunRecoveryError(
                    "finalize recovery 需要 outcome-ready checkpoint",
                    run_id=run.run_id,
                )
            terminal_sequence = first_sequence + 1
            updated = _replace_run(
                run,
                phase=RunPhase.TERMINAL,
                stop_reason=RunStopReason.COMPLETED,
                revision=run.revision + 1,
                current_activation_id=None,
                last_event_sequence=terminal_sequence,
                updated_at=command.now,
                finished_at=command.now,
            )
            terminal_event = _make_event(
                updated,
                RunEventKind.RUN_TERMINAL,
                command.now,
                sequence=terminal_sequence,
                payload={"stop_reason": RunStopReason.COMPLETED.value},
            )
            events = (abandoned_event, terminal_event)
            result = project_result(updated)
            output_checkpoint = terminal_checkpoint
            delete_lane = True
        else:
            if command.new_activation_id is None:
                raise IrisRunRecoveryError("resume recovery 缺少 new activation identity")
            if (
                self._select_activation(
                    connection,
                    command.new_activation_id,
                    operation=operation,
                )
                is not None
            ):
                raise IrisRunConflictError(
                    "activation_id 已存在",
                    activation_id=command.new_activation_id,
                )
            row = connection.execute(
                """SELECT COALESCE(MAX(ordinal), 0) AS max_ordinal
                FROM run_activations WHERE run_id = ?""",
                (run.run_id,),
            ).fetchone()
            activation_next = ActivationRecord(
                activation_id=command.new_activation_id,
                run_id=run.run_id,
                ordinal=int(row["max_ordinal"]) + 1,
                kind=ActivationKind.RECOVER,
                status=ActivationStatus.ACTIVE,
                started_at=command.now,
            )
            rebound = RunCheckpoint.model_validate(
                checkpoint.model_dump()
                | {
                    "sequence": checkpoint.sequence + 1,
                    "activation_id": activation_next.activation_id,
                    "resumability": CheckpointResumability.SAFE,
                }
            )
            start_sequence = first_sequence + 1
            updated = _replace_run(
                run,
                revision=run.revision + 1,
                current_activation_id=activation_next.activation_id,
                checkpoint_sequence=rebound.sequence,
                last_event_sequence=start_sequence,
                updated_at=command.now,
            )
            start_event = _make_event(
                updated,
                RunEventKind.ACTIVATION_STARTED,
                command.now,
                sequence=start_sequence,
                activation_id=activation_next.activation_id,
            )
            events = (abandoned_event, start_event)
            result = None
            output_checkpoint = rebound

        if updated_session is not None and current_session is not None:
            self._update_session(connection, current_session, updated_session, command.now)
            self._update_checkpoint(connection, checkpoint, terminal_checkpoint, command.now)
        else:
            self._touch_session(
                connection,
                run.session_id,
                checkpoint.session_revision,
                command.now,
            )
        self._update_run(connection, run, updated, output_checkpoint.session_revision)
        self._update_activation(connection, activation, abandoned)
        if delete_lane:
            self._delete_lane(connection, run, require_match=False)
        for current_call, unknown_call in unknown_pairs:
            self._update_tool_call(connection, current_call, unknown_call)
        if activation_next is not None and rebound is not None:
            self._insert_activation(connection, activation_next)
            self._update_checkpoint(connection, checkpoint, rebound, command.now)
        for event in events:
            self._insert_event(connection, event)
        return RunCommit(
            run=updated,
            session=updated_session,
            checkpoint=output_checkpoint,
            events=events,
            result=result,
        )

    def _close_interaction(
        self,
        connection: sqlite3.Connection,
        run: RunRecord,
        now: datetime,
        reason: str,
        *,
        operation: str,
    ) -> HumanInteraction:
        """读取并关闭当前 waiting interaction。"""
        interaction = self._require_interaction(
            connection,
            run.pending_interaction_id,
            operation=operation,
        )
        return _closed_interaction(interaction, now, reason)

    def _select_claimed_tool_calls(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        *,
        operation: str,
    ) -> list[RunToolCallRecord]:
        """只读取目标 run 的 unresolved durable claims。"""
        return [
            _decode_row(
                _row_to_tool_call,
                row,
                path=self.path,
                operation=operation,
            )
            for row in connection.execute(
                """SELECT * FROM run_tool_calls
                WHERE run_id = ? AND phase = 'claimed'
                ORDER BY step_index, ordinal""",
                (run_id,),
            )
        ]

    def _terminal_tool_closures(
        self,
        connection: sqlite3.Connection,
        run: RunRecord,
        now: datetime,
        *,
        operation: str,
    ) -> list[tuple[RunToolCallRecord, RunToolCallRecord, Msg]]:
        """构造当前 run 所有未闭合 tool call 的 fact/message。"""
        return [
            (record, *build_terminal_tool_closure(record, now=now))
            for record in (
                _decode_row(
                    _row_to_tool_call,
                    row,
                    path=self.path,
                    operation=operation,
                )
                for row in connection.execute(
                    """SELECT * FROM run_tool_calls
                    WHERE run_id = ? AND phase IN ('prepared', 'claimed')
                    ORDER BY step_index, ordinal""",
                    (run.run_id,),
                )
            )
        ]

    def _create_run(
        self,
        connection: sqlite3.Connection,
        command: CreateRun,
    ) -> RunCommit:
        """增量创建 run、lane、activation、checkpoint 与起始事件。"""
        run_id = command.request.run_id
        if run_id is None:
            raise IrisRunStateError("CreateRun request 缺少最终 run_id")
        if self._select_run(connection, run_id, operation="create_run") is not None:
            raise IrisRunConflictError("run_id 已存在", run_id=run_id)
        lane = connection.execute(
            "SELECT run_id FROM session_run_lanes WHERE session_id = ?",
            (command.request.session_id,),
        ).fetchone()
        if lane is not None:
            raise IrisRunConflictError(
                "session lane 已被 non-terminal run 占用",
                session_id=command.request.session_id,
                owner_run_id=lane["run_id"],
            )
        if (
            self._select_activation(
                connection,
                command.start_activation_id,
                operation="create_run",
            )
            is not None
        ):
            raise IrisRunConflictError(
                "activation_id 已存在",
                activation_id=command.start_activation_id,
            )
        session_row = connection.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (command.request.session_id,),
        ).fetchone()
        session = (
            SessionSnapshot(session_id=command.request.session_id)
            if session_row is None
            else _decode_row(
                _row_to_session,
                session_row,
                path=self.path,
                operation="create_run",
            )
        )
        if command.initial_checkpoint.session_revision != session.revision:
            raise IrisRunConflictError(
                "initial checkpoint session revision 不匹配",
                expected=session.revision,
                actual=command.initial_checkpoint.session_revision,
            )

        deadline = command.options.limits.deadline_at
        if deadline is not None and command.now >= deadline:
            run = RunRecord(
                run_id=run_id,
                session_id=command.request.session_id,
                agent_id=command.agent_id,
                request=command.request,
                options=command.options,
                phase=RunPhase.TERMINAL,
                stop_reason=RunStopReason.DEADLINE_EXCEEDED,
                revision=1,
                current_activation_id=None,
                pending_interaction_id=None,
                usage=RunUsage(),
                environment_fingerprint=command.environment_fingerprint,
                checkpoint_sequence=0,
                last_event_sequence=1,
                created_at=command.now,
                started_at=command.now,
                updated_at=command.now,
                finished_at=command.now,
            )
            event = _make_event(
                run,
                RunEventKind.RUN_TERMINAL,
                command.now,
                sequence=1,
            )
            self._persist_create_session(connection, session, session_row, command.now)
            self._insert_run(connection, run, session.revision)
            self._insert_event(connection, event)
            return RunCommit(
                run=run,
                events=(event,),
                result=project_result(run),
            )

        activation = ActivationRecord(
            activation_id=command.start_activation_id,
            run_id=run_id,
            ordinal=1,
            kind=ActivationKind.START,
            status=ActivationStatus.ACTIVE,
            started_at=command.now,
        )
        run = RunRecord(
            run_id=run_id,
            session_id=command.request.session_id,
            agent_id=command.agent_id,
            request=command.request,
            options=command.options,
            phase=RunPhase.ACTIVE,
            revision=1,
            current_activation_id=command.start_activation_id,
            usage=RunUsage(),
            environment_fingerprint=command.environment_fingerprint,
            checkpoint_sequence=1,
            last_event_sequence=2,
            created_at=command.now,
            started_at=command.now,
            updated_at=command.now,
        )
        events = (
            _make_event(run, RunEventKind.RUN_STARTED, command.now, sequence=1),
            _make_event(
                run,
                RunEventKind.ACTIVATION_STARTED,
                command.now,
                sequence=2,
                activation_id=activation.activation_id,
            ),
        )
        self._persist_create_session(connection, session, session_row, command.now)
        self._insert_run(connection, run, session.revision)
        _execute(
            connection,
            """INSERT INTO session_run_lanes(
                session_id, run_id, revision, acquired_at
            ) VALUES (?, ?, ?, ?)""",
            (session.session_id, run_id, run.revision, command.now.isoformat()),
        )
        self._insert_activation(connection, activation)
        self._insert_checkpoint(connection, command.initial_checkpoint, command.now)
        for event in events:
            self._insert_event(connection, event)
        return RunCommit(
            run=run,
            checkpoint=command.initial_checkpoint,
            events=events,
        )

    def _persist_create_session(
        self,
        connection: sqlite3.Connection,
        session: SessionSnapshot,
        session_row: sqlite3.Row | None,
        updated_at: datetime,
    ) -> None:
        """创建缺失 session，或只推进已有 session 的 aggregate 时间。"""
        if session_row is None:
            _execute(
                connection,
                """INSERT INTO sessions(
                    session_id, revision, messages_json, updated_at
                ) VALUES (?, ?, ?, ?)""",
                (
                    session.session_id,
                    session.revision,
                    _dump_json(session.messages),
                    updated_at.isoformat(),
                ),
            )
            return
        self._touch_session(
            connection,
            session.session_id,
            session.revision,
            updated_at,
        )

    def _touch_session(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        expected_revision: int,
        updated_at: datetime,
    ) -> None:
        """只推进 session aggregate 时间，不重写消息 JSON。"""
        timestamp = updated_at.isoformat()
        cursor = _execute(
            connection,
            """UPDATE sessions
            SET updated_at = CASE WHEN updated_at < ? THEN ? ELSE updated_at END
            WHERE session_id = ? AND revision = ?""",
            (timestamp, timestamp, session_id, expected_revision),
        )
        if cursor.rowcount != 1:
            raise IrisRunConflictError(
                "session revision 已变化",
                session_id=session_id,
                expected=expected_revision,
            )

    def _update_session(
        self,
        connection: sqlite3.Connection,
        current: SessionSnapshot,
        updated: SessionSnapshot,
        updated_at: datetime,
    ) -> None:
        """使用 revision CAS 更新当前 session history。"""
        cursor = _execute(
            connection,
            """UPDATE sessions
            SET revision = ?, messages_json = ?, updated_at = ?
            WHERE session_id = ? AND revision = ?""",
            (
                updated.revision,
                _dump_json(updated.messages),
                updated_at.isoformat(),
                current.session_id,
                current.revision,
            ),
        )
        if cursor.rowcount != 1:
            raise IrisRunConflictError(
                "session revision 已变化",
                session_id=current.session_id,
                expected=current.revision,
            )

    def _insert_run(
        self,
        connection: sqlite3.Connection,
        run: RunRecord,
        session_revision: int,
    ) -> None:
        """插入一条完整 run record。"""
        _execute(connection, _INSERT_RUN, _run_values(run, session_revision))

    def _update_run(
        self,
        connection: sqlite3.Connection,
        current: RunRecord,
        updated: RunRecord,
        session_revision: int,
    ) -> None:
        """使用 run revision CAS 替换一条 run record。"""
        values = _run_values(updated, session_revision)
        cursor = _execute(
            connection,
            _UPDATE_RUN,
            (*values[1:], current.run_id, current.revision),
        )
        if cursor.rowcount != 1:
            raise IrisRunConflictError(
                "run revision 已变化",
                run_id=current.run_id,
                expected=current.revision,
            )

    def _insert_activation(
        self,
        connection: sqlite3.Connection,
        activation: ActivationRecord,
    ) -> None:
        """插入一条 activation fact。"""
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

    def _update_activation(
        self,
        connection: sqlite3.Connection,
        current: ActivationRecord,
        updated: ActivationRecord,
    ) -> None:
        """按 activation 当前状态更新 fence outcome。"""
        cursor = _execute(
            connection,
            """UPDATE run_activations SET
                ordinal = ?, kind = ?, status = ?, outcome = ?, started_at = ?, ended_at = ?
            WHERE activation_id = ? AND status = ?""",
            (
                updated.ordinal,
                updated.kind.value,
                updated.status.value,
                _stored_activation_outcome(updated),
                updated.started_at.isoformat(),
                _iso(updated.ended_at),
                current.activation_id,
                current.status.value,
            ),
        )
        if cursor.rowcount != 1:
            raise IrisRunConflictError(
                "activation 不存在或 fence 已变化",
                activation_id=current.activation_id,
            )

    def _insert_checkpoint(
        self,
        connection: sqlite3.Connection,
        checkpoint: RunCheckpoint,
        updated_at: datetime,
    ) -> None:
        """插入一条 current checkpoint。"""
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
                updated_at.isoformat(),
            ),
        )

    def _update_checkpoint(
        self,
        connection: sqlite3.Connection,
        current: RunCheckpoint,
        updated: RunCheckpoint,
        updated_at: datetime,
    ) -> None:
        """使用 sequence CAS 更新 current checkpoint。"""
        cursor = _execute(
            connection,
            """UPDATE run_checkpoints SET
                sequence = ?, activation_id = ?, checkpoint_version = ?, cursor_json = ?,
                session_revision = ?, model_steps_reserved = ?, model_steps_committed = ?,
                environment_fingerprint = ?, resumability = ?, updated_at = ?
            WHERE run_id = ? AND sequence = ?""",
            (
                updated.sequence,
                updated.activation_id,
                updated.checkpoint_version,
                _dump_json(updated.engine_cursor),
                updated.session_revision,
                updated.model_steps_reserved,
                updated.model_steps_committed,
                updated.environment_fingerprint,
                updated.resumability.value,
                updated_at.isoformat(),
                current.run_id,
                current.sequence,
            ),
        )
        if cursor.rowcount != 1:
            raise IrisRunConflictError("checkpoint sequence 已变化", run_id=current.run_id)

    def _insert_tool_call(
        self,
        connection: sqlite3.Connection,
        call: RunToolCallRecord,
    ) -> None:
        """插入一条 prepared tool intent。"""
        _execute(connection, _INSERT_TOOL_CALL, _tool_call_values(call))

    def _update_tool_call(
        self,
        connection: sqlite3.Connection,
        current: RunToolCallRecord,
        updated: RunToolCallRecord,
    ) -> None:
        """使用 run/tool/version CAS 更新一条 tool fact。"""
        cursor = _execute(
            connection,
            """UPDATE run_tool_calls SET
                interaction_id = ?, phase = ?, claim_activation_id = ?,
                result_json = ?, version = ?, updated_at = ?, claimed_at = ?,
                committed_at = ?
            WHERE run_id = ? AND tool_call_id = ? AND version = ?""",
            (
                updated.interaction_id,
                updated.phase.value,
                updated.claim_activation_id,
                _dump_json(updated.result) if updated.result is not None else None,
                updated.version,
                updated.updated_at.isoformat(),
                _iso(updated.claimed_at),
                _iso(updated.committed_at),
                current.run_id,
                current.tool_call_id,
                current.version,
            ),
        )
        if cursor.rowcount != 1:
            raise IrisRunConflictError(
                "tool call version 已变化",
                tool_call_id=current.tool_call_id,
            )

    def _insert_interaction(
        self,
        connection: sqlite3.Connection,
        interaction: HumanInteraction,
    ) -> None:
        """插入一条 pending interaction。"""
        _execute(connection, _INSERT_INTERACTION, _interaction_values(interaction))

    def _update_interaction(
        self,
        connection: sqlite3.Connection,
        current: HumanInteraction,
        updated: HumanInteraction,
    ) -> None:
        """使用 version CAS 更新 interaction 状态。"""
        cursor = _execute(
            connection,
            """UPDATE run_interactions SET
                status = ?, response_json = ?, version = ?, resolved_at = ?,
                closed_at = ?, close_reason = ?
            WHERE interaction_id = ? AND version = ?""",
            (
                updated.status.value,
                _dump_json(updated.response) if updated.response is not None else None,
                updated.version,
                _iso(updated.resolved_at),
                _iso(updated.closed_at),
                updated.close_reason,
                current.interaction_id,
                current.version,
            ),
        )
        if cursor.rowcount != 1:
            raise IrisRunConflictError(
                "interaction version 已变化",
                interaction_id=current.interaction_id,
            )

    def _validate_prepared_calls(
        self,
        connection: sqlite3.Connection,
        run: RunRecord,
        calls: list[RunToolCallRecord],
        *,
        operation: str,
    ) -> list[RunToolCallRecord]:
        """校验本批 prepared tool identity 与数据库唯一性。"""
        identities: set[str] = set()
        result: list[RunToolCallRecord] = []
        for tool_call in calls:
            if tool_call.run_id != run.run_id or tool_call.phase is not ToolCallPhase.PREPARED:
                raise IrisRunStateError("prepared tool call 的 run/phase 不一致")
            if (
                tool_call.tool_call_id in identities
                or self._select_tool_call(
                    connection,
                    run.run_id,
                    tool_call.tool_call_id,
                    operation=operation,
                )
                is not None
            ):
                raise IrisRunConflictError(
                    "tool_call_id 已存在",
                    tool_call_id=tool_call.tool_call_id,
                )
            identities.add(tool_call.tool_call_id)
            result.append(deepcopy(tool_call))
        return result

    def _delete_lane(
        self,
        connection: sqlite3.Connection,
        run: RunRecord,
        *,
        require_match: bool,
    ) -> None:
        """只删除目标 run 持有的 session lane。"""
        cursor = _execute(
            connection,
            """DELETE FROM session_run_lanes
            WHERE session_id = ? AND run_id = ?""",
            (run.session_id, run.run_id),
        )
        if require_match and cursor.rowcount != 1:
            raise IrisRunConflictError("session lane owner 已变化", run_id=run.run_id)

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        event: RunEvent,
    ) -> None:
        """Append 一条不可变 run event。"""
        _execute(connection, _INSERT_EVENT, _event_values(event))

    def _read(
        self,
        operation: str,
        reader: Callable[[sqlite3.Connection], _ReadT],
    ) -> _ReadT:
        with self._lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN")
                    result = reader(connection)
                    connection.commit()
                    return deepcopy(result)
            except IrisRunPersistenceError:
                raise
            except sqlite3.Error as exc:
                raise IrisRunPersistenceError(
                    "lifecycle SQLite read 失败",
                    path=str(self.path),
                    operation=operation,
                ) from exc

    def _select_run(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        *,
        operation: str,
    ) -> RunRecord | None:
        row = connection.execute(
            "SELECT * FROM agent_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return _decode_row(_row_to_run, row, path=self.path, operation=operation)

    def _select_session(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        *,
        operation: str,
    ) -> SessionSnapshot:
        row = connection.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return SessionSnapshot(session_id=session_id)
        return _decode_row(_row_to_session, row, path=self.path, operation=operation)

    def _select_activation(
        self,
        connection: sqlite3.Connection,
        activation_id: str | None,
        *,
        operation: str,
    ) -> ActivationRecord | None:
        if activation_id is None:
            return None
        row = connection.execute(
            "SELECT * FROM run_activations WHERE activation_id = ?",
            (activation_id,),
        ).fetchone()
        if row is None:
            return None
        return _decode_row(_row_to_activation, row, path=self.path, operation=operation)

    def _select_checkpoint(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        *,
        operation: str,
    ) -> RunCheckpoint | None:
        row = connection.execute(
            "SELECT * FROM run_checkpoints WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return _decode_row(_row_to_checkpoint, row, path=self.path, operation=operation)

    def _select_tool_call(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        tool_call_id: str,
        *,
        operation: str,
    ) -> RunToolCallRecord | None:
        row = connection.execute(
            """SELECT * FROM run_tool_calls
            WHERE run_id = ? AND tool_call_id = ?""",
            (run_id, tool_call_id),
        ).fetchone()
        if row is None:
            return None
        return _decode_row(_row_to_tool_call, row, path=self.path, operation=operation)

    def _select_interaction(
        self,
        connection: sqlite3.Connection,
        interaction_id: str | None,
        *,
        operation: str,
    ) -> HumanInteraction | None:
        if interaction_id is None:
            return None
        row = connection.execute(
            "SELECT * FROM run_interactions WHERE interaction_id = ?",
            (interaction_id,),
        ).fetchone()
        if row is None:
            return None
        return _decode_row(_row_to_interaction, row, path=self.path, operation=operation)


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


def _decode_row[RowT](
    factory: Callable[[sqlite3.Row], RowT],
    row: sqlite3.Row,
    *,
    path: Path,
    operation: str,
) -> RowT:
    """把 durable row 的反序列化失败统一映射为 persistence error。"""
    try:
        return factory(row)
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


def _project_durable_result(
    run: RunRecord,
    interaction: HumanInteraction | None,
    *,
    path: Path,
    operation: str,
) -> RunResult:
    """从目标 durable facts 投影 waiting 或 terminal result。"""
    try:
        return project_result(run, interaction)
    except (ValidationError, ValueError, TypeError, IrisRunStateError) as exc:
        raise IrisRunPersistenceError(
            "lifecycle SQLite durable row 无法验证",
            path=str(path),
            operation=operation,
        ) from exc


def _row_to_session(row: sqlite3.Row) -> SessionSnapshot:
    return SessionSnapshot(
        session_id=row["session_id"],
        revision=row["revision"],
        messages=[Msg.from_dict(item) for item in _load_json(row["messages_json"])],
    )


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


def _row_to_run_control(row: sqlite3.Row) -> RunControlSnapshot:
    return RunControlSnapshot(
        run_id=row["run_id"],
        phase=row["phase"],
        revision=row["run_revision"],
        current_activation_id=row["current_activation_id"],
        cancellation_requested_at=row["cancellation_requested_at"],
        cancellation_reason=row["cancellation_reason"],
        last_event_sequence=row["last_event_sequence"],
        updated_at=row["updated_at"],
    )


def _row_to_activation(row: sqlite3.Row) -> ActivationRecord:
    stored_outcome = row["outcome"]
    return ActivationRecord(
        activation_id=row["activation_id"],
        run_id=row["run_id"],
        ordinal=row["ordinal"],
        kind=row["kind"],
        status=row["status"],
        outcome=(ActivationOutcome(stored_outcome) if stored_outcome is not None else None),
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
        updated_at=row["updated_at"],
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


_INSERT_RUN = """INSERT INTO agent_runs(
    run_id, session_id, agent_id, phase, stop_reason, request_json, options_json,
    environment_fingerprint, session_revision, run_revision, current_activation_id,
    pending_interaction_id, cancellation_requested_at, cancellation_reason,
    model_steps_reserved, model_steps_committed, tool_calls_committed, usage_json,
    assistant_message_json, error_json, checkpoint_sequence, last_event_sequence,
    created_at, started_at, updated_at, finished_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

_UPDATE_RUN = """UPDATE agent_runs SET
    session_id = ?, agent_id = ?, phase = ?, stop_reason = ?, request_json = ?,
    options_json = ?, environment_fingerprint = ?, session_revision = ?,
    run_revision = ?, current_activation_id = ?, pending_interaction_id = ?,
    cancellation_requested_at = ?, cancellation_reason = ?,
    model_steps_reserved = ?, model_steps_committed = ?, tool_calls_committed = ?,
    usage_json = ?, assistant_message_json = ?, error_json = ?, checkpoint_sequence = ?,
    last_event_sequence = ?, created_at = ?, started_at = ?, updated_at = ?, finished_at = ?
WHERE run_id = ? AND run_revision = ?"""

_INSERT_TOOL_CALL = """INSERT INTO run_tool_calls(
    run_id, tool_call_id, step_index, ordinal, tool_name, arguments_json, fingerprint,
    interaction_id, phase, claim_activation_id, result_json, version, prepared_at,
    updated_at, claimed_at, committed_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

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
        call.updated_at.isoformat(),
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


def _make_event(
    run: RunRecord,
    kind: RunEventKind,
    occurred_at: datetime,
    *,
    sequence: int,
    activation_id: str | None = None,
    step_index: int | None = None,
    correlation_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> RunEvent:
    """构造与 lifecycle mutation 同事务追加的 event。"""
    return RunEvent(
        run_id=run.run_id,
        session_id=run.session_id,
        sequence=sequence,
        kind=kind,
        occurred_at=occurred_at,
        activation_id=activation_id,
        step_index=step_index,
        correlation_id=correlation_id,
        payload={} if payload is None else payload,
    )


def _replace_run(run: RunRecord, **changes: Any) -> RunRecord:
    """以领域模型验证后的字段替换构造新 run。"""
    return RunRecord.model_validate(run.model_dump() | changes)


def _append_messages(session: SessionSnapshot, delta: list[Msg]) -> SessionSnapshot:
    """仅在存在 delta 时推进 session revision 并追加历史。"""
    if not delta:
        return deepcopy(session)
    return SessionSnapshot(
        session_id=session.session_id,
        revision=session.revision + 1,
        messages=deepcopy(session.messages) + deepcopy(delta),
    )


def _validate_checkpoint_replacement(
    run: RunRecord,
    current: RunCheckpoint,
    replacement: RunCheckpoint,
    activation_id: str,
    session_revision: int,
    usage: RunUsage,
) -> None:
    """保持 checkpoint identity、sequence 与计数器替换约束。"""
    if replacement.run_id != run.run_id:
        raise IrisRunConflictError("checkpoint run identity 不匹配")
    if replacement.sequence != current.sequence + 1:
        raise IrisRunConflictError("checkpoint sequence 必须恰好推进一次")
    if replacement.activation_id != activation_id:
        raise IrisRunConflictError("checkpoint activation fence 不匹配")
    if replacement.session_revision != session_revision:
        raise IrisRunConflictError("checkpoint session revision 不匹配")
    if replacement.environment_fingerprint != run.environment_fingerprint:
        raise IrisRunConflictError("checkpoint environment fingerprint 不匹配")
    if replacement.model_steps_reserved != usage.model_steps_reserved:
        raise IrisRunConflictError("checkpoint reserved counter 不匹配")
    if replacement.model_steps_committed != usage.model_steps_committed:
        raise IrisRunConflictError("checkpoint committed counter 不匹配")


def _validate_pending_interaction(run: RunRecord, interaction: HumanInteraction) -> None:
    """校验 suspend command 的 pending interaction identity。"""
    if interaction.run_id != run.run_id or interaction.session_id != run.session_id:
        raise IrisRunConflictError("interaction run/session identity 不匹配")
    if interaction.status is not InteractionStatus.PENDING:
        raise IrisRunStateError("suspend command 必须包含 pending interaction")
    if interaction.response is not None:
        raise IrisRunStateError("pending interaction 不能包含 response")


def _is_preflight_result(result: ToolResult) -> bool:
    """判断工具失败是否可证明发生在 effect claim 之前。"""
    if not result.is_error or result.error is None:
        return False
    return result.error.code in {
        "NOT_FOUND",
        "PERMISSION_ERROR",
        "TOOL_NOT_ALLOWED",
        "VALIDATION_ERROR",
    }


def _closed_interaction(
    interaction: HumanInteraction,
    now: datetime,
    reason: str,
) -> HumanInteraction:
    """构造当前 open interaction 的 closed replacement。"""
    if interaction.status not in {InteractionStatus.PENDING, InteractionStatus.RESOLVED}:
        raise IrisRunStateError("run 当前 interaction 已关闭")
    return interaction.model_copy(
        deep=True,
        update={
            "status": InteractionStatus.CLOSED,
            "version": interaction.version + 1,
            "closed_at": now,
            "close_reason": reason,
        },
    )


def _activation_outcome(stop_reason: RunStopReason) -> ActivationOutcome:
    """把 terminal stop reason 映射为 activation settlement outcome。"""
    return {
        RunStopReason.COMPLETED: ActivationOutcome.COMPLETED,
        RunStopReason.CANCELLED: ActivationOutcome.CANCELLED,
        RunStopReason.OUTCOME_UNKNOWN: ActivationOutcome.OUTCOME_UNKNOWN,
    }.get(stop_reason, ActivationOutcome.FAILED)


def _replay_key(operation: str, command: BaseModel) -> str:
    payload = command.model_dump(mode="json")
    return f"{operation}:{json.dumps(payload, allow_nan=False, sort_keys=True)}"


def _stored_activation_outcome(activation: ActivationRecord) -> str | None:
    return activation.outcome.value if activation.outcome is not None else None


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
