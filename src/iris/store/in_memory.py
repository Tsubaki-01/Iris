"""进程内 logical run aggregate reference store。

每个 command 在同一把进程内锁中完成检查、构造和替换；所有入口与返回值均深拷贝，
避免调用方修改 store 内部事实。

Example:
    store = InMemoryLifecycleStore()
    commit = store.create_run(command)
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from threading import RLock
from typing import Any, Protocol

from pydantic import BaseModel

from ..exceptions import (
    IrisRunConflictError,
    IrisRunNotFoundError,
    IrisRunRecoveryError,
    IrisRunStateError,
)
from ..hitl.models import (
    HumanInteraction,
    InteractionStatus,
    PermissionInteractionResponse,
    QuestionInteractionResponse,
)
from ..lifecycle.models import (
    ActivationKind,
    ActivationOutcome,
    ActivationRecord,
    ActivationStatus,
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


class _ActiveCommand(Protocol):
    """Active mutation 共用的 CAS 与 activation fence 输入。"""

    run_id: str
    expected_run_revision: int
    activation_id: str


class InMemoryLifecycleStore:
    """使用 process-local dict 实现完整 ``LifecycleStore`` contract。"""

    def __init__(self) -> None:
        self._lock = RLock()
        self._runs: dict[str, RunRecord] = {}
        self._sessions: dict[str, SessionSnapshot] = {}
        self._lanes: dict[str, str] = {}
        self._activations: dict[str, ActivationRecord] = {}
        self._checkpoints: dict[str, RunCheckpoint] = {}
        self._tool_calls: dict[tuple[str, str], RunToolCallRecord] = {}
        self._tool_call_ids_by_run: dict[str, list[str]] = {}
        self._interactions: dict[str, HumanInteraction] = {}
        self._events: dict[str, list[RunEvent]] = {}
        self._results: dict[str, RunResult] = {}
        self._replays: dict[str, RunCommit] = {}

    def create_run(self, command: CreateRun) -> RunCommit:
        """原子创建 run、lane、activation、checkpoint 与起始事件。"""
        command = deepcopy(command)
        with self._lock:
            run_id = command.request.run_id
            if run_id is None:
                raise IrisRunStateError("CreateRun request 缺少最终 run_id")
            if run_id in self._runs:
                raise IrisRunConflictError("run_id 已存在", run_id=run_id)
            owner = self._lanes.get(command.request.session_id)
            if owner is not None:
                raise IrisRunConflictError(
                    "session lane 已被 non-terminal run 占用",
                    session_id=command.request.session_id,
                    owner_run_id=owner,
                )
            if command.start_activation_id in self._activations:
                raise IrisRunConflictError(
                    "activation_id 已存在", activation_id=command.start_activation_id
                )
            session = self._sessions.get(
                command.request.session_id,
                SessionSnapshot(session_id=command.request.session_id),
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
                events = (self._event(run, RunEventKind.RUN_TERMINAL, command.now, sequence=1),)
                result = project_result(run)
                commit = RunCommit(run=run, events=events, result=result)
                self._runs[run_id] = deepcopy(run)
                self._sessions.setdefault(session.session_id, deepcopy(session))
                self._events[run_id] = deepcopy(list(events))
                self._results[run_id] = deepcopy(result)
                return deepcopy(commit)

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
                self._event(run, RunEventKind.RUN_STARTED, command.now, sequence=1),
                self._event(
                    run,
                    RunEventKind.ACTIVATION_STARTED,
                    command.now,
                    sequence=2,
                    activation_id=activation.activation_id,
                ),
            )
            commit = RunCommit(
                run=run,
                session=None,
                checkpoint=command.initial_checkpoint,
                events=events,
            )
            self._runs[run_id] = deepcopy(run)
            self._sessions.setdefault(session.session_id, deepcopy(session))
            self._lanes[session.session_id] = run_id
            self._activations[activation.activation_id] = deepcopy(activation)
            self._checkpoints[run_id] = deepcopy(command.initial_checkpoint)
            self._events[run_id] = deepcopy(list(events))
            return deepcopy(commit)

    def resume_waiting_run(self, command: ResumeWaitingRun) -> RunCommit:
        """从 resolved waiting run 建立新的 active fence。"""
        command = deepcopy(command)
        with self._lock:
            replay = self._load_replay("resume_waiting_run", command)
            if replay is not None:
                return replay
            run = self._require_run(command.run_id)
            self._require_revision(run, command.expected_run_revision)
            if run.phase is not RunPhase.WAITING:
                raise IrisRunStateError("只有 waiting run 可以开始新 activation", run_id=run.run_id)
            checkpoint = self._require_checkpoint(run.run_id)
            if checkpoint.sequence != command.expected_checkpoint_sequence:
                raise IrisRunConflictError("checkpoint sequence 已变化", run_id=run.run_id)
            if command.new_activation_id in self._activations:
                raise IrisRunConflictError(
                    "activation_id 已存在", activation_id=command.new_activation_id
                )
            interaction = self._require_interaction(run.pending_interaction_id)
            if interaction.status is not InteractionStatus.RESOLVED:
                raise IrisRunStateError(
                    "interaction 尚未 resolved", interaction_id=interaction.interaction_id
                )
            self._require_lane(run)

            ordinal = 1 + max(
                (item.ordinal for item in self._activations.values() if item.run_id == run.run_id),
                default=0,
            )
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
            updated = self._replace_run(
                run,
                phase=RunPhase.ACTIVE,
                revision=run.revision + 1,
                current_activation_id=activation.activation_id,
                pending_interaction_id=None,
                checkpoint_sequence=rebound.sequence,
                last_event_sequence=event_sequence,
                updated_at=command.now,
            )
            event = self._event(
                updated,
                RunEventKind.ACTIVATION_STARTED,
                command.now,
                sequence=event_sequence,
                activation_id=activation.activation_id,
            )
            commit = RunCommit(
                run=updated,
                checkpoint=rebound,
                interaction=closed,
                events=(event,),
            )
            self._runs[run.run_id] = deepcopy(updated)
            self._activations[activation.activation_id] = deepcopy(activation)
            self._checkpoints[run.run_id] = deepcopy(rebound)
            self._interactions[closed.interaction_id] = deepcopy(closed)
            self._events[run.run_id].append(deepcopy(event))
            self._results.pop(run.run_id, None)
            return self._store_replay("resume_waiting_run", command, commit)

    def reserve_model_step(self, command: ReserveModelStep) -> RunCommit:
        """在 provider effect 前增加 durable model-step reservation。"""
        command = deepcopy(command)
        with self._lock:
            replay = self._load_replay("reserve_model_step", command)
            if replay is not None:
                return replay
            run = self._require_active(command)
            if run.usage.model_steps_reserved >= run.options.limits.max_model_steps:
                return self._finish_budget_exhausted(run, command.now, command)
            usage = RunUsage.model_validate(
                run.usage.model_dump()
                | {"model_steps_reserved": run.usage.model_steps_reserved + 1}
            )
            checkpoint = self._require_checkpoint(run.run_id).model_copy(
                update={"model_steps_reserved": usage.model_steps_reserved}
            )
            sequence = run.last_event_sequence + 1
            updated = self._replace_run(
                run,
                revision=run.revision + 1,
                usage=usage,
                last_event_sequence=sequence,
                updated_at=command.now,
            )
            event = self._event(
                updated,
                RunEventKind.MODEL_STEP_RESERVED,
                command.now,
                sequence=sequence,
                activation_id=command.activation_id,
                step_index=usage.model_steps_reserved - 1,
            )
            commit = RunCommit(
                run=updated,
                checkpoint=checkpoint,
                events=(event,),
            )
            self._runs[run.run_id] = deepcopy(updated)
            self._checkpoints[run.run_id] = deepcopy(checkpoint)
            self._events[run.run_id].append(deepcopy(event))
            return self._store_replay("reserve_model_step", command, commit)

    def commit_model_step(self, command: CommitModelStep) -> RunCommit:
        """原子提交模型响应及其历史、tool intents 与 checkpoint。"""
        command = deepcopy(command)
        with self._lock:
            replay = self._load_replay("commit_model_step", command)
            if replay is not None:
                return replay
            run = self._require_active(command)
            session = self._require_history_preconditions(run, command.expected_session_revision)
            current_checkpoint = self._require_checkpoint(run.run_id)
            next_session = self._append_messages(session, command.message_delta)
            self._validate_checkpoint_replacement(
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
            prepared = self._validate_prepared_calls(run, command.prepared_tool_calls)
            sequence = run.last_event_sequence + 1
            updated = self._replace_run(
                run,
                revision=run.revision + 1,
                usage=command.usage,
                assistant_message=command.assistant_message,
                checkpoint_sequence=command.checkpoint.sequence,
                last_event_sequence=sequence,
                updated_at=command.now,
            )
            event = self._event(
                updated,
                RunEventKind.MODEL_STEP_COMMITTED,
                command.now,
                sequence=sequence,
                activation_id=command.activation_id,
                step_index=command.usage.model_steps_committed - 1,
            )
            commit = RunCommit(
                run=updated,
                session=next_session if command.message_delta else None,
                checkpoint=command.checkpoint,
                events=(event,),
            )
            self._runs[run.run_id] = deepcopy(updated)
            self._sessions[session.session_id] = deepcopy(next_session)
            self._checkpoints[run.run_id] = deepcopy(command.checkpoint)
            for tool_call in prepared:
                self._set_tool_call(tool_call)
            self._events[run.run_id].append(deepcopy(event))
            return self._store_replay("commit_model_step", command, commit)

    def claim_tool_call(self, command: ClaimToolCall) -> RunCommit:
        """将 prepared tool call durable 转为 claimed。"""
        command = deepcopy(command)
        with self._lock:
            replay = self._load_replay("claim_tool_call", command)
            if replay is not None:
                return replay
            run = self._require_active(command)
            tool_call = self._require_tool_call(run.run_id, command.tool_call_id)
            if tool_call.version != command.expected_tool_version:
                raise IrisRunConflictError(
                    "tool call version 已变化", tool_call_id=command.tool_call_id
                )
            if tool_call.fingerprint != command.fingerprint:
                raise IrisRunConflictError(
                    "tool call fingerprint 不匹配", tool_call_id=command.tool_call_id
                )
            if tool_call.phase is not ToolCallPhase.PREPARED:
                raise IrisRunStateError("只有 prepared tool call 可以 claim")
            if run.cancellation_requested_at is not None:
                raise IrisRunStateError("已请求取消的 run 不接受新的 tool call claim")
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
            updated = self._replace_run(
                run,
                revision=run.revision + 1,
                last_event_sequence=sequence,
                updated_at=command.now,
            )
            event = self._event(
                updated,
                RunEventKind.TOOL_CALL_CLAIMED,
                command.now,
                sequence=sequence,
                activation_id=command.activation_id,
                step_index=tool_call.step_index,
                correlation_id=tool_call.tool_call_id,
            )
            commit = RunCommit(
                run=updated,
                checkpoint=self._checkpoints.get(run.run_id),
                events=(event,),
            )
            self._runs[run.run_id] = deepcopy(updated)
            self._set_tool_call(claimed)
            self._events[run.run_id].append(deepcopy(event))
            return self._store_replay("claim_tool_call", command, commit)

    def commit_tool_result(self, command: CommitToolResult) -> RunCommit:
        """将 claimed 或无副作用失败的 prepared 调用转为 committed。"""
        command = deepcopy(command)
        with self._lock:
            replay = self._load_replay("commit_tool_result", command)
            if replay is not None:
                return replay
            run = self._require_active(command)
            session = self._require_history_preconditions(run, command.expected_session_revision)
            tool_call = self._require_tool_call(run.run_id, command.tool_call_id)
            if tool_call.version != command.expected_tool_version:
                raise IrisRunConflictError(
                    "tool call version 已变化", tool_call_id=command.tool_call_id
                )
            if command.result.tool_use_id != tool_call.tool_call_id:
                raise IrisRunConflictError(
                    "tool result identity 不匹配", tool_call_id=command.tool_call_id
                )
            if command.result.tool_name != tool_call.tool_name:
                raise IrisRunConflictError(
                    "tool result name 不匹配", tool_call_id=command.tool_call_id
                )
            if (
                tool_call.phase is ToolCallPhase.PREPARED
                and not self._is_preflight_result(command.result)
                and not self._is_interaction_result(tool_call, command.result)
            ):
                raise IrisRunStateError("可能包含副作用的工具结果必须先 claim")
            if tool_call.phase not in {ToolCallPhase.PREPARED, ToolCallPhase.CLAIMED}:
                raise IrisRunStateError("当前 tool call phase 不能提交 result")

            next_session = self._append_messages(session, command.message_delta)
            checkpoint = self._require_checkpoint(run.run_id)
            self._validate_checkpoint_replacement(
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
                run.usage.model_dump()
                | {"tool_calls_committed": run.usage.tool_calls_committed + 1}
            )
            sequence = run.last_event_sequence + 1
            updated = self._replace_run(
                run,
                revision=run.revision + 1,
                usage=usage,
                checkpoint_sequence=command.checkpoint.sequence,
                last_event_sequence=sequence,
                updated_at=command.now,
            )
            event = self._event(
                updated,
                RunEventKind.TOOL_CALL_COMMITTED,
                command.now,
                sequence=sequence,
                activation_id=command.activation_id,
                step_index=tool_call.step_index,
                correlation_id=tool_call.tool_call_id,
            )
            commit = RunCommit(
                run=updated,
                session=next_session if command.message_delta else None,
                checkpoint=command.checkpoint,
                events=(event,),
            )
            self._runs[run.run_id] = deepcopy(updated)
            self._sessions[session.session_id] = deepcopy(next_session)
            self._checkpoints[run.run_id] = deepcopy(command.checkpoint)
            self._set_tool_call(committed_call)
            self._events[run.run_id].append(deepcopy(event))
            return self._store_replay("commit_tool_result", command, commit)

    def suspend_run(self, command: SuspendRun) -> RunCommit:
        """原子提交当前 facts 并将 active run 转为 waiting。"""
        command = deepcopy(command)
        with self._lock:
            replay = self._load_replay("suspend_run", command)
            if replay is not None:
                return replay
            run = self._require_active(command)
            session = self._require_history_preconditions(run, command.expected_session_revision)
            checkpoint = self._require_checkpoint(run.run_id)
            next_session = self._append_messages(session, command.message_delta)
            self._validate_checkpoint_replacement(
                run,
                checkpoint,
                command.checkpoint,
                command.activation_id,
                next_session.revision,
                command.usage,
            )
            interaction = command.pending_interaction
            self._validate_pending_interaction(run, interaction)
            if any(
                item.run_id == run.run_id
                and item.status in {InteractionStatus.PENDING, InteractionStatus.RESOLVED}
                for item in self._interactions.values()
            ):
                raise IrisRunConflictError("run 已存在 open interaction", run_id=run.run_id)
            prepared = self._validate_prepared_calls(run, command.prepared_tool_calls)
            interaction_tool = next(
                (item for item in prepared if item.tool_call_id == interaction.tool_call_id),
                self._tool_calls.get((run.run_id, interaction.tool_call_id)),
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
            activation = self._require_activation(command.activation_id)
            settled = ActivationRecord.model_validate(
                activation.model_dump()
                | {
                    "status": ActivationStatus.SETTLED,
                    "outcome": ActivationOutcome.SUSPENDED,
                    "ended_at": command.now,
                }
            )
            sequence = run.last_event_sequence + 1
            updated = self._replace_run(
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
            event = self._event(
                updated,
                RunEventKind.INTERACTION_SUSPENDED,
                command.now,
                sequence=sequence,
                activation_id=command.activation_id,
                step_index=interaction.step_index,
                correlation_id=interaction.interaction_id,
            )
            result = project_result(updated, interaction)
            commit = RunCommit(
                run=updated,
                session=next_session if command.message_delta else None,
                checkpoint=command.checkpoint,
                interaction=interaction,
                events=(event,),
                result=result,
            )
            self._runs[run.run_id] = deepcopy(updated)
            self._sessions[session.session_id] = deepcopy(next_session)
            self._activations[activation.activation_id] = deepcopy(settled)
            self._checkpoints[run.run_id] = deepcopy(command.checkpoint)
            self._interactions[interaction.interaction_id] = deepcopy(interaction)
            for tool_call in prepared:
                self._set_tool_call(tool_call)
            self._set_tool_call(bound_interaction_tool)
            self._events[run.run_id].append(deepcopy(event))
            self._results[run.run_id] = deepcopy(result)
            return self._store_replay("suspend_run", command, commit)

    def resolve_interaction(self, command: ResolveInteraction) -> RunCommit:
        """以 version、kind 与 fingerprint CAS 写入人工响应。"""
        command = deepcopy(command)
        with self._lock:
            replay = self._load_replay("resolve_interaction", command)
            if replay is not None:
                return replay
            run = self._require_run(command.run_id)
            if run.phase is not RunPhase.WAITING:
                raise IrisRunStateError("只有 waiting run 可以 resolve interaction")
            interaction = self._require_interaction(command.interaction_id)
            if run.pending_interaction_id != interaction.interaction_id:
                raise IrisRunConflictError("interaction 已不再属于 run 当前等待")
            if interaction.request.tool_call.fingerprint != command.expected_fingerprint:
                raise IrisRunConflictError("interaction fingerprint 不匹配")
            if interaction.request.prompt.kind != command.response.kind:
                raise IrisRunConflictError("interaction response kind 不匹配")
            if interaction.status is InteractionStatus.RESOLVED:
                if interaction.response != command.response:
                    raise IrisRunConflictError("interaction 已保存不同 response")
                return self._current_commit(run, interaction=interaction)
            self._require_revision(run, command.expected_run_revision)
            if interaction.version != command.expected_interaction_version:
                raise IrisRunConflictError("interaction version 已变化")
            if interaction.status is not InteractionStatus.PENDING:
                raise IrisRunStateError("只有 pending interaction 可以 resolve")
            if interaction.expires_at is not None and command.now >= interaction.expires_at:
                raise IrisRunStateError(
                    "interaction 已过期", interaction_id=interaction.interaction_id
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
            updated = self._replace_run(
                run,
                revision=run.revision + 1,
                last_event_sequence=sequence,
                updated_at=command.now,
            )
            event = self._event(
                updated,
                RunEventKind.INTERACTION_RESOLVED,
                command.now,
                sequence=sequence,
                correlation_id=interaction.interaction_id,
            )
            result = project_result(updated, resolved)
            commit = RunCommit(
                run=updated,
                checkpoint=self._checkpoints.get(run.run_id),
                interaction=resolved,
                events=(event,),
                result=result,
            )
            self._runs[run.run_id] = deepcopy(updated)
            self._interactions[interaction.interaction_id] = deepcopy(resolved)
            self._events[run.run_id].append(deepcopy(event))
            self._results[run.run_id] = deepcopy(result)
            return self._store_replay("resolve_interaction", command, commit)

    def request_cancellation(self, command: RequestCancellation) -> RunCommit:
        """记录首次 cancellation request，并按显式要求结算 waiting run。"""
        command = deepcopy(command)
        with self._lock:
            replay = self._load_replay("request_cancellation", command)
            if replay is not None:
                return replay
            run = self._require_run(command.run_id)
            if run.phase is RunPhase.TERMINAL:
                raise IrisRunStateError("terminal run 不接受 cancellation request")
            if run.phase is RunPhase.ACTIVE:
                if command.activation_id != run.current_activation_id:
                    raise IrisRunConflictError("activation fence 已变化", run_id=run.run_id)
            elif command.activation_id is not None:
                raise IrisRunConflictError("waiting run 不应携带 activation fence")
            if run.cancellation_requested_at is not None:
                if run.cancellation_reason == command.reason and not command.settle_waiting:
                    replay_interaction = (
                        self._require_interaction(run.pending_interaction_id)
                        if run.phase is RunPhase.WAITING
                        else None
                    )
                    return self._current_commit(run, interaction=replay_interaction)
                raise IrisRunConflictError("cancellation 已由其他 command 请求", run_id=run.run_id)
            self._require_revision(run, command.expected_run_revision)
            checkpoint = self._require_checkpoint(run.run_id)
            sequence = run.last_event_sequence + 1
            events = [
                self._event(
                    run,
                    RunEventKind.CANCELLATION_REQUESTED,
                    command.now,
                    sequence=sequence,
                    activation_id=command.activation_id,
                    payload={"reason": command.reason},
                )
            ]
            interaction: HumanInteraction | None = None
            updated_session: SessionSnapshot | None = None
            updated_checkpoint = checkpoint
            claimed_closures: list[RunToolCallRecord] = []
            if run.phase is RunPhase.WAITING and command.settle_waiting:
                interaction = self._close_interaction(run, command.now, command.reason)
                closures = self._terminal_tool_closures(run, command.now)
                closure_messages = [message for _, _, message in closures]
                current_session = self._sessions.get(
                    run.session_id,
                    SessionSnapshot(session_id=run.session_id),
                )
                appended_session = self._append_messages(current_session, closure_messages)
                if closure_messages:
                    updated_session = appended_session
                    updated_checkpoint = checkpoint.model_copy(
                        deep=True,
                        update={"session_revision": appended_session.revision},
                    )
                claimed_closures = [
                    updated_call
                    for current_call, updated_call, _ in closures
                    if current_call.phase is ToolCallPhase.CLAIMED
                ]
                for index, record in enumerate(claimed_closures, start=1):
                    events.append(
                        self._event(
                            run,
                            RunEventKind.TOOL_CALL_OUTCOME_UNKNOWN,
                            command.now,
                            sequence=sequence + index,
                            activation_id=record.claim_activation_id,
                            step_index=record.step_index,
                            correlation_id=record.tool_call_id,
                        )
                    )
                sequence += len(claimed_closures) + 1
                updated = self._replace_run(
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
                    self._event(
                        updated,
                        RunEventKind.RUN_TERMINAL,
                        command.now,
                        sequence=sequence,
                    )
                )
                result = project_result(updated)
                self._lanes.pop(run.session_id, None)
                self._results[run.run_id] = deepcopy(result)
            else:
                updated = self._replace_run(
                    run,
                    revision=run.revision + 1,
                    cancellation_requested_at=command.now,
                    cancellation_reason=command.reason,
                    last_event_sequence=sequence,
                    updated_at=command.now,
                )
                result = self._results.get(run.run_id)
                if run.phase is RunPhase.WAITING:
                    result = project_result(
                        updated, self._require_interaction(run.pending_interaction_id)
                    )
                    self._results[run.run_id] = deepcopy(result)
            commit = RunCommit(
                run=updated,
                session=updated_session,
                checkpoint=updated_checkpoint,
                interaction=interaction,
                events=tuple(events),
                result=result,
            )
            self._runs[run.run_id] = deepcopy(updated)
            if (
                updated_session is not None
                and updated_session.revision != checkpoint.session_revision
            ):
                self._sessions[run.session_id] = deepcopy(updated_session)
                self._checkpoints[run.run_id] = deepcopy(updated_checkpoint)
            if interaction is not None:
                self._interactions[interaction.interaction_id] = deepcopy(interaction)
            for record in claimed_closures:
                self._set_tool_call(record)
            self._events[run.run_id].extend(deepcopy(events))
            return self._store_replay("request_cancellation", command, commit)

    def finish_run(self, command: FinishRun) -> RunCommit:
        """将 active/waiting run 原子结算并释放 session lane。"""
        command = deepcopy(command)
        with self._lock:
            replay = self._load_replay("finish_run", command)
            if replay is not None:
                return replay
            run = self._require_run(command.run_id)
            self._require_revision(run, command.expected_run_revision)
            if run.phase is RunPhase.TERMINAL:
                raise IrisRunStateError("terminal run 不接受进一步 mutation", run_id=run.run_id)
            self._require_lane(run)
            if run.phase is RunPhase.ACTIVE:
                if command.activation_id != run.current_activation_id:
                    raise IrisRunConflictError("activation fence 已变化", run_id=run.run_id)
            elif command.activation_id is not None:
                raise IrisRunConflictError("waiting run 不应携带 activation fence")
            if (
                command.stop_reason
                in {
                    RunStopReason.FAILED,
                    RunStopReason.OUTCOME_UNKNOWN,
                }
                and command.error is None
            ):
                raise IrisRunStateError("failed/outcome_unknown finish 必须包含 error")
            if command.stop_reason is RunStopReason.COMPLETED and command.error is not None:
                raise IrisRunStateError("completed finish 不能包含 error")

            interaction: HumanInteraction | None = None
            if run.phase is RunPhase.WAITING:
                close_reason = command.interaction_close_reason or command.stop_reason.value
                interaction = self._close_interaction(run, command.now, close_reason)
            if run.phase is RunPhase.ACTIVE:
                activation = self._require_activation(run.current_activation_id)
                outcome = self._activation_outcome(command.stop_reason)
                settled = ActivationRecord.model_validate(
                    activation.model_dump()
                    | {
                        "status": ActivationStatus.SETTLED,
                        "outcome": outcome,
                        "ended_at": command.now,
                    }
                )
            else:
                activation = None
                settled = None
            checkpoint = self._require_checkpoint(run.run_id)
            closures = self._terminal_tool_closures(run, command.now)
            closure_messages = [message for _, _, message in closures]
            current_session = self._sessions.get(
                run.session_id,
                SessionSnapshot(session_id=run.session_id),
            )
            updated_session = self._append_messages(current_session, closure_messages)
            updated_checkpoint = (
                checkpoint.model_copy(
                    deep=True,
                    update={"session_revision": updated_session.revision},
                )
                if closure_messages
                else checkpoint
            )
            unknown_calls = [
                updated_call
                for current_call, updated_call, _ in closures
                if current_call.phase is ToolCallPhase.CLAIMED
            ]
            sequence = run.last_event_sequence + len(unknown_calls) + 1
            updated = self._replace_run(
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
                self._event(
                    updated,
                    RunEventKind.TOOL_CALL_OUTCOME_UNKNOWN,
                    command.now,
                    sequence=run.last_event_sequence + index,
                    activation_id=record.claim_activation_id,
                    step_index=record.step_index,
                    correlation_id=record.tool_call_id,
                )
                for index, record in enumerate(unknown_calls, start=1)
            )
            terminal_event = self._event(
                updated,
                RunEventKind.RUN_TERMINAL,
                command.now,
                sequence=sequence,
                activation_id=command.activation_id,
                payload={"stop_reason": command.stop_reason.value},
            )
            result = project_result(updated)
            commit = RunCommit(
                run=updated,
                session=updated_session if closure_messages else None,
                checkpoint=updated_checkpoint,
                interaction=interaction,
                events=(*unknown_events, terminal_event),
                result=result,
            )
            self._runs[run.run_id] = deepcopy(updated)
            if closure_messages:
                self._sessions[run.session_id] = deepcopy(updated_session)
                self._checkpoints[run.run_id] = deepcopy(updated_checkpoint)
            self._lanes.pop(run.session_id, None)
            if activation is not None and settled is not None:
                self._activations[activation.activation_id] = deepcopy(settled)
            if interaction is not None:
                self._interactions[interaction.interaction_id] = deepcopy(interaction)
            for record in unknown_calls:
                self._set_tool_call(record)
            self._events[run.run_id].extend(deepcopy(commit.events))
            self._results[run.run_id] = deepcopy(result)
            return self._store_replay("finish_run", command, commit)

    def recover_active_run(self, command: RecoverActiveRun) -> RunCommit:
        """按 durable checkpoint/tool facts 放弃并恢复或终止旧 activation。"""
        command = deepcopy(command)
        with self._lock:
            replay = self._load_replay("recover_active_run", command)
            if replay is not None:
                return replay
            run = self._require_run(command.run_id)
            self._require_revision(run, command.expected_run_revision)
            if run.phase is not RunPhase.ACTIVE:
                raise IrisRunStateError("只有 active run 可以执行 active recovery")
            if run.current_activation_id != command.expected_activation_id:
                raise IrisRunConflictError("activation fence 已变化", run_id=run.run_id)
            checkpoint = self._require_checkpoint(run.run_id)
            if checkpoint.sequence != command.expected_checkpoint_sequence:
                raise IrisRunConflictError("checkpoint sequence 已变化", run_id=run.run_id)
            activation = self._require_activation(command.expected_activation_id)
            claimed = [
                item
                for item in self._tool_calls.values()
                if item.run_id == run.run_id and item.phase is ToolCallPhase.CLAIMED
            ]
            if command.recovery_disposition is RecoveryDisposition.RESUME and claimed:
                raise IrisRunRecoveryError(
                    "safe recovery 不能重放 unresolved durable claim", run_id=run.run_id
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
            abandoned_event = self._event(
                run,
                RunEventKind.ACTIVATION_ABANDONED,
                command.now,
                sequence=first_sequence,
                activation_id=activation.activation_id,
            )
            terminal_closures = (
                self._terminal_tool_closures(run, command.now)
                if command.recovery_disposition
                in {RecoveryDisposition.OUTCOME_UNKNOWN, RecoveryDisposition.FINALIZE}
                else []
            )
            closure_messages = [message for _, _, message in terminal_closures]
            updated_session: SessionSnapshot | None = None
            terminal_checkpoint = checkpoint
            if closure_messages:
                current_session = self._sessions.get(
                    run.session_id,
                    SessionSnapshot(session_id=run.session_id),
                )
                updated_session = self._append_messages(current_session, closure_messages)
                terminal_checkpoint = checkpoint.model_copy(
                    deep=True,
                    update={"session_revision": updated_session.revision},
                )
            if command.recovery_disposition is RecoveryDisposition.OUTCOME_UNKNOWN:
                if not claimed:
                    raise IrisRunRecoveryError(
                        "outcome_unknown recovery 缺少 unresolved durable claim", run_id=run.run_id
                    )
                unknown_calls = [
                    updated_call
                    for current_call, updated_call, _ in terminal_closures
                    if current_call.phase is ToolCallPhase.CLAIMED
                ]
                terminal_sequence = first_sequence + len(unknown_calls) + 1
                updated = self._replace_run(
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
                    self._event(
                        updated,
                        RunEventKind.TOOL_CALL_OUTCOME_UNKNOWN,
                        command.now,
                        sequence=first_sequence + index,
                        activation_id=record.claim_activation_id,
                        step_index=record.step_index,
                        correlation_id=record.tool_call_id,
                    )
                    for index, record in enumerate(unknown_calls, start=1)
                )
                terminal_event = self._event(
                    updated,
                    RunEventKind.RUN_TERMINAL,
                    command.now,
                    sequence=terminal_sequence,
                    payload={"stop_reason": RunStopReason.OUTCOME_UNKNOWN.value},
                )
                events = (abandoned_event, *unknown_events, terminal_event)
                result = project_result(updated)
                commit = RunCommit(
                    run=updated,
                    session=updated_session,
                    checkpoint=terminal_checkpoint,
                    events=events,
                    result=result,
                )
                self._lanes.pop(run.session_id, None)
                self._results[run.run_id] = deepcopy(result)
                for item in unknown_calls:
                    self._set_tool_call(item)
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
                updated = self._replace_run(
                    run,
                    phase=RunPhase.TERMINAL,
                    stop_reason=RunStopReason.COMPLETED,
                    revision=run.revision + 1,
                    current_activation_id=None,
                    last_event_sequence=terminal_sequence,
                    updated_at=command.now,
                    finished_at=command.now,
                )
                terminal_event = self._event(
                    updated,
                    RunEventKind.RUN_TERMINAL,
                    command.now,
                    sequence=terminal_sequence,
                    payload={"stop_reason": RunStopReason.COMPLETED.value},
                )
                events = (abandoned_event, terminal_event)
                result = project_result(updated)
                commit = RunCommit(
                    run=updated,
                    session=updated_session,
                    checkpoint=terminal_checkpoint,
                    events=events,
                    result=result,
                )
                self._lanes.pop(run.session_id, None)
                self._results[run.run_id] = deepcopy(result)
            else:
                if command.new_activation_id is None:
                    raise IrisRunRecoveryError("resume recovery 缺少 new activation identity")
                if command.new_activation_id in self._activations:
                    raise IrisRunConflictError(
                        "activation_id 已存在", activation_id=command.new_activation_id
                    )
                ordinal = 1 + max(
                    (
                        item.ordinal
                        for item in self._activations.values()
                        if item.run_id == run.run_id
                    ),
                    default=0,
                )
                activation_next = ActivationRecord(
                    activation_id=command.new_activation_id,
                    run_id=run.run_id,
                    ordinal=ordinal,
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
                updated = self._replace_run(
                    run,
                    revision=run.revision + 1,
                    current_activation_id=activation_next.activation_id,
                    checkpoint_sequence=rebound.sequence,
                    last_event_sequence=start_sequence,
                    updated_at=command.now,
                )
                start_event = self._event(
                    updated,
                    RunEventKind.ACTIVATION_STARTED,
                    command.now,
                    sequence=start_sequence,
                    activation_id=activation_next.activation_id,
                )
                events = (abandoned_event, start_event)
                result = None
                commit = RunCommit(run=updated, checkpoint=rebound, events=events)
                self._activations[activation_next.activation_id] = deepcopy(activation_next)
                self._checkpoints[run.run_id] = deepcopy(rebound)
            self._runs[run.run_id] = deepcopy(updated)
            if updated_session is not None:
                self._sessions[run.session_id] = deepcopy(updated_session)
                self._checkpoints[run.run_id] = deepcopy(terminal_checkpoint)
            self._activations[activation.activation_id] = deepcopy(abandoned)
            self._events[run.run_id].extend(deepcopy(list(events)))
            return self._store_replay("recover_active_run", command, commit)

    def load_run(self, run_id: str) -> RunRecord | None:
        """按 ID 返回 copy-isolated run record。"""
        with self._lock:
            return deepcopy(self._runs.get(run_id))

    def load_run_control(self, run_id: str) -> RunControlSnapshot | None:
        """返回 activation/cancellation 判断所需的最小 run 投影。"""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            return RunControlSnapshot(
                run_id=run.run_id,
                phase=run.phase,
                revision=run.revision,
                current_activation_id=run.current_activation_id,
                cancellation_requested_at=run.cancellation_requested_at,
                cancellation_reason=run.cancellation_reason,
                last_event_sequence=run.last_event_sequence,
                updated_at=run.updated_at,
            )

    def load_session(self, session_id: str) -> SessionSnapshot:
        """返回 session snapshot；缺失 session 表示 revision 0 的空历史。"""
        with self._lock:
            return deepcopy(self._sessions.get(session_id, SessionSnapshot(session_id=session_id)))

    def load_session_lane(self, session_id: str) -> str | None:
        """返回当前 session 的 non-terminal lane owner。"""
        with self._lock:
            return self._lanes.get(session_id)

    def load_interaction(self, interaction_id: str) -> HumanInteraction | None:
        """按 ID 返回 copy-isolated interaction。"""
        with self._lock:
            return deepcopy(self._interactions.get(interaction_id))

    def load_checkpoint(self, run_id: str) -> RunCheckpoint | None:
        """返回 run 的 current checkpoint。"""
        with self._lock:
            return deepcopy(self._checkpoints.get(run_id))

    def load_tool_call(
        self,
        run_id: str,
        tool_call_id: str,
    ) -> RunToolCallRecord | None:
        """按 composite identity 返回 copy-isolated tool call。"""
        with self._lock:
            return deepcopy(self._tool_calls.get((run_id, tool_call_id)))

    def list_tool_calls(self, run_id: str) -> list[RunToolCallRecord]:
        """按 step index 与 ordinal 返回 run 的全部工具调用。"""
        with self._lock:
            if run_id not in self._runs:
                raise IrisRunNotFoundError("run 不存在", run_id=run_id)
            calls = [
                self._tool_calls[(run_id, tool_call_id)]
                for tool_call_id in self._tool_call_ids_by_run.get(run_id, ())
            ]
            return deepcopy(sorted(calls, key=lambda item: (item.step_index, item.ordinal)))

    def load_result(self, run_id: str) -> RunResult | None:
        """Active run 返回 ``None``；waiting/terminal 返回 durable result。"""
        with self._lock:
            if run_id not in self._runs:
                return None
            return deepcopy(self._results.get(run_id))

    def list_events(self, run_id: str, after_sequence: int = 0) -> list[RunEvent]:
        """返回 sequence 严格大于游标的 durable events。"""
        if after_sequence < 0:
            raise IrisRunStateError("after_sequence 不能小于 0", after_sequence=after_sequence)
        with self._lock:
            if run_id not in self._runs:
                raise IrisRunNotFoundError("run 不存在", run_id=run_id)
            return deepcopy(
                [event for event in self._events[run_id] if event.sequence > after_sequence]
            )

    def _require_run(self, run_id: str) -> RunRecord:
        run = self._runs.get(run_id)
        if run is None:
            raise IrisRunNotFoundError("run 不存在", run_id=run_id)
        return run

    def _require_checkpoint(self, run_id: str) -> RunCheckpoint:
        checkpoint = self._checkpoints.get(run_id)
        if checkpoint is None:
            raise IrisRunStateError("run 不存在 current checkpoint", run_id=run_id)
        return checkpoint

    def _require_activation(self, activation_id: str | None) -> ActivationRecord:
        if activation_id is None or activation_id not in self._activations:
            raise IrisRunConflictError(
                "activation 不存在或 fence 已变化", activation_id=activation_id
            )
        return self._activations[activation_id]

    def _require_interaction(self, interaction_id: str | None) -> HumanInteraction:
        if interaction_id is None or interaction_id not in self._interactions:
            raise IrisRunConflictError("interaction 不存在或已变化", interaction_id=interaction_id)
        return self._interactions[interaction_id]

    def _require_tool_call(self, run_id: str, tool_call_id: str) -> RunToolCallRecord:
        tool_call = self._tool_calls.get((run_id, tool_call_id))
        if tool_call is None:
            raise IrisRunNotFoundError("tool call 不存在", run_id=run_id, tool_call_id=tool_call_id)
        return tool_call

    def _set_tool_call(self, record: RunToolCallRecord) -> None:
        """写入权威 tool fact，并在首次插入时登记 per-run identity。"""
        key = (record.run_id, record.tool_call_id)
        if key not in self._tool_calls:
            self._tool_call_ids_by_run.setdefault(record.run_id, []).append(record.tool_call_id)
        self._tool_calls[key] = deepcopy(record)

    @staticmethod
    def _require_revision(run: RunRecord, expected: int) -> None:
        if run.revision != expected:
            raise IrisRunConflictError(
                "run revision 已变化", run_id=run.run_id, expected=expected, actual=run.revision
            )

    def _require_active(self, command: _ActiveCommand) -> RunRecord:
        run = self._require_run(command.run_id)
        self._require_revision(run, command.expected_run_revision)
        if run.phase is RunPhase.TERMINAL:
            raise IrisRunStateError("terminal run 不接受进一步 mutation", run_id=run.run_id)
        if run.phase is not RunPhase.ACTIVE:
            raise IrisRunStateError("command 要求 active run", run_id=run.run_id)
        if run.current_activation_id != command.activation_id:
            raise IrisRunConflictError("activation fence 已变化", run_id=run.run_id)
        self._require_lane(run)
        return run

    def _require_lane(self, run: RunRecord) -> None:
        if self._lanes.get(run.session_id) != run.run_id:
            raise IrisRunConflictError("session lane owner 已变化", run_id=run.run_id)

    def _require_history_preconditions(
        self,
        run: RunRecord,
        expected_session_revision: int,
    ) -> SessionSnapshot:
        self._require_lane(run)
        session = self._sessions.get(run.session_id, SessionSnapshot(session_id=run.session_id))
        if session.revision != expected_session_revision:
            raise IrisRunConflictError(
                "session revision 已变化",
                session_id=run.session_id,
                expected=expected_session_revision,
                actual=session.revision,
            )
        return session

    @staticmethod
    def _append_messages(session: SessionSnapshot, delta: list[Msg]) -> SessionSnapshot:
        if not delta:
            return deepcopy(session)
        return SessionSnapshot(
            session_id=session.session_id,
            revision=session.revision + 1,
            messages=deepcopy(session.messages) + deepcopy(delta),
        )

    @staticmethod
    def _replace_run(run: RunRecord, **changes: Any) -> RunRecord:
        return RunRecord.model_validate(run.model_dump() | changes)

    @staticmethod
    def _event(
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

    @staticmethod
    def _validate_checkpoint_replacement(
        run: RunRecord,
        current: RunCheckpoint,
        replacement: RunCheckpoint,
        activation_id: str,
        session_revision: int,
        usage: RunUsage,
    ) -> None:
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

    def _validate_prepared_calls(
        self,
        run: RunRecord,
        calls: list[RunToolCallRecord],
    ) -> list[RunToolCallRecord]:
        identities: set[str] = set()
        result: list[RunToolCallRecord] = []
        for tool_call in calls:
            if tool_call.run_id != run.run_id or tool_call.phase is not ToolCallPhase.PREPARED:
                raise IrisRunStateError("prepared tool call 的 run/phase 不一致")
            if (
                tool_call.tool_call_id in identities
                or (
                    run.run_id,
                    tool_call.tool_call_id,
                )
                in self._tool_calls
            ):
                raise IrisRunConflictError(
                    "tool_call_id 已存在", tool_call_id=tool_call.tool_call_id
                )
            identities.add(tool_call.tool_call_id)
            result.append(deepcopy(tool_call))
        return result

    @staticmethod
    def _validate_pending_interaction(run: RunRecord, interaction: HumanInteraction) -> None:
        if interaction.run_id != run.run_id or interaction.session_id != run.session_id:
            raise IrisRunConflictError("interaction run/session identity 不匹配")
        if interaction.status is not InteractionStatus.PENDING:
            raise IrisRunStateError("suspend command 必须包含 pending interaction")
        if interaction.response is not None:
            raise IrisRunStateError("pending interaction 不能包含 response")

    def _close_interaction(
        self,
        run: RunRecord,
        now: datetime,
        reason: str,
    ) -> HumanInteraction:
        interaction = self._require_interaction(run.pending_interaction_id)
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

    def _terminal_tool_closures(
        self,
        run: RunRecord,
        now: datetime,
    ) -> list[tuple[RunToolCallRecord, RunToolCallRecord, Msg]]:
        """构造当前 run 所有未闭合 tool call 的 fact/message。"""
        records = sorted(
            (
                record
                for record in self._tool_calls.values()
                if record.run_id == run.run_id
                and record.phase in {ToolCallPhase.PREPARED, ToolCallPhase.CLAIMED}
            ),
            key=lambda record: (record.step_index, record.ordinal),
        )
        return [(record, *build_terminal_tool_closure(record, now=now)) for record in records]

    @staticmethod
    def _activation_outcome(stop_reason: RunStopReason) -> ActivationOutcome:
        return {
            RunStopReason.COMPLETED: ActivationOutcome.COMPLETED,
            RunStopReason.CANCELLED: ActivationOutcome.CANCELLED,
            RunStopReason.OUTCOME_UNKNOWN: ActivationOutcome.OUTCOME_UNKNOWN,
        }.get(stop_reason, ActivationOutcome.FAILED)

    @staticmethod
    def _is_preflight_result(result: ToolResult) -> bool:
        if not result.is_error or result.error is None:
            return False
        return result.error.code in {
            "NOT_FOUND",
            "PERMISSION_ERROR",
            "TOOL_NOT_ALLOWED",
            "VALIDATION_ERROR",
        }

    def _is_interaction_result(
        self,
        tool_call: RunToolCallRecord,
        result: ToolResult,
    ) -> bool:
        if tool_call.interaction_id is None:
            return False
        interaction = self._interactions.get(tool_call.interaction_id)
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

    def _current_commit(
        self,
        run: RunRecord,
        *,
        interaction: HumanInteraction | None = None,
    ) -> RunCommit:
        return deepcopy(
            RunCommit(
                run=run,
                checkpoint=self._checkpoints.get(run.run_id),
                interaction=interaction,
                events=(),
                result=self._results.get(run.run_id),
            )
        )

    def _finish_budget_exhausted(
        self,
        run: RunRecord,
        now: datetime,
        command: ReserveModelStep,
    ) -> RunCommit:
        activation = self._require_activation(run.current_activation_id)
        settled = ActivationRecord.model_validate(
            activation.model_dump()
            | {
                "status": ActivationStatus.SETTLED,
                "outcome": ActivationOutcome.FAILED,
                "ended_at": now,
            }
        )
        sequence = run.last_event_sequence + 1
        updated = self._replace_run(
            run,
            phase=RunPhase.TERMINAL,
            stop_reason=RunStopReason.BUDGET_EXHAUSTED,
            revision=run.revision + 1,
            current_activation_id=None,
            last_event_sequence=sequence,
            updated_at=now,
            finished_at=now,
        )
        event = self._event(
            updated,
            RunEventKind.RUN_TERMINAL,
            now,
            sequence=sequence,
            activation_id=activation.activation_id,
            payload={"stop_reason": RunStopReason.BUDGET_EXHAUSTED.value},
        )
        result = project_result(updated)
        commit = RunCommit(
            run=updated,
            checkpoint=self._checkpoints.get(run.run_id),
            events=(event,),
            result=result,
        )
        self._runs[run.run_id] = deepcopy(updated)
        self._activations[activation.activation_id] = deepcopy(settled)
        self._lanes.pop(run.session_id, None)
        self._events[run.run_id].append(deepcopy(event))
        self._results[run.run_id] = deepcopy(result)
        return self._store_replay("reserve_model_step", command, commit)

    @staticmethod
    def _replay_key(operation: str, command: BaseModel) -> str:
        payload = command.model_dump(mode="json")
        return f"{operation}:{json.dumps(payload, allow_nan=False, sort_keys=True)}"

    def _load_replay(self, operation: str, command: BaseModel) -> RunCommit | None:
        replay = self._replays.get(self._replay_key(operation, command))
        if replay is None:
            return None
        run = self._runs[replay.run.run_id]
        session = self._sessions.get(run.session_id) if replay.session is not None else None
        checkpoint = self._checkpoints.get(run.run_id)
        interaction = (
            self._interactions.get(replay.interaction.interaction_id)
            if replay.interaction is not None
            else None
        )
        current = replay.model_copy(
            deep=True,
            update={
                "run": deepcopy(run),
                "session": deepcopy(session),
                "checkpoint": deepcopy(checkpoint),
                "interaction": deepcopy(interaction),
                "events": (),
                "result": deepcopy(self._results.get(run.run_id)),
            },
        )
        return deepcopy(current)

    def _store_replay(
        self,
        operation: str,
        command: BaseModel,
        commit: RunCommit,
    ) -> RunCommit:
        isolated = deepcopy(commit)
        self._replays[self._replay_key(operation, command)] = isolated
        return deepcopy(isolated)


__all__ = ["InMemoryLifecycleStore"]
