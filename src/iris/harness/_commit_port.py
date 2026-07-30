"""LifecycleStore 到 RuntimeCommitPort 的 activation-bound 适配器。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

from ..exceptions import IrisRunConflictError, IrisRunNotFoundError, IrisRunStateError
from ..hitl import HumanInteraction, HumanInteractionService
from ..lifecycle import (
    CheckpointResumability,
    ClaimToolCall,
    CommitModelStep,
    CommitToolResult,
    LifecycleStore,
    ReserveModelStep,
    RunCheckpoint,
    RunCommit,
    RunEvent,
    RunPhase,
    RunRecord,
    RunToolCallRecord,
    RunUsage,
    SessionSnapshot,
    SuspendRun,
    ToolCallPhase,
    snapshot_run,
)
from ..runtime import (
    ModelStepReservation,
    RuntimeCommitPort,
    RuntimeCursor,
    RuntimeModelStepCommit,
    RuntimeSuspension,
    RuntimeSuspensionResult,
    RuntimeToolCall,
    RuntimeToolResultCommit,
    ToolCallClaim,
)


class StoreRuntimeCommitPort(RuntimeCommitPort):
    """把一个 activation 的 engine commits 映射到 lifecycle aggregate。"""

    def __init__(
        self,
        *,
        store: LifecycleStore,
        run: RunRecord,
        activation_id: str,
        clock: Callable[[], datetime],
        event_sink: list[RunEvent],
        interaction_service: HumanInteractionService | None = None,
    ) -> None:
        if run.phase is not RunPhase.ACTIVE or run.current_activation_id != activation_id:
            raise IrisRunStateError("commit port 必须绑定当前 active activation")
        checkpoint = store.load_checkpoint(run.run_id)
        if checkpoint is None or checkpoint.activation_id != activation_id:
            raise IrisRunConflictError("commit port checkpoint activation 不匹配")
        self._store = store
        self._run = run
        self._checkpoint = checkpoint
        self._session_revision = store.load_session(run.session_id).revision
        self._activation_id = activation_id
        self._clock = clock
        self._event_sink = event_sink
        self._interaction_service = interaction_service or HumanInteractionService()
        self._event_keys = {(event.run_id, event.sequence) for event in event_sink}
        self._reusable_model_reservation = (
            checkpoint.model_steps_reserved == checkpoint.model_steps_committed + 1
        )
        self._writable = True

    @property
    def run(self) -> RunRecord:
        """返回最近一次成功 store commit 的 run record。"""
        return self._run

    def revoke(self) -> None:
        """撤销该 activation 后续所有 mutation 权限。"""
        self._writable = False

    def load_session(self) -> SessionSnapshot:
        """读取 port 绑定 session 的权威 history。"""
        session = self._store.load_session(self._run.session_id)
        self._session_revision = session.revision
        return session

    def reserve_model_step(self, cursor: RuntimeCursor) -> ModelStepReservation:
        """在 provider effect 前通过 aggregate 预留一步预算。"""
        self._require_writable()
        self._require_cursor(cursor)
        if self._reusable_model_reservation:
            self._reusable_model_reservation = False
            return ModelStepReservation(
                granted=True,
                step_index=cursor.step_index,
                cursor=cursor,
                remaining_deadline_seconds=self.remaining_deadline_seconds(),
            )
        previous_reserved = self._run.usage.model_steps_reserved
        commit = self._store.reserve_model_step(
            ReserveModelStep(
                run_id=self._run.run_id,
                expected_run_revision=self._run.revision,
                activation_id=self._activation_id,
                now=self._clock(),
            )
        )
        self._accept(commit)
        granted = (
            commit.run.phase is RunPhase.ACTIVE
            and commit.run.usage.model_steps_reserved == previous_reserved + 1
        )
        if not granted:
            self._writable = False
        return ModelStepReservation(
            granted=granted,
            step_index=cursor.step_index,
            cursor=cursor,
            remaining_deadline_seconds=self.remaining_deadline_seconds(),
        )

    def commit_model_step(self, commit: RuntimeModelStepCommit) -> RuntimeCursor:
        """原子提交 provider response、history、tool intents 与 checkpoint。"""
        self._require_writable()
        self._require_cursor(commit.cursor_before)
        now = self._clock()
        usage = self._model_usage(commit)
        checkpoint = self._next_checkpoint(
            cursor=commit.cursor_after,
            usage=usage,
            message_count=len(commit.message_delta),
            resumability=commit.resumability,
        )
        stored = self._store.commit_model_step(
            CommitModelStep(
                run_id=self._run.run_id,
                expected_run_revision=self._run.revision,
                activation_id=self._activation_id,
                expected_session_revision=self._session_revision,
                message_delta=list(commit.message_delta),
                usage=usage,
                prepared_tool_calls=self._new_prepared_records(
                    commit.prepared_tool_calls,
                    now=now,
                ),
                checkpoint=checkpoint,
                assistant_message=commit.assistant_message,
                now=now,
            )
        )
        self._accept(stored)
        return self._stored_cursor(commit.cursor_after)

    def claim_tool_call(self, call: RuntimeToolCall) -> ToolCallClaim:
        """验证 exact prepared subject 并在 effect 前 durable claim。"""
        self._require_writable()
        self._require_runtime_call(call)
        record = self._tool_record(call.tool_call_id)
        self._require_tool_subject(record, call, expected_version=call.tool_version)
        stored = self._store.claim_tool_call(
            ClaimToolCall(
                run_id=self._run.run_id,
                expected_run_revision=self._run.revision,
                activation_id=self._activation_id,
                tool_call_id=call.tool_call_id,
                fingerprint=call.fingerprint,
                expected_tool_version=call.tool_version,
                now=self._clock(),
            )
        )
        self._accept(stored)
        claimed = self._tool_record(call.tool_call_id)
        return ToolCallClaim(
            run_id=self._run.run_id,
            activation_id=self._activation_id,
            tool_call_id=claimed.tool_call_id,
            tool_name=claimed.tool_name,
            fingerprint=claimed.fingerprint,
            tool_version=claimed.version,
        )

    def commit_tool_result(self, commit: RuntimeToolResultCommit) -> RuntimeCursor:
        """提交 claimed/preflight 工具结果并推进 checkpoint cursor。"""
        self._require_writable()
        self._require_runtime_call(commit.tool_call)
        expected_version = (
            commit.claim.tool_version if commit.claim is not None else commit.tool_call.tool_version
        )
        record = self._tool_record(commit.tool_call.tool_call_id)
        self._require_tool_subject(
            record,
            commit.tool_call,
            expected_version=expected_version,
        )
        if commit.claim is not None and (
            commit.claim.run_id != commit.tool_call.run_id
            or commit.claim.activation_id != commit.tool_call.activation_id
            or commit.claim.tool_call_id != commit.tool_call.tool_call_id
            or commit.claim.tool_name != commit.tool_call.tool_name
            or commit.claim.fingerprint != commit.tool_call.fingerprint
        ):
            raise IrisRunConflictError("tool result claim 与 runtime tool call 不匹配")
        checkpoint = self._next_checkpoint(
            cursor=commit.cursor_after,
            usage=self._run.usage,
            message_count=len(commit.message_delta),
        )
        stored = self._store.commit_tool_result(
            CommitToolResult(
                run_id=self._run.run_id,
                expected_run_revision=self._run.revision,
                activation_id=self._activation_id,
                expected_session_revision=self._session_revision,
                tool_call_id=commit.tool_call.tool_call_id,
                expected_tool_version=expected_version,
                result=commit.result,
                message_delta=list(commit.message_delta),
                checkpoint=checkpoint,
                now=self._clock(),
            )
        )
        self._accept(stored)
        return self._stored_cursor(commit.cursor_after)

    def suspend(self, suspension: RuntimeSuspension) -> RuntimeSuspensionResult:
        """原子提交模型事实、pending interaction 与 waiting result。"""
        self._require_writable()
        self._require_cursor(suspension.cursor_before)
        now = self._clock()
        usage = self._model_usage(suspension)
        checkpoint = self._next_checkpoint(
            cursor=suspension.cursor,
            usage=usage,
            message_count=len(suspension.message_delta),
            resumability=suspension.resumability,
        )
        interaction = self._pending_interaction(
            suspension,
            checkpoint=checkpoint,
            now=now,
        )
        prepared_tool_calls = self._new_prepared_records(
            suspension.prepared_tool_calls,
            now=now,
        )
        prepared_tool_calls = [
            record.model_copy(update={"interaction_id": interaction.interaction_id})
            if record.tool_call_id == interaction.tool_call_id
            else record
            for record in prepared_tool_calls
        ]
        stored = self._store.suspend_run(
            SuspendRun(
                run_id=self._run.run_id,
                expected_run_revision=self._run.revision,
                activation_id=self._activation_id,
                expected_session_revision=self._session_revision,
                message_delta=list(suspension.message_delta),
                prepared_tool_calls=prepared_tool_calls,
                checkpoint=checkpoint,
                pending_interaction=interaction,
                assistant_message=suspension.assistant_message,
                usage=usage,
                now=now,
            )
        )
        self._accept(stored)
        self._writable = False
        if stored.interaction is None:
            raise IrisRunStateError("suspend commit 缺少 durable interaction")
        return RuntimeSuspensionResult(
            cursor=self._stored_cursor(suspension.cursor),
            interaction=stored.interaction,
        )

    def cancellation_requested(self) -> bool:
        """返回最近 committed run 上的 durable cancellation fact。"""
        current = self._store.load_run(self._run.run_id)
        if current is None:
            raise IrisRunNotFoundError("commit port 绑定的 run 不存在", run_id=self._run.run_id)
        return current.cancellation_requested_at is not None

    def remaining_deadline_seconds(self) -> float | None:
        """按 absolute deadline 与 injected clock 计算非负剩余秒数。"""
        deadline = self._run.options.limits.deadline_at
        if deadline is None:
            return None
        return max(0.0, (deadline - self._clock()).total_seconds())

    def _accept(self, commit: RunCommit) -> None:
        self._run = commit.run
        if commit.checkpoint is not None:
            self._checkpoint = commit.checkpoint
        if commit.session is not None:
            self._session_revision = commit.session.revision
        for event in commit.events:
            key = (event.run_id, event.sequence)
            if key not in self._event_keys:
                self._event_keys.add(key)
                self._event_sink.append(event)

    def _require_writable(self) -> None:
        if not self._writable:
            raise IrisRunStateError("activation commit port 已结算或撤销")
        current = self._store.load_run(self._run.run_id)
        if current is None:
            raise IrisRunNotFoundError("commit port 绑定的 run 不存在", run_id=self._run.run_id)
        if current.revision == self._run.revision:
            return
        expected = self._run.model_copy(
            update={
                "revision": current.revision,
                "cancellation_requested_at": current.cancellation_requested_at,
                "cancellation_reason": current.cancellation_reason,
                "last_event_sequence": current.last_event_sequence,
                "updated_at": current.updated_at,
            }
        )
        if (
            self._run.cancellation_requested_at is not None
            or current.cancellation_requested_at is None
            or current.revision != self._run.revision + 1
            or current.last_event_sequence != self._run.last_event_sequence + 1
            or current != expected
        ):
            raise IrisRunConflictError("activation 期间出现非 cancellation mutation")
        self._run = current
        self._event_keys.update((event.run_id, event.sequence) for event in self._event_sink)
        for event in self._store.list_events(
            current.run_id,
            self._run.last_event_sequence - 1,
        ):
            key = (event.run_id, event.sequence)
            if key not in self._event_keys:
                self._event_keys.add(key)
                self._event_sink.append(event)

    def _require_cursor(self, cursor: RuntimeCursor) -> None:
        if RuntimeCursor.model_validate(self._checkpoint.engine_cursor) != cursor:
            raise IrisRunConflictError("runtime cursor 与 durable checkpoint 不匹配")

    def _require_runtime_call(self, call: RuntimeToolCall) -> None:
        if call.run_id != self._run.run_id or call.activation_id != self._activation_id:
            raise IrisRunConflictError("runtime tool call 跨越 commit port identity")

    def _tool_record(self, tool_call_id: str) -> RunToolCallRecord:
        match = next(
            (
                record
                for record in self._store.list_tool_calls(self._run.run_id)
                if record.tool_call_id == tool_call_id
            ),
            None,
        )
        if match is None:
            raise IrisRunConflictError("durable prepared tool call 不存在")
        return match

    def _next_checkpoint(
        self,
        *,
        cursor: RuntimeCursor,
        usage: RunUsage,
        message_count: int,
        resumability: CheckpointResumability | None = None,
    ) -> RunCheckpoint:
        return RunCheckpoint(
            run_id=self._run.run_id,
            sequence=self._checkpoint.sequence + 1,
            activation_id=self._activation_id,
            engine_cursor=cursor.model_dump(mode="json"),
            session_revision=self._session_revision + (1 if message_count else 0),
            model_steps_reserved=usage.model_steps_reserved,
            model_steps_committed=usage.model_steps_committed,
            environment_fingerprint=self._run.environment_fingerprint,
            resumability=resumability or self._checkpoint.resumability,
        )

    def _model_usage(
        self,
        commit: RuntimeModelStepCommit | RuntimeSuspension,
    ) -> RunUsage:
        usage = self._run.usage
        commits_model_step = commit.cursor_before.position == "before_model"
        if not commits_model_step and (
            commit.message_delta
            or commit.input_tokens
            or commit.output_tokens
            or commit.total_tokens
        ):
            raise IrisRunStateError("existing tool batch suspension 不能重复提交 model facts")
        return RunUsage(
            model_steps_reserved=usage.model_steps_reserved,
            model_steps_committed=(
                usage.model_steps_committed + 1
                if commits_model_step
                else usage.model_steps_committed
            ),
            tool_calls_committed=usage.tool_calls_committed,
            input_tokens=usage.input_tokens + commit.input_tokens,
            output_tokens=usage.output_tokens + commit.output_tokens,
            total_tokens=usage.total_tokens + commit.total_tokens,
        )

    def _new_prepared_records(
        self,
        calls: tuple[RuntimeToolCall, ...],
        *,
        now: datetime,
    ) -> list[RunToolCallRecord]:
        """验证已持久化 subject，并只构造本次尚未存在的 prepared records。"""
        existing = {
            record.tool_call_id: record for record in self._store.list_tool_calls(self._run.run_id)
        }
        records: list[RunToolCallRecord] = []
        for call in calls:
            stored = existing.get(call.tool_call_id)
            if stored is None:
                records.append(self._prepared_record(call, now=now))
                continue
            self._require_tool_subject(stored, call, expected_version=call.tool_version)
            if stored.phase is not ToolCallPhase.PREPARED:
                raise IrisRunConflictError("existing tool batch subject 不再是 prepared")
        return records

    @staticmethod
    def _require_tool_subject(
        record: RunToolCallRecord,
        call: RuntimeToolCall,
        *,
        expected_version: int,
    ) -> None:
        if (
            record.step_index != call.step_index
            or record.ordinal != call.ordinal
            or record.tool_name != call.tool_name
            or record.arguments != call.arguments
            or record.fingerprint != call.fingerprint
            or record.version != expected_version
        ):
            raise IrisRunConflictError("runtime tool call 与 durable prepared subject 不匹配")

    def _prepared_record(
        self,
        call: RuntimeToolCall,
        *,
        now: datetime,
    ) -> RunToolCallRecord:
        if call.run_id != self._run.run_id:
            raise IrisRunConflictError("prepared tool call run identity 不匹配")
        return RunToolCallRecord(
            run_id=call.run_id,
            step_index=call.step_index,
            ordinal=call.ordinal,
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            arguments=call.arguments,
            fingerprint=call.fingerprint,
            interaction_id=call.interaction_id,
            phase=ToolCallPhase.PREPARED,
            version=call.tool_version,
            created_at=now,
            updated_at=now,
        )

    def _pending_interaction(
        self,
        suspension: RuntimeSuspension,
        *,
        checkpoint: RunCheckpoint,
        now: datetime,
    ) -> HumanInteraction:
        expires_at = suspension.expires_at
        if expires_at is None:
            timeout = self._run.options.limits.interaction_timeout_seconds
            if timeout is not None:
                expires_at = now + timedelta(seconds=timeout)
        run = snapshot_run(self._run).model_copy(update={"updated_at": now})
        return self._interaction_service.create_pending(
            suspension.interaction_request,
            run=run,
            step_index=suspension.cursor.step_index,
            expires_at=expires_at,
        )

    def _stored_cursor(self, expected: RuntimeCursor) -> RuntimeCursor:
        actual = RuntimeCursor.model_validate(self._checkpoint.engine_cursor)
        if actual != expected:
            raise IrisRunConflictError("store 返回的 checkpoint cursor 与 commit 不匹配")
        return actual


__all__ = ["StoreRuntimeCommitPort"]
