"""单 session 的 process-local input admission facade。"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from collections.abc import AsyncIterator, Iterable
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator, model_validator

from ..exceptions import IrisRunNotFoundError, IrisRunStateError
from ..hitl import HumanInteractionResponse
from ..lifecycle import AgentRunOptions, AgentRunRequest, RunEvent, RunPhase, RunResult, RunSnapshot
from ..message import Msg
from ..runtime import SteeringInput
from .runner import AgentRunner

logger = logging.getLogger(__name__)

type SubmissionMode = Literal["steer", "follow_up"]
type SubmissionFailureReason = Literal[
    "target_terminal",
    "target_cancelling",
    "session_closed",
    "commit_failed",
    "start_failed",
]


class SubmitReceipt(BaseModel):
    """一次普通输入 admission 的不可变即时回执。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    submission_id: str
    run_id: str
    mode: SubmissionMode | None
    state: Literal["pending", "delivered"]

    @field_validator("submission_id", "run_id")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{info.field_name} 不能为空")
        return normalized

    @model_validator(mode="after")
    def _validate_state(self) -> SubmitReceipt:
        if (self.mode is None, self.state) not in {
            (True, "delivered"),
            (False, "pending"),
        }:
            raise ValueError("idle receipt 必须 delivered，busy receipt 必须 pending")
        return self


class SubmissionEvent(BaseModel):
    """Busy submission 的 process-local 状态事件。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    submission_id: str
    run_id: str
    mode: SubmissionMode
    state: Literal["pending", "delivered", "failed"]
    reason: SubmissionFailureReason | None = None

    @field_validator("submission_id", "run_id")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{info.field_name} 不能为空")
        return normalized

    @model_validator(mode="after")
    def _validate_reason(self) -> SubmissionEvent:
        if (self.state == "failed") != (self.reason is not None):
            raise ValueError("仅 failed submission event 必须包含 reason")
        return self


type SessionEvent = RunEvent | SubmissionEvent


class _PendingInput(BaseModel):
    """尚未证明 durable delivery 的单条 transient input。"""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    submission_id: str
    input: str
    mode: SubmissionMode
    run_id: str
    options: AgentRunOptions | None = None

    @field_validator("submission_id", "input", "run_id")
    @classmethod
    def _validate_required_text(cls, value: str, info: ValidationInfo) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{info.field_name} 不能为空")
        return normalized

    @model_validator(mode="after")
    def _validate_options(self) -> _PendingInput:
        if self.mode == "steer" and self.options is not None:
            raise ValueError("steer input 不接受 run options")
        return self


class _PendingInputQueue:
    """按 mode 分离的两条 FIFO；eligibility 由调用方决定。"""

    def __init__(self) -> None:
        self._steer: deque[_PendingInput] = deque()
        self._follow_up: deque[_PendingInput] = deque()

    def admit_steer(self, item: _PendingInput) -> None:
        self._steer.append(item)

    def admit_follow_up(self, item: _PendingInput) -> None:
        self._follow_up.append(item)

    def claim_steer(self, run_id: str) -> _PendingInput | None:
        if not self._steer or self._steer[0].run_id != run_id:
            return None
        return self._steer.popleft()

    def pop_follow_up(self) -> _PendingInput | None:
        return self._follow_up.popleft() if self._follow_up else None

    def fail_target(self, run_id: str) -> tuple[_PendingInput, ...]:
        failed = tuple(item for item in self._steer if item.run_id == run_id)
        self._steer = deque(item for item in self._steer if item.run_id != run_id)
        return failed

    def take_follow_ups(self) -> tuple[_PendingInput, ...]:
        items = tuple(self._follow_up)
        self._follow_up.clear()
        return items

    def fail_all(self) -> tuple[_PendingInput, ...]:
        items = (*self._steer, *self._follow_up)
        self._steer.clear()
        self._follow_up.clear()
        return items


class _EventStreamEnd:
    """Mixed event stream 的 private sentinel 类型。"""


_EVENT_STREAM_END = _EventStreamEnd()


@dataclass(slots=True)
class _FollowUpAdmission:
    """已从 FIFO 取出、等待 create admission 结算的 follow-up。"""

    item: _PendingInput
    task: asyncio.Task[RunResult]
    started: asyncio.Event
    outcome: asyncio.Future[Exception | None]
    waiter: asyncio.Task[None] | None = None
    finalizer: asyncio.Task[None] | None = None


class _SessionSteeringPort:
    """把 runtime safe-boundary protocol 映射到 manager transient state。"""

    def __init__(self, manager: SessionManager) -> None:
        self._manager = manager

    async def claim(self, run_id: str, activation_id: str) -> SteeringInput | None:
        manager = self._manager
        async with manager._lock:
            if manager._closed or manager._current_run_id != run_id:
                return None
            try:
                snapshot = manager._runner.get_run(run_id)
            except IrisRunNotFoundError:
                return None
            if (
                snapshot.phase is not RunPhase.ACTIVE
                or snapshot.current_activation_id != activation_id
                or snapshot.cancellation_requested_at is not None
            ):
                return None
            item = manager._pending.claim_steer(run_id)
            if item is None:
                return None
            manager._claimed_steer[item.submission_id] = item
            return SteeringInput(
                submission_id=item.submission_id,
                message=Msg.user(
                    item.input,
                    metadata={"submission_id": item.submission_id, "mode": "steer"},
                ),
            )

    def acknowledge(self, submission_id: str) -> None:
        manager = self._manager
        item = manager._claimed_steer.pop(submission_id, None)
        if item is None:
            logger.error(
                "steering acknowledge identity 不存在",
                extra={"session_id": manager._session_id, "submission_id": submission_id},
            )
            return
        manager._emit_submission_event(item, "delivered")

    def fail(self, submission_id: str, reason: str) -> None:
        manager = self._manager
        item = manager._claimed_steer.pop(submission_id, None)
        if item is None:
            logger.error(
                "steering fail identity 不存在",
                extra={"session_id": manager._session_id, "submission_id": submission_id},
            )
            return
        if reason != "commit_failed":
            logger.error(
                "steering runtime 返回未知 failure reason",
                extra={"submission_id": submission_id, "reason": reason},
            )
        manager._emit_submission_event(item, "failed", reason="commit_failed")


class SessionManager:
    """绑定 exact runner 与单个 session 的 process-local admission owner。"""

    def __init__(self, runner: AgentRunner, session_id: str) -> None:
        normalized_session_id = session_id.strip()
        if not normalized_session_id:
            raise IrisRunStateError("session_id 不能为空")
        self._runner = runner
        self._session_id = normalized_session_id
        self._lock = asyncio.Lock()
        self._pending = _PendingInputQueue()
        self._claimed_steer: dict[str, _PendingInput] = {}
        self._current_run_id: str | None = None
        self._current_task: asyncio.Task[RunResult] | None = None
        self._closed = False
        self._event_queue: asyncio.Queue[SessionEvent | _EventStreamEnd] = asyncio.Queue()
        self._relayed_event_keys: set[tuple[str, int]] = set()
        self._event_consumer_started = False
        self._event_stream_closed = False
        self._steering = _SessionSteeringPort(self)
        self._follow_up_admissions: dict[str, _FollowUpAdmission] = {}

    async def submit(
        self,
        input: str,
        *,
        mode: SubmissionMode | None = None,
        options: AgentRunOptions | None = None,
    ) -> SubmitReceipt:
        """提交 idle input，或向 busy run admission 一条 steer/follow-up。"""
        normalized_input = input.strip()
        if not normalized_input:
            raise IrisRunStateError("input 不能为空")
        async with self._lock:
            self._require_open()
            await self._reconcile_locked()
            if self._current_run_id is None:
                if mode is not None:
                    raise IrisRunStateError("idle submit 的 mode 必须为 None")
                submission_id = self._new_submission_id()
                run_id = self._new_run_id()
                self._current_run_id = run_id
                task, started = self._create_start_task_locked(
                    input=normalized_input,
                    run_id=run_id,
                    options=options,
                    submission=None,
                )
                try:
                    await self._wait_for_admission(task, started)
                except Exception:
                    if self._current_task is task and self._current_run_id == run_id:
                        self._current_task = None
                        self._current_run_id = None
                    raise
                return SubmitReceipt(
                    submission_id=submission_id,
                    run_id=run_id,
                    mode=None,
                    state="delivered",
                )

            if mode not in ("steer", "follow_up"):
                raise IrisRunStateError("busy submit 必须显式提供 steer 或 follow_up mode")
            if mode == "steer" and options is not None:
                raise IrisRunStateError("steer mode 不接受 options")
            try:
                current = self._runner.get_run(self._current_run_id)
            except IrisRunNotFoundError as exc:
                raise IrisRunStateError("current run admission 尚未完成") from exc
            if mode == "steer" and current.cancellation_requested_at is not None:
                self._fail_items(
                    self._pending.fail_target(current.run_id),
                    reason="target_cancelling",
                )
                raise IrisRunStateError("cancelling run 不接受新的 steer input")
            item = _PendingInput(
                submission_id=self._new_submission_id(),
                input=normalized_input,
                mode=mode,
                run_id=current.run_id if mode == "steer" else self._new_run_id(),
                options=options,
            )
            if mode == "steer":
                self._pending.admit_steer(item)
            else:
                self._pending.admit_follow_up(item)
            self._emit_submission_event(item, "pending")
            return SubmitReceipt(
                submission_id=item.submission_id,
                run_id=item.run_id,
                mode=item.mode,
                state="pending",
            )

    async def resume(
        self,
        *,
        interaction_id: str,
        response: HumanInteractionResponse,
    ) -> RunResult:
        """恢复 façade 当前 exact waiting run，并返回 runner 的 durable result。"""
        async with self._lock:
            self._require_open()
            await self._reconcile_locked()
            run_id = self._require_current_run()
            snapshot = self._runner.get_run(run_id)
            if snapshot.phase is not RunPhase.WAITING:
                raise IrisRunStateError("current run 不处于 waiting phase", run_id=run_id)
            if snapshot.pending_interaction_id != interaction_id.strip():
                raise IrisRunStateError("interaction_id 不是 current waiting interaction")
            started = asyncio.Event()
            task = asyncio.create_task(
                self._runner._resume_managed(
                    run_id,
                    interaction_id=interaction_id,
                    response=response,
                    steering=self._steering,
                    durable_event_callback=self._relay_run_event,
                    activation_started=started,
                )
            )
            self._current_task = task
            self._attach_settlement_callback(task, run_id, submission=None)
            await self._wait_for_admission(task, started)
        return await asyncio.shield(task)

    async def interrupt(self, *, reason: str | None = None) -> RunSnapshot:
        """请求取消 façade 当前 exact run，并保留 follow-up 到真实 terminal。"""
        async with self._lock:
            self._require_open()
            await self._reconcile_locked()
            run_id = self._require_current_run()
            before = self._runner.get_run(run_id)
            snapshot = self._runner.request_cancel(run_id, reason=reason)
            for event in self._runner.list_events(run_id, before.last_event_sequence):
                self._relay_run_event(event)
            self._fail_items(
                self._pending.fail_target(run_id),
                reason="target_cancelling",
            )
            if snapshot.phase is RunPhase.TERMINAL:
                await self._handle_terminal_locked(run_id)
            return snapshot

    def events(self) -> AsyncIterator[SessionEvent]:
        """返回唯一 mixed event consumer；不回放 manager 创建前的 durable events。"""
        if self._event_consumer_started:
            raise IrisRunStateError("SessionManager events 只允许一个 consumer")
        self._event_consumer_started = True
        return self._iterate_events()

    async def close(self) -> None:
        """关闭 façade admission/event stream，不取消或等待 durable run。"""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            assert not self._claimed_steer, "claim 到 callback 之间不得出现 await"
            for admission in tuple(self._follow_up_admissions.values()):
                try:
                    self._runner.get_run(admission.item.run_id)
                except IrisRunNotFoundError:
                    self._follow_up_admissions.pop(admission.item.submission_id, None)
                    self._emit_submission_event(
                        admission.item,
                        "failed",
                        reason="session_closed",
                    )
                    admission.task.cancel()
                    self._cancel_follow_up_helpers(admission)
                else:
                    self._complete_follow_up_success_locked(admission)
            self._fail_items(self._pending.fail_all(), reason="session_closed")
            self._current_run_id = None
            self._current_task = None
            self._event_stream_closed = True
            self._event_queue.put_nowait(_EVENT_STREAM_END)

    async def _iterate_events(self) -> AsyncIterator[SessionEvent]:
        while True:
            event = await self._event_queue.get()
            if isinstance(event, _EventStreamEnd):
                return
            yield event

    def _create_start_task_locked(
        self,
        *,
        input: str,
        run_id: str,
        options: AgentRunOptions | None,
        submission: _PendingInput | None,
    ) -> tuple[asyncio.Task[RunResult], asyncio.Event]:
        started = asyncio.Event()
        task = asyncio.create_task(
            self._runner._start_managed(
                AgentRunRequest(input=input, session_id=self._session_id, run_id=run_id),
                options=options,
                steering=self._steering,
                durable_event_callback=self._relay_run_event,
                activation_started=started,
            )
        )
        self._current_task = task
        self._attach_settlement_callback(task, run_id, submission=submission)
        return task, started

    def _attach_settlement_callback(
        self,
        task: asyncio.Task[RunResult],
        run_id: str,
        *,
        submission: _PendingInput | None,
    ) -> None:
        def schedule(completed: asyncio.Task[RunResult]) -> None:
            asyncio.create_task(
                self._settle_managed_task(completed, run_id, submission=submission)
            )

        task.add_done_callback(schedule)

    async def _wait_for_admission(
        self,
        task: asyncio.Task[RunResult],
        started: asyncio.Event,
    ) -> None:
        signal_waiter = asyncio.create_task(started.wait())
        try:
            await asyncio.wait(
                {task, signal_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if started.is_set():
                return
            task.result()
        finally:
            signal_waiter.cancel()
            with suppress(asyncio.CancelledError):
                await signal_waiter

    async def _settle_managed_task(
        self,
        task: asyncio.Task[RunResult],
        run_id: str,
        *,
        submission: _PendingInput | None,
    ) -> None:
        task_error = None if task.cancelled() else task.exception()
        try:
            async with self._lock:
                if self._current_task is not task or self._current_run_id != run_id:
                    return
                self._current_task = None
                try:
                    snapshot = self._runner.get_run(run_id)
                except IrisRunNotFoundError:
                    self._current_run_id = None
                    if submission is not None:
                        admission = self._follow_up_admissions.get(submission.submission_id)
                        if admission is not None:
                            if not admission.outcome.done():
                                admission.outcome.set_result(
                                    task_error or IrisRunStateError("follow-up create 未形成 run")
                                )
                            self._complete_follow_up_failure_locked(admission)
                    return
                if submission is not None:
                    admission = self._follow_up_admissions.get(submission.submission_id)
                    if admission is not None:
                        if not admission.outcome.done():
                            admission.outcome.set_result(None)
                        self._complete_follow_up_success_locked(admission)
                if snapshot.phase is RunPhase.TERMINAL:
                    await self._handle_terminal_locked(run_id)
                # waiting 保留 current owner；active task error 也不自动 recover/drain。
        except Exception:
            logger.exception(
                "SessionManager managed task settlement 失败",
                extra={"session_id": self._session_id, "run_id": run_id},
            )

    async def _reconcile_locked(self) -> None:
        while self._current_run_id is not None:
            run_id = self._current_run_id
            try:
                snapshot = self._runner.get_run(run_id)
            except IrisRunNotFoundError:
                task = self._current_task
                if task is not None and not task.done():
                    return
                self._current_task = None
                self._current_run_id = None
                return
            if snapshot.phase is RunPhase.TERMINAL:
                await self._handle_terminal_locked(run_id)
                continue
            task = self._current_task
            if task is not None and task.done():
                self._current_task = None
            return

    async def _handle_terminal_locked(self, run_id: str) -> None:
        if self._current_run_id != run_id:
            return
        self._fail_items(self._pending.fail_target(run_id), reason="target_terminal")
        self._current_run_id = None
        self._current_task = None
        if not self._closed:
            await self._start_next_follow_up_locked()

    async def _start_next_follow_up_locked(self) -> None:
        item = self._pending.pop_follow_up()
        if item is None:
            return
        self._current_run_id = item.run_id
        task, started = self._create_start_task_locked(
            input=item.input,
            run_id=item.run_id,
            options=item.options,
            submission=item,
        )
        admission = _FollowUpAdmission(
            item=item,
            task=task,
            started=started,
            outcome=asyncio.get_running_loop().create_future(),
        )
        self._follow_up_admissions[item.submission_id] = admission
        admission.waiter = asyncio.create_task(self._resolve_follow_up_admission(admission))
        admission.finalizer = asyncio.create_task(self._finalize_follow_up_admission(admission))
        try:
            await asyncio.shield(admission.outcome)
        except asyncio.CancelledError:
            raise
        self._apply_follow_up_outcome_locked(admission)

    async def _resolve_follow_up_admission(self, admission: _FollowUpAdmission) -> None:
        try:
            await self._wait_for_admission(admission.task, admission.started)
        except asyncio.CancelledError:
            if not admission.outcome.done():
                admission.outcome.cancel()
            return
        except Exception as exc:
            outcome: Exception | None = exc
        else:
            outcome = None
        if not admission.outcome.done():
            admission.outcome.set_result(outcome)

    async def _finalize_follow_up_admission(self, admission: _FollowUpAdmission) -> None:
        try:
            await asyncio.shield(admission.outcome)
        except asyncio.CancelledError:
            return
        async with self._lock:
            self._apply_follow_up_outcome_locked(admission)

    def _apply_follow_up_outcome_locked(self, admission: _FollowUpAdmission) -> None:
        if self._follow_up_admissions.get(admission.item.submission_id) is not admission:
            return
        if admission.outcome.result() is None:
            self._complete_follow_up_success_locked(admission)
            return
        self._complete_follow_up_failure_locked(admission)

    def _complete_follow_up_success_locked(self, admission: _FollowUpAdmission) -> None:
        if self._follow_up_admissions.pop(admission.item.submission_id, None) is not admission:
            return
        self._cancel_follow_up_helpers(admission)
        self._emit_submission_event(admission.item, "delivered")

    def _complete_follow_up_failure_locked(self, admission: _FollowUpAdmission) -> None:
        if self._follow_up_admissions.pop(admission.item.submission_id, None) is not admission:
            return
        self._cancel_follow_up_helpers(admission)
        if self._current_task is admission.task and self._current_run_id == admission.item.run_id:
            self._current_task = None
            self._current_run_id = None
        self._emit_submission_event(admission.item, "failed", reason="start_failed")
        self._fail_items(self._pending.take_follow_ups(), reason="start_failed")

    @staticmethod
    def _cancel_follow_up_helpers(admission: _FollowUpAdmission) -> None:
        current = asyncio.current_task()
        for helper in (admission.waiter, admission.finalizer):
            if helper is not None and helper is not current and not helper.done():
                helper.cancel()

    def _fail_items(
        self,
        items: Iterable[_PendingInput],
        *,
        reason: SubmissionFailureReason,
    ) -> None:
        for item in items:
            self._emit_submission_event(item, "failed", reason=reason)

    def _relay_run_event(self, event: RunEvent) -> None:
        if self._event_stream_closed:
            return
        key = (event.run_id, event.sequence)
        if key in self._relayed_event_keys:
            return
        self._relayed_event_keys.add(key)
        self._event_queue.put_nowait(event)

    def _emit_submission_event(
        self,
        item: _PendingInput,
        state: Literal["pending", "delivered", "failed"],
        *,
        reason: SubmissionFailureReason | None = None,
    ) -> None:
        if self._event_stream_closed:
            return
        self._event_queue.put_nowait(
            SubmissionEvent(
                submission_id=item.submission_id,
                run_id=item.run_id,
                mode=item.mode,
                state=state,
                reason=reason,
            )
        )

    def _require_open(self) -> None:
        if self._closed:
            raise IrisRunStateError("SessionManager is closed")

    def _require_current_run(self) -> str:
        if self._current_run_id is None:
            raise IrisRunStateError("SessionManager 没有 current run")
        return self._current_run_id

    @staticmethod
    def _new_submission_id() -> str:
        return f"sub_{uuid.uuid4().hex}"

    @staticmethod
    def _new_run_id() -> str:
        return f"run_{uuid.uuid4().hex}"


__all__ = ["SessionEvent", "SessionManager", "SubmissionEvent", "SubmitReceipt"]
