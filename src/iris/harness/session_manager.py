"""单 session 的 process-local input admission facade。

``SessionManager`` 只组合一个 exact ``AgentRunner`` 与一个 session id。
它不接管任何 durable ownership：run、history、checkpoint、interaction、cancellation、result
与 ``RunEvent`` 始终由 runner/store 权威负责。
队列、receipt 状态、submission 事件、steer claim 与 durable event 水位都只存在于当前进程，
新建 manager 不扫描也不接管既有 active/waiting lane。

Example:
    manager = SessionManager(runner, "default")
    receipt = await manager.submit("先分析现状")
    queued = await manager.submit("把重点改为并发边界", mode="steer")
    await manager.close()
"""

# region imports
from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator, model_validator

from ..exceptions import IrisRunNotFoundError, IrisRunStateError
from ..hitl import HumanInteractionResponse
from ..lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    RunEvent,
    RunEventKind,
    RunPhase,
    RunResult,
    RunSnapshot,
)
from ..message import Msg
from ..runtime import SteeringInput
from .runner import AgentRunner

# endregion

logger = logging.getLogger(__name__)

# ==========================================
#              Type Aliases
# ==========================================
# region aliases
type SubmissionMode = Literal["steer", "follow_up"]
_DEFAULT_MAX_PENDING_STEER = 64
_DEFAULT_MAX_PENDING_FOLLOW_UP = 64
_DEFAULT_MAX_BUFFERED_SUBMISSION_EVENTS = 256
_DEFAULT_MAX_TRACKED_DURABLE_RUNS = 64
_DURABLE_REPLAY_BATCH_SIZE = 64
# steer 只在 runtime 安全边界失败时报 commit_failed；follow_up 只在 create admission 失败时报
# start_failed。其余三种由 manager 侧的 target/session 状态变化产生。
type SubmissionFailureReason = Literal[
    "target_terminal",
    "target_cancelling",
    "session_closed",
    "commit_failed",
    "start_failed",
]
# endregion


def _require_positive_capacity(value: int, *, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} 必须是正整数")


class SubmitReceipt(BaseModel):
    """一次普通输入 admission 的不可变即时回执。

    只表达"输入是否已被接纳"，不表达 run 结果。idle submit 在 run create 已 durable commit
    后返回 ``delivered``；busy submit 一律返回 ``pending``，最终 delivery/failure 只通过
    ``SessionManager.events()`` 报告。

    Attributes:
        submission_id (str): 该次提交的 process-local 标识。
        run_id (str): 绑定的 run id；follow-up 是预生成的 future run id。
        mode (SubmissionMode | None): busy 模式；None 表示 idle submit。
        state (Literal["pending", "delivered"]): 接纳状态。
    """

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
        # mode 与 state 是绑定的：idle submit 已经完成 create，busy submit 一定还在排队。
        if (self.mode is None, self.state) not in {
            (True, "delivered"),
            (False, "pending"),
        }:
            raise ValueError("idle receipt 必须 delivered，busy receipt 必须 pending")
        return self


class SubmissionEvent(BaseModel):
    """Busy submission 的 process-local 状态事件。

    与 durable ``RunEvent`` 混在同一条 stream 中投递，但本身不持久化，也不参与 run 的
    sequence 编号。

    Attributes:
        submission_id (str): 对应 ``SubmitReceipt.submission_id``。
        run_id (str): 绑定的 run id。
        mode (SubmissionMode): busy 模式。
        state (Literal["pending", "delivered", "failed"]): 该 submission 的最新状态。
        reason (SubmissionFailureReason | None): 失败原因；仅 failed 状态携带。
    """

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


# 单消费者 stream 原样混合 durable run event 与 transient submission event，不合并 sequence。
type SessionEvent = RunEvent | SubmissionEvent


class _EventPageReader(Protocol):
    """按 durable sequence 游标读取有限 event page。"""

    def __call__(
        self,
        run_id: str,
        after_sequence: int = 0,
        *,
        limit: int | None = None,
    ) -> list[RunEvent]: ...


@dataclass(frozen=True, slots=True)
class _PendingInput:
    """尚未证明 durable delivery 的单条 transient input。

    Attributes:
        submission_id (str): 该次提交的 process-local 标识。
        input (str): 已 strip 的用户输入。
        mode (SubmissionMode): 决定它进入哪一条 FIFO。
        run_id (str): steer 绑定 exact current run；follow-up 是预生成的 future run id。
        options (AgentRunOptions | None): 仅 follow-up 可携带，用于其未来的 run create。
    """

    submission_id: str
    input: str
    mode: SubmissionMode
    run_id: str
    options: AgentRunOptions | None = None


class _PendingInputQueue:
    """按 mode 分离的两条 FIFO；eligibility 由调用方决定。

    两条队列各自保持 FIFO，但推进条件不同：steer 需要 exact current run 仍可接收，follow-up
    需要 current run 已 terminal。分开存放使较早的 follow-up 不会阻塞仍能进入当前 run 的
    steer。

    Attributes:
        _steer (deque[_PendingInput]): steer FIFO。
        _follow_up (deque[_PendingInput]): follow-up FIFO。
    """

    def __init__(self, *, max_steer: int, max_follow_up: int) -> None:
        _require_positive_capacity(max_steer, name="max_pending_steer")
        _require_positive_capacity(max_follow_up, name="max_pending_follow_up")
        self._max_steer = max_steer
        self._max_follow_up = max_follow_up
        self._steer: deque[_PendingInput] = deque()
        self._follow_up: deque[_PendingInput] = deque()

    @property
    def steer_count(self) -> int:
        """当前 steer FIFO 长度。"""
        return len(self._steer)

    @property
    def follow_up_count(self) -> int:
        """当前 follow-up FIFO 长度。"""
        return len(self._follow_up)

    def can_accept(self, mode: SubmissionMode) -> bool:
        """只读判断对应 FIFO 是否仍有一个槽位。"""
        if mode == "steer":
            return len(self._steer) < self._max_steer
        return len(self._follow_up) < self._max_follow_up

    def enqueue(self, item: _PendingInput) -> None:
        """把 input 追加到对应 FIFO；满载时拒绝且不修改队列。"""
        if not self.can_accept(item.mode):
            raise IrisRunStateError(f"{item.mode} input 队列容量已满")
        target = self._steer if item.mode == "steer" else self._follow_up
        target.append(item)

    def claim_steer(self, run_id: str) -> _PendingInput | None:
        """取出队首 steer，且仅当它绑定的正是 ``run_id`` 时。

        队首绑定其它 run 时直接返回 None 而不是跳过它，以保持 steer 的严格 FIFO 语义。
        """
        if not self._steer or self._steer[0].run_id != run_id:
            return None
        return self._steer.popleft()

    def pop_follow_up(self) -> _PendingInput | None:
        """取出队首 follow-up；队列为空时返回 None。"""
        return self._follow_up.popleft() if self._follow_up else None

    def peek_follow_up(self) -> _PendingInput | None:
        """只读返回队首 follow-up；队列为空时返回 None。"""
        return self._follow_up[0] if self._follow_up else None

    def drain_steers_for_run(self, run_id: str) -> tuple[_PendingInput, ...]:
        """移除并返回绑定指定 run 的全部 steer input。

        Returns:
            tuple[_PendingInput, ...]: 被移除的 steer input，供调用方发出 failed 事件。
        """
        items = tuple(item for item in self._steer if item.run_id == run_id)
        self._steer = deque(item for item in self._steer if item.run_id != run_id)
        return items

    def drain_follow_ups(self) -> tuple[_PendingInput, ...]:
        """清空并返回全部 follow-up input。"""
        items = tuple(self._follow_up)
        self._follow_up.clear()
        return items

    def drain_all_pending(self) -> tuple[_PendingInput, ...]:
        """清空两条队列并返回全部 input，用于 session 关闭。"""
        items = (*self._steer, *self._follow_up)
        self._steer.clear()
        self._follow_up.clear()
        return items


@dataclass(slots=True)
class _DurableRunTracker:
    """一个 run 的 durable relay 与 consumer 交付水位。"""

    observed_highest_sequence: int
    delivered_highest_sequence: int
    settled: bool = False


@dataclass(frozen=True, slots=True)
class _BufferedSubmissionEvent:
    """Transient event 及其入队前必须先交付的 durable 水位。"""

    event: SubmissionEvent
    durable_barriers: tuple[tuple[str, int], ...]


class _SessionEventBuffer:
    """有界保存 transient events，并从 store 按水位补读 durable events。"""

    def __init__(
        self,
        list_events: _EventPageReader,
        *,
        max_buffered_submission_events: int,
        max_tracked_durable_runs: int,
        on_tracker_released: Callable[[], None],
    ) -> None:
        _require_positive_capacity(
            max_buffered_submission_events,
            name="max_buffered_submission_events",
        )
        _require_positive_capacity(max_tracked_durable_runs, name="max_tracked_durable_runs")
        self._list_events = list_events
        self._max_buffered_submission_events = max_buffered_submission_events
        self._max_tracked_durable_runs = max_tracked_durable_runs
        self._on_tracker_released = on_tracker_released
        self._submission_events: deque[_BufferedSubmissionEvent] = deque()
        self._reserved_submission_ids: set[str] = set()
        self._run_trackers: OrderedDict[str, _DurableRunTracker] = OrderedDict()
        self._replayed_events: deque[RunEvent] = deque()
        self._wakeup = asyncio.Event()
        self._closed = False

    @property
    def buffered_submission_event_count(self) -> int:
        return len(self._submission_events)

    @property
    def reserved_terminal_slots(self) -> int:
        return len(self._reserved_submission_ids)

    @property
    def tracked_run_count(self) -> int:
        return len(self._run_trackers)

    @property
    def durable_wakeup_pending(self) -> bool:
        return self._wakeup.is_set()

    @property
    def durable_replay_batch_count(self) -> int:
        return len(self._replayed_events)

    def can_reserve_submission_lifecycle(self) -> bool:
        """判断 pending event 与其 terminal reservation 是否同时有空间。"""
        used = len(self._submission_events) + len(self._reserved_submission_ids)
        return used + 2 <= self._max_buffered_submission_events

    def add_pending(self, event: SubmissionEvent) -> None:
        """加入 pending event，并为 exact submission 保留一个 terminal 槽位。"""
        if not self.can_reserve_submission_lifecycle():
            raise IrisRunStateError("submission event buffer 容量已满")
        barriers = tuple(
            (run_id, tracker.observed_highest_sequence)
            for run_id, tracker in self._run_trackers.items()
            if tracker.observed_highest_sequence > tracker.delivered_highest_sequence
        )
        self._submission_events.append(_BufferedSubmissionEvent(event, barriers))
        self._reserved_submission_ids.add(event.submission_id)
        self._wakeup.set()

    def add_terminal(self, event: SubmissionEvent) -> None:
        """消费 exact reservation 后加入 delivered/failed event。"""
        if event.submission_id not in self._reserved_submission_ids:
            raise RuntimeError("submission terminal event 缺少 reservation")
        self._reserved_submission_ids.remove(event.submission_id)
        barriers = tuple(
            (run_id, tracker.observed_highest_sequence)
            for run_id, tracker in self._run_trackers.items()
            if tracker.observed_highest_sequence > tracker.delivered_highest_sequence
        )
        self._submission_events.append(_BufferedSubmissionEvent(event, barriers))
        self._wakeup.set()

    def can_register_run(self, run_id: str) -> bool:
        return (
            run_id in self._run_trackers or len(self._run_trackers) < self._max_tracked_durable_runs
        )

    def register_run(self, run_id: str, *, after_sequence: int) -> None:
        """在启动 managed source 前登记不回放旧事件的 baseline。"""
        if after_sequence < 0:
            raise ValueError("after_sequence 不能为负数")
        if run_id in self._run_trackers:
            return
        if not self.can_register_run(run_id):
            raise IrisRunStateError("durable run tracker 容量已满")
        self._run_trackers[run_id] = _DurableRunTracker(
            observed_highest_sequence=after_sequence,
            delivered_highest_sequence=after_sequence,
        )

    def discard_run(self, run_id: str) -> None:
        """移除未形成 durable run 的 tracker。"""
        self._replayed_events = deque(
            event for event in self._replayed_events if event.run_id != run_id
        )
        if self._run_trackers.pop(run_id, None) is not None:
            self._on_tracker_released()
        self._wakeup.set()

    def observe_run_event(self, event: RunEvent) -> None:
        """合并 committed event 水位；durable payload 始终留在 store。"""
        if self._closed:
            return
        tracker = self._run_trackers.get(event.run_id)
        if tracker is None:
            return
        tracker.observed_highest_sequence = max(
            tracker.observed_highest_sequence,
            event.sequence,
        )
        if event.kind is RunEventKind.RUN_TERMINAL:
            tracker.settled = True
        self._cleanup_run_if_caught_up(event.run_id, tracker)
        self._wakeup.set()

    def mark_run_settled(self, run_id: str) -> None:
        """标记 managed run 已无后续 callback，并在 consumer 追平后回收。"""
        tracker = self._run_trackers.get(run_id)
        if tracker is None:
            return
        tracker.settled = True
        self._cleanup_run_if_caught_up(run_id, tracker)
        self._wakeup.set()

    async def next_event(self) -> SessionEvent | None:
        """返回下一条 mixed event；closed 且保证状态耗尽时返回 None。"""
        while True:
            event = self._take_ready_event()
            if event is not None:
                return event
            if self._closed and not self._submission_events and not self._has_undelivered_event():
                return None
            self._wakeup.clear()
            await self._wakeup.wait()

    def close(self) -> None:
        """停止接受 relay，并在已观察事实交付后结束 consumer。"""
        if self._closed:
            return
        self._closed = True
        for run_id, tracker in tuple(self._run_trackers.items()):
            tracker.settled = True
            self._cleanup_run_if_caught_up(run_id, tracker)
        self._wakeup.set()

    def _take_ready_event(self) -> SessionEvent | None:
        buffered = self._submission_events[0] if self._submission_events else None
        if buffered is not None and self._barriers_satisfied(buffered.durable_barriers):
            return self._submission_events.popleft().event
        if self._replayed_events:
            return self._pop_replayed_event()

        barrier_limits = dict(buffered.durable_barriers) if buffered is not None else None
        for run_id, tracker in tuple(self._run_trackers.items()):
            target = tracker.observed_highest_sequence
            if barrier_limits is not None:
                target = min(target, barrier_limits.get(run_id, tracker.delivered_highest_sequence))
            if tracker.delivered_highest_sequence >= target:
                continue
            page_limit = min(
                _DURABLE_REPLAY_BATCH_SIZE,
                target - tracker.delivered_highest_sequence,
            )
            rows = self._list_events(
                run_id,
                tracker.delivered_highest_sequence,
                limit=page_limit,
            )
            if len(rows) > page_limit:
                raise IrisRunStateError("store 返回的 durable event 批次超过请求上限")
            replayed = sorted(
                {
                    event.sequence: event
                    for event in rows
                    if tracker.delivered_highest_sequence < event.sequence <= target
                }.values(),
                key=lambda event: event.sequence,
            )
            if not replayed:
                raise IrisRunStateError("durable event watermark 无法从 store 补齐", run_id=run_id)
            self._replayed_events.extend(replayed)
            return self._pop_replayed_event()
        return None

    def _pop_replayed_event(self) -> RunEvent:
        event = self._replayed_events.popleft()
        tracker = self._run_trackers[event.run_id]
        tracker.delivered_highest_sequence = event.sequence
        self._cleanup_run_if_caught_up(event.run_id, tracker)
        return event

    def _barriers_satisfied(self, barriers: tuple[tuple[str, int], ...]) -> bool:
        return all(
            (tracker := self._run_trackers.get(run_id)) is None
            or tracker.delivered_highest_sequence >= sequence
            for run_id, sequence in barriers
        )

    def _has_undelivered_event(self) -> bool:
        return any(
            tracker.delivered_highest_sequence < tracker.observed_highest_sequence
            for tracker in self._run_trackers.values()
        )

    def _cleanup_run_if_caught_up(
        self,
        run_id: str,
        tracker: _DurableRunTracker,
    ) -> None:
        if not tracker.settled:
            return
        if tracker.delivered_highest_sequence < tracker.observed_highest_sequence:
            return
        if self._run_trackers.pop(run_id, None) is tracker:
            self._on_tracker_released()


@dataclass(slots=True)
class _FollowUpAdmission:
    """已从 FIFO 取出、等待 create admission 结算的 follow-up。

    follow-up 的 create 可能失败，而结算既可能由 admission waiter 先观察到，也可能由 run
    settlement callback 先观察到；``outcome`` 作为唯一汇合点，让两条路径只结算一次。

    Attributes:
        item (_PendingInput): 对应的 pending input。
        task (asyncio.Task[RunResult]): 正在执行的 managed start task。
        started (asyncio.Event): runner 的 admission signal。
        outcome (asyncio.Future[Exception | None]): 结算结果，None 表示 create 成功。
        waiter (asyncio.Task[None] | None): 观察 admission 并填充 ``outcome`` 的 helper。
        finalizer (asyncio.Task[None] | None): 在锁外兜底应用 ``outcome`` 的 helper。
    """

    item: _PendingInput
    task: asyncio.Task[RunResult]
    started: asyncio.Event
    outcome: asyncio.Future[Exception | None]
    waiter: asyncio.Task[None] | None = None
    finalizer: asyncio.Task[None] | None = None


class _SessionSteeringPort:
    """把 runtime safe-boundary protocol 映射到 manager transient state。

    runtime 只在安全边界调用 ``claim()``，随后按写入 durable session history 的结果回调
    ``acknowledge()`` 或 ``fail()``。claim 到回调之间不允许出现 await，因此 claim 中的 item
    暂存在 ``SessionManager._claimed_steer``，由回调负责摘除。

    Attributes:
        _manager (SessionManager): 被映射的 manager，访问其锁与 transient 队列。
    """

    def __init__(self, manager: SessionManager) -> None:
        self._manager = manager

    async def claim(self, run_id: str, activation_id: str) -> SteeringInput | None:
        """在安全边界为 exact activation 取出至多一条 steer input。

        Args:
            run_id (str): 请求 steer 的 run id。
            activation_id (str): 请求 steer 的 activation id，用作 fence。

        Returns:
            SteeringInput | None: 待注入的用户消息；session 已关闭、run 身份不符、run 不再
                active、已请求取消或队首不匹配时返回 None。
        """
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
            # 已离开队列但尚未证明 durable delivery，暂存等待 acknowledge/fail 回调结算。
            manager._claimed_steer[item.submission_id] = item
            return SteeringInput.model_construct(
                submission_id=item.submission_id,
                message=Msg.user(
                    item.input,
                    metadata={"submission_id": item.submission_id, "mode": "steer"},
                ),
            )

    def acknowledge(self, submission_id: str) -> None:
        """确认 steer input 已写入 durable session history。

        Args:
            submission_id (str): 之前 claim 返回的 submission id。

        Notes:
            identity 不存在说明 claim/回调配对被破坏，只记录日志：此时既无法判定投递结果，
            也不应伪造 submission event。
        """
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
        """报告 steer input 未能写入 durable session history。

        Args:
            submission_id (str): 之前 claim 返回的 submission id。
            reason (str): runtime 给出的失败原因。

        Notes:
            对外只暴露 ``commit_failed`` 这一种 steer 失败原因；收到其它取值时额外记录日志，
            但仍按 ``commit_failed`` 上报，避免把内部字符串泄漏成新的公开状态。
        """
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
    """绑定 exact runner 与单个 session 的 process-local admission owner。

    面向"当前 run 执行期间还要接收新普通输入"的 host：idle 时直接创建 run，busy 时按
    ``steer`` / ``follow_up`` 入队。所有状态由单个 ``asyncio.Lock`` 串行化，durable 事实一律
    委托给 runner。

    Attributes:
        _runner (AgentRunner): 唯一 durable owner。
        _session_id (str): 绑定的 session id。
        _lock (asyncio.Lock): 串行化全部 transient 状态变更。
        _pending (_PendingInputQueue): steer / follow-up 两条 FIFO。
        _claimed_steer (dict[str, _PendingInput]): 已 claim 但未结算的 steer input。
        _current_run_id (str | None): facade 认定的当前 run；None 表示 idle。
        _current_task (asyncio.Task[RunResult] | None): 推进当前 run 的 managed task。
        _closed (bool): facade 是否已关闭。
        _event_buffer (_SessionEventBuffer): 有界 transient buffer 与 durable replay 水位。
        _event_consumer_started (bool): ``events()`` 是否已被取用。
        _steering (_SessionSteeringPort): 注入 runner 的安全边界 steering port。
        _follow_up_admissions (dict[str, _FollowUpAdmission]): 等待 create 结算的 follow-up。

    Example:
        manager = SessionManager(runner, "default")
        await manager.submit("先分析现状")
        await manager.close()
    """

    # ==========================================
    #               Initialization
    # ==========================================
    # region
    def __init__(
        self,
        runner: AgentRunner,
        session_id: str,
        *,
        max_pending_steer: int = _DEFAULT_MAX_PENDING_STEER,
        max_pending_follow_up: int = _DEFAULT_MAX_PENDING_FOLLOW_UP,
        max_buffered_submission_events: int = _DEFAULT_MAX_BUFFERED_SUBMISSION_EVENTS,
        max_tracked_durable_runs: int = _DEFAULT_MAX_TRACKED_DURABLE_RUNS,
    ) -> None:
        """绑定 runner 与 session id，初始化全部 process-local 状态。"""
        normalized_session_id = session_id.strip()
        if not normalized_session_id:
            raise IrisRunStateError("session_id 不能为空")
        self._runner = runner
        self._session_id = normalized_session_id
        self._lock = asyncio.Lock()
        self._pending = _PendingInputQueue(
            max_steer=max_pending_steer,
            max_follow_up=max_pending_follow_up,
        )
        self._claimed_steer: dict[str, _PendingInput] = {}
        self._current_run_id: str | None = None
        self._current_task: asyncio.Task[RunResult] | None = None
        self._closed = False
        self._event_consumer_started = False
        self._steering = _SessionSteeringPort(self)
        self._follow_up_admissions: dict[str, _FollowUpAdmission] = {}
        self._tracker_reconcile_task: asyncio.Task[None] | None = None
        self._event_buffer = _SessionEventBuffer(
            runner.list_events,
            max_buffered_submission_events=max_buffered_submission_events,
            max_tracked_durable_runs=max_tracked_durable_runs,
            on_tracker_released=self._schedule_tracker_reconcile,
        )

    # endregion

    # ==========================================
    #               Input Admission
    # ==========================================
    # region
    async def submit(
        self,
        input: str,
        *,
        mode: SubmissionMode | None = None,
        options: AgentRunOptions | None = None,
    ) -> SubmitReceipt:
        """提交 idle input，或向 busy run admission 一条 steer/follow-up。

        Args:
            input (str): 用户输入，不能为空白。
            mode (SubmissionMode | None): idle 时必须为 None；busy 时必须显式给出
                ``steer`` 或 ``follow_up``。
            options (AgentRunOptions | None): 新 run 的限额与 runtime 选项；``steer`` 不接受。

        Returns:
            SubmitReceipt: idle submit 返回 ``delivered``；busy submit 返回 ``pending``，最终
                结果只通过 ``events()`` 报告。

        Raises:
            IrisRunStateError: 当输入为空、mode 与 idle/busy 状态不符、steer 携带 options、
                current run admission 尚未完成，或 run 已请求取消而不再接收 steer 时。
        """
        # --- 1. 规范化输入并同步 facade 状态 ---
        normalized_input = input.strip()
        if not normalized_input:
            raise IrisRunStateError("input 不能为空")
        async with self._lock:
            self._require_open()
            await self._reconcile_locked()

            # --- 2. idle：直接创建并等待 run admission ---
            if self._current_run_id is None:
                if mode is not None:
                    raise IrisRunStateError("idle submit 的 mode 必须为 None")
                submission_id = self._new_submission_id()
                run_id = self._new_run_id()
                self._event_buffer.register_run(run_id, after_sequence=0)
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
                    # create 失败时必须回退 facade 的 current 占位，否则 session 永远停在 busy。
                    if self._current_task is task and self._current_run_id == run_id:
                        self._current_task = None
                        self._current_run_id = None
                    self._event_buffer.discard_run(run_id)
                    raise
                return SubmitReceipt(
                    submission_id=submission_id,
                    run_id=run_id,
                    mode=None,
                    state="delivered",
                )

            # --- 3. busy：校验 mode 与 target run 可接收性 ---
            # busy 语义差异很大（改写当前 run 还是排下一个 run），不提供默认值。
            if mode not in ("steer", "follow_up"):
                raise IrisRunStateError("busy submit 必须显式提供 steer 或 follow_up mode")
            if mode == "steer" and options is not None:
                raise IrisRunStateError("steer mode 不接受 options")
            try:
                current = self._runner.get_run(self._current_run_id)
            except IrisRunNotFoundError as exc:
                raise IrisRunStateError("current run admission 尚未完成") from exc
            # 已请求取消的 run 不会再到达安全边界，顺带清空其存量 steer 而不是让它们悬挂。
            if mode == "steer" and current.cancellation_requested_at is not None:
                self._fail_items(
                    self._pending.drain_steers_for_run(current.run_id),
                    reason="target_cancelling",
                )
                raise IrisRunStateError("cancelling run 不接受新的 steer input")
            if not self._pending.can_accept(mode):
                raise IrisRunStateError(f"{mode} input 队列容量已满")
            if not self._event_buffer.can_reserve_submission_lifecycle():
                raise IrisRunStateError("submission event buffer 容量已满")

            # --- 4. 入队并返回 pending receipt ---
            item = _PendingInput(
                submission_id=self._new_submission_id(),
                input=normalized_input,
                mode=mode,
                run_id=current.run_id if mode == "steer" else self._new_run_id(),
                options=options,
            )
            self._pending.enqueue(item)
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
        """恢复 facade 当前 exact waiting run，并返回 runner 的 durable result。

        HITL 响应不进入普通输入队列，只能经由本方法交付。

        Args:
            interaction_id (str): 必须是 current run 当前 pending 的 exact interaction id。
            response (HumanInteractionResponse): 人工决定。

        Returns:
            RunResult: runner 的 durable result；等待 result 时已释放 facade 锁。

        Raises:
            IrisRunStateError: 当 facade 已关闭、没有 current run、current run 不在 waiting
                phase，或 interaction_id 不是当前 waiting interaction 时。
        """
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
        # 在锁外等待完整 result；shield 保证调用方被取消时不会连带取消 durable activation。
        return await asyncio.shield(task)

    async def interrupt(self, *, reason: str | None = None) -> RunSnapshot:
        """请求取消 facade 当前 exact run，并保留 follow-up 到真实 terminal。

        active run 的 cancellation request 不是 terminal，因此只清空该 run 的 steer input，
        follow-up 仍等待真实 settlement。

        Args:
            reason (str | None): 取消原因，None 表示使用 runner 默认文案。

        Returns:
            RunSnapshot: 请求提交后的 run snapshot。

        Raises:
            IrisRunStateError: 当 facade 已关闭或没有 current run 时。
        """
        async with self._lock:
            self._require_open()
            await self._reconcile_locked()
            run_id = self._require_current_run()
            before = self._runner.get_run(run_id)
            snapshot = self._runner.request_cancel(run_id, reason=reason)
            # request_cancel 是同步 durable 写入，其 events 不经过 managed
            # callback，需要在此补 relay。
            for event in self._runner.list_events(run_id, before.last_event_sequence):
                self._relay_run_event(event)
            self._fail_items(
                self._pending.drain_steers_for_run(run_id),
                reason="target_cancelling",
            )
            if snapshot.phase is RunPhase.TERMINAL:
                await self._handle_terminal_locked(run_id)
            return snapshot

    def events(self) -> AsyncIterator[SessionEvent]:
        """返回唯一 mixed event consumer；不回放 manager 创建前的 durable events。

        Returns:
            AsyncIterator[SessionEvent]: 混合 durable ``RunEvent`` 与 transient
                ``SubmissionEvent`` 的异步迭代器，在 ``close()`` 之后正常结束。

        Raises:
            IrisRunStateError: 当已有 consumer 取用过该 stream 时；事件只入队一次，多个
                consumer 会互相吞掉事件。
        """
        if self._event_consumer_started:
            raise IrisRunStateError("SessionManager events 只允许一个 consumer")
        self._event_consumer_started = True
        return self._iterate_events()

    async def close(self) -> None:
        """关闭 facade admission/event stream，不取消或等待 durable run。

        幂等：重复调用直接返回。当前 durable run 继续由 runner 推进，manager 只放弃对它的
        观察与后续输入接纳。

        Notes:
            已进入 create 的 follow-up 按 run 是否已 durable 区分处理：已存在则视为投递成功，
            尚未成型才以 ``session_closed`` 失败并取消其 task。
        """
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            # claim 与 acknowledge/fail 之间不允许 await，因此持锁时不应存在悬挂 claim。
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
                    self._event_buffer.discard_run(admission.item.run_id)
                else:
                    self._complete_follow_up_success_locked(admission)
            self._fail_items(self._pending.drain_all_pending(), reason="session_closed")
            self._current_run_id = None
            self._current_task = None
            self._event_buffer.close()

    # endregion

    # ==========================================
    #            Managed Task Plumbing
    # ==========================================
    # region
    async def _iterate_events(self) -> AsyncIterator[SessionEvent]:
        """消费有界 mixed event buffer 直到其 closed 状态耗尽。"""
        while True:
            event = await self._event_buffer.next_event()
            if event is None:
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
        """创建 managed start task 并把它登记为 facade 的 current run。

        必须在持有 ``_lock`` 时调用：它直接改写 ``_current_task``。

        Args:
            input (str): 已规范化的用户输入。
            run_id (str): 预生成的 run id。
            options (AgentRunOptions | None): run 级限额与 runtime 选项。
            submission (_PendingInput | None): 触发该 run 的 follow-up；idle submit 为 None。

        Returns:
            tuple[asyncio.Task[RunResult], asyncio.Event]: managed task 与其 admission signal。
        """
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
        """挂上 done callback，在 managed task 结束后异步收口 facade 状态。

        Args:
            task (asyncio.Task[RunResult]): managed task。
            run_id (str): 该 task 推进的 run id。
            submission (_PendingInput | None): 触发该 run 的 follow-up；idle submit 为 None。
        """

        # done callback 是同步上下文，收口需要取锁，因此只在这里派发一个 task。
        def schedule(completed: asyncio.Task[RunResult]) -> None:
            asyncio.create_task(self._settle_managed_task(completed, run_id, submission=submission))

        task.add_done_callback(schedule)

    async def _wait_for_admission(
        self,
        task: asyncio.Task[RunResult],
        started: asyncio.Event,
    ) -> None:
        """等待 run admission 完成，或在 admission 前失败时抛出原因。

        Args:
            task (asyncio.Task[RunResult]): managed task。
            started (asyncio.Event): runner 的 admission signal。
        """
        signal_waiter = asyncio.create_task(started.wait())
        try:
            # signal 与 task 竞速：任一先到都说明 admission 已有结论，不必等完整 result。
            await asyncio.wait(
                {task, signal_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if started.is_set():
                return
            # task 先到说明 admission 失败，抛出异常让调用方知晓。
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
        """managed task 结束后收口 facade 的 current owner 与 follow-up 结算。

        Args:
            task (asyncio.Task[RunResult]): 已结束的 managed task。
            run_id (str): 该 task 推进的 run id。
            submission (_PendingInput | None): 触发该 run 的 follow-up；idle submit 为 None。

        Notes:
            全程吞掉异常并只记录日志：本方法运行在 done callback 派发的游离 task 中，没有
            调用方可以接收错误，抛出只会变成未处理异常。
        """
        task_error: Exception | None = None
        if not task.cancelled():
            exception = task.exception()
            if isinstance(exception, Exception):
                task_error = exception
        try:
            async with self._lock:
                # 取锁期间 facade 可能已换代或已关闭，非当前 owner 一律不再改状态。
                if self._current_task is not task or self._current_run_id != run_id:
                    return
                self._current_task = None
                try:
                    snapshot = self._runner.get_run(run_id)
                except IrisRunNotFoundError:
                    # run 从未成型：facade 回到 idle，并把 follow-up 判为 create 失败。
                    self._current_run_id = None
                    self._event_buffer.discard_run(run_id)
                    if submission is not None:
                        admission = self._follow_up_admissions.get(submission.submission_id)
                        if admission is not None:
                            if not admission.outcome.done():
                                admission.outcome.set_result(
                                    task_error or IrisRunStateError("follow-up create 未形成 run")
                                )
                            self._complete_follow_up_failure_locked(admission)
                    return
                # run 已 durable 存在即视为 follow-up 投递成功，run 自身成败与之无关。
                if submission is not None:
                    admission = self._follow_up_admissions.get(submission.submission_id)
                    if admission is not None:
                        if not admission.outcome.done():
                            admission.outcome.set_result(None)
                        self._complete_follow_up_success_locked(admission)
                if snapshot.phase is RunPhase.TERMINAL:
                    self._event_buffer.mark_run_settled(run_id)
                    await self._handle_terminal_locked(run_id)
                # waiting 保留 current owner；active task error 也不自动 recover/drain。
        except Exception:
            logger.exception(
                "SessionManager managed task settlement 失败",
                extra={"session_id": self._session_id, "run_id": run_id},
            )

    async def _reconcile_locked(self) -> None:
        """按 durable 事实校正 facade 的 current owner，并串行放行 follow-up。

        必须在持有 ``_lock`` 时调用。循环是必要的：结算一个 terminal run 会立即启动下一条
        follow-up，而它也可能已经是 terminal。
        """
        while self._current_run_id is not None:
            run_id = self._current_run_id
            try:
                snapshot = self._runner.get_run(run_id)
            except IrisRunNotFoundError:
                # task 仍在跑说明 create 尚未提交，保留占位等它出结果，不要误判为 idle。
                task = self._current_task
                if task is not None and not task.done():
                    return
                self._current_task = None
                self._current_run_id = None
                self._event_buffer.discard_run(run_id)
                return
            if snapshot.phase is RunPhase.TERMINAL:
                await self._handle_terminal_locked(run_id)
                continue
            # waiting 或 active 都保留 current owner；只清理已结束的 task 引用。
            task = self._current_task
            if task is not None and task.done():
                self._current_task = None
            return

    async def _handle_terminal_locked(self, run_id: str) -> None:
        """结算 terminal run：清空其 steer 并启动下一条 follow-up。"""
        if self._current_run_id != run_id:
            return
        self._event_buffer.mark_run_settled(run_id)
        self._fail_items(self._pending.drain_steers_for_run(run_id), reason="target_terminal")
        self._current_run_id = None
        self._current_task = None
        if not self._closed:
            await self._start_next_follow_up_locked()

    async def _start_next_follow_up_locked(self) -> None:
        """取出并启动至多一条 follow-up，等待其 create admission 有结论。

        follow-up 严格串行：只有本条 create 结算后，facade 才可能推进到下一条。
        """
        item = self._pending.peek_follow_up()
        if item is None:
            return
        if not self._event_buffer.can_register_run(item.run_id):
            return
        self._event_buffer.register_run(item.run_id, after_sequence=0)
        item = self._pending.pop_follow_up()
        assert item is not None, "peek 后的 follow-up 必须仍在队首"
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
        # waiter 负责观察 admission，finalizer 负责在本方法被取消后仍能兜底应用 outcome。
        admission.waiter = asyncio.create_task(self._resolve_follow_up_admission(admission))
        admission.finalizer = asyncio.create_task(self._finalize_follow_up_admission(admission))
        try:
            # shield 让 outcome 不因本 await 被取消而丢失，交由 finalizer 继续结算。
            await asyncio.shield(admission.outcome)
        except asyncio.CancelledError:
            raise
        self._apply_follow_up_outcome_locked(admission)

    async def _resolve_follow_up_admission(self, admission: _FollowUpAdmission) -> None:
        """观察 follow-up 的 create admission，把结论写入 ``outcome``。"""
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
        # settlement callback 可能已先写入结论，此处不覆盖。
        if not admission.outcome.done():
            admission.outcome.set_result(outcome)

    async def _finalize_follow_up_admission(self, admission: _FollowUpAdmission) -> None:
        """在 ``_start_next_follow_up_locked`` 之外兜底应用 follow-up 的 outcome。"""
        try:
            await asyncio.shield(admission.outcome)
        except asyncio.CancelledError:
            return
        async with self._lock:
            self._apply_follow_up_outcome_locked(admission)

    def _apply_follow_up_outcome_locked(self, admission: _FollowUpAdmission) -> None:
        """按已就绪的 ``outcome`` 把 follow-up 结算为 delivered 或 failed。"""
        if self._follow_up_admissions.get(admission.item.submission_id) is not admission:
            return
        if admission.outcome.result() is None:
            self._complete_follow_up_success_locked(admission)
            return
        self._complete_follow_up_failure_locked(admission)

    def _complete_follow_up_success_locked(self, admission: _FollowUpAdmission) -> None:
        """摘除 admission 并发出 delivered 事件。"""
        # pop 兼作幂等闸门：多条结算路径只有第一条能取到它。
        if self._follow_up_admissions.pop(admission.item.submission_id, None) is not admission:
            return
        self._cancel_follow_up_helpers(admission)
        self._emit_submission_event(admission.item, "delivered")

    def _complete_follow_up_failure_locked(self, admission: _FollowUpAdmission) -> None:
        """回退 facade 状态，并以 ``start_failed`` 结算该 follow-up 及其后继。

        create 失败通常源于 session lane 或配置层面的问题，后续 follow-up 大概率同样失败，
        因此一并结算而不是逐条重试。
        """
        if self._follow_up_admissions.pop(admission.item.submission_id, None) is not admission:
            return
        self._cancel_follow_up_helpers(admission)
        if self._current_task is admission.task and self._current_run_id == admission.item.run_id:
            self._current_task = None
            self._current_run_id = None
        self._event_buffer.discard_run(admission.item.run_id)
        self._emit_submission_event(admission.item, "failed", reason="start_failed")
        self._fail_items(self._pending.drain_follow_ups(), reason="start_failed")

    @staticmethod
    def _cancel_follow_up_helpers(admission: _FollowUpAdmission) -> None:
        """结算完成后取消该 admission 残留的观察 task。"""
        # 结算可能正发生在 helper 自身的栈上，取消自己会把结算路径一起中断。
        current = asyncio.current_task()
        for helper in (admission.waiter, admission.finalizer):
            if helper is not None and helper is not current and not helper.done():
                helper.cancel()

    def _schedule_tracker_reconcile(self) -> None:
        """合并 tracker 释放通知，异步继续被容量阻塞的 follow-up。"""
        if self._closed:
            return
        task = self._tracker_reconcile_task
        if task is not None and not task.done():
            return
        self._tracker_reconcile_task = asyncio.create_task(self._reconcile_after_tracker_release())

    async def _reconcile_after_tracker_release(self) -> None:
        """在 manager 锁内按最新 durable 事实恢复 FIFO 推进。"""
        try:
            async with self._lock:
                if self._closed:
                    return
                await self._reconcile_locked()
                if self._current_run_id is None:
                    await self._start_next_follow_up_locked()
        except Exception:
            logger.exception(
                "SessionManager tracker 释放后的 reconcile 失败",
                extra={"session_id": self._session_id},
            )
        finally:
            self._tracker_reconcile_task = None

    # endregion

    # ==========================================
    #              Event Emission
    # ==========================================
    # region
    def _fail_items(
        self,
        items: Iterable[_PendingInput],
        *,
        reason: SubmissionFailureReason,
    ) -> None:
        """批量把一组 pending input 结算为 failed。"""
        for item in items:
            self._emit_submission_event(item, "failed", reason=reason)

    def _relay_run_event(self, event: RunEvent) -> None:
        """把 durable ``RunEvent`` 合并为可从 store 补读的 per-run 水位。"""
        self._event_buffer.observe_run_event(event)

    def _emit_submission_event(
        self,
        item: _PendingInput,
        state: Literal["pending", "delivered", "failed"],
        *,
        reason: SubmissionFailureReason | None = None,
    ) -> None:
        """发出一条 transient ``SubmissionEvent``，报告某条输入的最终去向。"""
        event = SubmissionEvent(
            submission_id=item.submission_id,
            run_id=item.run_id,
            mode=item.mode,
            state=state,
            reason=reason,
        )
        if state == "pending":
            self._event_buffer.add_pending(event)
        else:
            self._event_buffer.add_terminal(event)

    # endregion

    # ==========================================
    #             Internal Helpers
    # ==========================================
    # region
    def _require_open(self) -> None:
        """断言 session 仍开放，拒绝 close 之后的新输入。"""
        if self._closed:
            raise IrisRunStateError("SessionManager is closed")

    def _require_current_run(self) -> str:
        """取出当前 run id，要求 facade 处于 busy。"""
        if self._current_run_id is None:
            raise IrisRunStateError("SessionManager 没有 current run")
        return self._current_run_id

    @staticmethod
    def _new_submission_id() -> str:
        """生成一个提交 id。"""
        return f"sub_{uuid.uuid4().hex}"

    @staticmethod
    def _new_run_id() -> str:
        """预生成一个 run id，供 follow-up 在 create 之前占位。"""
        return f"run_{uuid.uuid4().hex}"

    # endregion


__all__ = ["SessionEvent", "SessionManager", "SubmissionEvent", "SubmitReceipt"]
