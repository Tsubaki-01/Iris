"""进程内 live epoch、顺序、replay 与 fan-out owner。

Broker 只在绑定 event loop 的线程内同步发布，不执行网络 I/O。

Example:
    broker = LiveStreamBroker(
        replay_capacity_per_scope=256,
        subscription_capacity=64,
    )
"""

# region imports

from __future__ import annotations

import asyncio
import threading
import uuid
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ..exceptions import IrisRunStateError
from ..harness.streaming import LiveFact, LivePublisher
from .models import (
    LiveCursor,
    LiveEnvelope,
    LiveScope,
    LiveStreamItem,
    LiveSubscriptionRequest,
    ReplayGap,
    ReplayGapReason,
    SubscriptionTerminal,
)
from .projection import _ProjectedLiveFact, project_live_fact

# endregion

type _ScopeKey = tuple[LiveScope, str]


@dataclass(frozen=True, slots=True)
class _StoredEnvelope:
    """Ring 和 subscription 共用的 envelope delivery metadata。"""

    envelope: LiveEnvelope
    critical: bool
    coalescing_key: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class _PendingDelivery:
    """Subscription 内保留顺序的 data/control item。"""

    item: LiveStreamItem
    critical: bool
    coalescing_key: tuple[str, ...] | None = None


class LiveSubscription(AsyncIterator[LiveStreamItem]):
    """单 consumer 的有界、可异步消费 live delivery。"""

    def __init__(
        self,
        *,
        broker: LiveStreamBroker,
        scope: LiveScope,
        scope_id: str,
        capacity: int,
        last_delivered_sequence: int,
    ) -> None:
        """创建只归属于一个 broker/scope 的 subscription。

        Args:
            broker (LiveStreamBroker): 唯一 owner broker。
            scope (LiveScope): Run 或 session scope。
            scope_id (str): Scope identity。
            capacity (int): Future-live pending envelope 上限。
            last_delivered_sequence (int): 初始已知 live high-water。
        """
        self._broker = broker
        self._scope = scope
        self._scope_id = scope_id
        self._capacity = capacity
        self._last_delivered_sequence = last_delivered_sequence
        self._replay: deque[_StoredEnvelope] = deque()
        self._pending: deque[_PendingDelivery] = deque()
        self._data_count = 0
        self._ready = asyncio.Event()
        self._accepting = True
        self._closed = False

    def __aiter__(self) -> LiveSubscription:
        """返回当前 subscription iterator。"""
        return self

    async def __anext__(self) -> LiveStreamItem:
        """等待并返回下一条 envelope/control item。"""
        self._broker._ensure_loop()
        while True:
            if self._replay:
                stored = self._replay.popleft()
                self._last_delivered_sequence = stored.envelope.live_sequence
                if not self._replay and not self._pending:
                    self._ready.clear()
                return stored.envelope
            if self._pending:
                delivery = self._pending.popleft()
                if isinstance(delivery.item, LiveEnvelope):
                    self._data_count -= 1
                    self._last_delivered_sequence = delivery.item.live_sequence
                if not self._replay and not self._pending:
                    self._ready.clear()
                if isinstance(delivery.item, SubscriptionTerminal):
                    self._closed = True
                    self._broker._detach(self)
                return delivery.item
            if self._closed or not self._accepting:
                raise StopAsyncIteration
            self._ready.clear()
            await self._ready.wait()

    async def aclose(self) -> None:
        """幂等关闭并只移除当前 subscription。"""
        self._broker._ensure_loop()
        if self._closed:
            return
        self._closed = True
        self._accepting = False
        self._replay.clear()
        self._pending.clear()
        self._data_count = 0
        self._ready.set()
        self._broker._detach(self)

    def _offer(self, stored: _StoredEnvelope) -> None:
        """按 coalescing/critical 规则放入一条 future 或 replay envelope。"""
        if not self._accepting:
            return
        if not stored.critical:
            if stored.coalescing_key is not None and self._replace_partial(stored):
                return
            if self._data_count >= self._capacity:
                return
            self._append_envelope(stored)
            return
        if self._data_count >= self._capacity:
            self._evict_one_partial()
        if self._data_count >= self._capacity:
            self._mark_slow_consumer()
            return
        self._append_envelope(stored)

    def _enqueue_replay(self, stored: _StoredEnvelope) -> None:
        """把原子捕获的 bounded replay snapshot 放入独立队列。"""
        self._replay.append(stored)
        self._ready.set()

    def _append_envelope(self, stored: _StoredEnvelope) -> None:
        self._pending.append(
            _PendingDelivery(
                item=stored.envelope,
                critical=stored.critical,
                coalescing_key=stored.coalescing_key,
            )
        )
        self._data_count += 1
        self._ready.set()

    def _replace_partial(self, stored: _StoredEnvelope) -> bool:
        for delivery in self._pending:
            if delivery.coalescing_key == stored.coalescing_key:
                self._pending.remove(delivery)
                self._pending.append(
                    _PendingDelivery(
                        item=stored.envelope,
                        critical=False,
                        coalescing_key=stored.coalescing_key,
                    )
                )
                self._ready.set()
                return True
        return False

    def _evict_one_partial(self) -> None:
        for delivery in self._pending:
            if delivery.coalescing_key is not None:
                self._pending.remove(delivery)
                self._data_count -= 1
                return

    def _mark_slow_consumer(self) -> None:
        self._pending = deque(
            delivery for delivery in self._pending if not isinstance(delivery.item, LiveEnvelope)
        )
        self._data_count = 0
        replay_high_water = (
            self._replay[-1].envelope.live_sequence
            if self._replay
            else self._last_delivered_sequence
        )
        cursor = LiveCursor(
            stream_epoch=self._broker._epoch,
            scope=self._scope,
            scope_id=self._scope_id,
            after_live_sequence=max(self._last_delivered_sequence, replay_high_water),
        )
        self._pending.append(
            _PendingDelivery(
                item=ReplayGap(
                    reason="slow_consumer",
                    requested_cursor=cursor,
                    current_epoch=self._broker._epoch,
                ),
                critical=True,
            )
        )
        self._pending.append(
            _PendingDelivery(
                item=SubscriptionTerminal(
                    reason="slow_consumer",
                    message="Live subscription 因消费过慢已关闭，请执行 durable sync",
                ),
                critical=True,
            )
        )
        self._accepting = False
        self._ready.set()
        self._broker._detach(self)

    def _enqueue_gap(
        self,
        reason: ReplayGapReason,
        cursor: LiveCursor,
    ) -> None:
        self._pending.append(
            _PendingDelivery(
                item=ReplayGap(
                    reason=reason,
                    requested_cursor=cursor,
                    current_epoch=self._broker._epoch,
                ),
                critical=True,
            )
        )
        self._ready.set()

    def _broker_closed(self) -> None:
        if not self._accepting:
            return
        self._accepting = False
        self._pending.append(
            _PendingDelivery(
                item=SubscriptionTerminal(
                    reason="broker_closed",
                    message="Live stream broker 已关闭",
                ),
                critical=True,
            )
        )
        self._ready.set()


class LiveStreamBroker(LivePublisher):
    """唯一拥有 live epoch、sequence、rings 与 subscriptions 的 broker。"""

    def __init__(
        self,
        *,
        replay_capacity_per_scope: int,
        subscription_capacity: int,
    ) -> None:
        """创建显式有界的进程内 broker。

        Args:
            replay_capacity_per_scope (int): 每个 run/session ring 的 envelope 上限。
            subscription_capacity (int): 每个 consumer 的 pending envelope 上限。

        Raises:
            ValueError: 任一 capacity 不是有限正整数。
        """
        _require_positive_capacity(replay_capacity_per_scope, "replay_capacity_per_scope")
        _require_positive_capacity(subscription_capacity, "subscription_capacity")
        self._replay_capacity = replay_capacity_per_scope
        self._subscription_capacity = subscription_capacity
        self._epoch = uuid.uuid4().hex
        self._sequences: dict[_ScopeKey, int] = {}
        self._rings: dict[_ScopeKey, deque[_StoredEnvelope]] = {}
        self._subscriptions: set[LiveSubscription] = set()
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread_id: int | None = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            self._loop = loop
            self._thread_id = threading.get_ident()

    def publish(self, fact: LiveFact) -> None:
        """同步投影并 fan-out 一条 trusted fact。"""
        self._ensure_loop()
        if self._closed:
            raise IrisRunStateError("LiveStreamBroker 已关闭")
        for projected in project_live_fact(fact):
            self._publish_projected(projected)

    def subscribe(self, request: LiveSubscriptionRequest) -> LiveSubscription:
        """原子捕获 high-water、预填 replay/control 并登记 future live。"""
        self._ensure_loop()
        if self._closed:
            raise IrisRunStateError("LiveStreamBroker 已关闭")
        key = (request.scope, request.scope_id)
        high_water = self._sequences.get(key, 0)
        cursor = request.cursor
        initial_sequence = (
            cursor.after_live_sequence
            if cursor is not None
            and cursor.stream_epoch == self._epoch
            and key in self._sequences
            and cursor.after_live_sequence <= high_water
            else high_water
        )
        subscription = LiveSubscription(
            broker=self,
            scope=request.scope,
            scope_id=request.scope_id,
            capacity=self._subscription_capacity,
            last_delivered_sequence=initial_sequence,
        )
        if cursor is not None:
            self._prepare_replay(subscription, key, cursor, high_water)
        if subscription._accepting:
            self._subscriptions.add(subscription)
        return subscription

    def current_epoch(self) -> str:
        """返回当前进程内 broker epoch。"""
        self._ensure_loop()
        return self._epoch

    def close(self) -> None:
        """幂等关闭 broker，并为每个 active subscription 排入唯一终态。"""
        self._ensure_loop()
        if self._closed:
            return
        self._closed = True
        subscriptions = tuple(self._subscriptions)
        self._subscriptions.clear()
        for subscription in subscriptions:
            subscription._broker_closed()

    def _publish_projected(self, projected: _ProjectedLiveFact) -> None:
        key = (projected.scope, projected.scope_id)
        sequence = self._sequences.get(key, 0) + 1
        self._sequences[key] = sequence
        envelope = LiveEnvelope.model_construct(
            stream_epoch=self._epoch,
            scope=projected.scope,
            scope_id=projected.scope_id,
            live_sequence=sequence,
            kind=projected.kind,
            run_id=projected.run_id,
            session_id=projected.session_id,
            activation_id=projected.activation_id,
            durable_sequence=projected.durable_sequence,
            payload=projected.payload,
        )
        stored = _StoredEnvelope(
            envelope=envelope,
            critical=projected.critical,
            coalescing_key=projected.coalescing_key,
        )
        ring = self._rings.setdefault(
            key,
            deque(maxlen=self._replay_capacity),
        )
        ring.append(stored)
        for subscription in tuple(self._subscriptions):
            if (
                subscription._scope == projected.scope
                and subscription._scope_id == projected.scope_id
            ):
                subscription._offer(stored)

    def _prepare_replay(
        self,
        subscription: LiveSubscription,
        key: _ScopeKey,
        cursor: LiveCursor,
        high_water: int,
    ) -> None:
        if cursor.stream_epoch != self._epoch:
            subscription._enqueue_gap("epoch_changed", cursor)
            return
        if key not in self._sequences or cursor.after_live_sequence > high_water:
            subscription._enqueue_gap("unknown_cursor", cursor)
            return
        ring = self._rings[key]
        oldest_sequence = ring[0].envelope.live_sequence
        if cursor.after_live_sequence < oldest_sequence - 1:
            subscription._enqueue_gap("cursor_expired", cursor)
            return
        for stored in ring:
            if cursor.after_live_sequence < stored.envelope.live_sequence <= high_water:
                subscription._enqueue_replay(stored)

    def _detach(self, subscription: LiveSubscription) -> None:
        self._subscriptions.discard(subscription)

    def _ensure_loop(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise IrisRunStateError("LiveStreamBroker 操作必须发生在绑定 event loop 内") from exc
        thread_id = threading.get_ident()
        if self._loop is None:
            self._loop = loop
            self._thread_id = thread_id
            return
        if self._loop is not loop or self._thread_id != thread_id:
            raise IrisRunStateError("LiveStreamBroker 不支持跨 event loop/thread 调用")


def _require_positive_capacity(value: int, name: str) -> None:
    """只接受非 bool 的正整数 capacity。"""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")


__all__ = ["LiveStreamBroker", "LiveSubscription"]
