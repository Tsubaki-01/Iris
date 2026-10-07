"""Goal session 通知与可空取消回执的 live plane 契约。"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from iris.goal.models import GoalChanged
from iris.goal.service import GoalService
from iris.harness import AgentRunner, SessionManager
from iris.harness.streaming import LiveFact
from iris.lifecycle import RunEvent, RunEventKind
from iris.store import InMemoryLifecycleStore
from iris.streaming.broker import LiveStreamBroker
from iris.streaming.gateway import StreamingGateway
from iris.streaming.models import (
    CancelAccepted,
    CancelCommand,
    CommandReceipt,
    LiveSubscriptionRequest,
)
from iris.streaming.projection import project_live_fact
from tests.harness.fakes import StaticProvider, build_runtime


def _changed(objective: str, *, session_id: str = "session") -> GoalChanged:
    """使用权威读视图生成通知。"""
    service = GoalService(InMemoryLifecycleStore())
    service.create(session_id, objective)
    return GoalChanged(session_id=session_id, view=service.get_view(session_id))


def test_goal_projection_is_session_only_and_keeps_actual_view() -> None:
    """无 Run 也能通知，不编造运行身份或 durable sequence。"""
    event = _changed("完成目标")
    projected, tree = project_live_fact(event)
    assert tree.scope == "session_tree" and tree.payload == projected.payload
    assert (projected.scope, projected.scope_id, projected.kind) == (
        "session",
        "session",
        "goal.changed",
    )
    assert projected.run_id is None
    assert projected.activation_id is None
    assert projected.durable_sequence is None
    assert projected.session_id == "session"
    assert projected.coalescing_key == ("goal", "session")
    assert not projected.critical
    assert projected.payload["view"]["goal"]["objective"] == "完成目标"
    assert projected.payload["view"]["armed"] is False


@pytest.mark.asyncio
async def test_goal_coalescing_moves_latest_after_terminal_and_is_session_scoped() -> None:
    """连续快照复用 broker 合并规则，最新结果排在已发布终态之后。"""
    broker = LiveStreamBroker(replay_capacity_per_scope=16, subscription_capacity=3)
    first = broker.subscribe(LiveSubscriptionRequest(scope="session", scope_id="session"))
    other = broker.subscribe(LiveSubscriptionRequest(scope="session", scope_id="other"))
    broker.publish(_changed("开始"))
    broker.publish(
        RunEvent(
            run_id="run",
            session_id="session",
            sequence=3,
            kind=RunEventKind.RUN_TERMINAL,
            occurred_at=datetime.now(UTC),
        )
    )
    for index in range(10):
        broker.publish(_changed(f"最新 {index}"))
    broker.publish(_changed("另一个 session", session_id="other"))
    terminal = await asyncio.wait_for(anext(first), 1)
    latest = await asyncio.wait_for(anext(first), 1)
    assert terminal.kind == "run.terminal"
    assert latest.kind == "goal.changed"
    assert terminal.live_sequence < latest.live_sequence
    assert latest.payload["view"]["goal"]["objective"] == "最新 9"
    assert (await asyncio.wait_for(anext(other), 1)).payload["view"]["goal"]["objective"] == (
        "另一个 session"
    )
    await first.aclose()
    await other.aclose()
    broker.close()


@pytest.mark.asyncio
async def test_runner_goal_fact_dispatch_is_best_effort(tmp_path: Path) -> None:
    """失败 publisher 不会因 Goal 缺少 event/run_id 字段泄露分派异常。"""

    class Publisher:
        """记录已到达发布边界的事实后抛出宿主异常。"""

        def __init__(self) -> None:
            self.facts: list[LiveFact] = []

        def publish(self, fact: LiveFact) -> None:
            """模拟调用方通知异常。"""
            self.facts.append(fact)
            raise RuntimeError("publisher unavailable")

    publisher = Publisher()
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=StaticProvider()),
        store=InMemoryLifecycleStore(),
        live_publisher=publisher,
    )
    event = _changed("目标")
    runner._publish_live_fact(event)
    assert publisher.facts == [event]
    await runner.aclose()


@pytest.mark.asyncio
async def test_gateway_accepts_goal_only_interrupt_without_fabricating_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gateway 直接传递 manager 的可空取消结果，并可进行 wire roundtrip。"""
    runner = AgentRunner(
        runtime=build_runtime(tmp_path, provider=StaticProvider()), store=InMemoryLifecycleStore()
    )
    manager = SessionManager(runner, "session")
    broker = LiveStreamBroker(replay_capacity_per_scope=4, subscription_capacity=4)
    gateway = StreamingGateway(
        runner=runner, manager=manager, broker=broker, session_id="session", durable_page_size=4
    )

    async def interrupt(*, reason: str | None = None) -> None:
        """返回仅暂停 Goal 意图后的真实新契约。"""
        assert reason == "暂停目标"
        return None

    monkeypatch.setattr(manager, "interrupt", interrupt)
    receipt = await gateway.handle(CancelCommand(request_id="cancel", reason="暂停目标"))
    assert isinstance(receipt, CancelAccepted)
    assert receipt.run is None
    assert TypeAdapter(CommandReceipt).validate_json(receipt.model_dump_json()) == receipt
    await manager.close()
    await runner.aclose()
    broker.close()
