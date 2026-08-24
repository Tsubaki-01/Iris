"""Per-session ``SessionManager`` admission 与事件语义测试。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest

from iris.exceptions import IrisRunConflictError, IrisRunStateError
from iris.harness import (
    AgentRunner,
    AgentRunOptions,
    AgentRunRequest,
    RunEvent,
    RunPhase,
    RunResult,
    SessionEvent,
    SessionManager,
    SubmissionEvent,
)
from iris.hitl import PermissionInteractionResponse
from iris.lifecycle import (
    CreateRun,
    RunEventKind,
    RunStopReason,
    RuntimeExecutionOptions,
    ToolErrorPolicy,
)
from iris.message import ToolUseBlock
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolCapability, ToolRegistry

from .fakes import (
    BlockingProvider,
    StaticProvider,
    build_runtime,
    text_response,
    tool_response,
)


async def _wait_until(predicate: Callable[[], bool]) -> None:
    """让后台 settlement callback 获得运行机会，直到条件成立。"""
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("后台状态未在预期事件循环轮次内完成")


async def _collect_until_closed(stream: AsyncIterator[SessionEvent]) -> list[SessionEvent]:
    """收集 manager close 前已排队的全部事件。"""
    return [event async for event in stream]


def _submissions(events: list[SessionEvent]) -> list[SubmissionEvent]:
    return [event for event in events if isinstance(event, SubmissionEvent)]


@pytest.mark.asyncio
async def test_idle_submit_returns_after_create_while_provider_is_running(tmp_path: Path) -> None:
    """Idle receipt 证明 create 已提交，但不等待 provider settlement。"""
    provider = BlockingProvider()
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store),
        "session-managed",
    )

    receipt = await manager.submit("开始")

    run = store.load_run(receipt.run_id)
    assert receipt.mode is None and receipt.state == "delivered"
    assert run is not None and run.phase is RunPhase.ACTIVE
    assert store.load_result(receipt.run_id) is None
    await asyncio.wait_for(provider.started.wait(), timeout=1)
    provider.release.set()
    await _wait_until(lambda: store.load_result(receipt.run_id) is not None)
    await manager.close()


@pytest.mark.asyncio
async def test_submit_validates_idle_and_busy_modes(tmp_path: Path) -> None:
    """调用方必须显式区分 idle input 与 busy steer/follow-up。"""
    provider = BlockingProvider()
    manager = SessionManager(
        AgentRunner(
            runtime=build_runtime(tmp_path, provider=provider),
            store=InMemoryLifecycleStore(),
        ),
        "session-validation",
    )

    with pytest.raises(IrisRunStateError, match="idle"):
        await manager.submit("错误", mode="steer")
    current = await manager.submit("开始")
    with pytest.raises(IrisRunStateError, match="mode"):
        await manager.submit("缺少 mode")
    with pytest.raises(IrisRunStateError, match="options"):
        await manager.submit("错误 options", mode="steer", options=AgentRunOptions())
    steer = await manager.submit("调整", mode="steer")
    follow_up = await manager.submit("下一轮", mode="follow_up", options=AgentRunOptions())

    assert steer.run_id == current.run_id and steer.state == "pending"
    assert follow_up.run_id != current.run_id and follow_up.state == "pending"
    await manager.close()
    provider.release.set()


@pytest.mark.asyncio
async def test_steer_delivery_follows_durable_commit_and_preserves_fifo(tmp_path: Path) -> None:
    """每个 safe boundary 只 claim 队首，且 delivered 晚于 durable commit event。"""
    provider = BlockingProvider(text_response("一轮"))
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store),
        "session-steer",
    )
    stream = manager.events()
    current = await manager.submit("开始")
    first = await manager.submit("方向一", mode="steer")
    second = await manager.submit("方向二", mode="steer")

    provider.release.set()
    await _wait_until(lambda: store.load_result(current.run_id) is not None)
    await _wait_until(lambda: manager._current_run_id is None)
    await manager.close()
    events = await _collect_until_closed(stream)
    submissions = _submissions(events)

    assert [(event.submission_id, event.state) for event in submissions] == [
        (first.submission_id, "pending"),
        (second.submission_id, "pending"),
        (first.submission_id, "delivered"),
        (second.submission_id, "delivered"),
    ]
    for delivered in (first, second):
        delivered_index = next(
            index
            for index, event in enumerate(events)
            if isinstance(event, SubmissionEvent)
            and event.submission_id == delivered.submission_id
            and event.state == "delivered"
        )
        assert any(
            isinstance(event, RunEvent) and event.kind is RunEventKind.MODEL_STEP_COMMITTED
            for event in events[:delivered_index]
        )
    assert [message.text for message in store.load_session("session-steer").messages] == [
        "开始",
        "一轮",
        "方向一",
        "一轮",
        "方向二",
        "一轮",
    ]


@pytest.mark.asyncio
async def test_follow_up_waits_for_terminal_and_starts_one_run_at_a_time(tmp_path: Path) -> None:
    """Follow-up admission 不抢占 current run，并按 FIFO 串行 create。"""
    provider = BlockingProvider(text_response("完成"))
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store),
        "session-follow-up",
    )
    stream = manager.events()
    current = await manager.submit("第一轮")
    first = await manager.submit("第二轮", mode="follow_up")
    second = await manager.submit("第三轮", mode="follow_up")

    assert store.load_run(first.run_id) is None
    assert store.load_run(second.run_id) is None
    provider.release.set()
    await _wait_until(lambda: store.load_result(second.run_id) is not None)
    await manager.close()
    submissions = _submissions(await _collect_until_closed(stream))

    assert store.load_result(current.run_id) is not None
    assert store.load_result(first.run_id) is not None
    assert store.load_result(second.run_id) is not None
    assert [(event.submission_id, event.state) for event in submissions] == [
        (first.submission_id, "pending"),
        (second.submission_id, "pending"),
        (first.submission_id, "delivered"),
        (second.submission_id, "delivered"),
    ]


@pytest.mark.asyncio
async def test_follow_up_does_not_block_eligible_steer(tmp_path: Path) -> None:
    """两条 mode FIFO 独立，早到 follow-up 不阻塞 current-run steer。"""
    provider = BlockingProvider(text_response("完成"))
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store),
        "session-mixed",
    )
    stream = manager.events()
    current = await manager.submit("开始")
    follow_up = await manager.submit("以后", mode="follow_up")
    steer = await manager.submit("现在", mode="steer")

    provider.release.set()
    await _wait_until(lambda: store.load_result(follow_up.run_id) is not None)
    await manager.close()
    submissions = _submissions(await _collect_until_closed(stream))
    delivered = [event.submission_id for event in submissions if event.state == "delivered"]

    assert delivered == [steer.submission_id, follow_up.submission_id]
    first_run_messages = [
        message.text
        for message in store.load_session("session-mixed").messages
        if message.text != "以后"
    ]
    assert "现在" in first_run_messages
    assert store.load_result(current.run_id) is not None


@pytest.mark.asyncio
async def test_waiting_run_accepts_steer_only_at_resume_boundary(tmp_path: Path) -> None:
    """HITL response 使用 exact resume，queued steer 等到 resumed activation boundary。"""
    def write(value: str) -> str:
        return value

    registry = ToolRegistry()
    registry.register_function(
        write,
        name="write",
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="write-1", name="write", input={"value": "x"})),
        text_response("已恢复"),
        text_response("已调整"),
    )
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(
            runtime=build_runtime(tmp_path, provider=provider, registry=registry),
            store=store,
        ),
        "session-hitl",
    )
    stream = manager.events()
    current = await manager.submit("写入")
    await _wait_until(
        lambda: (result := store.load_result(current.run_id)) is not None
        and result.run.phase is RunPhase.WAITING
    )
    waiting = store.load_result(current.run_id)
    assert waiting is not None and waiting.pending_interaction is not None
    steer = await manager.submit("恢复后调整", mode="steer")

    result = await manager.resume(
        interaction_id=waiting.pending_interaction.interaction_id,
        response=PermissionInteractionResponse(decision="approve"),
    )
    await manager.close()
    submissions = _submissions(await _collect_until_closed(stream))

    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert [(event.submission_id, event.state) for event in submissions] == [
        (steer.submission_id, "pending"),
        (steer.submission_id, "delivered"),
    ]
    assert "恢复后调整" in [
        message.text for message in store.load_session("session-hitl").messages
    ]


@pytest.mark.asyncio
async def test_active_interrupt_fails_steer_and_waits_terminal_before_follow_up(
    tmp_path: Path,
) -> None:
    """Cancellation request 不是 early follow-up eligibility。"""
    provider = BlockingProvider(text_response("完成"))
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store),
        "session-interrupt",
    )
    stream = manager.events()
    current = await manager.submit("第一轮")
    steer = await manager.submit("取消前调整", mode="steer")
    follow_up = await manager.submit("取消后继续", mode="follow_up")

    snapshot = await manager.interrupt(reason="用户停止")
    assert snapshot.run_id == current.run_id
    assert snapshot.phase is RunPhase.ACTIVE
    assert store.load_run(follow_up.run_id) is None
    await _wait_until(lambda: store.load_run(follow_up.run_id) is not None)
    provider.release.set()
    await _wait_until(lambda: store.load_result(follow_up.run_id) is not None)
    await manager.close()
    submissions = _submissions(await _collect_until_closed(stream))

    assert any(
        event.submission_id == steer.submission_id
        and event.state == "failed"
        and event.reason == "target_cancelling"
        for event in submissions
    )
    assert any(
        event.submission_id == follow_up.submission_id and event.state == "delivered"
        for event in submissions
    )


@pytest.mark.asyncio
async def test_waiting_interrupt_settles_old_run_then_starts_follow_up(tmp_path: Path) -> None:
    """Waiting cancellation 同次 terminal 后才交付一条 follow-up，backfill 不重复事件。"""
    def write(value: str) -> str:
        return value

    registry = ToolRegistry()
    registry.register_function(
        write,
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(
            runtime=build_runtime(
                tmp_path,
                registry=registry,
                provider=StaticProvider(
                    tool_response(
                        ToolUseBlock(id="write-wait", name="write", input={"value": "x"})
                    ),
                    text_response("后续完成"),
                ),
            ),
            store=store,
        ),
        "session-waiting-interrupt",
    )
    stream = manager.events()
    current = await manager.submit("等待授权")
    await _wait_until(
        lambda: (result := store.load_result(current.run_id)) is not None
        and result.run.phase is RunPhase.WAITING
    )
    follow_up = await manager.submit("取消后继续", mode="follow_up")

    interrupted = await manager.interrupt(reason="放弃授权")
    await _wait_until(lambda: store.load_result(follow_up.run_id) is not None)
    await manager.close()
    events = await _collect_until_closed(stream)

    assert interrupted.run_id == current.run_id
    assert interrupted.stop_reason is RunStopReason.CANCELLED
    run_events = [event for event in events if isinstance(event, RunEvent)]
    assert len({(event.run_id, event.sequence) for event in run_events}) == len(run_events)
    delivered_index = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, SubmissionEvent)
        and event.submission_id == follow_up.submission_id
        and event.state == "delivered"
    )
    assert any(
        isinstance(event, RunEvent)
        and event.run_id == follow_up.run_id
        and event.kind is RunEventKind.RUN_STARTED
        for event in events[:delivered_index]
    )


@pytest.mark.asyncio
async def test_cancelled_waiting_interrupt_still_finalizes_follow_up_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public interrupt waiter 取消后，已取出的 follow-up 仍由 manager 结算。"""
    def write(value: str) -> str:
        return value

    registry = ToolRegistry()
    registry.register_function(
        write,
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(
                    ToolUseBlock(id="write-cancel-wait", name="write", input={"value": "x"})
                ),
                text_response("后续完成"),
            ),
        ),
        store=store,
    )
    manager = SessionManager(runner, "session-cancelled-interrupt")
    stream = manager.events()
    current = await manager.submit("等待授权")
    await _wait_until(
        lambda: (result := store.load_result(current.run_id)) is not None
        and result.run.phase is RunPhase.WAITING
    )
    follow_up = await manager.submit("仍应交付", mode="follow_up")
    original_start = runner._start_managed
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed_start(request: AgentRunRequest, **kwargs: object) -> RunResult:
        if request.run_id == follow_up.run_id:
            entered.set()
            await release.wait()
        return await original_start(request, **kwargs)

    monkeypatch.setattr(runner, "_start_managed", delayed_start)
    waiter = asyncio.create_task(manager.interrupt(reason="取消 waiting run"))
    await asyncio.wait_for(entered.wait(), timeout=1)

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    release.set()
    await _wait_until(lambda: store.load_result(follow_up.run_id) is not None)
    await manager.close()
    terminal_events = [
        event
        for event in _submissions(await _collect_until_closed(stream))
        if event.submission_id == follow_up.submission_id and event.state != "pending"
    ]

    assert [(event.state, event.reason) for event in terminal_events] == [("delivered", None)]


@pytest.mark.asyncio
async def test_follow_up_start_failure_fails_remaining_chain_and_returns_idle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future create 失败不会 blind retry 或遗留虚假 busy owner。"""
    provider = BlockingProvider(text_response("完成"))
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store),
        "session-start-failure",
    )
    stream = manager.events()
    current = await manager.submit("第一轮")
    first = await manager.submit("失败轮", mode="follow_up")
    second = await manager.submit("应被清理", mode="follow_up")
    original_create = store.create_run

    def fail_selected(command: CreateRun):
        if command.request.run_id == first.run_id:
            raise IrisRunConflictError("模拟 future create 失败")
        return original_create(command)

    monkeypatch.setattr(store, "create_run", fail_selected)
    provider.release.set()
    await _wait_until(
        lambda: store.load_result(current.run_id) is not None
        and manager._current_run_id is None
    )
    fresh = await manager.submit("新链")
    await _wait_until(lambda: store.load_result(fresh.run_id) is not None)
    await manager.close()
    submissions = _submissions(await _collect_until_closed(stream))

    assert [
        (event.submission_id, event.reason)
        for event in submissions
        if event.state == "failed"
    ] == [
        (first.submission_id, "start_failed"),
        (second.submission_id, "start_failed"),
    ]


@pytest.mark.asyncio
async def test_claimed_steer_commit_failure_emits_failed_without_false_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claim 后 durable commit 失败必须只产生 commit_failed。"""
    provider = BlockingProvider(text_response("准备提交"))
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store),
        "session-commit-failed",
    )
    stream = manager.events()
    current = await manager.submit("开始")
    steer = await manager.submit("不可误报", mode="steer")

    def fail_commit(command: object) -> object:
        del command
        raise IrisRunConflictError("模拟 steering commit 失败")

    monkeypatch.setattr(store, "commit_model_step", fail_commit)
    provider.release.set()
    await _wait_until(
        lambda: manager._current_task is None
        and (snapshot := store.load_run(current.run_id)) is not None
        and snapshot.phase is RunPhase.ACTIVE
    )
    await manager.close()
    terminal_events = [
        event
        for event in _submissions(await _collect_until_closed(stream))
        if event.submission_id == steer.submission_id and event.state != "pending"
    ]

    assert [(event.state, event.reason) for event in terminal_events] == [
        ("failed", "commit_failed")
    ]


@pytest.mark.asyncio
async def test_unclaimed_steer_fails_when_stop_policy_makes_target_terminal(
    tmp_path: Path,
) -> None:
    """STOP tool error 没有 safe-boundary claim，terminal callback 必须结算 pending steer。"""
    provider = BlockingProvider(
        tool_response(ToolUseBlock(id="missing-stop", name="missing", input={}))
    )
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store),
        "session-target-terminal",
    )
    stream = manager.events()
    current = await manager.submit(
        "触发 STOP",
        options=AgentRunOptions(
            runtime=RuntimeExecutionOptions(tool_error_policy=ToolErrorPolicy.STOP)
        ),
    )
    steer = await manager.submit("不会进入 history", mode="steer")

    provider.release.set()
    await _wait_until(lambda: store.load_result(current.run_id) is not None)
    await _wait_until(lambda: manager._current_run_id is None)
    await manager.close()
    terminal_events = [
        event
        for event in _submissions(await _collect_until_closed(stream))
        if event.submission_id == steer.submission_id and event.state != "pending"
    ]

    assert [(event.state, event.reason) for event in terminal_events] == [
        ("failed", "target_terminal")
    ]


@pytest.mark.asyncio
async def test_close_fails_all_pending_then_ends_stream_without_cancelling_run(
    tmp_path: Path,
) -> None:
    """Close 只结束 façade ownership；pending 不 silent drop，durable run 不受影响。"""
    provider = BlockingProvider()
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store),
        "session-close",
    )
    stream = manager.events()
    current = await manager.submit("继续运行")
    steer = await manager.submit("调整", mode="steer")
    follow_up = await manager.submit("以后", mode="follow_up")

    await manager.close()
    await manager.close()
    events = await _collect_until_closed(stream)
    failures = [event for event in _submissions(events) if event.state == "failed"]

    assert [(event.submission_id, event.reason) for event in failures] == [
        (steer.submission_id, "session_closed"),
        (follow_up.submission_id, "session_closed"),
    ]
    active = store.load_run(current.run_id)
    assert active is not None and active.phase is RunPhase.ACTIVE
    assert active.cancellation_requested_at is None
    with pytest.raises(IrisRunStateError, match="closed"):
        await manager.submit("拒绝")
    provider.release.set()
    await _wait_until(lambda: store.load_result(current.run_id) is not None)


@pytest.mark.asyncio
async def test_default_capacity_accounts_for_sixty_four_pending_submissions(
    tmp_path: Path,
) -> None:
    """默认容量保持既有 64 条负载，close 逐条结算所有已接纳 input。"""
    provider = BlockingProvider()
    manager = SessionManager(
        AgentRunner(
            runtime=build_runtime(tmp_path, provider=provider),
            store=InMemoryLifecycleStore(),
        ),
        "session-unbounded",
    )
    stream = manager.events()
    await manager.submit("占用")
    receipts = [await manager.submit(f"调整 {index}", mode="steer") for index in range(64)]

    await manager.close()
    submissions = _submissions(await _collect_until_closed(stream))
    pending_ids = [event.submission_id for event in submissions if event.state == "pending"]
    failed_ids = [event.submission_id for event in submissions if event.state == "failed"]

    assert pending_ids == [receipt.submission_id for receipt in receipts]
    assert failed_ids == pending_ids
    provider.release.set()


@pytest.mark.asyncio
async def test_events_has_one_consumer_and_new_manager_does_not_attach_lane(
    tmp_path: Path,
) -> None:
    """Event stream 与 current owner 均是 process-local，restart 不猜测 attach。"""
    provider = BlockingProvider()
    store = InMemoryLifecycleStore()
    runner = AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store)
    owner = SessionManager(runner, "session-restart")
    first_stream = owner.events()
    with pytest.raises(IrisRunStateError, match="consumer"):
        owner.events()
    current = await owner.submit("占用 lane")
    restarted = SessionManager(runner, "session-restart")

    with pytest.raises(IrisRunConflictError, match="lane"):
        await restarted.submit("不得 attach")
    await restarted.close()
    assert [event async for event in restarted.events()] == []
    await owner.close()
    assert await _collect_until_closed(first_stream)
    provider.release.set()
    await _wait_until(lambda: store.load_result(current.run_id) is not None)


@pytest.mark.asyncio
async def test_cancelling_submit_waiter_does_not_cancel_manager_owned_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """调用方取消 admission waiter 后，已交给 manager 的 task 仍完成 durable settlement。"""
    store = InMemoryLifecycleStore()
    runner = AgentRunner(runtime=build_runtime(tmp_path), store=store)
    manager = SessionManager(runner, "session-caller-cancel")
    original_start = runner._start_managed
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed_start(*args: object, **kwargs: object):
        entered.set()
        await release.wait()
        return await original_start(*args, **kwargs)

    monkeypatch.setattr(runner, "_start_managed", delayed_start)
    waiter = asyncio.create_task(manager.submit("继续完成"))
    await asyncio.wait_for(entered.wait(), timeout=1)
    run_id = manager._current_run_id
    assert run_id is not None

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    release.set()
    await _wait_until(lambda: store.load_result(run_id) is not None)
    await _wait_until(lambda: manager._current_run_id is None)

    fresh = await manager.submit("下一次")
    await _wait_until(lambda: store.load_result(fresh.run_id) is not None)
    await manager.close()


@pytest.mark.asyncio
async def test_cancelling_resume_waiter_does_not_cancel_managed_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume caller 取消只停止 waiter，exact resumed activation 继续 durable settlement。"""
    def write(value: str) -> str:
        return value

    registry = ToolRegistry()
    registry.register_function(
        write,
        description="写入",
        capabilities={ToolCapability.WRITE},
    )
    store = InMemoryLifecycleStore()
    runner = AgentRunner(
        runtime=build_runtime(
            tmp_path,
            registry=registry,
            provider=StaticProvider(
                tool_response(ToolUseBlock(id="write-resume", name="write", input={"value": "x"})),
                text_response("恢复完成"),
            ),
        ),
        store=store,
    )
    manager = SessionManager(runner, "session-resume-cancel")
    current = await manager.submit("等待授权")
    await _wait_until(
        lambda: (result := store.load_result(current.run_id)) is not None
        and result.run.phase is RunPhase.WAITING
    )
    waiting = store.load_result(current.run_id)
    assert waiting is not None and waiting.pending_interaction is not None
    original_resume = runner._resume_managed
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed_resume(*args: object, **kwargs: object) -> RunResult:
        entered.set()
        await release.wait()
        return await original_resume(*args, **kwargs)

    monkeypatch.setattr(runner, "_resume_managed", delayed_resume)
    waiter = asyncio.create_task(
        manager.resume(
            interaction_id=waiting.pending_interaction.interaction_id,
            response=PermissionInteractionResponse(decision="approve"),
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    release.set()
    await _wait_until(
        lambda: (result := store.load_result(current.run_id)) is not None
        and result.run.phase is RunPhase.TERMINAL
    )
    await _wait_until(lambda: manager._current_run_id is None)
    await manager.close()


@pytest.mark.asyncio
async def test_stale_task_settlement_cannot_replace_new_current_owner(tmp_path: Path) -> None:
    """旧 task callback 必须由 exact task object 与 run id 双重 fence 忽略。"""
    class FirstThenBlockingProvider:
        def __init__(self) -> None:
            self.calls = 0
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def complete(self, request: object):
            del request
            self.calls += 1
            if self.calls == 1:
                return text_response("第一轮完成")
            self.started.set()
            await self.release.wait()
            return text_response("第二轮完成")

    provider = FirstThenBlockingProvider()
    store = InMemoryLifecycleStore()
    manager = SessionManager(
        AgentRunner(runtime=build_runtime(tmp_path, provider=provider), store=store),
        "session-stale-task",
    )
    first = await manager.submit("第一轮")
    old_task = manager._current_task
    assert old_task is not None
    await _wait_until(lambda: store.load_result(first.run_id) is not None)
    await _wait_until(lambda: manager._current_run_id is None)
    second = await manager.submit("第二轮")
    await asyncio.wait_for(provider.started.wait(), timeout=1)
    new_task = manager._current_task

    await manager._settle_managed_task(old_task, first.run_id, submission=None)

    assert manager._current_run_id == second.run_id
    assert manager._current_task is new_task
    provider.release.set()
    await _wait_until(lambda: store.load_result(second.run_id) is not None)
    await manager.close()


def test_session_manager_rejects_blank_session_id(tmp_path: Path) -> None:
    runner = AgentRunner(runtime=build_runtime(tmp_path), store=InMemoryLifecycleStore())

    with pytest.raises(IrisRunStateError, match="session_id"):
        SessionManager(runner, "  ")
