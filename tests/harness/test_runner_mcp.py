"""MCP root runner 准备、恢复与宿主资源生命周期。"""

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest

from iris.exceptions import (
    IrisMCPError,
    IrisRunConflictError,
    IrisRunObservationTimeoutError,
    IrisRunPersistenceError,
    IrisRunStateError,
)
from iris.harness import AgentRunner, SessionHistory
from iris.hitl import PermissionInteractionResponse
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    ClaimToolCall,
    FinishRun,
    RunCommit,
    RunEvent,
    RunEventKind,
    RunLimits,
    RunPhase,
    RunStopReason,
    ToolCallPhase,
)
from iris.message import ToolUseBlock
from iris.store import InMemoryLifecycleStore

from ..mcp.fixtures.runtime import MCPPeer, mcp_agent
from ..mcp.fixtures.tools import AllowTools
from .fakes import BlockingProvider, FrozenClock, StaticProvider, text_response, tool_response


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_prepare_precedes_create_and_reuses_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit: bool,
) -> None:
    peer = MCPPeer(monkeypatch)
    store = InMemoryLifecycleStore()
    provider = StaticProvider(
        tool_response(ToolUseBlock(id="call", name="mcp__test__echo", input={})),
        text_response(),
        text_response(),
    )
    runner = AgentRunner.from_config(mcp_agent(tmp_path), store=store, provider=provider)
    assert not peer.events
    if explicit:
        await runner.aprepare()
        assert store.load_run("first") is None
    result = await runner.start(AgentRunRequest(input="call", run_id="first"))
    assert result.run.stop_reason is RunStopReason.COMPLETED
    assert store.list_tool_calls("first")[0].phase is ToolCallPhase.COMMITTED
    await runner.start(AgentRunRequest(input="again", run_id="second"))
    assert peer.events == ["open", "list", "call:echo"]
    await runner.aclose()
    await runner.aclose()
    assert peer.events[-1] == "close" and peer.events.count("close") == 1
    with pytest.raises(IrisRunStateError):
        await runner.start(AgentRunRequest(input="closed", run_id="third"))
    assert await runner.recover("first") == result


@pytest.mark.asyncio
async def test_required_prepare_failure_creates_no_run_and_closes_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = MCPPeer(monkeypatch)
    peer.fail_open = True
    store = InMemoryLifecycleStore()
    runner = AgentRunner.from_config(mcp_agent(tmp_path), store=store, provider=StaticProvider())
    with pytest.raises(IrisMCPError):
        await runner.start(AgentRunRequest(input="call", run_id="failed"))
    assert store.load_run("failed") is None
    assert peer.events == ["open", "close"]
    with pytest.raises(IrisRunStateError):
        await runner.aprepare()


@pytest.mark.asyncio
async def test_active_close_rejected_and_detached_durable_queries_survive_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = MCPPeer(monkeypatch)
    provider = BlockingProvider()
    runner = AgentRunner.from_config(mcp_agent(tmp_path), provider=provider)
    task = asyncio.create_task(runner.start(AgentRunRequest(input="wait", run_id="run")))
    await provider.started.wait()
    try:
        with pytest.raises(IrisRunStateError):
            await runner.aclose()
        assert "close" not in peer.events
    finally:
        provider.release.set()
        result = await task
        await runner.aclose()
    assert runner.get_result("run") == result


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["expiry", "deadline", "cancel", "resolved"])
async def test_resume_redispatches_fresh_state_after_slow_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    peer = MCPPeer(monkeypatch)
    config = mcp_agent(tmp_path, trust=False)
    clock = FrozenClock()
    store = InMemoryLifecycleStore()
    first = AgentRunner.from_config(
        config,
        store=store,
        clock=clock,
        provider=StaticProvider(
            tool_response(ToolUseBlock(id="call", name="mcp__test__echo", input={})),
            text_response(),
        ),
    )
    limits = RunLimits(
        interaction_timeout_seconds=10,
        deadline_at=clock.now() + timedelta(seconds=5) if change == "deadline" else None,
    )
    waiting = await first.start(
        AgentRunRequest(input="call", run_id="waiting"), options=AgentRunOptions(limits=limits)
    )
    interaction = waiting.pending_interaction
    assert interaction is not None
    second = AgentRunner.from_config(
        config, store=store, clock=clock, provider=StaticProvider(text_response())
    )
    peer.opened.clear()
    peer.release_open.clear()
    response = PermissionInteractionResponse(decision="approve")
    task = asyncio.create_task(
        second.resume("waiting", interaction_id=interaction.interaction_id, response=response)
    )
    try:
        await asyncio.wait_for(peer.opened.wait(), 1)
        if change in {"expiry", "deadline"}:
            clock.advance(seconds=10)
        elif change == "cancel":
            second.request_cancel("waiting")
            assert store.load_run("waiting").cancellation_requested_at is not None
        else:
            await first.resume(
                "waiting", interaction_id=interaction.interaction_id, response=response
            )
    finally:
        peer.release_open.set()
        result = await task
        await first.aclose()
        await second.aclose()
    expected = {
        "expiry": RunStopReason.INTERACTION_EXPIRED,
        "deadline": RunStopReason.DEADLINE_EXCEEDED,
        "cancel": RunStopReason.CANCELLED,
        "resolved": RunStopReason.COMPLETED,
    }[change]
    assert result.run.stop_reason is expected
    if change != "resolved":
        assert "call:echo" not in peer.events
        assert store.load_interaction(interaction.interaction_id).response is None


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["resume", "recover"])
async def test_expired_waiting_settles_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    peer = MCPPeer(monkeypatch)
    config = mcp_agent(tmp_path, trust=False)
    store = InMemoryLifecycleStore()
    clock = FrozenClock()
    first = AgentRunner.from_config(
        config,
        store=store,
        clock=clock,
        provider=StaticProvider(
            tool_response(ToolUseBlock(id="call", name="mcp__test__echo", input={}))
        ),
    )
    waiting = await first.start(
        AgentRunRequest(input="call", run_id="expired"),
        options=AgentRunOptions(limits=RunLimits(interaction_timeout_seconds=1)),
    )
    await first.aclose()
    peer.events.clear()
    peer.fail_open = True
    clock.advance(seconds=1)
    second = AgentRunner.from_config(config, store=store, clock=clock, provider=StaticProvider())
    if entry == "resume":
        result = await second.resume(
            "expired",
            interaction_id=waiting.pending_interaction.interaction_id,
            response=PermissionInteractionResponse(decision="approve"),
        )
    else:
        result = await second.recover("expired")
    assert result.run.stop_reason is RunStopReason.INTERACTION_EXPIRED
    assert peer.events == []


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["none", "terminal", "fence"])
async def test_active_recovery_refreshes_phase_and_keeps_callers_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    peer = MCPPeer(monkeypatch)
    config = mcp_agent(tmp_path)
    store = InMemoryLifecycleStore()
    blocked = BlockingProvider()
    first = AgentRunner.from_config(config, store=store, provider=blocked)
    running = asyncio.create_task(first.start(AgentRunRequest(input="recover", run_id="active")))
    await blocked.started.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    await first.aclose()
    fence = store.load_run("active").current_activation_id
    competitor_provider = (
        BlockingProvider() if change == "fence" else StaticProvider(text_response())
    )
    competitor = AgentRunner.from_config(config, store=store, provider=competitor_provider)
    await competitor.aprepare()
    second = AgentRunner.from_config(config, store=store, provider=StaticProvider(text_response()))
    peer.opened.clear()
    peer.release_open.clear()
    task = asyncio.create_task(second.recover("active", expected_activation_id=fence))
    other = None
    try:
        await asyncio.wait_for(peer.opened.wait(), 1)
        if change == "terminal":
            await competitor.recover("active", expected_activation_id=fence)
        elif change == "fence":
            other = asyncio.create_task(competitor.recover("active", expected_activation_id=fence))
            await competitor_provider.started.wait()
    finally:
        peer.release_open.set()
    try:
        if change == "fence":
            with pytest.raises(IrisRunConflictError, match="fence"):
                await task
        else:
            assert (await task).run.stop_reason is RunStopReason.COMPLETED
    finally:
        if other is not None:
            competitor_provider.release.set()
            await other
        await second.aclose()
        await competitor.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("claim_during_prepare", [False, True])
async def test_recovery_observes_unresolved_claim_without_replaying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    claim_during_prepare: bool,
) -> None:
    class InterruptedClaimStore(InMemoryLifecycleStore):
        command: ClaimToolCall

        def claim_tool_call(self, command: ClaimToolCall) -> RunCommit:
            self.command = command
            if not claim_during_prepare:
                super().claim_tool_call(command)
            raise IrisRunPersistenceError("interrupted claim")

    peer = MCPPeer(monkeypatch)
    config = mcp_agent(tmp_path)
    store = InterruptedClaimStore()
    first = AgentRunner.from_config(
        config,
        store=store,
        provider=StaticProvider(
            tool_response(ToolUseBlock(id="call", name="mcp__test__echo", input={}))
        ),
    )
    with pytest.raises(IrisRunPersistenceError):
        await first.start(AgentRunRequest(input="recover", run_id="claimed"))
    await first.aclose()
    peer.events.clear()
    second = AgentRunner.from_config(config, store=store, provider=StaticProvider())
    fence = store.load_run("claimed").current_activation_id
    if claim_during_prepare:
        peer.opened.clear()
        peer.release_open.clear()
        task = asyncio.create_task(second.recover("claimed", expected_activation_id=fence))
        try:
            await asyncio.wait_for(peer.opened.wait(), 1)
            InMemoryLifecycleStore.claim_tool_call(store, store.command)
        finally:
            peer.release_open.set()
        result = await task
    else:
        peer.fail_open = True
        result = await second.recover("claimed", expected_activation_id=fence)
        assert not peer.events
    assert result.run.stop_reason is RunStopReason.OUTCOME_UNKNOWN
    assert "call:echo" not in peer.events
    await second.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("known_result", [False, True])
async def test_public_cancel_waits_for_mcp_cleanup_and_preserves_known_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trust: bool,
    known_result: bool,
) -> None:
    peer = MCPPeer(monkeypatch)
    peer.block_call = True
    peer.return_on_cancel = known_result
    peer.release_cleanup.clear()
    runner = AgentRunner.from_config(
        mcp_agent(tmp_path, trust=trust),
        permission_policy=AllowTools(),
        provider=StaticProvider(
            tool_response(ToolUseBlock(id="call", name="mcp__test__echo", input={}))
        ),
    )
    task = asyncio.create_task(runner.start(AgentRunRequest(input="call", run_id="cancel")))
    try:
        await asyncio.wait_for(peer.called.wait(), 1)
        with pytest.raises(IrisRunObservationTimeoutError):
            await runner.cancel("cancel", settlement_timeout=0.05)
        assert peer.cleaning.is_set()
        assert runner.store.load_run("cancel").cancellation_requested_at is not None
        assert runner.get_run("cancel").phase is RunPhase.ACTIVE
        assert runner.get_result("cancel") is None
        assert runner.list_tool_calls("cancel")[0].phase is ToolCallPhase.CLAIMED
        assert "close" not in peer.events
    finally:
        peer.release_cleanup.set()
        result = await task
        await runner.aclose()
    assert result.run.stop_reason is (
        RunStopReason.CANCELLED if known_result else RunStopReason.OUTCOME_UNKNOWN
    )
    assert runner.list_tool_calls("cancel")[0].phase is (
        ToolCallPhase.COMMITTED if known_result else ToolCallPhase.OUTCOME_UNKNOWN
    )
    assert peer.events.count("call:echo") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", [False, True])
async def test_outcome_ready_recovery_prepares_current_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: bool,
) -> None:
    class FailFinishOnceStore(InMemoryLifecycleStore):
        failed = False

        def finish_run(self, command: FinishRun) -> RunCommit:
            if not self.failed:
                self.failed = True
                raise IrisRunPersistenceError("finish interrupted")
            return super().finish_run(command)

    peer = MCPPeer(monkeypatch)
    config = mcp_agent(tmp_path)
    store = FailFinishOnceStore()
    first = AgentRunner.from_config(config, store=store, provider=StaticProvider(text_response()))
    with pytest.raises(IrisRunPersistenceError):
        await first.start(AgentRunRequest(input="done", run_id="finalize"))
    await first.aclose()
    peer.events.clear()
    if drift:
        peer.protocol_version = "2025-11-25"
    provider = StaticProvider()
    second = AgentRunner.from_config(config, store=store, provider=provider)
    try:
        fence = store.load_run("finalize").current_activation_id
        assert (
            await second.recover("finalize", expected_activation_id=fence)
        ).run.stop_reason is RunStopReason.COMPLETED
        assert peer.events == ["open", "list"] and not provider.requests
    finally:
        await second.aclose()


@pytest.mark.asyncio
async def test_host_waits_original_call_after_cancel_result_before_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = MCPPeer(monkeypatch)
    delivering = asyncio.Event()
    release = asyncio.Event()

    class Observer:
        async def on_event(self, event: RunEvent) -> None:
            if event.kind is RunEventKind.RUN_TERMINAL:
                delivering.set()
                await release.wait()

    provider = BlockingProvider()
    runner = AgentRunner.from_config(mcp_agent(tmp_path), provider=provider, observers=[Observer()])
    task = asyncio.create_task(runner.start(AgentRunRequest(input="wait", run_id="delivery")))
    await provider.started.wait()
    try:
        result = await runner.cancel("delivery", settlement_timeout=1)
        await delivering.wait()
        assert result.run.stop_reason is RunStopReason.CANCELLED
        assert not task.done() and "close" not in peer.events
    finally:
        release.set()
        await task
        await runner.aclose()
    assert peer.events[-1] == "close"


@pytest.mark.asyncio
async def test_history_fork_and_terminal_reads_do_not_prepare_new_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = MCPPeer(monkeypatch)
    config = mcp_agent(tmp_path)
    store = InMemoryLifecycleStore()
    first = AgentRunner.from_config(config, store=store, provider=StaticProvider(text_response()))
    result = await first.start(AgentRunRequest(input="done", run_id="history"))
    await first.aclose()
    peer.events.clear()
    peer.fail_open = True
    second = AgentRunner.from_config(config, store=store, provider=StaticProvider())
    assert await second.recover("history") == result
    assert second.get_result("history") == result
    branch = SessionHistory(store).fork("history")
    assert branch.messages == store.load_session("default").messages
    second.request_cancel("history")
    assert not peer.events
