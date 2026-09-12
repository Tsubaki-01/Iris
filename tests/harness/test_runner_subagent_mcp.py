"""Child MCP 五个创建入口与宿主收尾的资源 ownership。"""

import asyncio
import json
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from iris.exceptions import (
    IrisMCPError,
    IrisRunConflictError,
    IrisRunPersistenceError,
    IrisRunRecoveryError,
)
from iris.harness import AgentRunner, SessionManager
from iris.hitl import PermissionInteractionResponse, QuestionInteractionResponse
from iris.hitl.models import SubagentExpiryOwner
from iris.lifecycle import (
    AdmitChildRun,
    AgentRunOptions,
    AgentRunRequest,
    CreateRun,
    FinishRun,
    RecoverActiveRun,
    RecoveryDisposition,
    RunLimits,
    RunPhase,
    RunStopReason,
    RuntimeExecutionOptions,
    ToolCallPhase,
)
from iris.message import ToolUseBlock
from iris.runtime import RuntimeCursor
from iris.store import InMemoryLifecycleStore, SQLiteStore
from iris.tools import DefaultPermissionPolicy, PermissionDecision, PermissionEffect

from ..mcp.fixtures.runtime import MCPPeer
from .fakes import FrozenClock, StaticProvider, text_response, tool_response
from .test_runner_subagent import (
    ChildProviders,
    _dispatch,
    _parent_provider,
    _prepare_parent,
    _write_configs,
)


def configs(tmp_path: Path, *, trust: bool = True, root_mcp: bool = False) -> Path:
    """在既有 parent/child 配置上添加普通 MCP 引用。"""
    path = _write_configs(tmp_path)
    (tmp_path / "mcp.json").write_text('{"servers":{"test":{"command":"fixture"}}}')
    mcp_yaml = (
        "\nmcp:\n  path: mcp.json\n  overrides:\n    test:\n      trust_annotations: "
        + str(trust).lower()
        + "\n"
    )
    child_path = tmp_path / "child.yaml"
    child_path.write_text(child_path.read_text(encoding="utf-8") + mcp_yaml, encoding="utf-8")
    if root_mcp:
        path.write_text(path.read_text(encoding="utf-8") + mcp_yaml, encoding="utf-8")
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "waiting", "prepare_failure"])
@pytest.mark.parametrize("close_failure", [False, True])
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_fresh_child_prepares_and_closes_without_overwriting_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    close_failure: bool,
    caplog: pytest.LogCaptureFixture,
    backend: str,
) -> None:
    peer = MCPPeer(monkeypatch)
    peer.fail_open = outcome == "prepare_failure"
    peer.fail_close = close_failure
    runner = AgentRunner.from_config_path(
        configs(tmp_path, trust=outcome != "waiting"),
        store=SQLiteStore(tmp_path / "child.db")
        if backend == "sqlite"
        else InMemoryLifecycleStore(),
        provider=_parent_provider(),
        child_provider_factory=ChildProviders(
            StaticProvider(
                tool_response(ToolUseBlock(id="call", name="mcp__test__echo", input={})),
                text_response(),
            )
        ),
    )
    result = await runner.start(AgentRunRequest(input="delegate", run_id="parent"))
    assert peer.events.count("open") == 1 and peer.events.count("close") == 1
    link = runner.store.load_subagent_link("parent", "delegate")
    if outcome == "prepare_failure":
        assert link is None
        assert (
            runner.store.load_tool_call("parent", "delegate").result.error.code
            == "SUBAGENT_PREPARE_ERROR"
        )
    else:
        assert link is not None
        child = runner.store.load_run(link.child_run_id)
        assert child.environment_fingerprint
        assert child.phase is (RunPhase.WAITING if outcome == "waiting" else RunPhase.TERMINAL)
        assert result.run.phase is child.phase
    if close_failure:
        assert "关闭" in caplog.text


@pytest.mark.asyncio
async def test_admission_exception_closes_child_before_propagating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = MCPPeer(monkeypatch)
    runner = AgentRunner.from_config_path(
        configs(tmp_path),
        provider=StaticProvider(),
        child_provider_factory=ChildProviders(StaticProvider()),
    )
    prepared = _prepare_parent(runner)

    def reject(command: object) -> None:
        raise IrisRunConflictError("admission changed")

    monkeypatch.setattr(runner.store, "admit_child_run", reject)
    with pytest.raises(IrisRunConflictError, match="admission changed"):
        await _dispatch(runner, prepared)
    assert peer.events == ["open", "list", "close"]
    assert runner.store.load_subagent_link("parent", "delegate") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", [False, True])
async def test_waiting_child_rebuilds_and_uses_ordinary_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: bool,
) -> None:
    peer = MCPPeer(monkeypatch)
    runner = AgentRunner.from_config_path(
        configs(tmp_path, trust=False),
        provider=_parent_provider(),
        child_provider_factory=ChildProviders(
            StaticProvider(
                tool_response(ToolUseBlock(id="call", name="mcp__test__echo", input={})),
                text_response(),
            )
        ),
    )
    waiting = await runner.start(AgentRunRequest(input="delegate", run_id="parent"))
    assert peer.events == ["open", "list", "close"]
    if drift:
        peer.protocol_version = "2025-11-25"
    proxy = waiting.pending_interaction
    if drift:
        with pytest.raises(IrisRunRecoveryError):
            await runner.resume(
                "parent",
                interaction_id=proxy.interaction_id,
                response=PermissionInteractionResponse(decision="approve"),
            )
    else:
        result = await runner.resume(
            "parent",
            interaction_id=proxy.interaction_id,
            response=PermissionInteractionResponse(decision="approve"),
        )
        assert result.run.stop_reason is RunStopReason.COMPLETED
        assert peer.events.count("call:echo") == 1
    assert peer.events.count("open") == peer.events.count("close") == 2


@pytest.mark.asyncio
async def test_real_sdk_child_waiting_closes_while_root_connection_remains(
    tmp_path: Path,
) -> None:
    path = configs(tmp_path, trust=False, root_mcp=True)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "trust_annotations: false", "trust_annotations: true"
        ),
        encoding="utf-8",
    )
    record = tmp_path / "server-events.jsonl"
    script = Path(__file__).parents[1] / "mcp" / "fixtures" / "servers.py"
    (tmp_path / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "test": {
                        "command": sys.executable,
                        "args": [str(script)],
                        "env": {"IRIS_MCP_TEST_LOG": str(record)},
                    }
                }
            }
        )
    )
    runner = AgentRunner.from_config_path(
        path,
        provider=StaticProvider(
            tool_response(
                ToolUseBlock(id="delegate", name="subagent", input={"prompt": "Child task"})
            ),
            text_response(),
            tool_response(ToolUseBlock(id="root-call", name="mcp__test__echo", input={"value": 5})),
            text_response(),
        ),
        child_provider_factory=ChildProviders(
            StaticProvider(
                tool_response(ToolUseBlock(id="call", name="mcp__test__echo", input={"value": 2})),
                text_response(),
            )
        ),
    )
    try:
        waiting = await runner.start(AgentRunRequest(input="delegate", run_id="parent"))
        events = [json.loads(line) for line in record.read_text().splitlines()]
        assert sum(event["event"] == "closed" for event in events) == 1
        await runner.resume(
            "parent",
            interaction_id=waiting.pending_interaction.interaction_id,
            response=PermissionInteractionResponse(decision="approve"),
        )
        events = [json.loads(line) for line in record.read_text().splitlines()]
        assert sum(event["event"] == "closed" for event in events) == 2
        result = await runner.start(AgentRunRequest(input="root call", run_id="next"))
        assert result.run.stop_reason is RunStopReason.COMPLETED
    finally:
        await runner.aclose()
    events = [json.loads(line) for line in record.read_text().splitlines()]
    assert sum(event["event"] == "closed" for event in events) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["continue", "cancel", "cancel_unavailable", "expiry"])
async def test_non_live_child_creation_scopes_close_and_cancel_persists_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    peer = MCPPeer(monkeypatch)
    clock = FrozenClock()
    runner = AgentRunner.from_config_path(
        configs(tmp_path),
        provider=StaticProvider(),
        clock=clock,
        child_provider_factory=ChildProviders(StaticProvider(text_response())),
    )
    build_start = AgentRunner._build_start_facts

    def limits(
        self: AgentRunner, request: AgentRunRequest, *, options: AgentRunOptions | None
    ) -> tuple[CreateRun, RuntimeCursor]:
        if self.runtime.environment.agent_config.name == "researcher" and operation == "expiry":
            options = AgentRunOptions(
                limits=RunLimits(deadline_at=clock.now() + timedelta(seconds=1))
            )
        return build_start(self, request, options=options)

    monkeypatch.setattr(AgentRunner, "_build_start_facts", limits)
    admitted_start = AgentRunner._run_admitted_start

    async def crash(*args: object, **kwargs: object) -> object:
        raise IrisRunPersistenceError("after admission")

    monkeypatch.setattr(AgentRunner, "_run_admitted_start", crash)
    prepared = _prepare_parent(runner)
    with pytest.raises(IrisRunPersistenceError):
        await _dispatch(runner, prepared)
    assert peer.events == ["open", "list", "close"]
    monkeypatch.setattr(AgentRunner, "_run_admitted_start", admitted_start)
    child_id = runner.store.load_subagent_link("parent", "delegate").child_run_id
    controller = runner._subagent_controller
    if operation == "continue":
        result = await _dispatch(runner, prepared, linked=True)
        assert not result.is_error
    elif operation == "expiry":
        clock.advance(seconds=2)
        result = await controller._expire_child(
            parent_run_id="parent",
            parent_tool_call_id="delegate",
            child_run_id=child_id,
            selector="researcher",
            owner=SubagentExpiryOwner.CHILD_EFFECTIVE_DEADLINE,
        )
        assert result.error.code == "SUBAGENT_TIMEOUT"
    else:
        peer.fail_open = operation == "cancel_unavailable"
        cancellation = controller.cancel_linked(
            parent_run_id="parent", parent_tool_call_id="delegate", reason="cancel"
        )
        if peer.fail_open:
            with pytest.raises(IrisMCPError):
                await cancellation
        else:
            await cancellation
        assert runner.store.load_run(child_id).cancellation_requested_at is not None
    assert peer.events.count("open") == peer.events.count("close") == 2


@pytest.mark.asyncio
async def test_admission_existing_child_closes_unused_runner_before_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = MCPPeer(monkeypatch)
    runner = AgentRunner.from_config_path(
        configs(tmp_path),
        provider=StaticProvider(),
        child_provider_factory=ChildProviders(StaticProvider()),
    )
    prepared = _prepare_parent(runner)
    admit = runner.store.admit_child_run

    def existing_child(command: AdmitChildRun) -> object:
        create = command.child_create
        other = replace(
            create,
            request=create.request.model_copy(update={"run_id": "existing"}),
            initial_checkpoint=create.initial_checkpoint.model_copy(update={"run_id": "existing"}),
        )
        link = admit(replace(command, child_create=other))
        child = runner.store.load_run(link.child_run_id)
        runner.store.finish_run(
            FinishRun(
                run_id=child.run_id,
                expected_run_revision=child.revision,
                activation_id=child.current_activation_id,
                stop_reason=RunStopReason.CANCELLED,
                now=runner.clock.now(),
            )
        )
        return link

    monkeypatch.setattr(runner.store, "admit_child_run", existing_child)
    continuation = runner._subagent_controller._continue_linked

    async def continue_after_close(*args: object, **kwargs: object) -> object:
        assert peer.events == ["open", "list", "close"]
        return await continuation(*args, **kwargs)

    monkeypatch.setattr(runner._subagent_controller, "_continue_linked", continue_after_close)
    result = await _dispatch(runner, prepared)
    assert result.metadata["child_run_id"] == "existing"


@pytest.mark.asyncio
async def test_deadline_before_child_task_still_closes_fresh_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = MCPPeer(monkeypatch)
    clock = FrozenClock()
    runner = AgentRunner.from_config_path(
        configs(tmp_path),
        provider=StaticProvider(),
        clock=clock,
        child_provider_factory=ChildProviders(StaticProvider()),
    )
    prepared = _prepare_parent(
        runner, options=AgentRunOptions(runtime=RuntimeExecutionOptions(tool_timeout_seconds=1))
    )
    admit = runner.store.admit_child_run

    def slow_admission(command: AdmitChildRun) -> object:
        link = admit(command)
        clock.advance(seconds=2)
        return link

    monkeypatch.setattr(runner.store, "admit_child_run", slow_admission)
    result = await _dispatch(runner, prepared)
    assert result.error.code == "SUBAGENT_TIMEOUT"
    assert peer.events.count("open") == peer.events.count("close") == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting_parent", [False, True])
async def test_parent_cancel_waits_child_mcp_cleanup_and_original_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    waiting_parent: bool,
) -> None:
    peer = MCPPeer(monkeypatch)
    peer.block_call = True
    peer.release_cleanup.clear()
    responses = [tool_response(ToolUseBlock(id="call", name="mcp__test__echo", input={}))]
    if waiting_parent:
        responses.insert(
            0,
            tool_response(
                ToolUseBlock(id="ask", name="ask_question", input={"question": "Continue?"})
            ),
        )
    runner = AgentRunner.from_config_path(
        configs(tmp_path, root_mcp=True),
        provider=_parent_provider(),
        child_provider_factory=ChildProviders(StaticProvider(*responses)),
    )
    manager = SessionManager(runner, "session")
    receipt = await manager.submit("delegate")
    initial = manager._current_task
    if waiting_parent:
        waiting = await initial
        assert peer.events.count("close") == 1
        running = asyncio.create_task(
            manager.resume(
                interaction_id=waiting.pending_interaction.interaction_id,
                response=QuestionInteractionResponse(answer="continue"),
            )
        )
    else:
        running = initial
    await asyncio.wait_for(peer.called.wait(), 1)
    child_id = runner.store.load_subagent_link(receipt.run_id, "delegate").child_run_id
    closing = asyncio.create_task(manager.close(cancel_run=True))
    try:
        await asyncio.wait_for(peer.cleaning.wait(), 1)
        assert not running.done() and not closing.done()
        assert runner.store.load_tool_call(child_id, "call").phase is ToolCallPhase.CLAIMED
        assert peer.events.count("close") == int(waiting_parent)
    finally:
        peer.release_cleanup.set()
        await closing
        result = await running
        await runner.aclose()
    assert result.run.stop_reason is RunStopReason.CANCELLED
    assert runner.store.load_run(child_id).stop_reason is RunStopReason.OUTCOME_UNKNOWN
    assert peer.events.count("close") == peer.events.count("open") == 2 + int(waiting_parent)


@pytest.mark.asyncio
async def test_repeated_outer_cancel_waits_owned_close_and_preserves_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = MCPPeer(monkeypatch)
    peer.release_close.clear()
    runner = AgentRunner.from_config_path(
        configs(tmp_path),
        provider=StaticProvider(),
        child_provider_factory=ChildProviders(StaticProvider(text_response())),
    )
    prepared = _prepare_parent(runner)
    task = asyncio.create_task(_dispatch(runner, prepared))
    await peer.closing.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    peer.release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert peer.events.count("close") == 1


@pytest.mark.asyncio
async def test_child_trusted_read_does_not_override_parent_permission_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ParentPolicy(DefaultPermissionPolicy):
        def check(self, tool: object, params: object, context: object) -> PermissionDecision:
            if tool.name == "mcp__test__echo":
                return PermissionDecision(effect=PermissionEffect.DENY)
            return super().check(tool, params, context)

        def fingerprint_payload(self) -> dict[str, object]:
            return {"policy": "deny-mcp-echo"}

    peer = MCPPeer(monkeypatch)
    runner = AgentRunner.from_config_path(
        configs(tmp_path),
        provider=_parent_provider(),
        permission_policy=ParentPolicy(),
        child_provider_factory=ChildProviders(
            StaticProvider(
                tool_response(ToolUseBlock(id="call", name="mcp__test__echo", input={})),
                text_response(),
            )
        ),
    )
    await runner.start(AgentRunRequest(input="delegate", run_id="parent"))
    child_id = runner.store.load_subagent_link("parent", "delegate").child_run_id
    assert runner.store.load_tool_call(child_id, "call").result.error.code == "PERMISSION_ERROR"
    assert "call:echo" not in peer.events


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["none", "cancel", "deadline", "drift"])
async def test_parent_child_owned_expiry_prepares_then_redispatches_fresh_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    peer = MCPPeer(monkeypatch)
    clock = FrozenClock()
    build_start = AgentRunner._build_start_facts

    def child_expiry(
        self: AgentRunner, request: AgentRunRequest, *, options: AgentRunOptions | None
    ) -> tuple[CreateRun, RuntimeCursor]:
        if self.runtime.environment.agent_config.name == "researcher":
            options = AgentRunOptions(limits=RunLimits(interaction_timeout_seconds=1))
        return build_start(self, request, options=options)

    monkeypatch.setattr(AgentRunner, "_build_start_facts", child_expiry)
    path = configs(tmp_path, trust=False, root_mcp=True)
    store = InMemoryLifecycleStore()
    first = AgentRunner.from_config_path(
        path,
        store=store,
        provider=_parent_provider(),
        clock=clock,
        child_provider_factory=ChildProviders(
            StaticProvider(tool_response(ToolUseBlock(id="call", name="mcp__test__echo", input={})))
        ),
    )
    waiting = await first.start(
        AgentRunRequest(input="delegate", run_id="parent"),
        options=AgentRunOptions(
            limits=RunLimits(
                deadline_at=clock.now() + timedelta(seconds=5) if change == "deadline" else None
            )
        ),
    )
    child_id = waiting.pending_interaction.request.subagent_origin.child_run_id
    await first.aclose()
    clock.advance(seconds=2)
    peer.events.clear()
    peer.opened.clear()
    peer.release_open.clear()
    if change == "drift":
        peer.protocol_version = "2025-11-25"
    second = AgentRunner.from_config_path(
        path,
        store=store,
        provider=StaticProvider(text_response()),
        clock=clock,
        child_provider_factory=ChildProviders(StaticProvider()),
    )
    task = asyncio.create_task(second.recover("parent"))
    try:
        await asyncio.wait_for(peer.opened.wait(), 1)
        if change == "cancel":
            second.request_cancel("parent")
        elif change == "deadline":
            clock.advance(seconds=4)
    finally:
        peer.release_open.set()
    try:
        if change == "drift":
            with pytest.raises(IrisRunRecoveryError):
                await task
            assert store.load_run(child_id).phase is RunPhase.WAITING
        else:
            result = await task
            assert (
                result.run.stop_reason
                is {
                    "none": RunStopReason.COMPLETED,
                    "cancel": RunStopReason.CANCELLED,
                    "deadline": RunStopReason.DEADLINE_EXCEEDED,
                }[change]
            )
            assert store.load_run(child_id).phase is RunPhase.TERMINAL
        assert peer.events == ["open", "list"]
        assert store.load_interaction(waiting.pending_interaction.interaction_id).response is None
    finally:
        await second.aclose()


@pytest.mark.asyncio
async def test_fresh_child_prepare_keeps_original_parent_activation_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """准备期间 parent 被接管后，旧调用不能借用新 activation 创建 child。"""
    peer = MCPPeer(monkeypatch)
    peer.release_open.clear()
    providers = ChildProviders(StaticProvider(text_response()))
    runner = AgentRunner.from_config_path(
        configs(tmp_path), provider=StaticProvider(), child_provider_factory=providers
    )
    prepared = _prepare_parent(runner)
    task = asyncio.create_task(_dispatch(runner, prepared))
    await peer.opened.wait()
    run = runner.store.load_run("parent")
    checkpoint = runner.store.load_checkpoint("parent")
    runner.store.recover_active_run(
        RecoverActiveRun(
            run_id="parent",
            expected_run_revision=run.revision,
            expected_activation_id=run.current_activation_id,
            expected_checkpoint_sequence=checkpoint.sequence,
            recovery_disposition=RecoveryDisposition.RESUME,
            new_activation_id="replacement",
            now=runner.clock.now(),
        )
    )
    peer.release_open.set()
    with pytest.raises(IrisRunConflictError):
        await task
    assert runner.store.load_subagent_link("parent", "delegate") is None
    assert not providers.provider.requests
    assert peer.events == ["open", "list", "close"]
