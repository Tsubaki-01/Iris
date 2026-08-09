"""Phase 2 工具 effect guard 与 cancellation 顺序测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris.exceptions import (
    IrisCancellationRequestedError,
    IrisRunConflictError,
    IrisRunPersistenceError,
    IrisRunStateError,
)
from iris.lifecycle import RuntimeExecutionOptions
from iris.message import Msg, ToolUseBlock
from iris.runtime import (
    CommitPortToolEffectGuard,
    RuntimeActivationInput,
    RuntimeCursor,
    RuntimeToolCall,
    ToolCallClaim,
)
from iris.tools import (
    DefaultPermissionPolicy,
    PreparedToolCall,
    ToolCapability,
    ToolExecutionContext,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
)


class MutableCancellationSignal:
    """测试用 activation-scope cancellation signal。"""

    def __init__(self) -> None:
        self.requested = False

    def raise_if_requested(self) -> None:
        if self.requested:
            raise IrisCancellationRequestedError("activation 已取消")


class RecordingGuard:
    """记录 guard 顺序并可在 claim 后触发取消。"""

    def __init__(
        self,
        events: list[str],
        *,
        signal: MutableCancellationSignal | None = None,
        error: Exception | None = None,
    ) -> None:
        self.events = events
        self.signal = signal
        self.error = error

    def before_effect(self, prepared: object) -> None:
        del prepared
        self.events.append("guard")
        if self.error is not None:
            raise self.error
        if self.signal is not None:
            self.signal.requested = True


class ClaimOnlyPort:
    """只实现 guard 测试所需 claim 方法的严格 fake。"""

    def __init__(self) -> None:
        self.calls: list[object] = []

    def claim_tool_call(self, call: RuntimeToolCall) -> ToolCallClaim:
        self.calls.append(call)
        return ToolCallClaim(
            run_id=call.run_id,
            activation_id=call.activation_id,
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            fingerprint=call.fingerprint,
            tool_version=2,
        )


def _prepared_call(
    tmp_path: Path,
    *,
    write_mode: str = "allow",
) -> tuple[ToolExecutor, PreparedToolCall, ToolExecutionContext]:
    registry = ToolRegistry()

    def echo(value: str) -> str:
        return value

    registry.register_function(
        echo,
        name="echo",
        description="回显",
        capabilities={ToolCapability.WRITE},
    )
    executor = ToolExecutor(
        registry,
        permission_policy=DefaultPermissionPolicy(write_mode=write_mode),
    )
    context = ToolExecutionContext(
        workspace_root=tmp_path,
        session_id="session_1",
        metadata={"run_id": "run_1"},
    )
    prepared = executor.prepare_many(
        [ToolUseBlock(id="call_1", name="echo", input={"value": "hello"})],
        context,
    ).calls[0]
    return executor, prepared, context


@pytest.mark.asyncio
async def test_effect_guard_runs_after_revalidation_before_middleware_and_tool(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    def effect(value: str) -> str:
        events.append("tool")
        return value

    class Middleware:
        def before_call(self, *args: object) -> None:
            del args
            events.append("middleware")

    registry = ToolRegistry()
    registry.register_function(effect, description="执行 effect")
    executor = ToolExecutor(registry, middleware=[Middleware()])
    context = ToolExecutionContext(workspace_root=tmp_path)
    prepared = executor.prepare_many(
        [ToolUseBlock(id="call_1", name="effect", input={"value": "hello"})],
        context,
    ).calls[0]

    result = await executor.execute_prepared(
        prepared,
        context,
        effect_guard=RecordingGuard(events),
    )

    assert result.model_content == "hello"
    assert events == ["guard", "middleware", "tool"]


@pytest.mark.asyncio
async def test_preflight_error_returns_without_effect_guard(tmp_path: Path) -> None:
    executor = ToolExecutor(ToolRegistry())
    context = ToolExecutionContext(workspace_root=tmp_path)
    prepared = executor.prepare_many(
        [ToolUseBlock(id="call_1", name="missing", input={})],
        context,
    ).calls[0]
    events: list[str] = []

    result = await executor.execute_prepared(
        prepared,
        context,
        effect_guard=RecordingGuard(events),
    )

    assert result.error is not None
    assert result.error.code == "NOT_FOUND"
    assert events == []


@pytest.mark.asyncio
async def test_guard_failure_propagates_without_tool_effect(tmp_path: Path) -> None:
    effects: list[str] = []
    registry = ToolRegistry()
    registry.register_function(
        lambda: effects.append("tool") or "done",
        name="effect",
        description="执行 effect",
    )
    executor = ToolExecutor(registry)
    context = ToolExecutionContext(workspace_root=tmp_path)
    prepared = executor.prepare_many(
        [ToolUseBlock(id="call_1", name="effect", input={})],
        context,
    ).calls[0]

    with pytest.raises(IrisRunPersistenceError):
        await executor.execute_prepared(
            prepared,
            context,
            effect_guard=RecordingGuard(
                [],
                error=IrisRunPersistenceError("claim failed"),
            ),
        )

    assert effects == []


@pytest.mark.asyncio
async def test_cancellation_after_claim_propagates_before_tool_body(tmp_path: Path) -> None:
    effects: list[str] = []
    signal = MutableCancellationSignal()
    registry = ToolRegistry()
    registry.register_function(
        lambda: effects.append("tool") or "done",
        name="effect",
        description="执行 effect",
    )
    executor = ToolExecutor(registry)
    context = ToolExecutionContext(workspace_root=tmp_path, cancellation=signal)
    prepared = executor.prepare_many(
        [ToolUseBlock(id="call_1", name="effect", input={})],
        context,
    ).calls[0]

    with pytest.raises(IrisCancellationRequestedError):
        await executor.execute_prepared(
            prepared,
            context,
            effect_guard=RecordingGuard([], signal=signal),
        )

    assert effects == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("write_mode", "approved_tool_call_id"),
    [("allow", None), ("confirm", "call_1")],
)
async def test_ordinary_and_approved_tools_share_effect_guard_path(
    tmp_path: Path,
    write_mode: str,
    approved_tool_call_id: str | None,
) -> None:
    executor, prepared, context = _prepared_call(tmp_path, write_mode=write_mode)
    events: list[str] = []

    result = await executor.execute_prepared(
        prepared,
        context,
        approved_tool_call_id=approved_tool_call_id,
        effect_guard=RecordingGuard(events),
    )

    assert result.is_error is False
    assert events == ["guard"]


def test_commit_port_guard_claims_exact_subject_once(tmp_path: Path) -> None:
    executor, prepared, _ = _prepared_call(tmp_path)
    del executor
    cursor = RuntimeCursor(
        position="tool_batch",
        step_index=2,
        tool_calls=(prepared.tool_use,),
        assistant_message=Msg.assistant([prepared.tool_use]),
    )
    activation = RuntimeActivationInput(
        run_id="run_1",
        activation_id="activation_1",
        session_id="session_1",
        kind="resume",
        input=None,
        cursor=cursor,
        options=RuntimeExecutionOptions(),
    )
    port = ClaimOnlyPort()
    guard = CommitPortToolEffectGuard(
        activation=activation,
        cursor=cursor,
        commits=port,
        workspace_root=tmp_path,
    )

    guard.before_effect(prepared)
    guard.before_effect(prepared)

    claim = guard.claim_for("call_1")
    assert claim is not None
    assert claim.tool_version == 2
    assert len(port.calls) == 1


def test_commit_port_guard_claims_indexed_uncommitted_suffix(tmp_path: Path) -> None:
    """显式索引只选择当前 batch 未提交后缀中的 exact subject。"""
    executor, first, context = _prepared_call(tmp_path)
    second_use = ToolUseBlock(id="call_2", name="echo", input={"value": "later"})
    second = executor.prepare_many([second_use], context).calls[0]
    cursor = RuntimeCursor(
        position="tool_batch",
        step_index=2,
        next_tool_index=0,
        tool_calls=(first.tool_use, second.tool_use),
        assistant_message=Msg.assistant([first.tool_use, second.tool_use]),
    )
    activation = RuntimeActivationInput(
        run_id="run_1",
        activation_id="activation_1",
        session_id="session_1",
        kind="resume",
        input=None,
        cursor=cursor,
        options=RuntimeExecutionOptions(),
    )
    port = ClaimOnlyPort()
    guard = CommitPortToolEffectGuard(
        activation=activation,
        cursor=cursor,
        commits=port,
        workspace_root=tmp_path,
        tool_index=1,
    )

    guard.before_effect(second)

    call = port.calls[0]
    assert isinstance(call, RuntimeToolCall)
    assert call.tool_call_id == "call_2"
    assert call.ordinal == 2
    with pytest.raises(IrisRunConflictError, match="subject"):
        guard.before_effect(first)


def test_commit_port_guard_rejects_committed_or_out_of_batch_index(tmp_path: Path) -> None:
    """显式索引不能回退到已提交 prefix，也不能跨出当前 batch。"""
    _, prepared, _ = _prepared_call(tmp_path)
    cursor = RuntimeCursor(
        position="tool_batch",
        step_index=2,
        next_tool_index=1,
        tool_calls=(prepared.tool_use, prepared.tool_use.model_copy(update={"id": "call_2"})),
        tool_results=(ToolResult(tool_use_id="call_1", tool_name="echo"),),
        assistant_message=Msg.assistant(
            [prepared.tool_use, prepared.tool_use.model_copy(update={"id": "call_2"})]
        ),
    )
    activation = RuntimeActivationInput(
        run_id="run_1",
        activation_id="activation_1",
        session_id="session_1",
        kind="resume",
        input=None,
        cursor=cursor,
        options=RuntimeExecutionOptions(),
    )

    with pytest.raises(IrisRunStateError, match="索引"):
        CommitPortToolEffectGuard(
            activation=activation,
            cursor=cursor,
            commits=ClaimOnlyPort(),
            workspace_root=tmp_path,
            tool_index=0,
        )
    with pytest.raises(IrisRunStateError, match="索引"):
        CommitPortToolEffectGuard(
            activation=activation,
            cursor=cursor,
            commits=ClaimOnlyPort(),
            workspace_root=tmp_path,
            tool_index=2,
        )
