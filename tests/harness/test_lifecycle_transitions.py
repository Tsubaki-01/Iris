"""定义 Phase 1 lifecycle aggregate 必须满足的状态转换 contract。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from importlib import import_module
from types import ModuleType
from typing import Any

import pytest

from iris.hitl import (
    HumanInteraction,
    HumanInteractionRequest,
    InteractionStatus,
    QuestionInteractionResponse,
    QuestionPrompt,
    ToolCallSnapshot,
    make_call_fingerprint,
)
from iris.message import Msg, TextBlock, ToolUseBlock
from iris.tools import ToolErrorInfo, ToolResult

_NOW = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
_T1 = _NOW + timedelta(seconds=1)
_T2 = _NOW + timedelta(seconds=2)
_T3 = _NOW + timedelta(seconds=3)
_ENVIRONMENT_FINGERPRINT = "environment-v1"
_TOOL_FINGERPRINT = "a" * 64


def _contracts() -> tuple[ModuleType, Any]:
    """加载 Phase 1 将提供的 contract 包与 reference store。"""
    lifecycle = import_module("iris.lifecycle")
    store_module = import_module("iris.store")
    return lifecycle, store_module.InMemoryLifecycleStore()


def _checkpoint(
    lifecycle: ModuleType,
    *,
    sequence: int,
    activation_id: str,
    session_revision: int,
    reserved: int = 0,
    committed: int = 0,
    resumability: str = "safe",
) -> Any:
    """构造手工确定的 lifecycle checkpoint。"""
    return lifecycle.RunCheckpoint(
        run_id="run-1",
        sequence=sequence,
        activation_id=activation_id,
        engine_cursor={"step_index": committed},
        session_revision=session_revision,
        model_steps_reserved=reserved,
        model_steps_committed=committed,
        environment_fingerprint=_ENVIRONMENT_FINGERPRINT,
        resumability=resumability,
    )


def _create_run(
    lifecycle: ModuleType,
    store: Any,
    *,
    deadline_at: datetime | None = None,
) -> Any:
    """通过公开 command 原子创建一个固定 logical run。"""
    return store.create_run(
        _create_run_command(
            lifecycle,
            run_id="run-1",
            session_id="session-1",
            activation_id="activation-1",
            deadline_at=deadline_at,
            now=_NOW,
        )
    )


def _create_run_command(
    lifecycle: ModuleType,
    *,
    run_id: str,
    session_id: str,
    activation_id: str,
    now: datetime,
    deadline_at: datetime | None = None,
) -> Any:
    """构造可用于 lane/identity contract 的固定 CreateRun command。"""
    options = lifecycle.AgentRunOptions(
        limits=lifecycle.RunLimits(deadline_at=deadline_at),
    )
    checkpoint = lifecycle.RunCheckpoint(
        run_id=run_id,
        sequence=1,
        activation_id=activation_id,
        engine_cursor={"step_index": 0},
        session_revision=0,
        model_steps_reserved=0,
        model_steps_committed=0,
        environment_fingerprint=_ENVIRONMENT_FINGERPRINT,
        resumability="safe",
    )
    return lifecycle.CreateRun(
        request=lifecycle.AgentRunRequest(
            input="start",
            session_id=session_id,
            run_id=run_id,
        ),
        options=options,
        agent_id="agent-1",
        environment_fingerprint=_ENVIRONMENT_FINGERPRINT,
        start_activation_id=activation_id,
        initial_checkpoint=checkpoint,
        now=now,
    )


def _pending_interaction() -> HumanInteraction:
    """构造 active -> waiting command 使用的固定 question interaction。"""
    arguments = {"question": "继续吗？"}
    fingerprint = make_call_fingerprint(
        session_id="session-1",
        run_id="run-1",
        tool_call_id="call-question",
        tool_name="ask_question",
        arguments=arguments,
        workspace_root="workspace",
    )
    request = HumanInteractionRequest(
        tool_call=ToolCallSnapshot(
            tool_call_id="call-question",
            tool_name="ask_question",
            arguments=arguments,
            workspace_root="workspace",
            fingerprint=fingerprint,
        ),
        prompt=QuestionPrompt(question="继续吗？"),
    )
    return HumanInteraction.model_construct(
        interaction_id="interaction-1",
        session_id="session-1",
        run_id="run-1",
        step_index=0,
        tool_call_id="call-question",
        status=InteractionStatus.PENDING,
        request=request,
        response=None,
        checkpoint={},
        version=1,
        expires_at=_NOW + timedelta(minutes=5),
        created_at=_T1,
        resolved_at=None,
        closed_at=None,
        close_reason=None,
    )


def _suspend_run(lifecycle: ModuleType, store: Any, active: Any) -> Any:
    """把固定 active run 原子转换为 waiting。"""
    interaction = _pending_interaction()
    prepared = lifecycle.RunToolCallRecord(
        run_id="run-1",
        step_index=0,
        ordinal=1,
        tool_call_id=interaction.tool_call_id,
        tool_name=interaction.request.tool_call.tool_name,
        arguments=interaction.request.tool_call.arguments,
        fingerprint=interaction.request.tool_call.fingerprint,
        phase="prepared",
        version=1,
        created_at=_T1,
        updated_at=_T1,
    )
    return store.suspend_run(
        lifecycle.SuspendRun(
            run_id="run-1",
            expected_run_revision=active.run.revision,
            activation_id="activation-1",
            expected_session_revision=0,
            message_delta=[],
            prepared_tool_calls=[prepared],
            checkpoint=_checkpoint(
                lifecycle,
                sequence=2,
                activation_id="activation-1",
                session_revision=0,
            ),
            pending_interaction=interaction,
            assistant_message=None,
            usage=active.run.usage,
            now=_T1,
        )
    )


def _finish_run(
    lifecycle: ModuleType,
    store: Any,
    current: Any,
    *,
    stop_reason: str,
    activation_id: str | None,
    interaction_close_reason: str | None = None,
) -> Any:
    """通过目标 command 终止 active 或 waiting run。"""
    return store.finish_run(
        lifecycle.FinishRun(
            run_id="run-1",
            expected_run_revision=current.run.revision,
            activation_id=activation_id,
            stop_reason=stop_reason,
            assistant_message=Msg.assistant("done") if stop_reason == "completed" else None,
            error=None,
            interaction_close_reason=interaction_close_reason,
            now=_T3,
        )
    )


@pytest.mark.parametrize(
    ("deadline_at", "expected_phase", "expected_reason", "expected_event_kinds"),
    [
        (None, "active", None, ["run.started", "activation.started"]),
        (_NOW, "terminal", "deadline_exceeded", ["run.terminal"]),
    ],
    ids=["active-start", "expired-start"],
)
def test_create_run_transition_matrix_target_contract(
    deadline_at: datetime | None,
    expected_phase: str,
    expected_reason: str | None,
    expected_event_kinds: list[str],
) -> None:
    """定义 absent -> active 与 expired absent -> terminal 的原子事实。"""
    lifecycle, store = _contracts()

    commit = _create_run(lifecycle, store, deadline_at=deadline_at)

    assert commit.run.phase == expected_phase
    assert commit.run.stop_reason == expected_reason
    assert [event.sequence for event in commit.events] == list(range(1, len(commit.events) + 1))
    assert [event.kind for event in commit.events] == expected_event_kinds
    if expected_phase == "active":
        assert commit.run.current_activation_id == "activation-1"
        assert commit.checkpoint is not None
        assert commit.result is None
    else:
        assert commit.run.current_activation_id is None
        assert commit.checkpoint is None
        assert commit.result is not None


def test_active_waiting_active_terminal_transition_matrix_target_contract() -> None:
    """定义同一 logical run 跨 interaction 创建新 activation 后终止的状态链。"""
    lifecycle, store = _contracts()
    active = _create_run(lifecycle, store)

    waiting = _suspend_run(lifecycle, store, active)
    assert waiting.run.phase == "waiting"
    assert waiting.run.current_activation_id is None
    assert waiting.run.pending_interaction_id == "interaction-1"
    assert waiting.result is not None
    assert waiting.result.pending_interaction is not None

    fingerprint = waiting.interaction.request.tool_call.fingerprint
    resolved = store.resolve_interaction(
        lifecycle.ResolveInteraction(
            run_id="run-1",
            expected_run_revision=waiting.run.revision,
            interaction_id="interaction-1",
            expected_interaction_version=waiting.interaction.version,
            response=QuestionInteractionResponse(answer="继续"),
            expected_fingerprint=fingerprint,
            now=_T2,
        )
    )
    resumed = store.begin_activation(
        lifecycle.BeginActivation(
            run_id="run-1",
            expected_run_revision=resolved.run.revision,
            new_activation_id="activation-2",
            kind="resume",
            expected_checkpoint_sequence=resolved.checkpoint.sequence,
            now=_T2,
        )
    )
    assert resumed.run.phase == "active"
    assert resumed.run.current_activation_id == "activation-2"
    assert resumed.run.pending_interaction_id is None
    assert resumed.checkpoint.activation_id == "activation-2"

    terminal = _finish_run(
        lifecycle,
        store,
        resumed,
        stop_reason="completed",
        activation_id="activation-2",
    )
    assert terminal.run.phase == "terminal"
    assert terminal.run.stop_reason == "completed"
    assert terminal.run.current_activation_id is None
    assert terminal.result is not None
    events = store.list_events("run-1")
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert events[-1].kind == "run.terminal"


@pytest.mark.parametrize("waiting", [False, True], ids=["active-lane", "waiting-lane"])
def test_non_terminal_run_keeps_exclusive_session_lane_target_contract(
    waiting: bool,
) -> None:
    """定义 active/waiting run 持有 session 唯一可写 lane。"""
    lifecycle, store = _contracts()
    current = _create_run(lifecycle, store)
    if waiting:
        current = _suspend_run(lifecycle, store, current)
    errors = import_module("iris.exceptions")
    events_before = store.list_events("run-1")

    with pytest.raises(errors.IrisRunConflictError):
        store.create_run(
            _create_run_command(
                lifecycle,
                run_id="run-2",
                session_id="session-1",
                activation_id="activation-2",
                now=_T2,
            )
        )

    assert store.load_run("run-2") is None
    assert store.load_run("run-1") == current.run
    assert store.list_events("run-1") == events_before


@pytest.mark.parametrize(
    ("expected_revision_delta", "activation_id"),
    [(-1, "activation-1"), (0, "activation-stale")],
    ids=["stale-run-revision", "stale-activation-fence"],
)
def test_active_mutation_rejects_stale_cas_target_contract(
    expected_revision_delta: int,
    activation_id: str,
) -> None:
    """定义 active mutation 同时检查 run revision 与 activation fence。"""
    lifecycle, store = _contracts()
    active = _create_run(lifecycle, store)
    errors = import_module("iris.exceptions")
    events_before = store.list_events("run-1")

    with pytest.raises(errors.IrisRunConflictError):
        store.reserve_model_step(
            lifecycle.ReserveModelStep(
                run_id="run-1",
                expected_run_revision=active.run.revision + expected_revision_delta,
                activation_id=activation_id,
                now=_T1,
            )
        )

    assert store.load_run("run-1") == active.run
    assert store.list_events("run-1") == events_before


def test_interaction_resolution_rejects_stale_version_target_contract() -> None:
    """定义 interaction response CAS 冲突不产生部分写。"""
    lifecycle, store = _contracts()
    waiting = _suspend_run(lifecycle, store, _create_run(lifecycle, store))
    errors = import_module("iris.exceptions")
    events_before = store.list_events("run-1")

    with pytest.raises(errors.IrisRunConflictError):
        store.resolve_interaction(
            lifecycle.ResolveInteraction(
                run_id="run-1",
                expected_run_revision=waiting.run.revision,
                interaction_id="interaction-1",
                expected_interaction_version=waiting.interaction.version + 1,
                response=QuestionInteractionResponse(answer="继续"),
                expected_fingerprint=waiting.interaction.request.tool_call.fingerprint,
                now=_T2,
            )
        )

    assert store.load_run("run-1") == waiting.run
    assert store.load_interaction("interaction-1") == waiting.interaction
    assert store.list_events("run-1") == events_before


def test_waiting_can_finish_deterministically_target_contract() -> None:
    """定义 waiting -> terminal 会关闭 interaction 并释放 run lane。"""
    lifecycle, store = _contracts()
    waiting = _suspend_run(lifecycle, store, _create_run(lifecycle, store))

    terminal = _finish_run(
        lifecycle,
        store,
        waiting,
        stop_reason="cancelled",
        activation_id=None,
        interaction_close_reason="cancelled",
    )

    assert terminal.run.phase == "terminal"
    assert terminal.run.pending_interaction_id is None
    assert terminal.interaction.status == "closed"
    assert terminal.result is not None
    next_run = store.create_run(
        _create_run_command(
            lifecycle,
            run_id="run-2",
            session_id="session-1",
            activation_id="activation-next",
            now=_T3,
        )
    )
    assert next_run.run.phase == "active"


def test_terminal_run_rejects_further_mutation_target_contract() -> None:
    """定义 terminal -> * 永远非法且不能产生新 event。"""
    lifecycle, store = _contracts()
    active = _create_run(lifecycle, store)
    terminal = _finish_run(
        lifecycle,
        store,
        active,
        stop_reason="completed",
        activation_id="activation-1",
    )
    errors = import_module("iris.exceptions")
    events_before = store.list_events("run-1")

    with pytest.raises(errors.IrisRunStateError):
        store.reserve_model_step(
            lifecycle.ReserveModelStep(
                run_id="run-1",
                expected_run_revision=terminal.run.revision,
                activation_id="activation-1",
                now=_T3,
            )
        )

    assert store.load_run("run-1") == terminal.run
    assert store.list_events("run-1") == events_before


def _prepare_tool_call(lifecycle: ModuleType, store: Any) -> Any:
    """提交一次 model step，使固定工具调用进入 prepared。"""
    created = _create_run(lifecycle, store)
    reserved = store.reserve_model_step(
        lifecycle.ReserveModelStep(
            run_id="run-1",
            expected_run_revision=created.run.revision,
            activation_id="activation-1",
            now=_T1,
        )
    )
    return store.commit_model_step(
        _commit_model_step_command(
            lifecycle,
            reserved,
            expected_session_revision=0,
        )
    )


def _commit_model_step_command(
    lifecycle: ModuleType,
    reserved: Any,
    *,
    expected_session_revision: int,
) -> Any:
    """构造可用于 session revision contract 的固定 model-step command。"""
    assistant = Msg.assistant([ToolUseBlock(id="call-tool", name="probe", input={"value": "A"})])
    prepared = lifecycle.RunToolCallRecord(
        run_id="run-1",
        step_index=0,
        ordinal=1,
        tool_call_id="call-tool",
        tool_name="probe",
        arguments={"value": "A"},
        fingerprint=_TOOL_FINGERPRINT,
        interaction_id=None,
        phase="prepared",
        claim_activation_id=None,
        result=None,
        version=1,
        created_at=_T1,
        updated_at=_T1,
        claimed_at=None,
        committed_at=None,
    )
    return lifecycle.CommitModelStep(
        run_id="run-1",
        expected_run_revision=reserved.run.revision,
        activation_id="activation-1",
        expected_session_revision=expected_session_revision,
        message_delta=[assistant],
        usage=lifecycle.RunUsage(
            model_steps_reserved=1,
            model_steps_committed=1,
        ),
        prepared_tool_calls=[prepared],
        checkpoint=_checkpoint(
            lifecycle,
            sequence=2,
            activation_id="activation-1",
            session_revision=1,
            reserved=1,
            committed=1,
        ),
        assistant_message=assistant,
        now=_T1,
    )


def _commit_tool_result_command(
    lifecycle: ModuleType,
    current: Any,
    *,
    result: ToolResult,
    expected_tool_version: int,
) -> Any:
    """构造 prepared/claimed -> committed 共用的固定 command。"""
    return lifecycle.CommitToolResult(
        run_id="run-1",
        expected_run_revision=current.run.revision,
        activation_id="activation-1",
        expected_session_revision=1,
        tool_call_id="call-tool",
        expected_tool_version=expected_tool_version,
        result=result,
        message_delta=[
            Msg.tool_result(
                tool_use_id="call-tool",
                name="probe",
                content=result.model_content,
                is_error=result.is_error,
            )
        ],
        checkpoint=_checkpoint(
            lifecycle,
            sequence=3,
            activation_id="activation-1",
            session_revision=2,
            reserved=1,
            committed=1,
        ),
        now=_T2,
    )


def test_history_mutation_rejects_stale_session_revision_target_contract() -> None:
    """定义 session delta 同时检查 lane owner 与 session revision。"""
    lifecycle, store = _contracts()
    created = _create_run(lifecycle, store)
    reserved = store.reserve_model_step(
        lifecycle.ReserveModelStep(
            run_id="run-1",
            expected_run_revision=created.run.revision,
            activation_id="activation-1",
            now=_T1,
        )
    )
    errors = import_module("iris.exceptions")
    session_before = store.load_session("session-1")
    events_before = store.list_events("run-1")

    with pytest.raises(errors.IrisRunConflictError):
        store.commit_model_step(
            _commit_model_step_command(
                lifecycle,
                reserved,
                expected_session_revision=session_before.revision + 1,
            )
        )

    assert store.load_run("run-1") == reserved.run
    assert store.load_session("session-1") == session_before
    assert store.list_tool_calls("run-1") == []
    assert store.list_events("run-1") == events_before


def test_tool_claim_rejects_stale_version_target_contract() -> None:
    """定义 tool version CAS 冲突不进入 claimed。"""
    lifecycle, store = _contracts()
    prepared = _prepare_tool_call(lifecycle, store)
    errors = import_module("iris.exceptions")
    events_before = store.list_events("run-1")

    with pytest.raises(errors.IrisRunConflictError):
        store.claim_tool_call(
            lifecycle.ClaimToolCall(
                run_id="run-1",
                expected_run_revision=prepared.run.revision,
                activation_id="activation-1",
                tool_call_id="call-tool",
                fingerprint=_TOOL_FINGERPRINT,
                expected_tool_version=2,
                now=_T2,
            )
        )

    assert store.list_tool_calls("run-1")[0].phase == "prepared"
    assert store.load_run("run-1") == prepared.run
    assert store.list_events("run-1") == events_before


def test_tool_prepared_claimed_committed_transition_target_contract() -> None:
    """定义有副作用工具必须先 durable claim，再提交精确 result。"""
    lifecycle, store = _contracts()
    prepared = _prepare_tool_call(lifecycle, store)

    claimed = store.claim_tool_call(
        lifecycle.ClaimToolCall(
            run_id="run-1",
            expected_run_revision=prepared.run.revision,
            activation_id="activation-1",
            tool_call_id="call-tool",
            fingerprint=_TOOL_FINGERPRINT,
            expected_tool_version=1,
            now=_T2,
        )
    )
    claimed_call = store.list_tool_calls("run-1")[0]
    assert claimed_call.phase == "claimed"
    assert claimed_call.claim_activation_id == "activation-1"

    result = ToolResult(
        tool_use_id="call-tool",
        tool_name="probe",
        content=[TextBlock(text="done")],
    )
    committed = store.commit_tool_result(
        _commit_tool_result_command(
            lifecycle,
            claimed,
            result=result,
            expected_tool_version=claimed_call.version,
        )
    )

    committed_call = store.list_tool_calls("run-1")[0]
    assert committed_call.phase == "committed"
    assert committed_call.result == result
    assert committed.run.usage.tool_calls_committed == 1


@pytest.mark.parametrize(
    ("result", "should_commit"),
    [
        (
            ToolResult(
                tool_use_id="call-tool",
                tool_name="probe",
                is_error=True,
                error=ToolErrorInfo(code="TOOL_NOT_ALLOWED", message="preflight rejected"),
            ),
            True,
        ),
        (
            ToolResult(
                tool_use_id="call-tool",
                tool_name="probe",
                content=[TextBlock(text="effect result")],
            ),
            False,
        ),
    ],
    ids=["preflight-result", "effect-without-claim"],
)
def test_prepared_tool_commit_guard_target_contract(
    result: ToolResult,
    should_commit: bool,
) -> None:
    """定义 prepared 仅允许无副作用结果直接提交。"""
    lifecycle, store = _contracts()
    prepared = _prepare_tool_call(lifecycle, store)
    command = _commit_tool_result_command(
        lifecycle,
        prepared,
        result=result,
        expected_tool_version=1,
    )

    if should_commit:
        committed = store.commit_tool_result(command)
        assert store.list_tool_calls("run-1")[0].phase == "committed"
        assert committed.run.usage.tool_calls_committed == 1
        return

    errors = import_module("iris.exceptions")
    with pytest.raises(errors.IrisRunStateError):
        store.commit_tool_result(command)
    assert store.list_tool_calls("run-1")[0].phase == "prepared"


def test_claimed_tool_recovery_marks_outcome_unknown_target_contract() -> None:
    """定义 claimed 且无 durable result 的恢复必须 fail closed，不能重放。"""
    lifecycle, store = _contracts()
    prepared = _prepare_tool_call(lifecycle, store)
    claimed = store.claim_tool_call(
        lifecycle.ClaimToolCall(
            run_id="run-1",
            expected_run_revision=prepared.run.revision,
            activation_id="activation-1",
            tool_call_id="call-tool",
            fingerprint=_TOOL_FINGERPRINT,
            expected_tool_version=1,
            now=_T2,
        )
    )

    recovered = store.recover_active_run(
        lifecycle.RecoverActiveRun(
            run_id="run-1",
            expected_run_revision=claimed.run.revision,
            expected_activation_id="activation-1",
            expected_checkpoint_sequence=claimed.checkpoint.sequence,
            recovery_disposition="outcome_unknown",
            new_activation_id=None,
            now=_T3,
        )
    )

    assert recovered.run.phase == "terminal"
    assert recovered.run.stop_reason == "outcome_unknown"
    assert store.list_tool_calls("run-1")[0].phase == "outcome_unknown"
    assert recovered.result is not None
