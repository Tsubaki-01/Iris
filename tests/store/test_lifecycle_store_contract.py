"""所有 ``LifecycleStore`` 实现必须共享的 aggregate contract。"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import pytest

from iris.exceptions import IrisRunConflictError, IrisRunRecoveryError, IrisRunStateError
from iris.hitl import (
    HumanInteraction,
    HumanInteractionRequest,
    InteractionStatus,
    QuestionInteractionResponse,
    QuestionPrompt,
    ToolCallSnapshot,
)
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
    BeginActivation,
    ClaimToolCall,
    CommitModelStep,
    CommitToolResult,
    CreateRun,
    FinishRun,
    LifecycleStore,
    RecoverActiveRun,
    RecoveryDisposition,
    RequestCancellation,
    ReserveModelStep,
    ResolveInteraction,
    RunCheckpoint,
    RunCommit,
    RunLimits,
    RunToolCallRecord,
    RunUsage,
    SuspendRun,
    project_result,
)
from iris.message import Msg, TextBlock, ToolUseBlock
from iris.store import InMemoryLifecycleStore
from iris.tools import ToolErrorInfo, ToolResult

_NOW = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
_T1 = _NOW + timedelta(seconds=1)
_T2 = _NOW + timedelta(seconds=2)
_T3 = _NOW + timedelta(seconds=3)
_ENVIRONMENT_FINGERPRINT = "environment-v1"
_TOOL_FINGERPRINT = "a" * 64
_INTERACTION_ID = "int_" + "1" * 32


class _StoreFactory(Protocol):
    def __call__(self) -> LifecycleStore: ...


@pytest.fixture(params=[InMemoryLifecycleStore], ids=["memory"])
def lifecycle_store(request: pytest.FixtureRequest) -> LifecycleStore:
    """参数化可复用 contract 当前已实现的 concrete stores。"""
    factory: _StoreFactory = request.param
    return factory()


def _checkpoint(
    *,
    run_id: str,
    sequence: int,
    activation_id: str,
    session_revision: int,
    reserved: int = 0,
    committed: int = 0,
) -> RunCheckpoint:
    return RunCheckpoint(
        run_id=run_id,
        sequence=sequence,
        activation_id=activation_id,
        engine_cursor={"step_index": committed},
        session_revision=session_revision,
        model_steps_reserved=reserved,
        model_steps_committed=committed,
        environment_fingerprint=_ENVIRONMENT_FINGERPRINT,
        resumability="safe",
    )


def _create_command(
    *,
    run_id: str = "run-1",
    session_id: str = "session-1",
    activation_id: str = "activation-1",
    max_model_steps: int = 20,
    metadata: dict[str, object] | None = None,
) -> CreateRun:
    return CreateRun(
        request=AgentRunRequest(
            input="start",
            session_id=session_id,
            run_id=run_id,
            metadata={} if metadata is None else metadata,
        ),
        options=AgentRunOptions(limits=RunLimits(max_model_steps=max_model_steps)),
        agent_id="agent-1",
        environment_fingerprint=_ENVIRONMENT_FINGERPRINT,
        start_activation_id=activation_id,
        initial_checkpoint=_checkpoint(
            run_id=run_id,
            sequence=1,
            activation_id=activation_id,
            session_revision=0,
        ),
        now=_NOW,
    )


def _create(store: LifecycleStore, **kwargs: object) -> RunCommit:
    return store.create_run(_create_command(**kwargs))


def _interaction() -> HumanInteraction:
    return HumanInteraction(
        interaction_id=_INTERACTION_ID,
        session_id="session-1",
        run_id="run-1",
        step_index=0,
        status=InteractionStatus.PENDING,
        request=HumanInteractionRequest(
            tool_call=ToolCallSnapshot(
                tool_call_id="call-question",
                tool_name="ask_question",
                arguments={"question": "继续吗？"},
                workspace_root="workspace",
                fingerprint=_TOOL_FINGERPRINT,
            ),
            prompt=QuestionPrompt(question="继续吗？"),
        ),
        checkpoint={},
        expires_at=_NOW + timedelta(minutes=5),
        created_at=_T1,
    )


def _suspend(store: LifecycleStore, created: RunCommit) -> RunCommit:
    return store.suspend_run(
        SuspendRun(
            run_id="run-1",
            expected_run_revision=created.run.revision,
            activation_id="activation-1",
            expected_session_revision=0,
            message_delta=[],
            prepared_tool_calls=[],
            checkpoint=_checkpoint(
                run_id="run-1",
                sequence=2,
                activation_id="activation-1",
                session_revision=0,
            ),
            pending_interaction=_interaction(),
            usage=created.run.usage,
            now=_T1,
        )
    )


def _resolve(store: LifecycleStore, waiting: RunCommit) -> tuple[ResolveInteraction, RunCommit]:
    command = ResolveInteraction(
        run_id="run-1",
        expected_run_revision=waiting.run.revision,
        interaction_id=_INTERACTION_ID,
        expected_interaction_version=waiting.interaction.version,
        response=QuestionInteractionResponse(answer="继续"),
        expected_fingerprint=_TOOL_FINGERPRINT,
        now=_T2,
    )
    return command, store.resolve_interaction(command)


def _prepare_tool(store: LifecycleStore) -> RunCommit:
    created = _create(store)
    reserved = store.reserve_model_step(
        ReserveModelStep(
            run_id="run-1",
            expected_run_revision=created.run.revision,
            activation_id="activation-1",
            now=_T1,
        )
    )
    assistant = Msg.assistant([ToolUseBlock(id="call-tool", name="probe", input={"value": "A"})])
    prepared = RunToolCallRecord(
        run_id="run-1",
        step_index=0,
        ordinal=1,
        tool_call_id="call-tool",
        tool_name="probe",
        arguments={"value": "A"},
        fingerprint=_TOOL_FINGERPRINT,
        phase="prepared",
        version=1,
        created_at=_T1,
        updated_at=_T1,
    )
    return store.commit_model_step(
        CommitModelStep(
            run_id="run-1",
            expected_run_revision=reserved.run.revision,
            activation_id="activation-1",
            expected_session_revision=0,
            message_delta=[assistant],
            usage=RunUsage(model_steps_reserved=1, model_steps_committed=1),
            prepared_tool_calls=[prepared],
            checkpoint=_checkpoint(
                run_id="run-1",
                sequence=2,
                activation_id="activation-1",
                session_revision=1,
                reserved=1,
                committed=1,
            ),
            assistant_message=assistant,
            now=_T1,
        )
    )


def test_protocol_exposes_every_required_operation() -> None:
    """删除任一 runner 所需 port method 都应破坏本 contract。"""
    expected = {
        "begin_activation",
        "claim_tool_call",
        "commit_model_step",
        "commit_tool_result",
        "create_run",
        "finish_run",
        "list_events",
        "list_tool_calls",
        "load_checkpoint",
        "load_interaction",
        "load_result",
        "load_run",
        "load_session",
        "recover_active_run",
        "request_cancellation",
        "reserve_model_step",
        "resolve_interaction",
        "suspend_run",
    }
    assert expected <= set(LifecycleStore.__dict__)


def test_runtime_and_lifecycle_share_one_tool_error_policy_enum() -> None:
    """两个旧/新入口必须解析到同一个 enum class。"""
    from iris.lifecycle import ToolErrorPolicy as LifecycleToolErrorPolicy
    from iris.runtime.models import ToolErrorPolicy as RuntimeToolErrorPolicy

    assert RuntimeToolErrorPolicy is LifecycleToolErrorPolicy


def test_lifecycle_source_has_no_forbidden_dependency_edges() -> None:
    """Lifecycle contract 不得反向依赖 owner、engine 或 concrete store。"""
    lifecycle_root = Path(__file__).parents[2] / "src" / "iris" / "lifecycle"
    imported_modules: set[str] = set()
    for path in lifecycle_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.level == 0:
                    imported_modules.add(node.module)
                elif node.level == 1:
                    imported_modules.add(f"iris.lifecycle.{node.module}")
                elif node.level == 2:
                    imported_modules.add(f"iris.{node.module}")
    assert not any(
        name == forbidden or name.startswith(f"{forbidden}.")
        for name in imported_modules
        for forbidden in ("iris.harness", "iris.runtime", "iris.store")
    )


def test_importing_lifecycle_does_not_load_owner_or_concrete_store_modules() -> None:
    """Contract import 的动态模块图也必须保持 dependency-neutral。"""
    script = """
import json
import sys
import iris.lifecycle

forbidden = (
    "iris.harness",
    "iris.runtime",
    "iris.store",
    "iris.hitl.in_memory",
    "iris.hitl.store",
    "iris.memory.sqlite",
    "iris.memory.store",
)
loaded = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
)
print(json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


def test_create_rejects_duplicate_id_but_allows_independent_sessions(
    lifecycle_store: LifecycleStore,
) -> None:
    """覆盖 duplicate identity、lane 与不同 session 独立性。"""
    command = _create_command()
    _create(lifecycle_store)

    with pytest.raises(IrisRunConflictError):
        lifecycle_store.create_run(command)

    second = _create(
        lifecycle_store,
        run_id="run-2",
        session_id="session-2",
        activation_id="activation-2",
    )
    assert second.run.phase == "active"


def test_create_and_read_are_copy_isolated(lifecycle_store: LifecycleStore) -> None:
    """修改 command 或 read snapshot 不得改写 store 内部事实。"""
    command = _create_command(metadata={"nested": {"value": "original"}})
    lifecycle_store.create_run(command)
    command.request.metadata["nested"]["value"] = "changed"
    first = lifecycle_store.load_run("run-1")
    assert first is not None
    first.request.metadata["nested"]["value"] = "also-changed"

    loaded = lifecycle_store.load_run("run-1")
    assert loaded is not None
    assert loaded.request.metadata == {"nested": {"value": "original"}}


def test_reserve_exact_replay_is_noop_and_budget_exhaustion_is_terminal(
    lifecycle_store: LifecycleStore,
) -> None:
    """精确 replay 不增加 revision/event，下一 reservation 报预算耗尽。"""
    created = _create(lifecycle_store, max_model_steps=1)
    command = ReserveModelStep(
        run_id="run-1",
        expected_run_revision=created.run.revision,
        activation_id="activation-1",
        now=_T1,
    )
    first = lifecycle_store.reserve_model_step(command)
    events_after_first = lifecycle_store.list_events("run-1")

    replay = lifecycle_store.reserve_model_step(command)
    assert replay.events == ()
    assert replay.run.revision == first.run.revision
    assert lifecycle_store.list_events("run-1") == events_after_first

    terminal = lifecycle_store.reserve_model_step(
        ReserveModelStep(
            run_id="run-1",
            expected_run_revision=first.run.revision,
            activation_id="activation-1",
            now=_T2,
        )
    )
    assert terminal.run.stop_reason == "budget_exhausted"
    assert terminal.result is not None


def test_commit_model_step_updates_history_checkpoint_and_tool_intents_atomically(
    lifecycle_store: LifecycleStore,
) -> None:
    """缺少任一 history/checkpoint/tool replacement 都应被本断言捕获。"""
    committed = _prepare_tool(lifecycle_store)

    session = lifecycle_store.load_session("session-1")
    assert session.revision == 1
    assert len(session.messages) == 1
    assert committed.checkpoint.sequence == 2
    assert lifecycle_store.list_tool_calls("run-1")[0].phase == "prepared"

    session.messages.clear()
    assert len(lifecycle_store.load_session("session-1").messages) == 1


def test_claim_and_commit_tool_result_cover_effect_fence(
    lifecycle_store: LifecycleStore,
) -> None:
    """工具 effect 必须存在 durable claim，result 提交推进 tool/session/checkpoint。"""
    prepared = _prepare_tool(lifecycle_store)
    claim = ClaimToolCall(
        run_id="run-1",
        expected_run_revision=prepared.run.revision,
        activation_id="activation-1",
        tool_call_id="call-tool",
        fingerprint=_TOOL_FINGERPRINT,
        expected_tool_version=1,
        now=_T2,
    )
    claimed = lifecycle_store.claim_tool_call(claim)
    assert lifecycle_store.claim_tool_call(claim).events == ()
    claimed_call = lifecycle_store.list_tool_calls("run-1")[0]
    result = ToolResult(
        tool_use_id="call-tool",
        tool_name="probe",
        content=[TextBlock(text="done")],
    )

    committed = lifecycle_store.commit_tool_result(
        CommitToolResult(
            run_id="run-1",
            expected_run_revision=claimed.run.revision,
            activation_id="activation-1",
            expected_session_revision=1,
            tool_call_id="call-tool",
            expected_tool_version=claimed_call.version,
            result=result,
            message_delta=[
                Msg.tool_result(
                    tool_use_id="call-tool",
                    name="probe",
                    content="done",
                )
            ],
            checkpoint=_checkpoint(
                run_id="run-1",
                sequence=3,
                activation_id="activation-1",
                session_revision=2,
                reserved=1,
                committed=1,
            ),
            now=_T3,
        )
    )
    assert committed.run.usage.tool_calls_committed == 1
    assert lifecycle_store.list_tool_calls("run-1")[0].result == result
    assert lifecycle_store.load_session("session-1").revision == 2


def test_zero_tool_version_reaches_store_cas_boundary(
    lifecycle_store: LifecycleStore,
) -> None:
    """Stale version 0 由领域 conflict 处理，而不是泄漏模型 ValidationError。"""
    prepared = _prepare_tool(lifecycle_store)
    with pytest.raises(IrisRunConflictError):
        lifecycle_store.claim_tool_call(
            ClaimToolCall(
                run_id="run-1",
                expected_run_revision=prepared.run.revision,
                activation_id="activation-1",
                tool_call_id="call-tool",
                fingerprint=_TOOL_FINGERPRINT,
                expected_tool_version=0,
                now=_T2,
            )
        )


def test_prepared_tool_rejects_execution_error_without_claim(
    lifecycle_store: LifecycleStore,
) -> None:
    """只有 preflight failure 可以绕过 claim，execution failure 不可冒充无副作用。"""
    prepared = _prepare_tool(lifecycle_store)
    with pytest.raises(IrisRunStateError):
        lifecycle_store.commit_tool_result(
            CommitToolResult(
                run_id="run-1",
                expected_run_revision=prepared.run.revision,
                activation_id="activation-1",
                expected_session_revision=1,
                tool_call_id="call-tool",
                expected_tool_version=1,
                result=ToolResult(
                    tool_use_id="call-tool",
                    tool_name="probe",
                    is_error=True,
                    error=ToolErrorInfo(code="EXECUTION_ERROR", message="effect failed"),
                ),
                message_delta=[],
                checkpoint=_checkpoint(
                    run_id="run-1",
                    sequence=3,
                    activation_id="activation-1",
                    session_revision=1,
                    reserved=1,
                    committed=1,
                ),
                now=_T2,
            )
        )


def test_resolve_exact_response_replays_but_different_response_conflicts(
    lifecycle_store: LifecycleStore,
) -> None:
    """同 response 幂等，改变 response 不能覆盖已经 durable 的人工事实。"""
    waiting = _suspend(lifecycle_store, _create(lifecycle_store))
    command, resolved = _resolve(lifecycle_store, waiting)
    replay = lifecycle_store.resolve_interaction(command)

    assert replay.events == ()
    assert replay.run.revision == resolved.run.revision
    refreshed_replay = lifecycle_store.resolve_interaction(
        command.model_copy(
            update={
                "expected_run_revision": resolved.run.revision,
                "expected_interaction_version": resolved.interaction.version,
            }
        )
    )
    assert refreshed_replay.events == ()
    assert refreshed_replay.run == resolved.run
    with pytest.raises(IrisRunConflictError):
        lifecycle_store.resolve_interaction(
            command.model_copy(update={"response": QuestionInteractionResponse(answer="停止")})
        )


def test_begin_activation_rebinds_checkpoint_and_clears_waiting_result(
    lifecycle_store: LifecycleStore,
) -> None:
    """Resume 创建新 fence，并关闭旧 interaction 与 active result。"""
    waiting = _suspend(lifecycle_store, _create(lifecycle_store))
    _, resolved = _resolve(lifecycle_store, waiting)
    resumed = lifecycle_store.begin_activation(
        BeginActivation(
            run_id="run-1",
            expected_run_revision=resolved.run.revision,
            new_activation_id="activation-2",
            kind="resume",
            expected_checkpoint_sequence=resolved.checkpoint.sequence,
            now=_T3,
        )
    )

    assert resumed.checkpoint.sequence == resolved.checkpoint.sequence + 1
    assert resumed.interaction.status == "closed"
    assert lifecycle_store.load_result("run-1") is None


def test_cancellation_request_is_once_only_and_does_not_release_active_lane(
    lifecycle_store: LifecycleStore,
) -> None:
    """Active cancellation 只记录 durable intent，不冒充已结算 terminal。"""
    created = _create(lifecycle_store)
    command = RequestCancellation(
        run_id="run-1",
        expected_run_revision=created.run.revision,
        activation_id="activation-1",
        reason="user requested",
        now=_T1,
    )
    requested = lifecycle_store.request_cancellation(command)
    event_count = len(lifecycle_store.list_events("run-1"))

    assert lifecycle_store.request_cancellation(command).events == ()
    refreshed_replay = lifecycle_store.request_cancellation(
        command.model_copy(update={"expected_run_revision": requested.run.revision})
    )
    assert refreshed_replay.events == ()
    assert refreshed_replay.run == requested.run
    assert len(lifecycle_store.list_events("run-1")) == event_count
    assert requested.run.phase == "active"
    with pytest.raises(IrisRunConflictError):
        _create(
            lifecycle_store,
            run_id="run-2",
            session_id="session-1",
            activation_id="activation-2",
        )


def test_finish_exact_replay_is_noop_and_releases_lane(
    lifecycle_store: LifecycleStore,
) -> None:
    """Terminal event 只能出现一次，精确 finish replay 不制造新事实。"""
    created = _create(lifecycle_store)
    command = FinishRun(
        run_id="run-1",
        expected_run_revision=created.run.revision,
        activation_id="activation-1",
        stop_reason="completed",
        assistant_message=Msg.assistant("done"),
        now=_T1,
    )
    terminal = lifecycle_store.finish_run(command)
    replay = lifecycle_store.finish_run(command)

    assert replay.events == ()
    assert replay.run == terminal.run
    next_run = _create(
        lifecycle_store,
        run_id="run-2",
        session_id="session-1",
        activation_id="activation-2",
    )
    assert next_run.run.phase == "active"


def test_safe_recovery_abandons_old_fence_and_rebinds_checkpoint(
    lifecycle_store: LifecycleStore,
) -> None:
    """无未知 effect 时 recovery 必须创建全新 activation fence。"""
    created = _create(lifecycle_store)
    recovered = lifecycle_store.recover_active_run(
        RecoverActiveRun(
            run_id="run-1",
            expected_run_revision=created.run.revision,
            expected_activation_id="activation-1",
            expected_checkpoint_sequence=1,
            recovery_disposition=RecoveryDisposition.RESUME,
            new_activation_id="activation-recovery",
            now=_T1,
        )
    )
    assert recovered.run.current_activation_id == "activation-recovery"
    assert recovered.checkpoint.sequence == 2
    assert [event.kind for event in recovered.events] == [
        "activation.abandoned",
        "activation.started",
    ]


def test_safe_recovery_rejects_unresolved_durable_claim(
    lifecycle_store: LifecycleStore,
) -> None:
    """Claimed effect 无 durable result 时只能 outcome_unknown，不能安全重放。"""
    prepared = _prepare_tool(lifecycle_store)
    claimed = lifecycle_store.claim_tool_call(
        ClaimToolCall(
            run_id="run-1",
            expected_run_revision=prepared.run.revision,
            activation_id="activation-1",
            tool_call_id="call-tool",
            fingerprint=_TOOL_FINGERPRINT,
            expected_tool_version=1,
            now=_T2,
        )
    )

    with pytest.raises(IrisRunRecoveryError):
        lifecycle_store.recover_active_run(
            RecoverActiveRun(
                run_id="run-1",
                expected_run_revision=claimed.run.revision,
                expected_activation_id="activation-1",
                expected_checkpoint_sequence=claimed.checkpoint.sequence,
                recovery_disposition="resume",
                new_activation_id="activation-recovery",
                now=_T3,
            )
        )


def test_read_methods_apply_cursor_and_validation_contract(
    lifecycle_store: LifecycleStore,
) -> None:
    """Read API 不重排/修改 durable truth，且负游标 fail closed。"""
    _create(lifecycle_store)
    assert lifecycle_store.load_session("missing").revision == 0
    assert lifecycle_store.load_checkpoint("run-1").sequence == 1
    assert lifecycle_store.load_interaction("missing") is None
    assert lifecycle_store.load_result("run-1") is None
    assert [event.sequence for event in lifecycle_store.list_events("run-1", 1)] == [2]
    with pytest.raises(IrisRunStateError):
        lifecycle_store.list_events("run-1", -1)


def test_project_result_rejects_active_record_with_domain_error(
    lifecycle_store: LifecycleStore,
) -> None:
    """Projection helper 不应向调用方泄漏通用 ValueError。"""
    created = _create(lifecycle_store)

    with pytest.raises(IrisRunStateError):
        project_result(created.run)
