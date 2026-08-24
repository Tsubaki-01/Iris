"""所有 ``LifecycleStore`` 实现必须共享的 aggregate contract。"""

from __future__ import annotations

import ast
import json
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol, cast

import pytest
from pydantic import ValidationError

import iris.store.sqlite as sqlite_module
from iris.exceptions import (
    IrisRunConflictError,
    IrisRunNotFoundError,
    IrisRunPersistenceError,
    IrisRunRecoveryError,
    IrisRunStateError,
)
from iris.hitl import (
    HumanInteraction,
    HumanInteractionRequest,
    InteractionStatus,
    PermissionInteractionResponse,
    PermissionPrompt,
    QuestionInteractionResponse,
    QuestionPrompt,
    ToolCallSnapshot,
)
from iris.lifecycle import (
    AgentRunOptions,
    AgentRunRequest,
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
    ResumeWaitingRun,
    RunCheckpoint,
    RunCommit,
    RunControlSnapshot,
    RunErrorInfo,
    RunLimits,
    RunToolCallRecord,
    RunUsage,
    SuspendRun,
    project_result,
)
from iris.message import Msg, TextBlock, ToolUseBlock
from iris.store import InMemoryLifecycleStore, SQLiteStore
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


@pytest.fixture(params=["memory", "sqlite"])
def lifecycle_store(request: pytest.FixtureRequest, tmp_path: Path) -> LifecycleStore:
    """让进程内与 SQLite 实现运行完全相同的 aggregate contract。"""
    if request.param == "sqlite":
        return SQLiteStore(tmp_path / "lifecycle.db")
    return InMemoryLifecycleStore()


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
    session_revision: int = 0,
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
            session_revision=session_revision,
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
        tool_call_id="call-question",
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
        expires_at=_NOW + timedelta(minutes=5),
        created_at=_T1,
    )


def _suspend(
    store: LifecycleStore,
    created: RunCommit,
    *,
    include_tool_history: bool = False,
) -> RunCommit:
    prepared = RunToolCallRecord(
        run_id="run-1",
        step_index=0,
        ordinal=1,
        tool_call_id="call-question",
        tool_name="ask_question",
        arguments={"question": "继续吗？"},
        fingerprint=_TOOL_FINGERPRINT,
        phase="prepared",
        version=1,
        created_at=_T1,
        updated_at=_T1,
    )
    assistant = Msg.assistant(
        [ToolUseBlock(id="call-question", name="ask_question", input={"question": "继续吗？"})]
    )
    session_revision = 1 if include_tool_history else 0
    return store.suspend_run(
        SuspendRun(
            run_id="run-1",
            expected_run_revision=created.run.revision,
            activation_id="activation-1",
            expected_session_revision=0,
            message_delta=[assistant] if include_tool_history else [],
            prepared_tool_calls=[prepared],
            checkpoint=_checkpoint(
                run_id="run-1",
                sequence=2,
                activation_id="activation-1",
                session_revision=session_revision,
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


def _prepare_tool_batch(store: LifecycleStore) -> RunCommit:
    """为 claim 顺序测试持久化三条同 batch prepared call。"""
    created = _create(store)
    reserved = store.reserve_model_step(
        ReserveModelStep(
            run_id="run-1",
            expected_run_revision=created.run.revision,
            activation_id="activation-1",
            now=_T1,
        )
    )
    uses = tuple(
        ToolUseBlock(id=f"call-{ordinal}", name="probe", input={"value": ordinal})
        for ordinal in range(1, 4)
    )
    assistant = Msg.assistant(uses)
    prepared = [
        RunToolCallRecord(
            run_id="run-1",
            step_index=0,
            ordinal=ordinal,
            tool_call_id=tool_use.id,
            tool_name=tool_use.name,
            arguments=dict(tool_use.input),
            fingerprint=_TOOL_FINGERPRINT,
            phase="prepared",
            version=1,
            created_at=_T1,
            updated_at=_T1,
        )
        for ordinal, tool_use in enumerate(uses, start=1)
    ]
    return store.commit_model_step(
        CommitModelStep(
            run_id="run-1",
            expected_run_revision=reserved.run.revision,
            activation_id="activation-1",
            expected_session_revision=0,
            message_delta=[assistant],
            usage=RunUsage(model_steps_reserved=1, model_steps_committed=1),
            prepared_tool_calls=prepared,
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
        "resume_waiting_run",
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
        "load_run_control",
        "load_session",
        "load_session_lane",
        "load_tool_call",
        "recover_active_run",
        "request_cancellation",
        "reserve_model_step",
        "resolve_interaction",
        "suspend_run",
    }
    assert expected <= set(LifecycleStore.__dict__)


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


def test_exact_tool_call_read_is_copy_isolated_and_missing_is_none(
    lifecycle_store: LifecycleStore,
) -> None:
    """Exact composite-key read 不扫描或模糊匹配其他 durable subject。"""
    _prepare_tool(lifecycle_store)
    _create(
        lifecycle_store,
        run_id="run-2",
        session_id="session-2",
        activation_id="activation-2",
    )

    loaded = lifecycle_store.load_tool_call("run-1", "call-tool")

    assert loaded == lifecycle_store.list_tool_calls("run-1")[0]
    assert loaded is not None
    loaded.arguments["value"] = "changed"
    reloaded = lifecycle_store.load_tool_call("run-1", "call-tool")
    assert reloaded is not None
    assert reloaded.arguments == {"value": "A"}
    assert lifecycle_store.load_tool_call("run-1", "missing") is None
    assert lifecycle_store.load_tool_call("missing", "call-tool") is None
    assert lifecycle_store.load_tool_call("run-2", "call-tool") is None
    with pytest.raises(IrisRunNotFoundError):
        lifecycle_store.list_tool_calls("missing")


def test_run_control_read_projects_exact_frozen_fields(
    lifecycle_store: LifecycleStore,
) -> None:
    """Control read 只暴露 fence/cancellation 所需的八个不可变字段。"""
    created = _create(lifecycle_store)

    control = lifecycle_store.load_run_control("run-1")

    assert control == RunControlSnapshot(
        run_id=created.run.run_id,
        phase=created.run.phase,
        revision=created.run.revision,
        current_activation_id=created.run.current_activation_id,
        cancellation_requested_at=created.run.cancellation_requested_at,
        cancellation_reason=created.run.cancellation_reason,
        last_event_sequence=created.run.last_event_sequence,
        updated_at=created.run.updated_at,
    )
    assert set(RunControlSnapshot.model_fields) == {
        "run_id",
        "phase",
        "revision",
        "current_activation_id",
        "cancellation_requested_at",
        "cancellation_reason",
        "last_event_sequence",
        "updated_at",
    }
    assert lifecycle_store.load_run_control("missing") is None
    assert control is not None
    with pytest.raises(ValidationError, match="frozen"):
        control.revision = 99
    with pytest.raises(ValidationError, match="Extra inputs"):
        RunControlSnapshot.model_validate(control.model_dump() | {"unexpected": True})
    with pytest.raises(ValidationError, match="同时存在"):
        RunControlSnapshot.model_validate(
            control.model_dump() | {"cancellation_reason": "unpaired"}
        )


def test_sqlite_run_control_read_skips_aggregate_json_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control projection 不读取 request/options/usage/message/error JSON。"""
    store = SQLiteStore(tmp_path / "control.db")
    created = _create(store)

    def fail_json_decode(value: str) -> object:
        del value
        raise AssertionError("control projection 不应 decode aggregate JSON")

    monkeypatch.setattr(sqlite_module, "_load_json", fail_json_decode)

    assert store.load_run_control("run-1") == RunControlSnapshot(
        run_id=created.run.run_id,
        phase=created.run.phase,
        revision=created.run.revision,
        current_activation_id=created.run.current_activation_id,
        cancellation_requested_at=None,
        cancellation_reason=None,
        last_event_sequence=created.run.last_event_sequence,
        updated_at=created.run.updated_at,
    )


def test_sqlite_corrupt_run_control_maps_validation_to_persistence_error(tmp_path: Path) -> None:
    """窄投影的 durable validation 失败沿用 lifecycle persistence error。"""
    store = SQLiteStore(tmp_path / "corrupt-control.db")
    _create(store)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE agent_runs SET cancellation_reason = 'unpaired' WHERE run_id = 'run-1'"
        )

    with pytest.raises(IrisRunPersistenceError) as captured:
        store.load_run_control("run-1")

    assert captured.value.context["operation"] == "load_run_control"
    assert captured.value.context["path"] == str(store.path)


def test_sqlite_corrupt_point_read_maps_decode_to_persistence_error(tmp_path: Path) -> None:
    """Exact tool read 保留既有 corrupt-row error context。"""
    store = SQLiteStore(tmp_path / "corrupt-tool.db")
    _prepare_tool(store)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE run_tool_calls SET arguments_json = '{' WHERE tool_call_id = 'call-tool'"
        )

    with pytest.raises(IrisRunPersistenceError) as captured:
        store.load_tool_call("run-1", "call-tool")

    assert captured.value.context["operation"] == "load_tool_call"
    assert captured.value.context["path"] == str(store.path)


def test_session_lane_read_tracks_non_terminal_owner_without_mutation(
    lifecycle_store: LifecycleStore,
) -> None:
    """Lane read 只观察 active/waiting owner，并在 terminal 后返回空。"""
    assert lifecycle_store.load_session_lane("session-1") is None
    created = _create(lifecycle_store)
    run_before = lifecycle_store.load_run("run-1")
    session_before = lifecycle_store.load_session("session-1")
    checkpoint_before = lifecycle_store.load_checkpoint("run-1")
    events_before = lifecycle_store.list_events("run-1")

    assert lifecycle_store.load_session_lane("session-1") == "run-1"
    assert lifecycle_store.load_session_lane("session-1") == "run-1"
    assert lifecycle_store.load_run("run-1") == run_before
    assert lifecycle_store.load_session("session-1") == session_before
    assert lifecycle_store.load_checkpoint("run-1") == checkpoint_before
    assert lifecycle_store.list_events("run-1") == events_before
    if isinstance(lifecycle_store, SQLiteStore):
        assert SQLiteStore(lifecycle_store.path).load_session_lane("session-1") == "run-1"

    waiting = _suspend(lifecycle_store, created)
    assert lifecycle_store.load_session_lane("session-1") == "run-1"
    if isinstance(lifecycle_store, SQLiteStore):
        assert SQLiteStore(lifecycle_store.path).load_session_lane("session-1") == "run-1"

    lifecycle_store.finish_run(
        FinishRun(
            run_id="run-1",
            expected_run_revision=waiting.run.revision,
            activation_id=None,
            stop_reason="cancelled",
            now=_T2,
        )
    )
    assert lifecycle_store.load_session_lane("session-1") is None
    if isinstance(lifecycle_store, SQLiteStore):
        assert SQLiteStore(lifecycle_store.path).load_session_lane("session-1") is None

    second = _create(
        lifecycle_store,
        run_id="run-2",
        session_id="session-1",
        activation_id="activation-2",
        session_revision=lifecycle_store.load_session("session-1").revision,
    )
    requested = lifecycle_store.request_cancellation(
        RequestCancellation(
            run_id="run-2",
            expected_run_revision=second.run.revision,
            activation_id="activation-2",
            reason="stop",
            now=_T2,
        )
    )
    assert lifecycle_store.load_session_lane("session-1") == "run-2"
    lifecycle_store.finish_run(
        FinishRun(
            run_id="run-2",
            expected_run_revision=requested.run.revision,
            activation_id="activation-2",
            stop_reason="cancelled",
            now=_T3,
        )
    )
    assert lifecycle_store.load_session_lane("session-1") is None


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
    assert committed.session is not None
    committed.session.messages[0].metadata["caller-mutated"] = True
    assert lifecycle_store.load_session("session-1").messages[0].metadata == {}


def test_nonempty_message_delta_advances_session_revision_once(
    lifecycle_store: LifecycleStore,
) -> None:
    created = _create(lifecycle_store)
    reserved = lifecycle_store.reserve_model_step(
        ReserveModelStep(
            run_id="run-1",
            expected_run_revision=created.run.revision,
            activation_id="activation-1",
            now=_T1,
        )
    )
    assistant = Msg.assistant("second")

    committed = lifecycle_store.commit_model_step(
        CommitModelStep(
            run_id="run-1",
            expected_run_revision=reserved.run.revision,
            activation_id="activation-1",
            expected_session_revision=0,
            message_delta=[Msg.user("first"), assistant],
            usage=RunUsage(model_steps_reserved=1, model_steps_committed=1),
            checkpoint=_checkpoint(
                run_id="run-1",
                sequence=2,
                activation_id="activation-1",
                session_revision=1,
                reserved=1,
                committed=1,
            ),
            assistant_message=assistant,
            now=_T2,
        )
    )

    assert committed.session is not None
    assert committed.session.revision == 1
    assert [message.text for message in committed.session.messages] == ["first", "second"]
    assert lifecycle_store.load_session("session-1") == committed.session


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


def test_claim_batch_respects_durable_cancellation_fence(
    lifecycle_store: LifecycleStore,
) -> None:
    """取消前可多 claim；取消后只允许 exact replay 与既有 claim result。"""
    prepared = _prepare_tool_batch(lifecycle_store)
    first_command = ClaimToolCall(
        run_id="run-1",
        expected_run_revision=prepared.run.revision,
        activation_id="activation-1",
        tool_call_id="call-1",
        fingerprint=_TOOL_FINGERPRINT,
        expected_tool_version=1,
        now=_T2,
    )
    first = lifecycle_store.claim_tool_call(first_command)
    second = lifecycle_store.claim_tool_call(
        first_command.model_copy(
            update={
                "expected_run_revision": first.run.revision,
                "tool_call_id": "call-2",
            }
        )
    )
    cancelled = lifecycle_store.request_cancellation(
        RequestCancellation(
            run_id="run-1",
            expected_run_revision=second.run.revision,
            activation_id="activation-1",
            reason="user requested",
            now=_T3,
        )
    )
    events_after_cancel = lifecycle_store.list_events("run-1")

    replay = lifecycle_store.claim_tool_call(first_command)
    assert replay.events == ()
    with pytest.raises(IrisRunStateError, match="取消"):
        lifecycle_store.claim_tool_call(
            first_command.model_copy(
                update={
                    "expected_run_revision": cancelled.run.revision,
                    "tool_call_id": "call-3",
                }
            )
        )
    assert lifecycle_store.list_events("run-1") == events_after_cancel
    calls = {call.tool_call_id: call for call in lifecycle_store.list_tool_calls("run-1")}
    assert calls["call-1"].phase == "claimed"
    assert calls["call-2"].phase == "claimed"
    assert calls["call-3"].phase == "prepared"

    committed = lifecycle_store.commit_tool_result(
        CommitToolResult(
            run_id="run-1",
            expected_run_revision=cancelled.run.revision,
            activation_id="activation-1",
            expected_session_revision=1,
            tool_call_id="call-1",
            expected_tool_version=2,
            result=ToolResult(tool_use_id="call-1", tool_name="probe"),
            message_delta=[Msg.tool_result(tool_use_id="call-1", name="probe")],
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


@pytest.mark.parametrize(
    ("tool_call_id", "fingerprint", "error_type"),
    [
        ("missing-call", _TOOL_FINGERPRINT, IrisRunNotFoundError),
        ("call-1", "b" * 64, IrisRunConflictError),
    ],
)
def test_cancelled_claim_preserves_exact_subject_error_priority(
    lifecycle_store: LifecycleStore,
    tool_call_id: str,
    fingerprint: str,
    error_type: type[Exception],
) -> None:
    """Cancellation fence 不得遮蔽不存在或 fingerprint 错误。"""
    prepared = _prepare_tool_batch(lifecycle_store)
    cancelled = lifecycle_store.request_cancellation(
        RequestCancellation(
            run_id="run-1",
            expected_run_revision=prepared.run.revision,
            activation_id="activation-1",
            reason="user requested",
            now=_T2,
        )
    )
    events_after_cancel = lifecycle_store.list_events("run-1")

    with pytest.raises(error_type):
        lifecycle_store.claim_tool_call(
            ClaimToolCall(
                run_id="run-1",
                expected_run_revision=cancelled.run.revision,
                activation_id="activation-1",
                tool_call_id=tool_call_id,
                fingerprint=fingerprint,
                expected_tool_version=1,
                now=_T3,
            )
        )

    assert lifecycle_store.list_events("run-1") == events_after_cancel
    assert all(call.phase == "prepared" for call in lifecycle_store.list_tool_calls("run-1"))


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


def test_resume_waiting_run_rebinds_checkpoint_and_clears_waiting_result(
    lifecycle_store: LifecycleStore,
) -> None:
    """Resume 创建新 fence，并关闭旧 interaction 与 active result。"""
    waiting = _suspend(lifecycle_store, _create(lifecycle_store))
    _, resolved = _resolve(lifecycle_store, waiting)
    resumed = lifecycle_store.resume_waiting_run(
        ResumeWaitingRun(
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


def test_suspend_rejects_interaction_subject_that_differs_from_prepared_call(
    lifecycle_store: LifecycleStore,
) -> None:
    created = _create(lifecycle_store)
    interaction = _interaction()
    mismatched_subject = interaction.request.tool_call.model_copy(
        update={"arguments": {"question": "另一个问题"}}
    )
    interaction = interaction.model_copy(
        update={"request": interaction.request.model_copy(update={"tool_call": mismatched_subject})}
    )
    prepared = RunToolCallRecord(
        run_id="run-1",
        step_index=0,
        ordinal=1,
        tool_call_id="call-question",
        tool_name="ask_question",
        arguments={"question": "继续吗？"},
        fingerprint=_TOOL_FINGERPRINT,
        phase="prepared",
        version=1,
        created_at=_T1,
        updated_at=_T1,
    )

    with pytest.raises(IrisRunConflictError, match="subject"):
        lifecycle_store.suspend_run(
            SuspendRun(
                run_id="run-1",
                expected_run_revision=created.run.revision,
                activation_id="activation-1",
                expected_session_revision=0,
                prepared_tool_calls=[prepared],
                checkpoint=_checkpoint(
                    run_id="run-1",
                    sequence=2,
                    activation_id="activation-1",
                    session_revision=0,
                ),
                pending_interaction=interaction,
                usage=created.run.usage,
                now=_T1,
            )
        )


def test_approved_permission_cannot_commit_rejection_without_claim(
    lifecycle_store: LifecycleStore,
) -> None:
    """若只看 USER_REJECTED error code，approve 可绕过 required effect claim。"""
    created = _create(lifecycle_store)
    interaction = HumanInteraction(
        interaction_id=_INTERACTION_ID,
        session_id="session-1",
        run_id="run-1",
        step_index=0,
        tool_call_id="call-write",
        status=InteractionStatus.PENDING,
        request=HumanInteractionRequest(
            tool_call=ToolCallSnapshot(
                tool_call_id="call-write",
                tool_name="write",
                arguments={"value": "x"},
                workspace_root="workspace",
                fingerprint=_TOOL_FINGERPRINT,
            ),
            prompt=PermissionPrompt(reason="写入"),
        ),
        expires_at=_NOW + timedelta(minutes=5),
        created_at=_T1,
    )
    prepared = RunToolCallRecord(
        run_id="run-1",
        step_index=0,
        ordinal=1,
        tool_call_id="call-write",
        tool_name="write",
        arguments={"value": "x"},
        fingerprint=_TOOL_FINGERPRINT,
        phase="prepared",
        version=1,
        created_at=_T1,
        updated_at=_T1,
    )
    waiting = lifecycle_store.suspend_run(
        SuspendRun(
            run_id="run-1",
            expected_run_revision=created.run.revision,
            activation_id="activation-1",
            expected_session_revision=0,
            prepared_tool_calls=[prepared],
            checkpoint=_checkpoint(
                run_id="run-1",
                sequence=2,
                activation_id="activation-1",
                session_revision=0,
            ),
            pending_interaction=interaction,
            usage=created.run.usage,
            now=_T1,
        )
    )
    resolved = lifecycle_store.resolve_interaction(
        ResolveInteraction(
            run_id="run-1",
            expected_run_revision=waiting.run.revision,
            interaction_id=_INTERACTION_ID,
            expected_interaction_version=1,
            response=PermissionInteractionResponse(decision="approve"),
            expected_fingerprint=_TOOL_FINGERPRINT,
            now=_T2,
        )
    )
    begun = lifecycle_store.resume_waiting_run(
        ResumeWaitingRun(
            run_id="run-1",
            expected_run_revision=resolved.run.revision,
            new_activation_id="activation-resume",
            kind="resume",
            expected_checkpoint_sequence=2,
            now=_T3,
        )
    )
    assert begun.checkpoint is not None

    with pytest.raises(IrisRunStateError, match="claim"):
        lifecycle_store.commit_tool_result(
            CommitToolResult(
                run_id="run-1",
                expected_run_revision=begun.run.revision,
                activation_id="activation-resume",
                expected_session_revision=0,
                tool_call_id="call-write",
                expected_tool_version=1,
                result=ToolResult(
                    tool_use_id="call-write",
                    tool_name="write",
                    is_error=True,
                    error=ToolErrorInfo(
                        code="USER_REJECTED",
                        message="用户拒绝了工具调用",
                    ),
                ),
                checkpoint=_checkpoint(
                    run_id="run-1",
                    sequence=begun.checkpoint.sequence + 1,
                    activation_id="activation-resume",
                    session_revision=0,
                ),
                now=_T3,
            )
        )


def test_question_projection_must_match_exact_durable_answer(
    lifecycle_store: LifecycleStore,
) -> None:
    waiting = _suspend(lifecycle_store, _create(lifecycle_store))
    _, resolved = _resolve(lifecycle_store, waiting)
    begun = lifecycle_store.resume_waiting_run(
        ResumeWaitingRun(
            run_id="run-1",
            expected_run_revision=resolved.run.revision,
            new_activation_id="activation-resume",
            kind="resume",
            expected_checkpoint_sequence=2,
            now=_T3,
        )
    )
    assert begun.checkpoint is not None

    with pytest.raises(IrisRunStateError, match="claim"):
        lifecycle_store.commit_tool_result(
            CommitToolResult(
                run_id="run-1",
                expected_run_revision=begun.run.revision,
                activation_id="activation-resume",
                expected_session_revision=0,
                tool_call_id="call-question",
                expected_tool_version=1,
                result=ToolResult(
                    tool_use_id="call-question",
                    tool_name="ask_question",
                    content=[TextBlock(text="伪造答案")],
                    data={"answer": "伪造答案"},
                ),
                checkpoint=_checkpoint(
                    run_id="run-1",
                    sequence=begun.checkpoint.sequence + 1,
                    activation_id="activation-resume",
                    session_revision=0,
                ),
                now=_T3,
            )
        )


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


def test_waiting_cancel_closes_prepared_tool_history_without_tool_event(
    lifecycle_store: LifecycleStore,
) -> None:
    """Waiting terminal cancellation 要闭合 history，但不能伪造 tool effect。"""
    waiting = _suspend(
        lifecycle_store,
        _create(lifecycle_store),
        include_tool_history=True,
    )
    cancelled = lifecycle_store.request_cancellation(
        RequestCancellation(
            run_id="run-1",
            expected_run_revision=waiting.run.revision,
            activation_id=None,
            reason="user requested",
            settle_waiting=True,
            now=_T2,
        )
    )

    session = lifecycle_store.load_session("session-1")
    [tool_result] = session.messages[-1].tool_results
    [record] = lifecycle_store.list_tool_calls("run-1")
    assert cancelled.run.stop_reason == "cancelled"
    assert session.revision == 2
    assert tool_result.tool_use_id == "call-question"
    assert tool_result.is_error is True
    assert tool_result.metadata["error"]["code"] == "TOOL_NOT_STARTED"
    assert tool_result.metadata["error"]["retryable"] is True
    assert record.phase == "prepared"
    assert record.result is None
    assert all(event.kind != "tool_call.outcome_unknown" for event in cancelled.events)
    assert cancelled.session == session
    assert cancelled.checkpoint is not None
    assert cancelled.checkpoint.sequence == waiting.checkpoint.sequence
    assert cancelled.checkpoint.session_revision == session.revision


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


def test_terminal_finish_closes_claimed_and_prepared_history_atomically(
    lifecycle_store: LifecycleStore,
) -> None:
    """Terminal settlement 必须原子闭合全部 unresolved tool history。"""
    prepared = _prepare_tool_batch(lifecycle_store)
    claim_revision = prepared.run.revision
    for tool_call_id in ("call-1", "call-2"):
        claimed = lifecycle_store.claim_tool_call(
            ClaimToolCall(
                run_id="run-1",
                expected_run_revision=claim_revision,
                activation_id="activation-1",
                tool_call_id=tool_call_id,
                fingerprint=_TOOL_FINGERPRINT,
                expected_tool_version=1,
                now=_T2,
            )
        )
        claim_revision = claimed.run.revision

    command = FinishRun(
        run_id="run-1",
        expected_run_revision=claim_revision,
        activation_id="activation-1",
        stop_reason="outcome_unknown",
        error=RunErrorInfo(
            code="TOOL_OUTCOME_UNKNOWN",
            message="工具结果不可证明",
            source="tool",
        ),
        now=_T3,
    )
    terminal = lifecycle_store.finish_run(command)

    records = lifecycle_store.list_tool_calls("run-1")
    assert [record.phase for record in records] == [
        "outcome_unknown",
        "outcome_unknown",
        "prepared",
    ]
    assert [event.kind for event in terminal.events] == [
        "tool_call.outcome_unknown",
        "tool_call.outcome_unknown",
        "run.terminal",
    ]
    assert terminal.events[-1].kind == "run.terminal"
    assert {
        event.correlation_id
        for event in terminal.events
        if event.kind == "tool_call.outcome_unknown"
    } == {"call-1", "call-2"}
    assert all(record.result is None for record in records)

    session = lifecycle_store.load_session("session-1")
    tool_results = [result for message in session.messages for result in message.tool_results]
    assert session.revision == 2
    assert [result.tool_use_id for result in tool_results] == ["call-1", "call-2", "call-3"]
    assert [result.is_error for result in tool_results] == [True, True, True]
    assert [result.metadata["error"]["code"] for result in tool_results] == [
        "TOOL_OUTCOME_UNKNOWN",
        "TOOL_OUTCOME_UNKNOWN",
        "TOOL_NOT_STARTED",
    ]
    assert [result.metadata["error"]["retryable"] for result in tool_results] == [
        False,
        False,
        True,
    ]
    assert terminal.session == session
    assert terminal.checkpoint is not None
    assert terminal.checkpoint.sequence == 2
    assert terminal.checkpoint.session_revision == session.revision
    assert terminal.run.usage.tool_calls_committed == 0

    replay = lifecycle_store.finish_run(command)
    assert replay.events == ()
    assert lifecycle_store.load_session("session-1") == session


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


def test_outcome_unknown_recovery_roundtrips_exact_activation_and_tool_facts(
    lifecycle_store: LifecycleStore,
) -> None:
    """Recovery 原子关闭多个 claim，重开后保留 activation/tool 精确事实。"""
    prepared = _prepare_tool_batch(lifecycle_store)
    first_claimed = lifecycle_store.claim_tool_call(
        ClaimToolCall(
            run_id="run-1",
            expected_run_revision=prepared.run.revision,
            activation_id="activation-1",
            tool_call_id="call-1",
            fingerprint=_TOOL_FINGERPRINT,
            expected_tool_version=1,
            now=_T2,
        )
    )
    claimed = lifecycle_store.claim_tool_call(
        ClaimToolCall(
            run_id="run-1",
            expected_run_revision=first_claimed.run.revision,
            activation_id="activation-1",
            tool_call_id="call-2",
            fingerprint=_TOOL_FINGERPRINT,
            expected_tool_version=1,
            now=_T2,
        )
    )

    recovered = lifecycle_store.recover_active_run(
        RecoverActiveRun(
            run_id="run-1",
            expected_run_revision=claimed.run.revision,
            expected_activation_id="activation-1",
            expected_checkpoint_sequence=claimed.checkpoint.sequence,
            recovery_disposition=RecoveryDisposition.OUTCOME_UNKNOWN,
            now=_T3,
        )
    )

    assert [event.kind for event in recovered.events] == [
        "activation.abandoned",
        "tool_call.outcome_unknown",
        "tool_call.outcome_unknown",
        "run.terminal",
    ]
    records = lifecycle_store.list_tool_calls("run-1")
    assert [record.phase for record in records] == [
        "outcome_unknown",
        "outcome_unknown",
        "prepared",
    ]
    session = lifecycle_store.load_session("session-1")
    tool_results = [result for message in session.messages for result in message.tool_results]
    assert session.revision == 2
    assert [result.tool_use_id for result in tool_results] == ["call-1", "call-2", "call-3"]
    assert recovered.session == session
    assert recovered.checkpoint is not None
    assert recovered.checkpoint.sequence == claimed.checkpoint.sequence
    assert recovered.checkpoint.session_revision == session.revision
    if not isinstance(lifecycle_store, SQLiteStore):
        return
    reopened = SQLiteStore(lifecycle_store.path)
    reopened_records = reopened.list_tool_calls("run-1")
    with sqlite3.connect(lifecycle_store.path) as connection:
        activation_fact = connection.execute(
            "SELECT status, outcome FROM run_activations WHERE activation_id = ?",
            ("activation-1",),
        ).fetchone()

    assert activation_fact == ("abandoned", "outcome_unknown")
    assert [record.phase for record in reopened_records] == [
        "outcome_unknown",
        "outcome_unknown",
        "prepared",
    ]
    assert [record.updated_at for record in reopened_records[:2]] == [_T3, _T3]


def test_read_methods_apply_cursor_and_validation_contract(
    lifecycle_store: LifecycleStore,
) -> None:
    """Read API 不重排/修改 durable truth，且负游标 fail closed。"""
    _create(lifecycle_store)
    assert lifecycle_store.load_session("missing").revision == 0
    assert lifecycle_store.load_checkpoint("run-1").sequence == 1
    assert lifecycle_store.load_interaction("missing") is None
    assert lifecycle_store.load_result("run-1") is None
    assert [event.sequence for event in lifecycle_store.list_events("run-1", limit=1)] == [1]
    assert [event.sequence for event in lifecycle_store.list_events("run-1", 1, limit=1)] == [2]
    assert [event.sequence for event in lifecycle_store.list_events("run-1", 1)] == [2]
    with pytest.raises(IrisRunStateError):
        lifecycle_store.list_events("run-1", -1)
    for invalid_limit in (0, -1, True, 1.5):
        with pytest.raises(IrisRunStateError):
            lifecycle_store.list_events("run-1", limit=cast(int, invalid_limit))


def test_project_result_rejects_active_record_with_domain_error(
    lifecycle_store: LifecycleStore,
) -> None:
    """Projection helper 不应向调用方泄漏通用 ValueError。"""
    created = _create(lifecycle_store)

    with pytest.raises(IrisRunStateError):
        project_result(created.run)
