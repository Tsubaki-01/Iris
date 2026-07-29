"""定义 Phase 1 lifecycle 公共模型必须满足的目标 contract。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from importlib import import_module
from types import ModuleType

import pytest
from pydantic import ValidationError

from iris.hitl import HumanInteraction

_NOW = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
_LATER = _NOW + timedelta(minutes=1)


def _lifecycle() -> ModuleType:
    """加载尚待 Phase 1 实现的 dependency-neutral lifecycle 契约。"""
    return import_module("iris.lifecycle")


def _snapshot_values(
    lifecycle: ModuleType,
    *,
    phase: str,
    stop_reason: str | None,
) -> dict[str, object]:
    """构造手工固定的 RunSnapshot 输入，不复用生产投影函数。"""
    is_active = phase == "active"
    is_waiting = phase == "waiting"
    is_terminal = phase == "terminal"
    return {
        "run_id": "run-1",
        "session_id": "session-1",
        "agent_id": "agent-1",
        "phase": phase,
        "stop_reason": stop_reason,
        "revision": 1,
        "current_activation_id": "activation-1" if is_active else None,
        "pending_interaction_id": "interaction-1" if is_waiting else None,
        "cancellation_requested_at": None,
        "cancellation_reason": None,
        "limits": lifecycle.RunLimits(),
        "usage": lifecycle.RunUsage(),
        "environment_fingerprint": "environment-v1",
        "checkpoint_sequence": 1,
        "last_event_sequence": 2,
        "created_at": _NOW,
        "started_at": _NOW,
        "updated_at": _LATER,
        "finished_at": _LATER if is_terminal else None,
    }


def _pending_interaction() -> HumanInteraction:
    """构造只供 RunResult 互斥校验使用的 pending interaction。"""
    return HumanInteraction.model_construct(
        interaction_id="interaction-1",
        session_id="session-1",
        run_id="run-1",
        step_index=0,
        status="pending",
        version=1,
        created_at=_NOW,
    )


@pytest.mark.parametrize(
    "invalid_values",
    [
        {"max_model_steps": 0},
        {"deadline_at": datetime(2026, 1, 2, 3, 4)},
        {"interaction_timeout_seconds": 0},
    ],
    ids=["zero-model-budget", "naive-deadline", "zero-interaction-timeout"],
)
def test_run_limits_reject_invalid_boundaries_target_contract(
    invalid_values: dict[str, object],
) -> None:
    """定义预算、deadline 和 interaction timeout 的验证边界。"""
    lifecycle = _lifecycle()

    with pytest.raises(ValidationError):
        lifecycle.RunLimits(**invalid_values)


@pytest.mark.parametrize(
    "invalid_values",
    [
        {"input": "   "},
        {"input": "hello", "session_id": "   "},
        {"input": "hello", "metadata": {"invalid": object()}},
    ],
    ids=["blank-input", "blank-session", "non-json-metadata"],
)
def test_agent_run_request_rejects_invalid_inputs_target_contract(
    invalid_values: dict[str, object],
) -> None:
    """定义 request 标识与 metadata 的持久化安全边界。"""
    lifecycle = _lifecycle()

    with pytest.raises(ValidationError):
        lifecycle.AgentRunRequest(**invalid_values)


@pytest.mark.parametrize(
    "invalid_values",
    [
        {"model_steps_reserved": -1},
        {"model_steps_reserved": 1, "model_steps_committed": 2},
        {"model_steps_reserved": 2, "model_steps_committed": 0},
        {"tool_calls_committed": -1},
        {"input_tokens": -1},
    ],
    ids=[
        "negative-reserved",
        "committed-over-reserved",
        "multiple-uncommitted-reservations",
        "negative-tools",
        "negative-tokens",
    ],
)
def test_run_usage_rejects_impossible_counters_target_contract(
    invalid_values: dict[str, object],
) -> None:
    """定义 usage 非负且 committed 不超过 reserved 的规则。"""
    lifecycle = _lifecycle()

    with pytest.raises(ValidationError):
        lifecycle.RunUsage(**invalid_values)


@pytest.mark.parametrize(
    ("phase", "stop_reason", "updates"),
    [
        ("terminal", None, {}),
        ("terminal", "completed", {"finished_at": None}),
        ("active", "completed", {}),
        ("waiting", "completed", {}),
        ("waiting", None, {"current_activation_id": "activation-stale"}),
    ],
    ids=[
        "terminal-without-reason",
        "terminal-without-finished-time",
        "active-with-reason",
        "waiting-with-reason",
        "waiting-with-active-fence",
    ],
)
def test_run_snapshot_rejects_inconsistent_state_target_contract(
    phase: str,
    stop_reason: str | None,
    updates: dict[str, object],
) -> None:
    """定义 run phase、stop reason、时间与 activation fence 的互斥关系。"""
    lifecycle = _lifecycle()
    values = _snapshot_values(lifecycle, phase=phase, stop_reason=stop_reason)
    values.update(updates)

    with pytest.raises(ValidationError):
        lifecycle.RunSnapshot(**values)


@pytest.mark.parametrize(
    ("phase", "stop_reason", "has_interaction", "has_error"),
    [
        ("active", None, False, False),
        ("waiting", None, False, False),
        ("terminal", "completed", True, False),
        ("terminal", "failed", False, False),
        ("terminal", "completed", False, True),
    ],
    ids=[
        "active-result",
        "waiting-without-interaction",
        "terminal-with-interaction",
        "failed-without-error",
        "completed-with-error",
    ],
)
def test_run_result_rejects_inconsistent_outcome_target_contract(
    phase: str,
    stop_reason: str | None,
    has_interaction: bool,
    has_error: bool,
) -> None:
    """定义 waiting/terminal result 的 interaction 与 error 互斥关系。"""
    lifecycle = _lifecycle()
    run = lifecycle.RunSnapshot(**_snapshot_values(lifecycle, phase=phase, stop_reason=stop_reason))
    error = (
        lifecycle.RunErrorInfo(
            code="RUN_FAILED",
            message="run failed",
            source="runtime",
        )
        if has_error
        else None
    )

    with pytest.raises(ValidationError):
        lifecycle.RunResult(
            run=run,
            pending_interaction=_pending_interaction() if has_interaction else None,
            error=error,
        )


def test_run_result_rejects_interaction_from_another_run_target_contract() -> None:
    """Waiting result 不得混入其他 run 的 interaction。"""
    lifecycle = _lifecycle()
    run = lifecycle.RunSnapshot(**_snapshot_values(lifecycle, phase="waiting", stop_reason=None))
    interaction = _pending_interaction().model_copy(update={"run_id": "run-other"})

    with pytest.raises(ValidationError):
        lifecycle.RunResult(run=run, pending_interaction=interaction)


def test_snapshot_rejects_usage_above_run_limit_target_contract() -> None:
    """Snapshot 不得报告超过固定模型预算的 usage。"""
    lifecycle = _lifecycle()
    values = _snapshot_values(lifecycle, phase="active", stop_reason=None)
    values["usage"] = lifecycle.RunUsage(
        model_steps_reserved=21,
        model_steps_committed=20,
    )

    with pytest.raises(ValidationError):
        lifecycle.RunSnapshot(**values)


def test_checkpoint_rejects_multiple_uncommitted_reservations_target_contract() -> None:
    """Checkpoint 与 run usage 共享 single-owner counter invariant。"""
    lifecycle = _lifecycle()

    with pytest.raises(ValidationError):
        lifecycle.RunCheckpoint(
            run_id="run-1",
            sequence=1,
            activation_id="activation-1",
            engine_cursor={},
            session_revision=0,
            model_steps_reserved=2,
            model_steps_committed=0,
            environment_fingerprint="environment-v1",
            resumability="safe",
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"sequence": 0},
        {"occurred_at": datetime(2026, 1, 2, 3, 4)},
        {"payload": {"invalid": object()}},
    ],
    ids=["zero-sequence", "naive-time", "non-json-payload"],
)
def test_run_event_rejects_invalid_persisted_facts_target_contract(
    updates: dict[str, object],
) -> None:
    """定义 event sequence、时间与 payload 的持久化边界。"""
    lifecycle = _lifecycle()
    values: dict[str, object] = {
        "run_id": "run-1",
        "session_id": "session-1",
        "sequence": 1,
        "kind": lifecycle.RunEventKind.RUN_STARTED,
        "occurred_at": _NOW,
        "payload": {},
    }
    values.update(updates)

    with pytest.raises(ValidationError):
        lifecycle.RunEvent(**values)


def test_public_models_are_frozen_and_extra_forbid_target_contract() -> None:
    """定义公共 lifecycle model 不接受额外字段且实例不可变。"""
    lifecycle = _lifecycle()

    with pytest.raises(ValidationError):
        lifecycle.RunLimits(unexpected=True)

    limits = lifecycle.RunLimits()
    with pytest.raises(ValidationError):
        limits.max_model_steps = 2
