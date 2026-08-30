"""Lifecycle 已验证对象的进程内状态转换。

Example:
    updated = reserve_model_step(usage)
"""

# region imports
from __future__ import annotations

from datetime import datetime
from typing import Any

from ..tools.base import ToolResult
from .models import (
    ActivationOutcome,
    ActivationRecord,
    ActivationStatus,
    RunCheckpoint,
    RunRecord,
    RunToolCallRecord,
    RunUsage,
    ToolCallPhase,
)

# endregion


def replace_run(run: RunRecord, **changes: Any) -> RunRecord:
    """应用已由 store 验证的 run 状态增量。"""
    return run.model_copy(update=changes)


def reserve_model_step(usage: RunUsage) -> RunUsage:
    """把 model-step reservation 计数推进一次。"""
    return usage.model_copy(update={"model_steps_reserved": usage.model_steps_reserved + 1})


def commit_tool_usage(usage: RunUsage) -> RunUsage:
    """把已提交工具调用计数推进一次。"""
    return usage.model_copy(update={"tool_calls_committed": usage.tool_calls_committed + 1})


def rebind_checkpoint(
    checkpoint: RunCheckpoint,
    *,
    activation_id: str,
) -> RunCheckpoint:
    """把已验证 checkpoint 绑定到新的 activation fence。"""
    return checkpoint.model_copy(update={"activation_id": activation_id})


def settle_activation(
    activation: ActivationRecord,
    *,
    outcome: ActivationOutcome,
    ended_at: datetime,
) -> ActivationRecord:
    """结算当前 active activation。"""
    return activation.model_copy(
        update={
            "status": ActivationStatus.SETTLED,
            "outcome": outcome,
            "ended_at": ended_at,
        }
    )


def abandon_activation(
    activation: ActivationRecord,
    *,
    outcome: ActivationOutcome,
    ended_at: datetime,
) -> ActivationRecord:
    """把旧 activation 标记为恢复流程已放弃。"""
    return activation.model_copy(
        update={
            "status": ActivationStatus.ABANDONED,
            "outcome": outcome,
            "ended_at": ended_at,
        }
    )


def claim_tool_call(
    tool_call: RunToolCallRecord,
    *,
    activation_id: str,
    now: datetime,
) -> RunToolCallRecord:
    """应用 prepared 到 claimed 的已验证增量。"""
    return tool_call.model_copy(
        update={
            "phase": ToolCallPhase.CLAIMED,
            "claim_activation_id": activation_id,
            "version": tool_call.version + 1,
            "updated_at": now,
            "claimed_at": now,
        }
    )


def commit_tool_call(
    tool_call: RunToolCallRecord,
    *,
    result: ToolResult,
    now: datetime,
) -> RunToolCallRecord:
    """应用 prepared/claimed 到 committed 的已验证增量。"""
    return tool_call.model_copy(
        update={
            "phase": ToolCallPhase.COMMITTED,
            "result": result,
            "version": tool_call.version + 1,
            "updated_at": now,
            "committed_at": now,
        }
    )


def mark_tool_call_outcome_unknown(
    tool_call: RunToolCallRecord,
    *,
    now: datetime,
) -> RunToolCallRecord:
    """把已 claim 且结果未知的工具调用标记为 outcome unknown。"""
    return tool_call.model_copy(
        update={
            "phase": ToolCallPhase.OUTCOME_UNKNOWN,
            "version": tool_call.version + 1,
            "updated_at": now,
        }
    )


__all__ = [
    "abandon_activation",
    "claim_tool_call",
    "commit_tool_call",
    "commit_tool_usage",
    "mark_tool_call_outcome_unknown",
    "rebind_checkpoint",
    "replace_run",
    "reserve_model_step",
    "settle_activation",
]
