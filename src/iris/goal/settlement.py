"""用本 Run 的已提交证据判定目标结果，不执行工具或读取外部状态。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import ValidationError

from ..lifecycle.history import is_ordinary_user
from ..lifecycle.models import (
    RunEvent,
    RunEventKind,
    RunResult,
    RunStopReason,
    RunToolCallRecord,
    ToolCallPhase,
)
from ..message.message import Msg
from .models import (
    GoalDecision,
    GoalReason,
    GoalReport,
    GoalRunBinding,
    GoalSettlement,
    GoalSnapshot,
    GoalStatus,
)

_GOAL_TOOLS = frozenset({"get_goal", "report_goal"})


@dataclass(frozen=True, slots=True, kw_only=True)
class GoalRunEvidence:
    """存储在同一个快照中取得的终态运行证据。"""

    result: RunResult
    calls: tuple[RunToolCallRecord, ...]
    messages: tuple[Msg, ...]
    events: tuple[RunEvent, ...]


def _latest_report(evidence: GoalRunEvidence) -> tuple[RunToolCallRecord, GoalReport] | None:
    """只解析成功提交报告的原始 payload，按模型步和 ordinal 取最后一份。"""
    for call in sorted(
        evidence.calls, key=lambda item: (item.step_index, item.ordinal), reverse=True
    ):
        if call.tool_name != "report_goal" or call.phase is not ToolCallPhase.COMMITTED:
            continue
        result = call.result
        if result.is_error or "goal_report" not in result.data:
            continue
        try:
            report = GoalReport.model_validate(result.data["goal_report"])
        except ValidationError:
            continue
        return call, report
    return None


def _report_window_is_current(call: RunToolCallRecord, evidence: GoalRunEvidence) -> bool:
    """新工作 intent、交付输入或人工回答会使报告的结果判断过时。"""
    if any(
        item.step_index >= call.step_index and item.tool_name not in _GOAL_TOOLS
        for item in evidence.calls
    ):
        return False
    assistant_index = next(
        (
            index
            for index, message in enumerate(evidence.messages)
            if any(use.id == call.tool_call_id for use in message.tool_calls)
        ),
        None,
    )
    if assistant_index is None or any(
        is_ordinary_user(message) for message in evidence.messages[assistant_index + 1 :]
    ):
        return False
    model_sequence = next(
        (
            event.sequence
            for event in evidence.events
            if event.kind is RunEventKind.MODEL_STEP_COMMITTED
            and event.step_index == call.step_index
        ),
        None,
    )
    return model_sequence is not None and not any(
        event.kind is RunEventKind.INTERACTION_RESOLVED and event.sequence > model_sequence
        for event in evidence.events
    )


def decide_settlement(
    binding: GoalRunBinding,
    current: GoalSnapshot | None,
    evidence: GoalRunEvidence,
    *,
    now: datetime,
) -> GoalSettlement:
    """消费存储已确认的未结算终态证据，保留用户决定并生成候选结算。"""
    settled = binding.model_copy(update={"settled_at": now})
    if (
        current is None
        or current.goal_id != binding.goal_id
        or current.status is not GoalStatus.ACTIVE
    ):
        return GoalSettlement(binding=settled, goal=current, goal_changed=False)

    status, reason = current.status, current.reason
    applied_call_id = None
    stop_reason = evidence.result.run.stop_reason
    if stop_reason is not RunStopReason.COMPLETED:
        status = GoalStatus.PAUSED
        reason = GoalReason.model_construct(
            code=stop_reason.value,
            text=f"本轮执行以 {stop_reason.value} 结束，等待显式恢复",
        )
    else:
        latest = _latest_report(evidence)
        if latest is not None:
            call, report = latest
            current_report = (
                report.goal_id == current.goal_id and report.revision == current.revision
            )
            if not current_report or (
                report.decision is not GoalDecision.CONTINUE
                and not _report_window_is_current(call, evidence)
            ):
                reason = GoalReason.model_construct(
                    code="report_superseded", text="最新申报已过时，需要重新申报"
                )
            elif report.decision is not GoalDecision.CONTINUE:
                status = (
                    GoalStatus.COMPLETED
                    if report.decision is GoalDecision.COMPLETE
                    else GoalStatus.BLOCKED
                )
                reason = GoalReason.model_construct(code=report.decision.value, text=report.reason)
                applied_call_id = call.tool_call_id
        if status is GoalStatus.ACTIVE and current.rounds_started >= current.max_rounds:
            status = GoalStatus.PAUSED
            reason = GoalReason.model_construct(code="round_limit", text="目标自动执行轮数已用尽")
    changed = current.status is not status or current.reason != reason
    updated = (
        current.model_copy(
            update={
                "status": status,
                "reason": reason,
                "revision": current.revision + 1,
                "updated_at": now,
            }
        )
        if changed
        else current
    )
    if applied_call_id is not None:
        settled = settled.model_copy(update={"applied_report_call_id": applied_call_id})
    return GoalSettlement(binding=settled, goal=updated, goal_changed=changed)


__all__ = ["GoalRunEvidence", "decide_settlement"]
