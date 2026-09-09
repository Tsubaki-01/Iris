"""两种 store 共用的 Sub Agent 操作边界，不拥有运行控制流。"""

from datetime import datetime

from ..exceptions import IrisRunConflictError, IrisRunStateError
from ..hitl.models import HumanInteraction, InteractionStatus, SubagentExpiryOwner
from ..lifecycle.models import (
    RunPhase,
    RunRecord,
    RunToolCallRecord,
    SubagentRunLink,
    ToolCallPhase,
)
from ..lifecycle.store import FinalizeSubagentResult


def validate_parent_tool(
    run: RunRecord,
    tool: RunToolCallRecord,
    *,
    activation_id: str | None,
    tool_version: int,
    now: datetime,
) -> None:
    """检查本次 parent mutation 的 phase、fence 与受影响工具。"""
    expected_phase = RunPhase.WAITING if activation_id is None else RunPhase.ACTIVE
    if run.phase is not expected_phase:
        raise IrisRunStateError("Sub Agent parent phase 不匹配", run_id=run.run_id)
    if run.current_activation_id != activation_id:
        raise IrisRunConflictError("activation fence 已变化", run_id=run.run_id)
    if run.cancellation_requested_at is not None:
        raise IrisRunStateError("parent 已请求取消", run_id=run.run_id)
    if run.options.limits.deadline_at is not None and now >= run.options.limits.deadline_at:
        raise IrisRunStateError("parent deadline 已到期", run_id=run.run_id)
    if tool.version != tool_version:
        raise IrisRunConflictError("tool call version 已变化", tool_call_id=tool.tool_call_id)
    if tool.phase is not ToolCallPhase.PREPARED or tool.tool_name != "subagent":
        raise IrisRunStateError("Sub Agent mutation 要求 prepared subagent tool")


def validate_proxy_binding(
    run: RunRecord,
    tool: RunToolCallRecord,
    link: SubagentRunLink,
    child: RunRecord,
    child_interaction: HumanInteraction,
    proxy: HumanInteraction,
) -> None:
    """核对新 proxy 与唯一 child 当前等待、父工具 subject 的关联。"""
    if child_interaction.status is not InteractionStatus.PENDING:
        raise IrisRunStateError("child 必须等待 pending interaction")
    origin = proxy.request.subagent_origin
    if (
        origin is None
        or origin.child_run_id != link.child_run_id
        or origin.child_interaction_id != child.pending_interaction_id
        or origin.child_interaction_id != child_interaction.interaction_id
    ):
        raise IrisRunConflictError("proxy origin 与 linked child interaction 不匹配")
    subject = proxy.request.tool_call
    if (
        proxy.run_id != run.run_id
        or proxy.session_id != run.session_id
        or proxy.tool_call_id != tool.tool_call_id
        or proxy.step_index != tool.step_index
        or subject.tool_call_id != tool.tool_call_id
        or subject.tool_name != tool.tool_name
        or subject.arguments != tool.arguments
        or subject.fingerprint != tool.fingerprint
    ):
        raise IrisRunConflictError("proxy 与 parent prepared tool subject 不匹配")
    if proxy.status is not InteractionStatus.PENDING:
        raise IrisRunStateError("新 proxy 必须 pending")


def validate_current_proxy(
    run: RunRecord,
    tool: RunToolCallRecord,
    link: SubagentRunLink,
    proxy: HumanInteraction,
    expected_id: str | None,
) -> None:
    """核对当前 parent proxy 的 durable identity。"""
    origin = proxy.request.subagent_origin
    if (
        expected_id != run.pending_interaction_id
        or proxy.interaction_id != expected_id
        or tool.interaction_id != expected_id
        or origin is None
        or origin.child_run_id != link.child_run_id
    ):
        raise IrisRunConflictError("当前 subagent proxy identity 不匹配")


def validate_final_result(
    command: FinalizeSubagentResult, tool: RunToolCallRecord, child: RunRecord
) -> None:
    """Child terminal 是提交唯一 parent result 的前提。"""
    if child.phase is not RunPhase.TERMINAL:
        raise IrisRunStateError("child terminal 前不能提交 parent result")
    if (
        command.result.tool_use_id != tool.tool_call_id
        or command.result.tool_name != tool.tool_name
    ):
        raise IrisRunConflictError("subagent result identity 不匹配")
    if command.parent_activation_id is not None:
        if command.proxy_interaction_id is not None or command.resume_activation_id is not None:
            raise IrisRunStateError("ACTIVE finalize 不能包含 proxy/resume activation")
    elif command.resume_activation_id is None:
        raise IrisRunStateError("WAITING finalize 必须包含 fresh RESUME activation")


def validate_final_proxy(proxy: HumanInteraction, now: datetime) -> None:
    """回答完成或 child-owned 到期时允许关闭当前 proxy。"""
    if proxy.status is InteractionStatus.RESOLVED:
        return
    origin = proxy.request.subagent_origin
    if (
        proxy.status is InteractionStatus.PENDING
        and proxy.expires_at is not None
        and now >= proxy.expires_at
        and (
            origin.expiry_owner == SubagentExpiryOwner.CHILD_INTERACTION_EXPIRY
            or origin.expiry_owner == SubagentExpiryOwner.CHILD_EFFECTIVE_DEADLINE
            or origin.expiry_owner == SubagentExpiryOwner.OUTER_TOOL_TIMEOUT
        )
    ):
        return
    raise IrisRunStateError("proxy 尚未回答或未到 child-owned 期限")
