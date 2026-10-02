"""命令 Hook 的事件 stdin、单个 JSON 输出与进程控制事实适配。"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import ValidationError

from ..command.models import (
    CommandOutcome,
    CommandRequest,
    CommandScope,
    CommandStatus,
    CommandStopSlot,
    ShellCommand,
)
from ..command.service import CommandBinding
from ..exceptions import (
    IrisCancellationRequestedError,
    IrisCommandCleanupError,
    IrisHookError,
    IrisHookProtocolError,
    IrisToolOutcomeUnknownError,
)
from ._dispatch_types import HookControl, HookControlOrigin, HookInvocationOutcome
from .models import HookEvent, HookResult, ToolAfterResult, ToolBeforeResult, event_to_dict

if TYPE_CHECKING:
    from ..tools.base import CancellationSignal

logger = logging.getLogger(__name__)
_CONTROL_ERRORS = (
    asyncio.CancelledError,
    IrisCancellationRequestedError,
    IrisToolOutcomeUnknownError,
    IrisCommandCleanupError,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CommandHookAdapter:
    """持有已装配的 Agent 命令依赖；不通过工具入口，不拥有 Run 状态。"""

    binding: CommandBinding
    workspace: Path
    command: str
    timeout_seconds: float = 10.0

    async def __call__(
        self, event: HookEvent, *, cancellation: CancellationSignal | None
    ) -> HookInvocationOutcome:
        """执行一次命令；先交接控制与收据，再解析完整 stdout。"""
        call_id = f"hook_{uuid4().hex}"
        slot = CommandStopSlot()
        request = CommandRequest(
            call_id=call_id,
            payload=ShellCommand(self.command),
            cwd=self.workspace,
            timeout_seconds=self.timeout_seconds,
            stdin=json.dumps(event_to_dict(event), ensure_ascii=False, allow_nan=False).encode(
                "utf-8"
            ),
        )
        scope = CommandScope(event.run_id or "", event.session_id)
        try:
            outcome = await self.binding.service.execute(scope, request)
        except _CONTROL_ERRORS as error:
            return HookInvocationOutcome(control=_error_control(error, slot, call_id, cancellation))
        slot.record(status=outcome.status, receipt=outcome.stop_receipt)
        if outcome.stderr:
            logger.info("Hook 命令 stderr [%s]：%s", call_id, outcome.stderr)

        control = _pending_control(slot, call_id, cancellation)
        if control is None and outcome.status is CommandStatus.CANCELLED:
            control = HookControl(
                "task_cancelled", asyncio.CancelledError("Hook 命令已取消"), slot, call_id
            )
        if control is None and (
            outcome.status is CommandStatus.ENVIRONMENT_INTERRUPTED
            or outcome.status is CommandStatus.EXITED
            and outcome.exit_code == 0
            and slot.receipt is not None
        ):
            # Docker 前台可能恰已退出，但该调用仍参加了共享 stop。
            control = HookControl(
                "environment_interrupted",
                IrisCancellationRequestedError("Hook 命令参与了命令环境停止"),
                slot,
                call_id,
            )
        if control is not None:
            return HookInvocationOutcome(control=control)

        # 只有普通可继续的失败走这里；没有收口的命令事实不能交给下一处理器。
        if slot.receipt is not None:
            try:
                await self.binding.service.wait_drained(slot.receipt)
            except _CONTROL_ERRORS as error:
                return HookInvocationOutcome(
                    control=_error_control(error, slot, call_id, cancellation)
                )
            slot.receipt = None
            control = _pending_control(slot, call_id, cancellation)
            if control is not None:
                return HookInvocationOutcome(control=control)

        if outcome.status is CommandStatus.TIMED_OUT:
            raise IrisHookError(f"Hook 命令超过 {self.timeout_seconds:g} 秒期限")
        if outcome.exit_code != 0:
            raise IrisHookError(f"Hook 命令非零退出：{outcome.exit_code}")
        return HookInvocationOutcome(result=_parse_output(event, outcome))


def _pending_control(
    slot: CommandStopSlot,
    call_id: str,
    cancellation: CancellationSignal | None,
) -> HookControl | None:
    """命令服务可能消费取消后返回成功，因此独立检查 live 取消来源。"""
    if cancellation is not None and cancellation.requested:
        return HookControl(
            "run_cancelled", IrisCancellationRequestedError("Hook 响应 Run 取消"), slot, call_id
        )
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        return HookControl(
            "task_cancelled", asyncio.CancelledError("Hook 调用 task 已取消"), slot, call_id
        )
    return None


def _error_control(
    error: BaseException,
    slot: CommandStopSlot,
    call_id: str,
    cancellation: CancellationSignal | None,
) -> HookControl:
    """将原异常及命令事实合入本次唯一槽，取消与 unknown 可同时存在。"""
    unknown: IrisToolOutcomeUnknownError | None = None
    origin: HookControlOrigin
    if isinstance(error, IrisToolOutcomeUnknownError):
        slot.record(receipt=error.stop_receipt)
        unknown = error
        origin = "unknown"
    elif isinstance(error, IrisCommandCleanupError):
        slot.record(cleanup_error=error)
        if error.command_outcome is not None:
            slot.record(
                status=error.command_outcome.status,
                receipt=error.command_outcome.stop_receipt,
            )
        origin = "cleanup"
    elif isinstance(error, IrisCancellationRequestedError):
        origin = "run_cancelled"
    else:
        origin = "task_cancelled"
    pending = _pending_control(slot, call_id, cancellation)
    if pending is not None and not isinstance(error, asyncio.CancelledError):
        origin, error = pending.origin, pending.error
    return HookControl(origin, error, slot, call_id, unknown_error=unknown)


def _parse_output(event: HookEvent, outcome: CommandOutcome) -> HookResult:
    """在唯一 raw stdout 边界解析一个完整对象，再校验对应事件结果。"""
    stats = outcome.output_stats
    if stats.stdout_bytes > stats.stdout_retained_bytes or stats.truncation_reasons - {
        "byte_limit"
    }:
        raise IrisHookProtocolError("Hook 命令 stdout 或输出采集不完整")
    try:
        payload = json.loads(outcome.stdout)
    except json.JSONDecodeError as error:
        raise IrisHookProtocolError("Hook 命令 stdout 必须是单个 JSON object") from error
    if not isinstance(payload, dict):
        raise IrisHookProtocolError("Hook 命令 stdout 必须是 JSON object")
    if not payload:
        return None
    try:
        if event.event == "tool.before":
            return ToolBeforeResult.model_validate(payload)
        if event.event == "tool.after":
            return ToolAfterResult.model_validate(payload)
    except ValidationError as error:
        raise IrisHookProtocolError(f"{event.event} 命令返回字段无效：{error}") from error
    raise IrisHookProtocolError(f"{event.event} 命令只能返回空 JSON object")
