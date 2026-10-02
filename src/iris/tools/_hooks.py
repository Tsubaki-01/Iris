"""工具阶段的 Hook 停止事实交接与已知 body 后的命令收口。"""

from __future__ import annotations

import logging

from ..command.models import CommandScope
from ..command.service import CommandBinding
from ..exceptions import IrisCommandCleanupError
from ..hooks._dispatch_types import HookControl
from .base import ToolExecutionContext

logger = logging.getLogger(__name__)


async def drain_hook_commands(
    context: ToolExecutionContext,
    binding: CommandBinding | None,
    *,
    stop_if_missing: bool = False,
) -> None:
    """消费当前唯一收据；未知附加动作没有收据时，先停止其命令 scope。"""
    slot = context.command_stop_slot
    if slot.cleanup_error is not None:
        raise slot.cleanup_error
    if binding is None:
        if slot.receipt is not None:
            error = IrisCommandCleanupError("工具 Hook 有待收口的命令收据，但未配置命令依赖")
            slot.record(cleanup_error=error)
            raise error
        return
    try:
        if slot.receipt is None and stop_if_missing:
            operation = binding.service.stop(
                CommandScope(str(context.metadata.get("run_id", "")), context.session_id)
            )
            slot.record(receipt=await operation.wait_stopped())
        if slot.receipt is not None:
            await binding.service.wait_drained(slot.receipt)
            slot.receipt = None
    except IrisCommandCleanupError as error:
        slot.record(cleanup_error=error)
        raise


async def consume_tool_hook_control(
    control: HookControl,
    context: ToolExecutionContext,
    binding: CommandBinding | None,
    *,
    body_known: bool,
) -> None:
    """before 传播原控制；after 保留已知 body，未知附加动作先收口再决定停止。"""
    incoming = control.stop_slot
    if incoming is not None:
        context.command_stop_slot.record(
            status=incoming.status,
            receipt=incoming.receipt,
            cleanup_error=incoming.cleanup_error,
        )
    if control.unknown_error is not None:
        if not body_known:
            raise control.unknown_error
        try:
            await drain_hook_commands(context, binding, stop_if_missing=True)
        except IrisCommandCleanupError:
            if control.origin in {"run_cancelled", "task_cancelled", "environment_interrupted"}:
                raise control.error from None
            raise
        logger.warning("tool.after 附加动作结果未知，命令已收口：%s", control.unknown_error)
        if control.origin == "unknown":
            return
    raise control.error
