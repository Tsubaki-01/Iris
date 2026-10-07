"""收集已有计量和分支结果，结束时尽力发布一次准备快照。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import replace
from types import TracebackType
from typing import Literal
from uuid import uuid4

from ..lifecycle import RunErrorInfo
from ._request_measurement import MeasuredRequest
from .diagnostics import ContextDecision, ContextPreparation, ContextStage
from .models import RuntimeActivationOutcome, RuntimeActivationResult

_logger = logging.getLogger(__name__)


class ContextPreparationRecorder:
    """单次准备的局部诊断收集器，不调用 estimator 或修改请求。"""

    def __init__(
        self,
        *,
        configuration_snapshot_id: str,
        session_id: str,
        run_id: str,
        activation_id: str,
        step_index: int,
        input_budget_tokens: int,
        trigger_tokens: int,
        publish: Callable[[ContextPreparation], None] | None,
    ) -> None:
        self.value = ContextPreparation(
            f"preparation_{uuid4().hex}",
            configuration_snapshot_id,
            session_id,
            run_id,
            activation_id,
            step_index,
            "preparing",
            input_budget_tokens,
            trigger_tokens,
        )
        self._publish = publish

    def __enter__(self) -> ContextPreparationRecorder:
        """进入准备区间。"""
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """保留真实退出原因；观察出口失败不改变运行。"""
        if error is not None:
            self.value = replace(
                self.value,
                phase="cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
                error=RunErrorInfo(
                    code="CONTEXT_PREPARATION_FAILED",
                    message=str(error) or type(error).__name__,
                    source="context",
                ),
            )
        if self._publish is not None:
            try:
                self._publish(self.value)
            except Exception:
                _logger.warning("上下文准备诊断发布失败", exc_info=True)

    def record(
        self,
        kind: str,
        before: int | None,
        after: int | None,
        *,
        outcome: Literal["applied", "skipped", "candidate", "rejected"] = "applied",
        decisions: tuple[ContextDecision, ...] = (),
        compaction_ref: str | None = None,
    ) -> None:
        """追加真实阶段，不补估算或预判未执行分支。"""
        stage = ContextStage(
            len(self.value.stages), kind, outcome, before, after, decisions, compaction_ref
        )
        self.value = replace(self.value, stages=(*self.value.stages, stage))

    def protect(self, refs: tuple[str, ...]) -> None:
        """保存本次原始历史与 required 来源的保护身份。"""
        self.value = replace(self.value, protected_refs=refs)

    def ready(self, measured: MeasuredRequest, contributions: tuple[str, ...]) -> None:
        """最终完整请求已被选中，provider 实际内容由其原观察接点采集。"""
        self.record("final", measured.input_tokens, measured.input_tokens)
        self.value = replace(
            self.value,
            phase="ready",
            final_input_tokens=measured.input_tokens,
            selected_tool_names=tuple(tool.name for tool in measured.request.tools),
            selected_contribution_keys=contributions,
        )

    def stop(self, result: RuntimeActivationResult) -> RuntimeActivationResult:
        """记录准备期间的真实停止结果，并原样返回业务结果。"""
        self.value = replace(
            self.value,
            error=result.error,
            phase="cancelled"
            if result.outcome
            in {RuntimeActivationOutcome.CANCELLED, RuntimeActivationOutcome.DEADLINE_EXCEEDED}
            else "failed",
        )
        return result
