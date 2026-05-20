"""工具执行结构化日志事件。"""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field

from ..log import logger

if TYPE_CHECKING:
    from ..tools.base import BaseTool, ToolExecutionContext, ToolResult


class ToolLogEventType(StrEnum):
    """工具日志事件类型。"""

    STARTED = "started"
    FINISHED = "finished"
    FAILED = "failed"
    PERMISSION_DENIED = "permission_denied"
    CIRCUIT_OPEN = "circuit_open"
    ARTIFACT_CREATED = "artifact_created"
    MIDDLEWARE_ERROR = "middleware_error"


class ToolLogEvent(BaseModel):
    """单条工具执行日志事件。"""

    event_type: ToolLogEventType
    timestamp: float = Field(default_factory=time.time)
    trace_id: str = ""
    session_id: str = ""
    agent_id: str = ""
    call_id: str = ""
    tool_name: str = ""
    tool_group: str = ""
    capabilities: list[str] = Field(default_factory=list)
    elapsed_ms: float | None = None
    is_error: bool = False
    error_code: str = ""
    artifact_path: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolLogSink(Protocol):
    """工具日志事件输出端协议。"""

    def emit(self, event: ToolLogEvent) -> None | Awaitable[None]:
        """输出单条工具日志事件。"""


class ToolLogRedactor:
    """生成工具日志安全摘要。"""

    @staticmethod
    def params_summary(params: dict[str, Any]) -> dict[str, Any]:
        """返回参数键名摘要，不记录参数值。"""
        return {"param_keys": sorted(params)}

    @staticmethod
    def result_summary(result: ToolResult) -> dict[str, Any]:
        """返回工具结果摘要，不记录结果正文。"""
        return {
            "content_length": len(result.model_content),
            "has_artifact": result.artifact is not None,
            "artifact_size_bytes": result.artifact.size_bytes if result.artifact else 0,
        }


class ToolLogEmitter:
    """构造并分发工具日志事件。"""

    def __init__(self, sinks: Sequence[ToolLogSink] = ()) -> None:
        """初始化工具日志分发器。

        Args:
            sinks: 工具日志事件输出端列表。
        """
        self.sinks = list(sinks)
        self.redactor = ToolLogRedactor()

    async def emit(
        self,
        event_type: ToolLogEventType,
        *,
        context: ToolExecutionContext,
        tool: BaseTool | None = None,
        tool_name: str = "",
        result: ToolResult | None = None,
        error_code: str = "",
        metadata: dict[str, Any] | None = None,
        elapsed_ms: float | None = None,
        artifact_path: str = "",
    ) -> None:
        """构造并发送单条工具日志事件。"""
        event = ToolLogEvent(
            event_type=event_type,
            trace_id=str(context.metadata.get("trace_id", "")),
            session_id=context.session_id,
            agent_id=context.agent_id,
            call_id=context.call_id,
            tool_name=tool.name if tool is not None else tool_name or context.tool_name,
            tool_group=tool.definition.group if tool is not None else "",
            capabilities=(
                sorted(capability.value for capability in tool.definition.capabilities)
                if tool is not None
                else []
            ),
            elapsed_ms=elapsed_ms,
            is_error=event_type
            in {
                ToolLogEventType.FAILED,
                ToolLogEventType.PERMISSION_DENIED,
                ToolLogEventType.CIRCUIT_OPEN,
                ToolLogEventType.MIDDLEWARE_ERROR,
            },
            error_code=error_code,
            artifact_path=artifact_path,
            metadata=self._metadata(result=result, metadata=metadata),
        )
        for sink in self.sinks:
            try:
                emitted = sink.emit(event)
                if inspect.isawaitable(emitted):
                    await emitted
            except Exception as exc:
                logger.warning("工具日志 sink 输出失败: {}", exc)

    def _metadata(
        self,
        *,
        result: ToolResult | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """合并调用方摘要与结果摘要。"""
        safe_metadata = dict(metadata or {})
        if result is not None:
            safe_metadata.update(self.redactor.result_summary(result))
        return safe_metadata


class LoguruToolLogSink:
    """将工具日志事件写入 Iris 的 Loguru logger。"""

    def __init__(self, level: str = "INFO") -> None:
        """初始化 Loguru 输出端。

        Args:
            level: Loguru 日志等级。
        """
        self.level = level

    def emit(self, event: ToolLogEvent) -> None:
        """输出单条工具日志事件。"""
        logger.bind(component="tools", tool_log=event.model_dump(mode="json")).log(
            self.level,
            "tool_log {} {} {}",
            event.event_type.value,
            event.tool_name,
            event.call_id,
        )
