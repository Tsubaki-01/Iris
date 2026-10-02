"""MCP 工具的输入边界、SDK 调用和有序文本/图片结果投影。"""

from __future__ import annotations

import base64
import json
from typing import Any, cast

from jsonschema import ValidationError
from mcp import types
from pydantic import BaseModel

from ..exceptions import (
    IrisImageError,
    IrisMCPCallError,
    IrisMCPToolError,
    IrisToolExecutionError,
    IrisToolOutcomeUnknownError,
    IrisToolValidationError,
)
from ..message import DataBlock, TextBlock
from ..tools._io import run_tool_io
from ..tools.artifacts import artifact_store_for
from ..tools.base import BaseTool, ToolErrorInfo, ToolExecutionContext, ToolResult
from ..tools.images import import_tool_image
from .connection import MCPConnection
from .models import MCPToolDescriptor


class MCPTool(BaseTool):
    """使用当前 executor 的普通 BaseTool，不另建权限、claim 或取消路径。"""

    def __init__(self, descriptor: MCPToolDescriptor, connection: MCPConnection) -> None:
        self.descriptor = descriptor
        self.definition = descriptor.definition
        self.connection = connection

    def validate_input(self, params: dict[str, Any]) -> dict[str, Any]:
        """在 LLM 参数边界校验一次，不做类型 coercion。"""
        try:
            self.descriptor.input_validator.validate(params)
        except ValidationError as error:
            raise IrisToolValidationError(error.message) from error
        return params

    def is_read_only(self, params: dict[str, Any]) -> bool:
        """消费 catalog 已确定的本地信任结论。"""
        return self.descriptor.trusted_read_only

    def is_concurrency_safe(self, params: dict[str, Any]) -> bool:
        """首版 MCP 不进入 runtime 的并行工具窗口。"""
        return False

    async def arun(
        self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
    ) -> ToolResult:
        """调用已发布的原始 wire 名，保留取消与 unknown 控制流。

        Raises:
            IrisToolOutcomeUnknownError: SDK 无确定结果且非受信只读，或 Iris 调用期限耗尽。
        """
        try:
            result = await self.connection.call_tool(
                self.descriptor.wire_name, cast(dict[str, Any], params)
            )
        except IrisMCPToolError as error:
            return self._error(context, error.code, error.message)
        except IrisMCPCallError as error:
            if self.descriptor.trusted_read_only:
                return self._error(context, "MCP_CALL_FAILED", "MCP 调用未取得可用结果")
            raise IrisToolOutcomeUnknownError(error.message, **error.context) from error
        except TimeoutError as error:
            raise IrisToolOutcomeUnknownError("MCP 调用期限耗尽，结果无法确认") from error
        try:
            return await run_tool_io(lambda: self._project_result(result, context))
        except IrisImageError as error:
            return self._error(context, "IMAGE_ERROR", error.message)
        except IrisToolExecutionError as error:
            return self._error(context, "ARTIFACT_ERROR", error.message)

    def _error(self, context: ToolExecutionContext, code: str, message: str) -> ToolResult:
        """让可操作的错误说明进入模型实际读取的 error.message。"""
        return ToolResult(
            tool_use_id=context.call_id,
            tool_name=self.name,
            is_error=True,
            error=ToolErrorInfo(code=code, message=message),
        )

    def _project_result(
        self, result: types.CallToolResult, context: ToolExecutionContext
    ) -> ToolResult:
        """保留有序文字与本地图片，完整 SDK JSON 交给现有 artifact store。"""
        parts: list[DataBlock] = []
        preserve = bool(result.model_extra) or result.meta is not None
        for block in result.content:
            if isinstance(block, types.TextContent):
                parts.append(TextBlock(text=block.text))
                preserve |= (
                    bool(block.model_extra)
                    or block.meta is not None
                    or block.annotations is not None
                )
            elif isinstance(block, types.ImageContent):
                try:
                    data = base64.b64decode(block.data, validate=True)
                except ValueError as error:
                    raise IrisImageError("MCP 图片不是合法 base64") from error
                parts.append(import_tool_image(data, context))
                preserve = True
            elif isinstance(block, types.ResourceLink):
                parts.append(
                    TextBlock(
                        text=f"Resource: {block.name} {block.uri} ({block.mime_type or 'unknown'})"
                    )
                )
                preserve = True
            elif isinstance(block, types.EmbeddedResource):
                resource = block.resource
                text = (
                    resource.text if isinstance(resource, types.TextResourceContents) else "binary"
                )
                parts.append(
                    TextBlock(
                        text=f"Resource: {resource.uri} ({resource.mime_type or 'unknown'})\n{text}"
                    )
                )
                preserve = True
            else:
                parts.append(TextBlock(text=f"{block.type}: {block.mime_type}，完整内容见结果文件"))
                preserve = True
        if result.structured_content is not None or "structured_content" in result.model_fields_set:
            parts.append(TextBlock(text=json.dumps(result.structured_content, ensure_ascii=False)))
            preserve = True
        text = "\n".join(part.text for part in parts if isinstance(part, TextBlock))
        if result.is_error and not text:
            text = "MCP 工具返回业务错误"
        projected = ToolResult(
            tool_use_id=context.call_id,
            tool_name=self.name,
            content=parts,
            is_error=result.is_error,
            error=ToolErrorInfo(code="MCP_TOOL_ERROR", message=text) if result.is_error else None,
        )
        store = artifact_store_for(context, preview_chars=self.definition.preview_chars)
        if preserve or len(projected.model_content) > self.definition.max_result_chars:
            artifact = store.persist_json(
                context.call_id,
                result.model_dump(mode="json", by_alias=True),
                preview=text[: self.definition.preview_chars],
            )
            note = f"[完整 MCP 结果：{artifact.path}]"
            projected.artifact = artifact
            projected.content.append(TextBlock(text=f"\n{note}" if text else note))
            if projected.error is not None:
                projected.error = projected.error.model_copy(
                    update={"message": f"{text}\n\n{note}"}
                )
        return projected


__all__ = ["MCPTool"]
