"""工具最终结果在全部 middleware 后统一生成 artifact。"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from iris.message import TextBlock, ToolUseBlock
from iris.tools import (
    BaseTool,
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutor,
    ToolMiddleware,
    ToolRegistry,
    ToolResult,
)


@pytest.mark.asyncio
async def test_after_call_expansion_is_persisted_once(tmp_path: Path) -> None:
    """正常扩展结果的 hook 也遵守工具声明的长度阈值。"""

    class ExpandResult(ToolMiddleware):
        """为结果补充描述。"""

        async def after_call(
            self, tool: BaseTool, result: ToolResult, context: ToolExecutionContext
        ) -> ToolResult:
            """补充正文，保持工具 identity。"""
            return result.model_copy(update={"content": [TextBlock(text="expanded result")]})

    class SmallTool(BaseTool):
        """声明短结果阈值的正常工具。"""

        definition = ToolDefinition(
            name="small",
            description="短结果",
            input_schema={"type": "object", "properties": {}},
            max_result_chars=3,
        )

        async def arun(
            self, params: BaseModel | dict[str, Any], context: ToolExecutionContext
        ) -> ToolResult:
            """返回阈值内的原始结果。"""
            return ToolResult(
                tool_use_id=context.call_id, tool_name=self.name, content=[TextBlock(text="ok")]
            )

    registry = ToolRegistry()
    registry.register(SmallTool())
    result = await ToolExecutor(
        registry, middleware=[ExpandResult()], artifact_preview_chars=2
    ).execute_one(
        ToolUseBlock(id="call", name="small", input={}),
        ToolExecutionContext(workspace_root=tmp_path),
    )
    assert not result.is_error
    assert result.artifact is not None
    assert result.artifact.path.read_text(encoding="utf-8") == "expanded result"
    assert result.artifact.preview == "ex"
